#include <algorithm>
#include <array>
#include <iomanip>
#include "cxxopts.hpp"
#include "disassembler/llvm_disasm.hpp"
#include "gpu/pipeline.hpp"
#include "gpu/pipeline_ats.hpp"
#include "gpu/pipeline_execute.hpp"
#include "gpu/pipeline_instr_fetch.hpp"
#include "gpu/pipeline_op_fetch.hpp"
#include "gpu/pipeline_op_latch.hpp"
#include "gpu/pipeline_warp_scheduler.hpp"
#include "gpu/pipeline_writeback.hpp"
#include "host/host_register_file.hpp"
#include "images/bmp.hpp"
#include "mem/mem_coalesce.hpp"
#include "mem/mem_data.hpp"
#include "mem/mem_instr.hpp"
#include "trace/trace.hpp"
#include "utils.hpp"

// Initialize pipeline (CPU is modelled as 1x1 GPU for simplicity)
Pipeline *initialize_pipeline(InstructionMemory *im, CoalescingUnit *cu,
                              RegisterFile *rf, LLVMDisassembler *disasm,
                              HostGPUControl *gpu_controller, bool is_cpu,
                              Tracer *instr_tracer = nullptr) {
  Pipeline *p = new Pipeline();

  if (is_cpu) {
    p->add_stage<WarpScheduler>(1, 1, im->get_base_addr(), cu);
  } else {
    p->add_stage<WarpScheduler>(NUM_LANES, NUM_WARPS, im->get_base_addr(), cu, false);
  }
  p->add_stage<ActiveThreadSelection>();
  p->add_stage<InstructionFetch>(im, disasm);                // Stage 2
  p->add_stage<OperandFetch>();                              // Stage 3
  p->add_stage<OperandLatch>();                              // Stage 4
  p->add_stage<ExecuteSuspend>(cu, rf, im->get_max_addr(), disasm,
                               gpu_controller);              // Stage 5
  p->add_stage<WritebackResume>(cu, rf, is_cpu);            // Stage 6

  std::shared_ptr<WarpScheduler> warp_scheduler_stage =
      std::dynamic_pointer_cast<WarpScheduler>(p->get_stage(0));
  std::shared_ptr<ExecuteSuspend> execute_stage =
      std::dynamic_pointer_cast<ExecuteSuspend>(p->get_stage(5));
  std::shared_ptr<WritebackResume> writeback_stage =
      std::dynamic_pointer_cast<WritebackResume>(p->get_stage(6));

  if (instr_tracer) {
    execute_stage->set_instr_tracer(instr_tracer);
    writeback_stage->set_instr_tracer(instr_tracer);
  }
  
  auto insert_warp_callback = [ws = warp_scheduler_stage](Warp *warp) {
    ws->insert_warp(warp);
  };
  execute_stage->insert_warp = insert_warp_callback;

  execute_stage->insert_warp_retry = [ws = warp_scheduler_stage](Warp *warp) {
    ws->insert_warp_retry(warp);
  };

  if (!is_cpu) {
    execute_stage->notify_warp_terminated = [pipeline = p]() {
      pipeline->notify_warp_terminated();
    };
  }

  // Set up warp insertion for writeback stage
  writeback_stage->insert_warp = insert_warp_callback;

  writeback_stage->insert_warp_with_susp_delay = [ws = warp_scheduler_stage](Warp *warp) {
    ws->insert_warp_retry(warp);
  };

  // Initialize latches (7 stages = 7 latches)
  // Note: Latches must persist for the lifetime of the pipeline, so we use heap allocation
  // The Pipeline should ideally own these, but for now we allocate them here
  PipelineLatch *latches[7];
  for (int i = 0; i < 7; i++) {
    latches[i] = new PipelineLatch();
  }

  // Connect latches in a circular pattern (stage N output -> stage N+1 input)
  p->get_stage(0)->set_latches(latches[6], latches[0]);
  p->get_stage(1)->set_latches(latches[0], latches[1]);
  p->get_stage(2)->set_latches(latches[1], latches[2]);
  p->get_stage(3)->set_latches(latches[2], latches[3]);
  p->get_stage(4)->set_latches(latches[3], latches[4]);
  p->get_stage(5)->set_latches(latches[4], latches[5]);
  p->get_stage(6)->set_latches(latches[5], latches[6]);

  return p;
}

int main(int argc, char *argv[]) {
  srand((unsigned) time(NULL));

  cxxopts::Options options("RISCVGpuSim",
                           "A software simulator for a RISC-V GPU");

  options.add_options()("filename", "Input filename",
                        cxxopts::value<std::string>())(
      "d,debug", "Turn on debugging logs")(
      "c,cpu-debug", "Turn on CPU debugging logs (requires --debug enabled)")(
      "r,regdump", "Dump the register values after each writeback stage")(
      "s,statsonly", "Do not print anything aside from the final stats")(
      "framebuffer-addr", "Base address of framebuffer in memory (hex, e.g. 0x80001000)",
                          cxxopts::value<std::string>())(
      "framebuffer-width", "Width of framebuffer in pixels",
                           cxxopts::value<uint64_t>()->default_value("64"))(
      "framebuffer-height", "Height of framebuffer in pixels",
                            cxxopts::value<uint64_t>()->default_value("64"))(
      "framebuffer-output", "Output BMP filename for framebuffer",
                            cxxopts::value<std::string>()->default_value("framebuffer.bmp"))(
      "trace-file", "Enable coalescing unit address tracing (specify filename, e.g. --trace-file=trace.log)",
                            cxxopts::value<std::string>())(
      "trace-coalesce", "Write coalesce (MEM_REQ_ISSUE, DRAM_REQ_ISSUE) to trace-file; by default coalesce logs are hidden")(
      "instr-trace-file", "Trace all GPU instruction execution (specify filename, e.g. --instr-trace-file=instr.log)",
                            cxxopts::value<std::string>())(
      "dram-trace-file", "Trace DRAM/SRAM accesses exiting CU pipeline (specify filename, e.g. --dram-trace-file=dram.log)",
                            cxxopts::value<std::string>())(
      "q,quick", "Disable buffering for outputting earlier than simulation end")(
      "warp-scheduler", "Choose a warp scheduler from 'baseline' or 'random'",
                            cxxopts::value<std::string>())(
      "h,help", "Show help");
  options.parse_positional({"filename"});
  options.positional_help("<Input File>");
  auto result = options.parse(argc, argv);

  if (result.count("help") || !result.count("filename")) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  auto &config = Config::instance();
  config.setDebug(result.count("debug") > 0);
  config.setCPUDebug(result.count("cpu-debug") > 0);
  config.setRegisterDump(result.count("regdump") > 0);
  config.setStatsOnly(result.count("statsonly") > 0);
  config.setQuick(result.count("quick") > 0);
  if (result.count("warp-scheduler") > 0) {
    std::string value = result["warp-scheduler"].as<std::string>();
    if (value == "random") {
      config.setWarpScheduler(RANDOM);
    }
  }

  std::string filename = result["filename"].as<std::string>();

  // Initialize LLVM machine code decoding (RISC-V only)
  LLVMInitializeRISCVTargetInfo();
  LLVMInitializeRISCVTargetMC();
  LLVMInitializeRISCVDisassembler();

  std::string target_id = "riscv64-unknown-elf";
  std::string cpu = "generic-rv64";
  std::string features = "+m,+a,+zfinx";
  LLVMDisassembler disasm(target_id, cpu, features);

  debug_log("Loading ELF file...");
  parse_output out;
  parse_error parse_err = parse_binary(filename, disasm, &out);
  if (parse_err != PARSE_SUCCESS) {
    std::cout << "Failed to load/parse file: " << filename << std::endl;
    return 1;
  }
  debug_log("Successfully loaded ELF file!");

  InstructionMemory tcim(&out);
  debug_log("Instruction memory has base_addr " +
            std::to_string(tcim.get_base_addr()));

  DataMemory scratchpad_mem;
  
  // Initialize data memory with sections from ELF file (rodata, data, etc.)
  for (const auto& section : out.data_sections) {
    uint64_t addr = section.first;
    const std::vector<uint8_t>& data = section.second;
    for (size_t i = 0; i < data.size(); i++) {
      // Store each byte individually (store function handles little-endian, but for 1 byte it's fine)
      scratchpad_mem.store(addr + i, 1, static_cast<uint64_t>(data[i]));
    }
    debug_log("Loaded data section at 0x" + 
              std::to_string(addr) + " (" + 
              std::to_string(data.size()) + " bytes)");
  }
  
  debug_log("Instantiated memory scratchpad for the SM");
  
  // Set up tracing if requested (coalesce logs hidden unless --trace-coalesce)
  std::string trace_file;
  const std::string *trace_file_ptr = nullptr;
  if (result.count("trace-file")) {
    trace_file = result["trace-file"].as<std::string>();
    if (result.count("trace-coalesce")) {
      trace_file_ptr = &trace_file;
      debug_log("Coalescing unit tracing enabled: " + trace_file);
    }
  }

  std::unique_ptr<Tracer> instr_tracer;
  if (result.count("instr-trace-file")) {
    instr_tracer = std::make_unique<Tracer>(result["instr-trace-file"].as<std::string>());
    debug_log("Instruction tracing (warp 1 thread 1) enabled");
  }
  
  CoalescingUnit cu(&scratchpad_mem, trace_file_ptr);
  if (instr_tracer) {
    cu.set_instr_tracer(instr_tracer.get());
  }
  std::unique_ptr<std::ofstream> dram_trace_file;
  if (result.count("dram-trace-file")) {
    dram_trace_file = std::make_unique<std::ofstream>(result["dram-trace-file"].as<std::string>());
    *dram_trace_file << "cycle,warp,type,beats,groups,dest,addr\n";
    cu.set_dram_trace(dram_trace_file.get());
    debug_log("DRAM trace enabled");
  }
  debug_log("Instantiated memory coalescing unit");

  RegisterFile rf(NUM_REGISTERS, NUM_LANES);
  HostRegisterFile hrf(&rf, NUM_REGISTERS);
  debug_log("Register file instantiated with " +
            std::to_string(NUM_REGISTERS) + " registers");

  // Initialization
  HostGPUControl gpu_controller;
  Pipeline *gpu_pipeline =
      initialize_pipeline(&tcim, &cu, &rf, &disasm, &gpu_controller, false,
                          instr_tracer ? instr_tracer.get() : nullptr);
  Pipeline *cpu_pipeline =
      initialize_pipeline(&tcim, &cu, &hrf, &disasm, &gpu_controller, true);

  gpu_pipeline->set_debug(true);
  cpu_pipeline->set_debug(config.isCPUDebug());

  gpu_controller.set_scheduler(
      std::dynamic_pointer_cast<WarpScheduler>(gpu_pipeline->get_stage(0)));
  gpu_controller.set_pipeline(gpu_pipeline);
  gpu_controller.set_coalescing_unit(&cu);

  // Execute the threads
  while (cpu_pipeline->has_active_stages() ||
         gpu_pipeline->has_active_stages() ||
         gpu_pipeline->is_pipeline_active()) {
    
    gpu_pipeline->apply_deferred_deactivation();

    GPUStatisticsManager::instance().set_gpu_pipeline_active(gpu_pipeline->is_pipeline_active());

    cpu_pipeline->execute();
    gpu_pipeline->execute();
    cu.tick();

    GPUStatisticsManager::instance().tick_instr_pipeline();

    if (gpu_pipeline->is_pipeline_active()) {
      GPUStatisticsManager::instance().increment_gpu_cycles();
    }
  }

  std::string output = gpu_controller.get_buffer();
  bool statsOnly = config.isStatsOnly();
  if (!config.isQuick()) {
    if (!statsOnly) {
      std::cout << "[Output]" << std::endl;
    }
    std::cout << output;
  }

  // Render framebuffer if address was specified
  if (result.count("framebuffer-addr")) {
    std::string addr_str = result["framebuffer-addr"].as<std::string>();
    uint64_t fb_addr = std::stoull(addr_str, nullptr, 0);  // Handles 0x prefix
    uint64_t fb_width = result["framebuffer-width"].as<uint64_t>();
    uint64_t fb_height = result["framebuffer-height"].as<uint64_t>();
    std::string fb_output = result["framebuffer-output"].as<std::string>();
    
    if (!statsOnly) {
      std::cout << "[Framebuffer]" << std::endl;
      std::cout << "Rendering framebuffer from address 0x" << std::hex << fb_addr << std::dec << std::endl;
      std::cout << "Dimensions: " << fb_width << "x" << fb_height << std::endl;
      std::cout << "Output: " << fb_output << std::endl;
    }
    
    render_framebuffer(scratchpad_mem, fb_addr, fb_width, fb_height, fb_output);
    
    if (!statsOnly) {
      std::cout << "Framebuffer rendered successfully!" << std::endl;
    }
  }

  delete cpu_pipeline;
  delete gpu_pipeline;

  return 0;
}