#include <algorithm>
#include <array>
#include <iomanip>
#include "cxxopts.hpp"
#include "custom_instrs.hpp"
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
                              Tracer *instr_tracer = nullptr,
                              const std::vector<CustomInstrEntry> *custom_instrs = nullptr) {
  Pipeline *p = new Pipeline();

  // Construct stages (matching SIMTight's 7-stage pipeline)
  if (is_cpu) {
    p->add_stage<WarpScheduler>(1, 1, im->get_base_addr(), cu);  // Stage 0: CPU = 1x1
  } else {
    p->add_stage<WarpScheduler>(NUM_LANES, NUM_WARPS, im->get_base_addr(), cu, false);  // Stage 0: GPU
  }
  p->add_stage<ActiveThreadSelection>();                     // Stage 1
  p->add_stage<InstructionFetch>(im, disasm);                // Stage 2
  p->add_stage<OperandFetch>();                              // Stage 3
  p->add_stage<OperandLatch>();                              // Stage 4
  p->add_stage<ExecuteSuspend>(cu, rf, im->get_max_addr(), disasm,
                               gpu_controller, custom_instrs); // Stage 5
  p->add_stage<WritebackResume>(cu, rf, is_cpu);            // Stage 6

  std::shared_ptr<WarpScheduler> warp_scheduler_stage =
      std::dynamic_pointer_cast<WarpScheduler>(p->get_stage(0));
  std::shared_ptr<ExecuteSuspend> execute_stage =
      std::dynamic_pointer_cast<ExecuteSuspend>(p->get_stage(5));
  std::shared_ptr<WritebackResume> writeback_stage =
      std::dynamic_pointer_cast<WritebackResume>(p->get_stage(6));

  if (instr_tracer) {
    execute_stage->set_instr_tracer(instr_tracer);
  }
  
  // Set up warp insertion callback (used by both execute and writeback stages)
  auto insert_warp_callback = [ws = warp_scheduler_stage](Warp *warp) {
    ws->insert_warp(warp);
  };
  execute_stage->insert_warp = insert_warp_callback;
  
  // Set up warp insertion for writeback stage
  writeback_stage->insert_warp = insert_warp_callback;

  // Initialize latches (7 stages = 7 latches)
  // Note: Latches must persist for the lifetime of the pipeline, so we use heap allocation
  // The Pipeline should ideally own these, but for now we allocate them here
  PipelineLatch *latches[7];
  for (int i = 0; i < 7; i++) {
    latches[i] = new PipelineLatch();
  }
  
  // Connect latches in a circular pattern (stage N output -> stage N+1 input)
  p->get_stage(0)->set_latches(latches[6], latches[0]);  // WarpScheduler
  p->get_stage(1)->set_latches(latches[0], latches[1]);  // ActiveThreadSelection
  p->get_stage(2)->set_latches(latches[1], latches[2]);  // InstructionFetch
  p->get_stage(3)->set_latches(latches[2], latches[3]);  // OperandFetch
  p->get_stage(4)->set_latches(latches[3], latches[4]);  // OperandLatch
  p->get_stage(5)->set_latches(latches[4], latches[5]);  // ExecuteSuspend
  p->get_stage(6)->set_latches(latches[5], latches[6]);  // WritebackResume

  return p;
}

int main(int argc, char *argv[]) {
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
      "custom-instrs", "Custom instructions config file (name, opcode, byte pattern, handler). Default: custom_instrs.txt in cwd",
                            cxxopts::value<std::string>())(
      "q,quick", "Disable buffering for outputting earlier than simulation end")(
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

  std::string filename = result["filename"].as<std::string>();

  // Load custom instructions (optional)
  std::vector<CustomInstrEntry> custom_instrs_storage;
  const std::vector<CustomInstrEntry> *custom_instrs_ptr = nullptr;
  std::string custom_instrs_path =
      result.count("custom-instrs") ? result["custom-instrs"].as<std::string>() : "custom_instrs.txt";
  custom_instrs_storage = load_custom_instrs(custom_instrs_path);
  if (!custom_instrs_storage.empty()) {
    custom_instrs_ptr = &custom_instrs_storage;
    debug_log("Loaded " + std::to_string(custom_instrs_storage.size()) +
              " custom instruction(s) from " + custom_instrs_path);
  }

  // Initialize LLVM machine code decoding (RISC-V only)
  LLVMInitializeRISCVTargetInfo();
  LLVMInitializeRISCVTargetMC();
  LLVMInitializeRISCVDisassembler();

  std::string target_id = "riscv64-unknown-elf";
  std::string cpu = "generic-rv64";
  std::string features = "+m,+a,+zfinx";
  LLVMDisassembler disasm(target_id, cpu, features, custom_instrs_ptr);

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
  debug_log("Instantiated memory coalescing unit");

  RegisterFile rf(NUM_REGISTERS, NUM_LANES);
  HostRegisterFile hrf(&rf, NUM_REGISTERS);
  debug_log("Register file instantiated with " +
            std::to_string(NUM_REGISTERS) + " registers");

  // Initialization
  HostGPUControl gpu_controller;
  Pipeline *gpu_pipeline =
      initialize_pipeline(&tcim, &cu, &rf, &disasm, &gpu_controller, false,
                          instr_tracer ? instr_tracer.get() : nullptr, custom_instrs_ptr);
  Pipeline *cpu_pipeline =
      initialize_pipeline(&tcim, &cu, &hrf, &disasm, &gpu_controller, true, nullptr, custom_instrs_ptr);

  gpu_pipeline->set_debug(true);
  cpu_pipeline->set_debug(config.isCPUDebug());

  gpu_controller.set_scheduler(
      std::dynamic_pointer_cast<WarpScheduler>(gpu_pipeline->get_stage(0)));
  gpu_controller.set_pipeline(gpu_pipeline);

  // Execute the threads
  while (cpu_pipeline->has_active_stages() ||
         gpu_pipeline->has_active_stages() ||
         gpu_pipeline->is_pipeline_active()) {
    
    cpu_pipeline->execute();
    gpu_pipeline->execute();
    cu.tick();

    if (gpu_pipeline->is_pipeline_active()) {
      GPUStatisticsManager::instance().increment_gpu_cycles();
    }
    
    if (gpu_pipeline->is_pipeline_active() && !gpu_pipeline->has_active_stages()) {
      gpu_pipeline->set_pipeline_active(false);
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