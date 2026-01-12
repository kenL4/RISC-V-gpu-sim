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
#include "mem/mem_coalesce.hpp"
#include "mem/mem_data.hpp"
#include "mem/mem_instr.hpp"
#include "utils.hpp"

// Initialize pipeline (CPU is modelled as 1x1 GPU for simplicity)
Pipeline *initialize_pipeline(InstructionMemory *im, CoalescingUnit *cu,
                              RegisterFile *rf, LLVMDisassembler *disasm,
                              HostGPUControl *gpu_controller, bool is_cpu) {
  Pipeline *p = new Pipeline();

  // Construct stages (matching SIMTight's 7-stage pipeline)
  if (is_cpu) {
    p->add_stage<WarpScheduler>(1, 1, im->get_base_addr());  // Stage 0: CPU = 1x1
  } else {
    p->add_stage<WarpScheduler>(NUM_LANES, NUM_WARPS, im->get_base_addr(), false);  // Stage 0: GPU
  }
  p->add_stage<ActiveThreadSelection>();                     // Stage 1
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
  
  // Set up warp insertion callback (used by both execute and writeback stages)
  auto insert_warp_callback = [ws = warp_scheduler_stage](Warp *warp) {
    ws->insert_warp(warp);
  };
  execute_stage->insert_warp = insert_warp_callback;
  
  // Connect WritebackResume to ExecutionUnit and set up warp insertion
  writeback_stage->set_execution_unit(execute_stage->get_execution_unit());
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
      "d,debug", "Turn on debugging logs")("c,cpu-debug",
                                           "Turn on CPU debugging logs")(
      "r,regdump", "Dump the register values after each writeback stage")(
      "s,statsonly", "Do not print anything aside from the final stats")(
      "simtight-format", "Output statistics in SIMTight format (hex, 8 digits)")(
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
  config.setSimtightFormat(result.count("simtight-format") > 0);

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
  debug_log("Instantiated memory scratchpad for the SM");
  CoalescingUnit cu(&scratchpad_mem);
  debug_log("Instantiated memory coalescing unit");

  RegisterFile rf(NUM_REGISTERS, NUM_LANES);
  HostRegisterFile hrf(&rf, NUM_REGISTERS);
  debug_log("Register file instantiated with " +
            std::to_string(NUM_REGISTERS) + " registers");

  // Initialization
  HostGPUControl gpu_controller;
  Pipeline *gpu_pipeline =
      initialize_pipeline(&tcim, &cu, &rf, &disasm, &gpu_controller, false);
  Pipeline *cpu_pipeline =
      initialize_pipeline(&tcim, &cu, &hrf, &disasm, &gpu_controller, true);

  gpu_pipeline->set_debug(true);
  cpu_pipeline->set_debug(config.isCPUDebug());

  gpu_controller.set_scheduler(
      std::dynamic_pointer_cast<WarpScheduler>(gpu_pipeline->get_stage(0)));
  gpu_controller.set_pipeline(gpu_pipeline);

  // Execute the threads
  // Matching SIMTight: pipelineActive stays true from kernel launch until all warps terminate
  while (cpu_pipeline->has_active_stages() ||
         gpu_pipeline->has_active_stages() ||
         gpu_pipeline->is_pipeline_active()) {
    
    cpu_pipeline->execute();
    gpu_pipeline->execute();
    cu.tick();

    // Count cycles every cycle the GPU pipeline is active (matching SIMTight)
    // SIMTight counts cycles when pipelineActive is true, which is active
    // from kernel launch until all warps terminate
    if (gpu_pipeline->is_pipeline_active()) {
      GPUStatisticsManager::instance().increment_gpu_cycles();
    }
    
    if (gpu_pipeline->is_pipeline_active() && !gpu_pipeline->has_active_stages()) {
      // All warps have completed - set pipeline_active = false
      gpu_pipeline->set_pipeline_active(false);
    }
  }

  std::string output = gpu_controller.get_buffer();
  bool stats_only = config.isStatsOnly();
  if (!stats_only) {
    std::cout << std::endl << "[Results]" << std::endl;
    std::cout << output;
  }
  uint64_t sum = std::count(output.begin(), output.end(), '1');
  if (!stats_only) {
    std::cout << ((sum == 0)
                      ? "All passed!"
                      : std::to_string(NUM_LANES * NUM_WARPS - sum) +
                            " passed, " + std::to_string(sum) + " failed")
              << std::endl
              << std::endl;
  }

  bool simtight_format = config.isSimtightFormat();
  
  if (simtight_format) {
    // SIMTight format: hex, 8 digits, specific labels
    uint64_t cycles = GPUStatisticsManager::instance().get_gpu_cycles();
    uint64_t gpu_instrs = GPUStatisticsManager::instance().get_gpu_instrs();
    uint64_t gpu_susps = GPUStatisticsManager::instance().get_gpu_susps();
    uint64_t gpu_retries = GPUStatisticsManager::instance().get_gpu_retries();
    uint64_t gpu_dram_accs = GPUStatisticsManager::instance().get_gpu_dram_accs();
    
    // Format as hex with 8 digits (matching SIMTight's puthex output)
    std::cout << "Cycles: " << std::hex << std::setfill('0') << std::setw(8) << cycles << std::dec << std::endl;
    std::cout << "Instrs: " << std::hex << std::setfill('0') << std::setw(8) << gpu_instrs << std::dec << std::endl;
    std::cout << "Susps: " << std::hex << std::setfill('0') << std::setw(8) << gpu_susps << std::dec << std::endl;
    std::cout << "Retries: " << std::hex << std::setfill('0') << std::setw(8) << gpu_retries << std::dec << std::endl;
    std::cout << "DRAMAccs: " << std::hex << std::setfill('0') << std::setw(8) << gpu_dram_accs << std::dec << std::endl;
  } else {
    // Original format
    std::cout << "[Statistics]" << std::endl;
    uint64_t cycles = GPUStatisticsManager::instance().get_gpu_cycles();
    uint64_t gpu_instrs = GPUStatisticsManager::instance().get_gpu_instrs();
    uint64_t cpu_instrs = GPUStatisticsManager::instance().get_cpu_instrs();
    double ipc = static_cast<double>(gpu_instrs) / static_cast<double>(cycles);
    uint64_t gpu_dram_accs = GPUStatisticsManager::instance().get_gpu_dram_accs();
    uint64_t cpu_dram_accs = GPUStatisticsManager::instance().get_cpu_dram_accs();
    uint64_t gpu_retries = GPUStatisticsManager::instance().get_gpu_retries();
    uint64_t gpu_susps = GPUStatisticsManager::instance().get_gpu_susps();
    std::cout << "GPU Cycles: " << cycles << std::endl;
    std::cout << "GPU Instrs: " << gpu_instrs << std::endl;
    std::cout << "CPU Instrs: " << cpu_instrs << std::endl;
    std::cout << "IPC: " << ipc << std::endl;
    std::cout << "GPU DRAMAccs: " << gpu_dram_accs << std::endl;
    std::cout << "CPU DRAMAccs: " << cpu_dram_accs << std::endl;
    std::cout << "GPU Retries: " << gpu_retries << std::endl;
    std::cout << "GPU Susps: " << gpu_susps << std::endl;
  }

  delete cpu_pipeline;
  delete gpu_pipeline;

  return 0;
}