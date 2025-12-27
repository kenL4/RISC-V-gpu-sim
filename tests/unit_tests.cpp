#include "test_host.hpp"
#include "test_memory.hpp"
#include "test_pipeline.hpp"
#include "test_pipeline_execute.hpp"
#include "test_pipeline_scheduler.hpp"
#include "llvm/Support/TargetSelect.h"
#include <iostream>

int main() {
  std::cout << "Starting Unit Tests..." << std::endl;

  // Initialize LLVM for disassembler tests
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllDisassemblers();

  test_data_memory_load_store();
  test_instr_memory();
  test_coalesce_latency();

  test_host_register_file();
  test_host_gpu_control();

  test_instr_fetch_latch();
  test_ats_latch();
  test_op_fetch_latch();
  test_writeback_latch();
  test_warp_scheduler();
  test_execution_unit();

  std::cout << "All tests passed!" << std::endl;
  return 0;
}
