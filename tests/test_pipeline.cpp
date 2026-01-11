#include "test_pipeline.hpp"

#include "disassembler/llvm_disasm.hpp"
#include "gpu/pipeline.hpp"
#include "gpu/pipeline_ats.hpp"
#include "gpu/pipeline_instr_fetch.hpp"
#include "gpu/pipeline_op_fetch.hpp"
#include "gpu/pipeline_writeback.hpp"
#include "mem/mem_coalesce.hpp"
#include "mem/mem_data.hpp"
#include "mem/mem_instr.hpp"
#include "parser.hpp"
#include <cassert>
#include <iostream>
#include <vector>

// Helper to create dummy code
// ADDI x1, x0, 10
// 00a00093
static parse_output create_dummy_code() {
  parse_output p;
  p.base_addr = 0x1000;
  p.max_addr = 0x1004;
  // 00a00093 in little endian: 93 00 a0 00
  p.code = {0x93, 0x00, 0xA0, 0x00};
  return p;
}

void test_instr_fetch_latch() {
  std::cout << "Running test_instr_fetch_latch..." << std::endl;

  parse_output p = create_dummy_code();
  InstructionMemory im(&p);
  LLVMDisassembler disasm("riscv32", "generic-rv32", "");
  InstructionFetch stage(&im, &disasm);

  PipelineLatch input, output;
  // Initialize latches
  input.updated = false;
  output.updated = false;

  stage.set_latches(&input, &output);

  Warp warp(0, 32, 0x1000, false);
  input.warp = &warp;
  input.updated = true;
  input.active_threads.assign(32, 1); // Enable all threads

  stage.execute();

  assert(output.updated == true);
  assert(output.warp == &warp);
  // Check if instruction was fetched. MCInst opcode should be non-zero
  // (assuming valid ADDI)
  assert(output.inst.getOpcode() != 0);

  std::cout << "test_instr_fetch_latch passed!" << std::endl;
}

void test_ats_latch() {
  std::cout << "Running test_ats_latch..." << std::endl;
  ActiveThreadSelection stage;
  PipelineLatch input, output;
  input.updated = false;
  output.updated = false;
  stage.set_latches(&input, &output);

  Warp warp(0, 32, 0x0, false);
  // Simulate divergence
  // Thread 0: level 0
  // Thread 1: level 1
  warp.nesting_level[0] = 0;
  warp.nesting_level[1] = 1;
  // Set others to 0
  for (size_t i = 2; i < 32; ++i)
    warp.nesting_level[i] = 0;

  input.warp = &warp;
  input.updated = true;

  // ActiveThreadSelection has 2-cycle latency (2 substages)
  // 1st cycle: Compute and store in buffer
  stage.execute();
  assert(output.updated == false); // Not available yet (in buffer)

  // 2nd cycle: Output from buffer
  input.updated = false; // Clear input for second cycle
  stage.execute();

  assert(output.updated == true);
  // ATS should pick threads with deepest nesting level => level 1 => Thread 1
  assert(output.active_threads.size() == 1);
  assert(output.active_threads[0] == 1); // Thread 1 should be active

  std::cout << "test_ats_latch passed!" << std::endl;
}

void test_op_fetch_latch() {
  std::cout << "Running test_op_fetch_latch..." << std::endl;
  OperandFetch stage;
  PipelineLatch input, output;
  input.updated = false;
  output.updated = false;
  stage.set_latches(&input, &output);

  Warp warp(0, 32, 0x0, false);
  input.warp = &warp;
  input.updated = true;

  stage.execute();

  assert(output.updated == true);
  assert(output.warp == &warp);

  std::cout << "test_op_fetch_latch passed!" << std::endl;
}

void test_writeback_latch() {
  std::cout << "Running test_writeback_latch..." << std::endl;

  DataMemory dm;
  CoalescingUnit cu(&dm);
  RegisterFile rf(32, 32);
  WritebackResume stage(&cu, &rf, true);  // true = CPU pipeline for test

  PipelineLatch input, output;
  input.updated = false;
  output.updated = false;
  stage.set_latches(&input, &output);

  // Case 1: Pass through from input latch
  Warp warp1(0, 32, 0x0, false);
  input.warp = &warp1;
  input.updated = true;

  stage.execute();

  assert(output.updated == true);
  assert(output.warp == &warp1);

  // Case 2: Resume from suspended warp (when input not updated)
  input.updated = false;
  output.updated = false;

  Warp warp2(1, 32, 0x0, true);  // Use CPU warp to match CPU pipeline WritebackResume
  std::vector<uint64_t> addrs = {0x1000};
  cu.load(&warp2, addrs, 4);
  assert(warp2.suspended == true);

  // Advance time to satisfy latency
  for (int i = 0; i < 1000; ++i)
    cu.tick();

  stage.execute();

  assert(output.updated == true);  // When resuming from memory, output_latch signals warp resume
  assert(output.warp == &warp2);
  assert(warp2.suspended == false);

  std::cout << "test_writeback_latch passed!" << std::endl;
}
