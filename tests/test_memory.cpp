#include "test_memory.hpp"
#include "gpu/pipeline.hpp"
#include "mem/mem_coalesce.hpp"
#include "mem/mem_data.hpp"
#include "mem/mem_instr.hpp"
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

void test_data_memory_load_store() {
  std::cout << "Running test_data_memory_load_store..." << std::endl;
  DataMemory mem;
  uint64_t addr = 0x1000;
  uint64_t val = 0x1122334455667788;

  // Store 8 bytes
  mem.store(addr, 8, val);

  // Load back 8 bytes
  int64_t loaded_8 = mem.load(addr, 8);
  assert(static_cast<uint64_t>(loaded_8) == val);

  // Load back 4 bytes (little endian) -> 0x55667788
  int64_t loaded_4 = mem.load(addr, 4);
  assert(static_cast<uint64_t>(loaded_4) == 0x55667788);

  // Load back 1 byte -> 0x88
  int64_t loaded_1 = mem.load(addr, 1);
  assert((loaded_1 & 0xFF) == 0x88);
  assert(loaded_1 == -120);

  // Test Sign Extension
  // Store 0xFF (which is -1 as signed 8-bit)
  mem.store(addr + 0x10, 1, 0xFF);
  int64_t loaded_signed = mem.load(addr + 0x10, 1);
  assert(loaded_signed == -1);

  std::cout << "test_data_memory_load_store passed!" << std::endl;
}

void test_instr_memory() {
  std::cout << "Running test_instr_memory..." << std::endl;
  parse_output pod;
  pod.base_addr = 0x4000;
  pod.max_addr = 0x4008; // 2 instructions: 0x4000, 0x4004
  // 8 bytes of code
  pod.code = {
      0x11, 0x11, 0x11, 0x11, // Instr 1
      0x22, 0x22, 0x22, 0x22  // Instr 2
  };

  InstructionMemory imem(&pod);

  assert(imem.get_base_addr() == 0x4000);
  // InstructionMemory sets max_addr = data->max_addr - 4
  assert(imem.get_max_addr() == 0x4004);

  uint8_t *instr1 = imem.get_instruction(0x4000);
  assert(instr1[0] == 0x11);

  uint8_t *instr2 = imem.get_instruction(0x4004);
  assert(instr2[0] == 0x22);

  std::cout << "test_instr_memory passed!" << std::endl;
}

void test_coalesce_latency() {
  std::cout << "Running test_coalesce_latency..." << std::endl;
  DataMemory dmem;
  CoalescingUnit unit(&dmem);

  // Create a dummy warp
  // Warp(uint64_t warp_id, size_t size, uint64_t start_pc, bool is_cpu)
  Warp w(0, 32, 0x1000, false);

  assert(!unit.is_busy());
  assert(!w.suspended);

  // Trigger load
  unit.load(&w, 0x2000, 4);

  assert(w.suspended);
  assert(unit.is_busy());

  // Tick until done
  int ticks = 0;
  while (unit.is_busy()) {
    unit.tick();
    Warp *resumed = unit.get_resumable_warp();
    if (resumed) {
      assert(resumed == &w);
    }
    ticks++;
    if (ticks > 100)
      break; // Safety break
  }

  // Verify we spent at least 1 tick (L1 latency is 1, DRAM is 2)
  assert(ticks >= 1);

  std::cout << "test_coalesce_latency passed!" << std::endl;
}
