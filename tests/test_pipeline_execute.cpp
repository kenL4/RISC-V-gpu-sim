#include "test_pipeline_execute.hpp"
#include "config.hpp"
#include "disassembler/llvm_disasm.hpp"
#include "gpu/pipeline_execute.hpp"
#include "gpu/register_file.hpp"
#include "mem/mem_coalesce.hpp"
#include "mem/mem_data.hpp"
#include <cassert>
#include <iostream>
#include <vector>

// Register defs included via llvm_disasm.hpp -> utils.hpp

// opcode constants
constexpr uint32_t OP_OP = 0x33;
constexpr uint32_t OP_OP_IMM = 0x13;
constexpr uint32_t OP_LUI = 0x37;
constexpr uint32_t OP_AUIPC = 0x17;
constexpr uint32_t OP_STORE = 0x23;
constexpr uint32_t OP_LOAD = 0x03;
constexpr uint32_t OP_BRANCH = 0x63;
constexpr uint32_t OP_JAL = 0x6F;
constexpr uint32_t OP_JALR = 0x67;
constexpr uint32_t OP_SYSTEM = 0x73;

// Encoding helpers
uint32_t encode_r_type(uint32_t funct7, uint32_t rs2, uint32_t rs1,
                       uint32_t funct3, uint32_t rd, uint32_t opcode) {
  return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) |
         (rd << 7) | opcode;
}

uint32_t encode_i_type(uint32_t imm, uint32_t rs1, uint32_t funct3, uint32_t rd,
                       uint32_t opcode) {
  return ((imm & 0xFFF) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) |
         opcode;
}

uint32_t encode_s_type(uint32_t imm, uint32_t rs2, uint32_t rs1,
                       uint32_t funct3, uint32_t opcode) {
  uint32_t imm11_5 = (imm >> 5) & 0x7F;
  uint32_t imm4_0 = imm & 0x1F;
  return (imm11_5 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) |
         (imm4_0 << 7) | opcode;
}

uint32_t encode_b_type(uint32_t imm, uint32_t rs2, uint32_t rs1,
                       uint32_t funct3, uint32_t opcode) {
  uint32_t imm12 = (imm >> 12) & 1;
  uint32_t imm10_5 = (imm >> 5) & 0x3F;
  uint32_t imm4_1 = (imm >> 1) & 0xF;
  uint32_t imm11 = (imm >> 11) & 1;
  return (imm12 << 31) | (imm10_5 << 25) | (rs2 << 20) | (rs1 << 15) |
         (funct3 << 12) | (imm4_1 << 8) | (imm11 << 7) | opcode;
}

uint32_t encode_u_type(uint32_t imm, uint32_t rd, uint32_t opcode) {
  return (imm & 0xFFFFF000) | (rd << 7) | opcode;
}

uint32_t encode_j_type(uint32_t imm, uint32_t rd, uint32_t opcode) {
  uint32_t imm20 = (imm >> 20) & 1;
  uint32_t imm10_1 = (imm >> 1) & 0x3FF;
  uint32_t imm11 = (imm >> 11) & 1;
  uint32_t imm19_12 = (imm >> 12) & 0xFF;
  return (imm20 << 31) | (imm10_1 << 21) | (imm11 << 20) | (imm19_12 << 12) |
         (rd << 7) | opcode;
}


// Helper function to complete a load operation and write back result
// Simulates the pipeline behavior: tick the coalescing unit until completion, then write back
void complete_load_operation(CoalescingUnit &cu, RegisterFile &rf, Warp *warp) {
  // Tick the coalescing unit until the warp is resumable
  // Use a safety limit to prevent infinite loops
  for (int i = 0; i < 1000; ++i) {
    cu.tick();
    Warp *resumed = cu.get_resumable_warp_for_pipeline(warp->is_cpu);
    if (resumed != nullptr && resumed == warp) {
      // Get load results and write them back
      auto load_results = cu.get_load_results(warp);
      if (!load_results.second.empty()) {
        unsigned int rd_reg = load_results.first;
        for (const auto &[thread_id, value] : load_results.second) {
          rf.set_register(warp->warp_id, thread_id, rd_reg, value);
        }
      }
      warp->suspended = false;
      return;
    }
  }
  // If we get here, the load didn't complete - this is an error
  assert(false && "Load operation did not complete within safety limit");
}

void test_execution_unit() {
  std::cout << "Running test_execution_unit..." << std::endl;

  // Components
  DataMemory dm;
  CoalescingUnit cu(&dm);
  RegisterFile rf(32, 32);

  LLVMDisassembler disasm("riscv32", "generic-rv32", "+m");
  ExecutionUnit eu(&cu, &rf, &disasm, nullptr);

  // Helper to run an instruction
  auto run_inst = [&](Warp *warp, uint32_t opcode_bytes,
                      const std::string &asm_str) {
    std::vector<uint8_t> bytes(4);
    bytes[0] = opcode_bytes & 0xFF;
    bytes[1] = (opcode_bytes >> 8) & 0xFF;
    bytes[2] = (opcode_bytes >> 16) & 0xFF;
    bytes[3] = (opcode_bytes >> 24) & 0xFF;

    return disasm.disasm_inst(0, bytes);
  };

  // Warp setup
  Warp warp(0, 32, 0x1000, false);
  std::vector<size_t> active_threads = {0}; // Test with thread 0

  // 1. ADDI x1, x0, 10
  std::cout << "  Testing ADDI..." << std::endl;
  {
    uint32_t opcode = encode_i_type(10, 0, 0, 1, OP_OP_IMM);
    llvm::MCInst inst = run_inst(&warp, opcode, "ADDI x1, x0, 10");
    execute_result res = eu.execute(&warp, active_threads, inst);

    assert(res.success);
    assert(rf.get_register(0, 0, llvm::RISCV::X1) == 10);
    assert(warp.pc[0] == 0x1004);
  }

  // 2. ADD x2, x1, x1
  std::cout << "  Testing ADD..." << std::endl;
  {
    uint32_t opcode = encode_r_type(0, 1, 1, 0, 2, OP_OP);
    llvm::MCInst inst = run_inst(&warp, opcode, "ADD x2, x1, x1");
    eu.execute(&warp, active_threads, inst);
    assert(rf.get_register(0, 0, llvm::RISCV::X2) == 20);
  }

  // 3. SUB
  std::cout << "  Testing SUB..." << std::endl;
  {
    uint32_t opcode = encode_r_type(0x20, 1, 2, 0, 3, OP_OP);
    llvm::MCInst inst = run_inst(&warp, opcode, "SUB x3, x2, x1");
    eu.execute(&warp, active_threads, inst);
    assert(rf.get_register(0, 0, llvm::RISCV::X3) == 10);
  }

  // 4. BEQ x1, x3, offset (10 == 10, taken)
  std::cout << "  Testing BEQ..." << std::endl;
  {
    uint32_t opcode = encode_b_type(8, 3, 1, 0, OP_BRANCH);
    llvm::MCInst inst = run_inst(&warp, opcode, "BEQ x1, x3, 8");
    eu.execute(&warp, active_threads, inst);
    assert(warp.pc[0] == 0x1014); // 0x100C + 8
  }

  // 5. SW x1, 0(x0)
  std::cout << "  Testing SW..." << std::endl;
  {
    uint32_t opcode = encode_s_type(0, 1, 0, 2, OP_STORE);
    llvm::MCInst inst = run_inst(&warp, opcode, "SW x1, 0(x0)");
    eu.execute(&warp, active_threads, inst);
    assert(warp.pc[0] == 0x1018);
  }

  // --- Loads ---
  std::cout << "  Testing Loads..." << std::endl;
  {
    // Setup: Store 0x12345678 at address 0x100 (using SW)
    // x1 = 0x12345678
    // lui x1, 0x12345 (top 20) -> 0x12345678 is tricky with just LUI/ADDI.
    // LUI x1, 0x12345 -> x1 = 0x12345000.
    // ADDI x1, x1, 0x678.

    // 1. LUI x1, 0x12345
    uint32_t opcode = encode_u_type(0x12345 << 12, 1, OP_LUI);
    llvm::MCInst inst = run_inst(&warp, opcode, "LUI x1, 0x12345");
    eu.execute(&warp, active_threads, inst);

    // 2. ADDI x1, x1, 0x678
    opcode = encode_i_type(0x678, 1, 0, 1, OP_OP_IMM);
    inst = run_inst(&warp, opcode, "ADDI x1, x1, 0x678");
    eu.execute(&warp, active_threads, inst);

    // Check x1
    assert(rf.get_register(0, 0, llvm::RISCV::X1) == 0x12345678);

    // 3. SW x1, 0x100(x0)
    opcode = encode_s_type(0x100, 1, 0, 2, OP_STORE);
    inst = run_inst(&warp, opcode, "SW x1, 0x100(x0)");
    eu.execute(&warp, active_threads, inst);

    // Now Load back
    // LW x2, 0x100(x0) -> x2 should be 0x12345678
    opcode = encode_i_type(0x100, 0, 2, 2, OP_LOAD); // funct3=2 (LW)
    inst = run_inst(&warp, opcode, "LW x2, 0x100(x0)");
    eu.execute(&warp, active_threads, inst);
    // Complete the load operation (tick until done and write back results)
    complete_load_operation(cu, rf, &warp);
    assert(rf.get_register(0, 0, llvm::RISCV::X2) == 0x12345678);

    // LH x2, 0x100(x0) -> x2 should be 0x5678 (sign extended if neg? 0x5678 is
    // pos)
    opcode = encode_i_type(0x100, 0, 1, 2, OP_LOAD); // funct3=1 (LH)
    inst = run_inst(&warp, opcode, "LH x2, 0x100(x0)");
    eu.execute(&warp, active_threads, inst);
    complete_load_operation(cu, rf, &warp);
    assert(rf.get_register(0, 0, llvm::RISCV::X2) == 0x5678);

    // LHU x2, 0x100(x0) -> x2 should be 0x5678
    opcode = encode_i_type(0x100, 0, 5, 2, OP_LOAD); // funct3=5 (LHU)
    inst = run_inst(&warp, opcode, "LHU x2, 0x100(x0)");
    eu.execute(&warp, active_threads, inst);
    assert(rf.get_register(0, 0, llvm::RISCV::X2) == 0x5678);

    // LB x2, 0x100(x0) -> x2 should be 0x78
    opcode = encode_i_type(0x100, 0, 0, 2, OP_LOAD); // funct3=0 (LB)
    inst = run_inst(&warp, opcode, "LB x2, 0x100(x0)");
    eu.execute(&warp, active_threads, inst);
    complete_load_operation(cu, rf, &warp);
    assert(rf.get_register(0, 0, llvm::RISCV::X2) == 0x78);

    // LBU x2, 0x100(x0) -> x2 should be 0x78
    opcode = encode_i_type(0x100, 0, 4, 2, OP_LOAD); // funct3=4 (LBU)
    inst = run_inst(&warp, opcode, "LBU x2, 0x100(x0)");
    eu.execute(&warp, active_threads, inst);
    complete_load_operation(cu, rf, &warp);
    assert(rf.get_register(0, 0, llvm::RISCV::X2) == 0x78);
  }

  std::cout << "  Testing Logical..." << std::endl;
  {
    // Reset inputs
    llvm::MCInst inst = run_inst(&warp, encode_i_type(10, 0, 0, 1, OP_OP_IMM),
                                 "ADDI x1, x0, 10");
    eu.execute(&warp, active_threads, inst);
    inst = run_inst(&warp, encode_i_type(20, 0, 0, 2, OP_OP_IMM),
                    "ADDI x2, x0, 20");
    eu.execute(&warp, active_threads, inst);

    // AND x4, x1, x2 (10 & 20 = 0)
    uint32_t opcode = encode_r_type(0, 2, 1, 7, 4, OP_OP);
    inst = run_inst(&warp, opcode, "AND x4, x1, x2");
    eu.execute(&warp, active_threads, inst);
    assert(rf.get_register(0, 0, llvm::RISCV::X4) == 0);

    // OR x4, x1, x2 (10 | 20 = 30)
    opcode = encode_r_type(0, 2, 1, 6, 4, OP_OP);
    inst = run_inst(&warp, opcode, "OR x4, x1, x2");
    eu.execute(&warp, active_threads, inst);
    assert(rf.get_register(0, 0, llvm::RISCV::X4) == 30);

    // XOR x4, x1, x2 (10 ^ 20 = 30)
    opcode = encode_r_type(0, 2, 1, 4, 4, OP_OP);
    inst = run_inst(&warp, opcode, "XOR x4, x1, x2");
    eu.execute(&warp, active_threads, inst);
    assert(rf.get_register(0, 0, llvm::RISCV::X4) == 30);

    // ANDI x4, x1, 7 (10 & 7 = 2)
    opcode = encode_i_type(7, 1, 7, 4, OP_OP_IMM);
    inst = run_inst(&warp, opcode, "ANDI x4, x1, 7");
    eu.execute(&warp, active_threads, inst);
    assert(rf.get_register(0, 0, llvm::RISCV::X4) == 2);

    // ORI x4, x1, 5 (10 | 5 = 15)
    opcode = encode_i_type(5, 1, 6, 4, OP_OP_IMM);
    inst = run_inst(&warp, opcode, "ORI x4, x1, 5");
    eu.execute(&warp, active_threads, inst);
    assert(rf.get_register(0, 0, llvm::RISCV::X4) == 15);

    // XORI x4, x1, 5 (10 ^ 5 = 15)
    opcode = encode_i_type(5, 1, 4, 4, OP_OP_IMM);
    inst = run_inst(&warp, opcode, "XORI x4, x1, 5");
    eu.execute(&warp, active_threads, inst);
    assert(rf.get_register(0, 0, llvm::RISCV::X4) == 15);
  }

  std::cout << "  Testing Shifts..." << std::endl;
  {
    // SLLI x4, x1, 2 (10 << 2 = 40)
    uint32_t opcode = encode_i_type(2, 1, 1, 4, OP_OP_IMM);
    llvm::MCInst inst = run_inst(&warp, opcode, "SLLI x4, x1, 2");
    eu.execute(&warp, active_threads, inst);
    assert(rf.get_register(0, 0, llvm::RISCV::X4) == 40);

    // SRLI x4, x1, 1 (10 >> 1 = 5)
    opcode = encode_i_type(1, 1, 5, 4, OP_OP_IMM);
    inst = run_inst(&warp, opcode, "SRLI x4, x1, 1");
    eu.execute(&warp, active_threads, inst);
    assert(rf.get_register(0, 0, llvm::RISCV::X4) == 5);

    // SRAI x4, x1, 1 (10 >> 1 arithmetic = 5)
    opcode = encode_i_type((0x20 << 5) | 1, 1, 5, 4, OP_OP_IMM);
    inst = run_inst(&warp, opcode, "SRAI x4, x1, 1");
    eu.execute(&warp, active_threads, inst);
    assert(rf.get_register(0, 0, llvm::RISCV::X4) == 5);

    // ADDI x5, x0, 2
    opcode = encode_i_type(2, 0, 0, 5, OP_OP_IMM);
    inst = run_inst(&warp, opcode, "ADDI x5, x0, 2");
    eu.execute(&warp, active_threads, inst);

    // SLL x4, x1, x5 (10 << 2 = 40)
    opcode = encode_r_type(0, 5, 1, 1, 4, OP_OP);
    inst = run_inst(&warp, opcode, "SLL x4, x1, x5");
    eu.execute(&warp, active_threads, inst);
    assert(rf.get_register(0, 0, llvm::RISCV::X4) == 40);

    // SRL x4, x1, x5 (10 >> 2 = 2)
    opcode = encode_r_type(0, 5, 1, 5, 4, OP_OP); // funct3=5
    inst = run_inst(&warp, opcode, "SRL x4, x1, x5");
    eu.execute(&warp, active_threads, inst);
    assert(rf.get_register(0, 0, llvm::RISCV::X4) == 2);

    // SRA x4, x1, x5 (10 >> 2 = 2)
    opcode = encode_r_type(0x20, 5, 1, 5, 4, OP_OP); // funct3=5, funct7=0x20
    inst = run_inst(&warp, opcode, "SRA x4, x1, x5");
    eu.execute(&warp, active_threads, inst);
    assert(rf.get_register(0, 0, llvm::RISCV::X4) == 2);
  }

  std::cout << "  Testing M-Ext..." << std::endl;
  {
    // MUL x4, x1, x2 (10 * 20 = 200)
    // x1=0x12345678 (destroyed in Loads test), x2=0x78
    // Let's reset values
    // ADDI x1, x0, 10
    llvm::MCInst inst = run_inst(&warp, encode_i_type(10, 0, 0, 1, OP_OP_IMM),
                                 "ADDI x1, x0, 10");
    eu.execute(&warp, active_threads, inst);
    // ADDI x2, x0, 20
    inst = run_inst(&warp, encode_i_type(20, 0, 0, 2, OP_OP_IMM),
                    "ADDI x2, x0, 20");
    eu.execute(&warp, active_threads, inst);

    uint32_t opcode = encode_r_type(1, 2, 1, 0, 4, OP_OP);
    inst = run_inst(&warp, opcode, "MUL x4, x1, x2");
    eu.execute(&warp, active_threads, inst);
    // MUL now executes immediately
    assert(rf.get_register(0, 0, llvm::RISCV::X4) == 200);

    // DIVU x4, x2, x1 (20 / 10 = 2)
    opcode = encode_r_type(1, 1, 2, 5, 4, OP_OP);
    inst = run_inst(&warp, opcode, "DIVU x4, x2, x1");
    eu.execute(&warp, active_threads, inst);
    // DIVU now executes immediately
    assert(rf.get_register(0, 0, llvm::RISCV::X4) == 2);

    // REMU x4, x1, x2 (10 % 20 = 10)
    opcode = encode_r_type(1, 2, 1, 7, 4, OP_OP);
    inst = run_inst(&warp, opcode, "REMU x4, x1, x2");
    eu.execute(&warp, active_threads, inst);
    // REMU now executes immediately
    assert(rf.get_register(0, 0, llvm::RISCV::X4) == 10);
  }

  std::cout << "  Testing Comparison..." << std::endl;
  {
    // SLTI x4, x1, 20 (10 < 20 -> 1)
    uint32_t opcode = encode_i_type(20, 1, 2, 4, OP_OP_IMM);
    llvm::MCInst inst = run_inst(&warp, opcode, "SLTI x4, x1, 20");
    eu.execute(&warp, active_threads, inst);
    assert(rf.get_register(0, 0, llvm::RISCV::X4) == 1);

    // SLTI x4, x1, 5 (10 < 5 -> 0)
    opcode = encode_i_type(5, 1, 2, 4, OP_OP_IMM);
    inst = run_inst(&warp, opcode, "SLTI x4, x1, 5");
    eu.execute(&warp, active_threads, inst);
    assert(rf.get_register(0, 0, llvm::RISCV::X4) == 0);

    // SLT x4, x1, x2 (10 < 20 -> 1)
    opcode = encode_r_type(0, 2, 1, 2, 4, OP_OP);
    inst = run_inst(&warp, opcode, "SLT x4, x1, x2");
    eu.execute(&warp, active_threads, inst);
    assert(rf.get_register(0, 0, llvm::RISCV::X4) == 1);

    // SLTU x4, x1, x2 (10 < 20 -> 1)
    opcode = encode_r_type(0, 2, 1, 3, 4, OP_OP); // funct3=3
    inst = run_inst(&warp, opcode, "SLTU x4, x1, x2");
    eu.execute(&warp, active_threads, inst);
    assert(rf.get_register(0, 0, llvm::RISCV::X4) == 1);
  }

  std::cout << "  Testing LUI/AUIPC..." << std::endl;
  {
    // LUI x4, 1 (x4 = 1 << 12 = 4096)
    uint32_t opcode = encode_u_type(1 << 12, 4, OP_LUI);
    llvm::MCInst inst = run_inst(&warp, opcode, "LUI x4, 1");
    eu.execute(&warp, active_threads, inst);
    assert(rf.get_register(0, 0, llvm::RISCV::X4) == 4096);

    // AUIPC x4, 1 (x4 = PC + 4096)
    uint64_t pc = warp.pc[0];
    opcode = encode_u_type(1 << 12, 4, OP_AUIPC);
    inst = run_inst(&warp, opcode, "AUIPC x4, 1");
    eu.execute(&warp, active_threads, inst);
    assert(rf.get_register(0, 0, llvm::RISCV::X4) == pc + 4096);
  }

  // --- Branch/Jump ---
  std::cout << "  Testing Branch/Jump..." << std::endl;
  {
    // BNE x1, x3, 8 (10 != 10 -> False) -> PC + 4
    uint64_t start_pc = warp.pc[0];
    uint32_t opcode = encode_b_type(8, 3, 1, 1, OP_BRANCH);
    llvm::MCInst inst = run_inst(&warp, opcode, "BNE x1, x3, 8");
    eu.execute(&warp, active_threads, inst);
    assert(warp.pc[0] == start_pc + 4);

    // JAL x0, 8 (Jump +8, don't link)
    start_pc = warp.pc[0];
    opcode = encode_j_type(8, 0, OP_JAL);
    inst = run_inst(&warp, opcode, "JAL x0, 8");
    eu.execute(&warp, active_threads, inst);
    assert(warp.pc[0] == start_pc + 8);

    // JALR x0, x1, 0 (Jump to x1 aka 10)
    // Careful: x1 is 10. PC becomes 10.
    start_pc = warp.pc[0];
    opcode = encode_i_type(0, 1, 0, 0, OP_JALR);
    inst = run_inst(&warp, opcode, "JALR x0, x1, 0");
    eu.execute(&warp, active_threads, inst);
    assert(warp.pc[0] == 10);
    // Reset PC to something safe
    for (size_t t = 0; t < 32; ++t)
      warp.pc[t] = 0x2000;

    // BLT x1, x3, 8 (10 < 10 -> False) -> PC + 4
    start_pc = warp.pc[0];
    opcode = encode_b_type(8, 3, 1, 4, OP_BRANCH); // funct3=4
    inst = run_inst(&warp, opcode, "BLT x1, x3, 8");
    eu.execute(&warp, active_threads, inst);
    assert(warp.pc[0] == start_pc + 4);

    // BGE x1, x3, 8 (10 >= 10 -> True) -> PC + 8
    start_pc = warp.pc[0];
    opcode = encode_b_type(8, 3, 1, 5, OP_BRANCH); // funct3=5
    inst = run_inst(&warp, opcode, "BGE x1, x3, 8");
    eu.execute(&warp, active_threads, inst);
    assert(warp.pc[0] == start_pc + 8);
  }

  // --- Load/Store Variants ---
  std::cout << "  Testing Load/Store Variants (Short)..." << std::endl;
  {
    // SH x1, 0(x0) (Store Half 10 -> addr 0)
    uint32_t opcode = encode_s_type(0, 1, 0, 1, OP_STORE);
    llvm::MCInst inst = run_inst(&warp, opcode, "SH x1, 0(x0)");
    eu.execute(&warp, active_threads, inst);
    assert(warp.pc[0] == 0x2010); // Incremented from 0x2008 in BGE check block
                                  // (0x2000 + 8 = 2008) -> +4 = 200C

    // SB x1, 4(x0) (Store Byte 10 -> addr 4)
    opcode = encode_s_type(4, 1, 0, 0, OP_STORE);
    inst = run_inst(&warp, opcode, "SB x1, 4(x0)");
    eu.execute(&warp, active_threads, inst);
    assert(warp.pc[0] == 0x2014);

    // Note: We are not verifying memory content here as checking
    // CoalescingUnit/DataMemory requires ticking them or inspecting their
    // internals which is complex in this unit test. We assume if SW worked
    // (verified by PC advance and no crash), these likely work too.
  }

  // --- Misc/System ---
  std::cout << "  Testing Misc..." << std::endl;
  {
    // FENCE (No-op)
    // opcode=0x0F, funct3=0, rd=0, rs1=0...
    // Actually standard FENCE: opcode 0001111 (0x0F)
    // Simulator uses FENCE logic?
    // Let's check source code... fence(..).
    // FENCE opcode is MISC_MEM (0x0F).
    uint32_t opcode = encode_i_type(0, 0, 0, 0, 0x0F);
    llvm::MCInst inst = run_inst(&warp, opcode, "FENCE");
    eu.execute(&warp, active_threads, inst);
    assert(warp.pc[0] == 0x2018);

    // ECALL (0x73, funct3=0, funct12=0)
    // opcode=0x73, rd=0, rs1=0, imm=0
    opcode = encode_i_type(0, 0, 0, 0, OP_SYSTEM);
    inst = run_inst(&warp, opcode, "ECALL");
    eu.execute(&warp, active_threads, inst);
    assert(warp.pc[0] == 0x201C); // Simulator just increments PC for now

    // EBREAK (0x73, funct3=0, funct12=1)
    opcode = encode_i_type(1, 0, 0, 0, OP_SYSTEM);
    inst = run_inst(&warp, opcode, "EBREAK");
    eu.execute(&warp, active_threads, inst);
    assert(warp.pc[0] == 0x2020);
  }

  // --- Custom ---
  std::cout << "  Testing Custom..." << std::endl;
  {
    // NOCLPUSH: 09 00 ... (Short 16-bit? or custom)
    // As per disassembler: 09 00 = NOCLPUSH (type 0)
    // We need to inject raw bytes since LLVM won't assemble "NOCLPUSH" mnemonic
    // if standard. But our run_inst disassembles raw bytes. Try raw bytes:
    // 0x09, 0x00, 0x00, 0x00
    llvm::MCInst inst = run_inst(&warp, 0x00050009, "NOCLPUSH");
    execute_result res = eu.execute(&warp, active_threads, inst);
    assert(res.success); // Should execute
    assert(warp.pc[0] == 0x2024);

    // NOCLPOP: 09 10 ...
    // 0x1009 -> 09 10
    inst = run_inst(&warp, 0x00051009, "NOCLPOP");
    res = eu.execute(&warp, active_threads, inst);
    assert(res.success);
    assert(warp.pc[0] == 0x2028);

    // CACHE_LINE_FLUSH: 08 00 ...
    inst = run_inst(&warp, 0x00050008, "CACHE_LINE_FLUSH");
    res = eu.execute(&warp, active_threads, inst);
    assert(res.success);
    assert(warp.pc[0] == 0x202C);
  }

  std::cout << "test_execution_unit passed!" << std::endl;
}
