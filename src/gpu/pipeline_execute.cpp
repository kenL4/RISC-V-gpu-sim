#include "pipeline_execute.hpp"
#include "../disassembler/llvm_disasm.hpp"
#include "../stats/stats.hpp"
#include "../config.hpp"
#include <climits>
#include <sstream>
#include <iomanip>

// In RISC-V, Word is always 32-bit (4 bytes)
#define WORD_SIZE 4

ExecutionUnit::ExecutionUnit(CoalescingUnit *cu, RegisterFile *rf,
                             LLVMDisassembler *disasm,
                             HostGPUControl *gpu_controller)
    : cu(cu), rf(rf), disasm(disasm), gpu_controller(gpu_controller) {}

execute_result ExecutionUnit::execute(Warp *warp,
                                      std::vector<size_t> active_threads,
                                      MCInst &inst) {
  execute_result res{true, false, true};

  std::string mnemonic = disasm->getOpcodeName(inst.getOpcode());
  if (mnemonic == "ADDI") {
    res.write_required = addi(warp, active_threads, &inst);
  } else if (mnemonic == "ADD") {
    res.write_required = add(warp, active_threads, &inst);
  } else if (mnemonic == "SUB") {
    res.write_required = sub(warp, active_threads, &inst);
  } else if (mnemonic == "MUL") {
    res.write_required = mul(warp, active_threads, &inst);
  } else if (mnemonic == "AND") {
    // Name followed by an underscore as and is a reserved keyword
    res.write_required = and_(warp, active_threads, &inst);
  } else if (mnemonic == "ANDI") {
    res.write_required = andi(warp, active_threads, &inst);
  } else if (mnemonic == "OR") {
    res.write_required = or_(warp, active_threads, &inst);
  } else if (mnemonic == "ORI") {
    res.write_required = ori(warp, active_threads, &inst);
  } else if (mnemonic == "XOR") {
    res.write_required = xor_(warp, active_threads, &inst);
  } else if (mnemonic == "XORI") {
    res.write_required = xori(warp, active_threads, &inst);
  } else if (mnemonic == "SLL") {
    res.write_required = sll(warp, active_threads, &inst);
  } else if (mnemonic == "SLLI") {
    res.write_required = slli(warp, active_threads, &inst);
  } else if (mnemonic == "SRL") {
    res.write_required = srl(warp, active_threads, &inst);
  } else if (mnemonic == "SRLI") {
    res.write_required = srli(warp, active_threads, &inst);
  } else if (mnemonic == "SRA") {
    res.write_required = sra(warp, active_threads, &inst);
  } else if (mnemonic == "SRAI") {
    res.write_required = srai(warp, active_threads, &inst);
  } else if (mnemonic == "LUI") {
    res.write_required = lui(warp, active_threads, &inst);
  } else if (mnemonic == "AUIPC") {
    res.write_required = auipc(warp, active_threads, &inst);
  } else if (mnemonic == "LW") {
    res.write_required = lw(warp, active_threads, &inst);
    // If memory operation returns false AND warp is not suspended, queue was full - need to retry
    // If memory operation returns false AND warp is suspended, operation succeeded (writeback/resume happens later)
    // Matching SIMTight: retry when queue full (warp not suspended), suspend when operation accepted
    if (!res.write_required && !warp->suspended) {
      res.success = false;
      res.counted = false;
    }
  } else if (mnemonic == "LH") {
    res.write_required = lh(warp, active_threads, &inst);
    if (!res.write_required && !warp->suspended) {
      res.success = false;
      res.counted = false;
    }
  } else if (mnemonic == "LHU") {
    res.write_required = lhu(warp, active_threads, &inst);
    if (!res.write_required && !warp->suspended) {
      res.success = false;
      res.counted = false;
    }
  } else if (mnemonic == "LB") {
    res.write_required = lb(warp, active_threads, &inst);
    if (!res.write_required && !warp->suspended) {
      res.success = false;
      res.counted = false;
    }
  } else if (mnemonic == "LBU") {
    res.write_required = lbu(warp, active_threads, &inst);
    if (!res.write_required && !warp->suspended) {
      res.success = false;
      res.counted = false;
    }
  } else if (mnemonic == "SW") {
    res.write_required = sw(warp, active_threads, &inst);
    if (!res.write_required && !warp->suspended) {
      res.success = false;
      res.counted = false;
    }
  } else if (mnemonic == "SH") {
    res.write_required = sh(warp, active_threads, &inst);
    if (!res.write_required && !warp->suspended) {
      res.success = false;
      res.counted = false;
    }
  } else if (mnemonic == "SB") {
    res.write_required = sb(warp, active_threads, &inst);
    if (!res.write_required && !warp->suspended) {
      res.success = false;
      res.counted = false;
    }
  } else if (mnemonic == "AMOADD_W") {
    res.write_required = amoadd_w(warp, active_threads, &inst);
    if (!res.write_required && !warp->suspended) {
      res.success = false;
      res.counted = false;
    }
  } else if (mnemonic == "JAL") {
    res.write_required = jal(warp, active_threads, &inst);
  } else if (mnemonic == "JALR") {
    res.write_required = jalr(warp, active_threads, &inst);
  } else if (mnemonic == "BEQ") {
    res.write_required = beq(warp, active_threads, &inst);
  } else if (mnemonic == "BNE") {
    res.write_required = bne(warp, active_threads, &inst);
  } else if (mnemonic == "BLT") {
    res.write_required = blt(warp, active_threads, &inst);
  } else if (mnemonic == "BLTU") {
    res.write_required = bltu(warp, active_threads, &inst);
  } else if (mnemonic == "BGE") {
    res.write_required = bge(warp, active_threads, &inst);
  } else if (mnemonic == "BGEU") {
    res.write_required = bgeu(warp, active_threads, &inst);
  } else if (mnemonic == "SLT") {
    res.write_required = slt(warp, active_threads, &inst);
  } else if (mnemonic == "SLTI") {
    res.write_required = slti(warp, active_threads, &inst);
  } else if (mnemonic == "SLTIU") {
    res.write_required = sltiu(warp, active_threads, &inst);
  } else if (mnemonic == "SLTU") {
    res.write_required = sltu(warp, active_threads, &inst);
  } else if (mnemonic == "REMU") {
    res.write_required = remu(warp, active_threads, &inst);
  } else if (mnemonic == "DIVU") {
    res.write_required = divu(warp, active_threads, &inst);
  } else if (mnemonic == "DIV") {
    res.write_required = div_(warp, active_threads, &inst);
  } else if (mnemonic == "REM") {
    res.write_required = rem_(warp, active_threads, &inst);
  } else if (mnemonic == "FENCE") {
    res.write_required = fence(warp, active_threads, &inst);
    // If memory operation returns false AND warp is not suspended, queue was full - need to retry
    // If memory operation returns false AND warp is suspended, operation succeeded (resume happens later)
    // Matching SIMTight: retry when queue full (warp not suspended), suspend when operation accepted
    if (!res.write_required && !warp->suspended) {
      res.success = false;
      res.counted = false;
    }
  } else if (mnemonic == "ECALL") {
    res.write_required = ecall(warp, active_threads, &inst);
  } else if (mnemonic == "EBREAK") {
    res.write_required = ebreak(warp, active_threads, &inst);
  } else if (mnemonic == "CSRRW") {
    res.write_required = csrrw(warp, active_threads, &inst);
  } else if (mnemonic == "NOCLPUSH") {
    res.write_required = noclpush(warp, active_threads, &inst);
  } else if (mnemonic == "NOCLPOP") {
    res.write_required = noclpop(warp, active_threads, &inst);
  } else if (mnemonic == "CACHE_LINE_FLUSH") {
    res.write_required = cache_line_flush(warp, active_threads, &inst);
  } else {
    // Default to skip instruction
    for (auto thread : active_threads) {
      warp->pc[thread] += 4;
    }
    res.success = false;
    res.counted = false;
    if (!Config::instance().isStatsOnly())
      std::cout << "[WARNING] Unknown instruction " << mnemonic << std::endl;
  }
  return res;
}

bool ExecutionUnit::add(Warp *warp, std::vector<size_t> active_threads,
                        MCInst *in) {
  assert(in->getNumOperands() == 3);
  for (auto thread : active_threads) {
    unsigned int rd = in->getOperand(0).getReg();
    unsigned int rs1_reg = in->getOperand(1).getReg();
    unsigned int rs2_reg = in->getOperand(2).getReg();
    int rs1 =
        rf->get_register(warp->warp_id, thread, rs1_reg, warp->is_cpu);
    int rs2 =
        rf->get_register(warp->warp_id, thread, rs2_reg, warp->is_cpu);
      int result = rs1 + rs2;
      rf->set_register(warp->warp_id, thread, rd, result, warp->is_cpu);

    warp->pc[thread] += 4;
  }
  return active_threads.size() > 0;
}
bool ExecutionUnit::addi(Warp *warp, std::vector<size_t> active_threads,
                         MCInst *in) {
  assert(in->getNumOperands() == 3);
  for (auto thread : active_threads) {
    unsigned int rd = in->getOperand(0).getReg();
    int rs1 =
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg(), warp->is_cpu);
    int64_t imm = in->getOperand(2).getImm();
    int result = rs1 + imm;
    rf->set_register(warp->warp_id, thread, rd, result, warp->is_cpu);

    warp->pc[thread] += 4;
  }
  return active_threads.size() > 0;
}
bool ExecutionUnit::sub(Warp *warp, std::vector<size_t> active_threads,
                        MCInst *in) {
  assert(in->getNumOperands() == 3);
  for (auto thread : active_threads) {
    unsigned int rd = in->getOperand(0).getReg();
    unsigned int rs1_reg = in->getOperand(1).getReg();
    unsigned int rs2_reg = in->getOperand(2).getReg();
    int rs1 =
        rf->get_register(warp->warp_id, thread, rs1_reg, warp->is_cpu);
    int rs2 =
        rf->get_register(warp->warp_id, thread, rs2_reg, warp->is_cpu);
      int result = rs1 - rs2;
      rf->set_register(warp->warp_id, thread, rd, result, warp->is_cpu);

    warp->pc[thread] += 4;
  }
  return active_threads.size() > 0;
}

bool ExecutionUnit::mul(Warp *warp, std::vector<size_t> active_threads,
                        MCInst *in) {
  assert(in->getNumOperands() == 3);
  for (auto thread : active_threads) {
    unsigned int rd = in->getOperand(0).getReg();
    int rs1 =
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg(), warp->is_cpu);
    int rs2 =
        rf->get_register(warp->warp_id, thread, in->getOperand(2).getReg(), warp->is_cpu);
    int result = rs1 * rs2;
    rf->set_register(warp->warp_id, thread, rd, result, warp->is_cpu);

    warp->pc[thread] += 4;
  }
  return active_threads.size() > 0;
}
bool ExecutionUnit::and_(Warp *warp, std::vector<size_t> active_threads,
                         MCInst *in) {
  assert(in->getNumOperands() == 3);
  for (auto thread : active_threads) {
    unsigned int rd = in->getOperand(0).getReg();
    int rs1 =
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg(), warp->is_cpu);
    int rs2 =
        rf->get_register(warp->warp_id, thread, in->getOperand(2).getReg(), warp->is_cpu);
    rf->set_register(warp->warp_id, thread, rd, rs1 & rs2, warp->is_cpu);

    warp->pc[thread] += 4;
  }
  return active_threads.size() > 0;
}
bool ExecutionUnit::andi(Warp *warp, std::vector<size_t> active_threads,
                         MCInst *in) {
  assert(in->getNumOperands() == 3);
  for (auto thread : active_threads) {
    unsigned int rd = in->getOperand(0).getReg();
    int rs1 =
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg(), warp->is_cpu);
    int64_t imm = in->getOperand(2).getImm();
    rf->set_register(warp->warp_id, thread, rd, rs1 & imm, warp->is_cpu);

    warp->pc[thread] += 4;
  }
  return active_threads.size() > 0;
}
bool ExecutionUnit::or_(Warp *warp, std::vector<size_t> active_threads,
                        MCInst *in) {
  assert(in->getNumOperands() == 3);
  for (auto thread : active_threads) {
    unsigned int rd = in->getOperand(0).getReg();
    int rs1 =
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg(), warp->is_cpu);
    int rs2 =
        rf->get_register(warp->warp_id, thread, in->getOperand(2).getReg(), warp->is_cpu);
    rf->set_register(warp->warp_id, thread, rd, rs1 | rs2, warp->is_cpu);

    warp->pc[thread] += 4;
  }
  return active_threads.size() > 0;
}
bool ExecutionUnit::ori(Warp *warp, std::vector<size_t> active_threads,
                        MCInst *in) {
  assert(in->getNumOperands() == 3);
  for (auto thread : active_threads) {
    unsigned int rd = in->getOperand(0).getReg();
    int rs1 =
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg(), warp->is_cpu);
    int64_t imm = in->getOperand(2).getImm();
    rf->set_register(warp->warp_id, thread, rd, rs1 | imm, warp->is_cpu);

    warp->pc[thread] += 4;
  }
  return active_threads.size() > 0;
}
bool ExecutionUnit::xor_(Warp *warp, std::vector<size_t> active_threads,
                         MCInst *in) {
  assert(in->getNumOperands() == 3);
  for (auto thread : active_threads) {
    unsigned int rd = in->getOperand(0).getReg();
    int rs1 =
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg(), warp->is_cpu);
    int rs2 =
        rf->get_register(warp->warp_id, thread, in->getOperand(2).getReg(), warp->is_cpu);
    rf->set_register(warp->warp_id, thread, rd, rs1 ^ rs2, warp->is_cpu);

    warp->pc[thread] += 4;
  }
  return active_threads.size() > 0;
}
bool ExecutionUnit::xori(Warp *warp, std::vector<size_t> active_threads,
                         MCInst *in) {
  assert(in->getNumOperands() == 3);
  for (auto thread : active_threads) {
    unsigned int rd = in->getOperand(0).getReg();
    int rs1 =
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg(), warp->is_cpu);
    int64_t imm = in->getOperand(2).getImm();
    rf->set_register(warp->warp_id, thread, rd, rs1 ^ imm, warp->is_cpu);

    warp->pc[thread] += 4;
  }
  return active_threads.size() > 0;
}
bool ExecutionUnit::sll(Warp *warp, std::vector<size_t> active_threads,
                        MCInst *in) {
  assert(in->getNumOperands() == 3);
  for (auto thread : active_threads) {
    unsigned int rd = in->getOperand(0).getReg();
    int rs1 =
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg(), warp->is_cpu);
    int rs2 =
        rf->get_register(warp->warp_id, thread, in->getOperand(2).getReg(), warp->is_cpu);
    // RISC-V: shift amount is masked to 5 bits (rs2 & 0x1F)
    unsigned int shamt = static_cast<unsigned int>(rs2) & 0x1F;
    rf->set_register(warp->warp_id, thread, rd,
                     static_cast<uint64_t>(rs1) << shamt, warp->is_cpu);

    warp->pc[thread] += 4;
  }
  return active_threads.size() > 0;
}
bool ExecutionUnit::slli(Warp *warp, std::vector<size_t> active_threads,
                         MCInst *in) {
  assert(in->getNumOperands() == 3);
  for (auto thread : active_threads) {
    unsigned int rd = in->getOperand(0).getReg();
    int rs1 =
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg(), warp->is_cpu);
    int64_t imm = in->getOperand(2).getImm();
    uint64_t result = static_cast<uint64_t>(rs1) << imm;
    rf->set_register(warp->warp_id, thread, rd, static_cast<int>(result), warp->is_cpu);

    warp->pc[thread] += 4;
  }
  return active_threads.size() > 0;
}
bool ExecutionUnit::srl(Warp *warp, std::vector<size_t> active_threads,
                        MCInst *in) {
  assert(in->getNumOperands() == 3);
  for (auto thread : active_threads) {
    unsigned int rd = in->getOperand(0).getReg();
    int rs1 =
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg(), warp->is_cpu);
    int rs2 =
        rf->get_register(warp->warp_id, thread, in->getOperand(2).getReg(), warp->is_cpu);
    // RISC-V: shift amount is masked to 5 bits (rs2 & 0x1F)
    // SRL is logical right shift - treat rs1 as unsigned 32-bit value
    unsigned int shamt = static_cast<unsigned int>(rs2) & 0x1F;
    uint32_t rs1_unsigned = static_cast<uint32_t>(rs1);
    uint64_t result = static_cast<uint64_t>(rs1_unsigned) >> shamt;
    rf->set_register(warp->warp_id, thread, rd, static_cast<int>(result), warp->is_cpu);

    warp->pc[thread] += 4;
  }
  return active_threads.size() > 0;
}
bool ExecutionUnit::srli(Warp *warp, std::vector<size_t> active_threads,
                         MCInst *in) {
  assert(in->getNumOperands() == 3);
  for (auto thread : active_threads) {
    unsigned int rd = in->getOperand(0).getReg();
    int rs1 =
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg(), warp->is_cpu);
    int64_t imm = in->getOperand(2).getImm();
    // SRLI is a logical right shift - treat rs1 as unsigned 32-bit value
    // Mask to 32 bits first to avoid sign extension when casting negative values
    uint32_t rs1_unsigned = static_cast<uint32_t>(rs1);
    uint64_t result = static_cast<uint64_t>(rs1_unsigned) >> imm;
    rf->set_register(warp->warp_id, thread, rd, static_cast<int>(result), warp->is_cpu);

    warp->pc[thread] += 4;
  }
  return active_threads.size() > 0;
}
bool ExecutionUnit::sra(Warp *warp, std::vector<size_t> active_threads,
                        MCInst *in) {
  assert(in->getNumOperands() == 3);
  for (auto thread : active_threads) {
    unsigned int rd = in->getOperand(0).getReg();
    int rs1 =
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg(), warp->is_cpu);
    int rs2 =
        rf->get_register(warp->warp_id, thread, in->getOperand(2).getReg(), warp->is_cpu);
    // RISC-V: shift amount is masked to 5 bits (rs2 & 0x1F)
    // SRA is arithmetic right shift - preserve sign
    unsigned int shamt = static_cast<unsigned int>(rs2) & 0x1F;
    rf->set_register(warp->warp_id, thread, rd, rs1 >> shamt, warp->is_cpu);

    warp->pc[thread] += 4;
  }
  return active_threads.size() > 0;
}
bool ExecutionUnit::srai(Warp *warp, std::vector<size_t> active_threads,
                         MCInst *in) {
  assert(in->getNumOperands() == 3);
  for (auto thread : active_threads) {
    unsigned int rd = in->getOperand(0).getReg();
    int rs1 =
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg(), warp->is_cpu);
    int64_t imm = in->getOperand(2).getImm();
    rf->set_register(warp->warp_id, thread, rd, rs1 >> imm, warp->is_cpu);

    warp->pc[thread] += 4;
  }
  return active_threads.size() > 0;
}
bool ExecutionUnit::lui(Warp *warp, std::vector<size_t> active_threads,
                        MCInst *in) {
  assert(in->getNumOperands() == 2);
  for (auto thread : active_threads) {
    unsigned int rd = in->getOperand(0).getReg();
    int64_t imm = in->getOperand(1).getImm();
    rf->set_register(warp->warp_id, thread, rd,
                     static_cast<uint64_t>(imm) << 12);

    warp->pc[thread] += 4;
  }
  return active_threads.size() > 0;
}
bool ExecutionUnit::auipc(Warp *warp, std::vector<size_t> active_threads,
                          MCInst *in) {
  assert(in->getNumOperands() == 2);
  for (auto thread : active_threads) {
    unsigned int rd = in->getOperand(0).getReg();
    int64_t imm = in->getOperand(1).getImm();
    rf->set_register(warp->warp_id, thread, rd,
                     warp->pc[thread] + (static_cast<uint64_t>(imm) << 12));

    warp->pc[thread] += 4;
  }
  return active_threads.size() > 0;
}
bool ExecutionUnit::lw(Warp *warp, std::vector<size_t> active_threads,
                       MCInst *in) {
  assert(in->getNumOperands() == 3);

  // Matching SIMTight: check canPut before accepting memory request
  // If queue is full, return false to trigger retry (PC should NOT advance)
  if (!cu->can_put()) {
    return false;  // Memory system busy, need to retry - PC stays unchanged
  }

  std::vector<uint64_t> addresses;
  std::vector<size_t> valid_threads;
  unsigned int rd = in->getOperand(0).getReg();
  unsigned int base = in->getOperand(1).getReg();
  int64_t disp = in->getOperand(2).getImm();

  for (auto thread : active_threads) {
    int rs1 = rf->get_register(warp->warp_id, thread, base, warp->is_cpu);
    // RISC-V 64-bit: addresses are zero-extended from 32-bit register values
    // rs1 is stored as int (32-bit signed), but addresses are unsigned - zero-extend to 64-bit
    uint64_t rs1_64 = static_cast<uint32_t>(rs1);  // Zero-extend from 32-bit
    uint64_t addr = rs1_64 + static_cast<uint64_t>(static_cast<int64_t>(disp));
    addresses.push_back(addr);
    valid_threads.push_back(thread);
    
    // Debug logging for GPU Warp 0 Thread 0 (not CPU)
    if (warp->warp_id == 0 && thread == 0 && !warp->is_cpu && Config::instance().isDebug()) {
      std::ostringstream oss;
      oss << "GPU Warp 0 Thread 0: LW addr=0x" << std::hex << addr 
          << " rd=" << std::dec << (rd - llvm::RISCV::X0)
          << " (rs1=0x" << std::hex << static_cast<uint32_t>(rs1) 
          << " disp=0x" << static_cast<int64_t>(disp) << std::dec << ")";
      log("Load", oss.str());
    }
  }

  // Queue the load request (warp will be suspended, results written on resume)
  // LW is signed, so use sign-extension (default)
  cu->load(warp, addresses, WORD_SIZE, rd, valid_threads, false);

  // Advance PC only after successful memory request (matching SIMTight)
  for (auto thread : valid_threads) {
    warp->pc[thread] += 4;
  }

  // Return false (no immediate write required - results written on resume)
  return false;
}

bool ExecutionUnit::lh(Warp *warp, std::vector<size_t> active_threads,
                       MCInst *in) {
  assert(in->getNumOperands() == 3);

  // Matching SIMTight: check canPut before accepting memory request
  if (!cu->can_put()) {
    return false;  // Memory system busy, need to retry - PC stays unchanged
  }

  std::vector<uint64_t> addresses;
  std::vector<size_t> valid_threads;
  unsigned int rd = in->getOperand(0).getReg();
  unsigned int base = in->getOperand(1).getReg();
  int64_t disp = in->getOperand(2).getImm();

  for (auto thread : active_threads) {
    int rs1 = rf->get_register(warp->warp_id, thread, base, warp->is_cpu);
    // RISC-V 64-bit: zero-extend 32-bit register value to 64-bit address
    uint64_t rs1_64 = static_cast<uint32_t>(rs1);
    uint64_t addr = rs1_64 + static_cast<uint64_t>(static_cast<int64_t>(disp));
    addresses.push_back(addr);
    valid_threads.push_back(thread);
  }

  // Queue the load request (warp will be suspended, results written on resume)
  // LH is signed, so use sign-extension (default)
  cu->load(warp, addresses, WORD_SIZE / 2, rd, valid_threads, false);

  // Advance PC only after successful memory request
  for (auto thread : valid_threads) {
    warp->pc[thread] += 4;
  }

  // Return false (no immediate write required - results written on resume)
  return false;
}

bool ExecutionUnit::lhu(Warp *warp, std::vector<size_t> active_threads,
                       MCInst *in) {
  assert(in->getNumOperands() == 3);

  // Matching SIMTight: check canPut before accepting memory request
  if (!cu->can_put()) {
    return false;  // Memory system busy, need to retry - PC stays unchanged
  }

  std::vector<uint64_t> addresses;
  std::vector<size_t> valid_threads;
  unsigned int rd = in->getOperand(0).getReg();
  unsigned int base = in->getOperand(1).getReg();
  int64_t disp = in->getOperand(2).getImm();

  for (auto thread : active_threads) {
    int rs1 = rf->get_register(warp->warp_id, thread, base, warp->is_cpu);
    // RISC-V 64-bit: zero-extend 32-bit register value to 64-bit address
    uint64_t rs1_64 = static_cast<uint32_t>(rs1);
    uint64_t addr = rs1_64 + static_cast<uint64_t>(static_cast<int64_t>(disp));
    addresses.push_back(addr);
    valid_threads.push_back(thread);
  }

  // Queue the load request (warp will be suspended, results written on resume)
  // LHU is unsigned, so use zero-extension
  cu->load(warp, addresses, WORD_SIZE / 2, rd, valid_threads, true);

  // Advance PC only after successful memory request
  for (auto thread : valid_threads) {
    warp->pc[thread] += 4;
  }

  // Return false (no immediate write required - results written on resume)
  return false;
}

bool ExecutionUnit::lb(Warp *warp, std::vector<size_t> active_threads,
                       MCInst *in) {
  assert(in->getNumOperands() == 3);

  // Matching SIMTight: check canPut before accepting memory request
  if (!cu->can_put()) {
    return false;  // Memory system busy, need to retry - PC stays unchanged
  }

  std::vector<uint64_t> addresses;
  std::vector<size_t> valid_threads;
  unsigned int rd = in->getOperand(0).getReg();
  unsigned int base = in->getOperand(1).getReg();
  int64_t disp = in->getOperand(2).getImm();

  for (auto thread : active_threads) {
    int rs1 = rf->get_register(warp->warp_id, thread, base, warp->is_cpu);
    // RISC-V 64-bit: zero-extend 32-bit register value to 64-bit address
    uint64_t rs1_64 = static_cast<uint32_t>(rs1);
    uint64_t addr = rs1_64 + static_cast<uint64_t>(static_cast<int64_t>(disp));
    addresses.push_back(addr);
    valid_threads.push_back(thread);
  }

  // Queue the load request (warp will be suspended, results written on resume)
  // LB is signed, so use sign-extension (default)
  cu->load(warp, addresses, 1, rd, valid_threads, false);

  // Advance PC only after successful memory request
  for (auto thread : valid_threads) {
    warp->pc[thread] += 4;
  }

  // Return false (no immediate write required - results written on resume)
  return false;
}

bool ExecutionUnit::lbu(Warp *warp, std::vector<size_t> active_threads,
                        MCInst *in) {
  assert(in->getNumOperands() == 3);

  // Matching SIMTight: check canPut before accepting memory request
  if (!cu->can_put()) {
    return false;  // Memory system busy, need to retry - PC stays unchanged
  }

  std::vector<uint64_t> addresses;
  std::vector<size_t> valid_threads;
  unsigned int rd = in->getOperand(0).getReg();
  unsigned int base = in->getOperand(1).getReg();
  int64_t disp = in->getOperand(2).getImm();

  for (auto thread : active_threads) {
    int rs1 = rf->get_register(warp->warp_id, thread, base, warp->is_cpu);
    // RISC-V 64-bit: zero-extend 32-bit register value to 64-bit address
    uint64_t rs1_64 = static_cast<uint32_t>(rs1);
    uint64_t addr = rs1_64 + static_cast<uint64_t>(static_cast<int64_t>(disp));
    addresses.push_back(addr);
    valid_threads.push_back(thread);
  }

  // Queue the load request (warp will be suspended, results written on resume)
  // LBU is unsigned, so use zero-extension
  cu->load(warp, addresses, 1, rd, valid_threads, true);

  // Advance PC only after successful memory request
  for (auto thread : valid_threads) {
    warp->pc[thread] += 4;
  }

  // Return false (no immediate write required - results written on resume)
  return false;
}

bool ExecutionUnit::sw(Warp *warp, std::vector<size_t> active_threads,
                       MCInst *in) {
  assert(in->getNumOperands() == 3);

  // Matching SIMTight: check canPut before accepting memory request
  if (!cu->can_put()) {
    return false;  // Memory system busy, need to retry - PC stays unchanged
  }

  std::vector<uint64_t> addresses;
  std::vector<int> values;
  std::vector<size_t> valid_threads;
  unsigned int base = in->getOperand(1).getReg();
  int64_t disp = in->getOperand(2).getImm();
  unsigned int rs2_reg = in->getOperand(0).getReg();

  for (auto thread : active_threads) {
    int rs2 = rf->get_register(warp->warp_id, thread, rs2_reg, warp->is_cpu);
    int rs1 = rf->get_register(warp->warp_id, thread, base, warp->is_cpu);
    // RISC-V 64-bit: zero-extend 32-bit register value to 64-bit address
    uint64_t rs1_64 = static_cast<uint32_t>(rs1);
    uint64_t addr = rs1_64 + static_cast<uint64_t>(static_cast<int64_t>(disp));
    addresses.push_back(addr);
    values.push_back(rs2);
    valid_threads.push_back(thread);
    
    // Debug logging for GPU Warp 0 Thread 0 (not CPU)
    if (warp->warp_id == 0 && thread == 0 && !warp->is_cpu && Config::instance().isDebug()) {
      std::ostringstream oss;
      oss << "GPU Warp 0 Thread 0: SW addr=0x" << std::hex << addr 
          << " value=0x" << static_cast<uint32_t>(rs2)
          << " (rs1=0x" << static_cast<uint32_t>(rs1) 
          << " rs2=0x" << static_cast<uint32_t>(rs2)
          << " disp=0x" << static_cast<int64_t>(disp) << std::dec << ")";
      log("Store", oss.str());
    }
  }

  cu->store(warp, addresses, WORD_SIZE, values, valid_threads);

  // Advance PC only after successful memory request
  for (auto thread : valid_threads) {
    warp->pc[thread] += 4;
  }
  return !warp->suspended;
}

bool ExecutionUnit::sh(Warp *warp, std::vector<size_t> active_threads,
                       MCInst *in) {
  assert(in->getNumOperands() == 3);

  // Matching SIMTight: check canPut before accepting memory request
  if (!cu->can_put()) {
    return false;  // Memory system busy, need to retry - PC stays unchanged
  }

  std::vector<uint64_t> addresses;
  std::vector<int> values;
  std::vector<size_t> valid_threads;
  unsigned int base = in->getOperand(1).getReg();
  int64_t disp = in->getOperand(2).getImm();
  unsigned int rs2_reg = in->getOperand(0).getReg();

  for (auto thread : active_threads) {
    int rs2 = rf->get_register(warp->warp_id, thread, rs2_reg, warp->is_cpu);
    int rs1 = rf->get_register(warp->warp_id, thread, base, warp->is_cpu);
    // RISC-V 64-bit: zero-extend 32-bit register value to 64-bit address
    uint64_t rs1_64 = static_cast<uint32_t>(rs1);
    uint64_t addr = rs1_64 + static_cast<uint64_t>(static_cast<int64_t>(disp));
    addresses.push_back(addr);
    values.push_back(rs2);
    valid_threads.push_back(thread);
  }

  cu->store(warp, addresses, WORD_SIZE / 2, values, valid_threads);

  // Advance PC only after successful memory request
  for (auto thread : valid_threads) {
    warp->pc[thread] += 4;
  }
  return !warp->suspended;
}

bool ExecutionUnit::sb(Warp *warp, std::vector<size_t> active_threads,
                       MCInst *in) {
  assert(in->getNumOperands() == 3);

  // Matching SIMTight: check canPut before accepting memory request
  if (!cu->can_put()) {
    return false;  // Memory system busy, need to retry - PC stays unchanged
  }

  std::vector<uint64_t> addresses;
  std::vector<int> values;
  std::vector<size_t> valid_threads;
  unsigned int base = in->getOperand(1).getReg();
  int64_t disp = in->getOperand(2).getImm();
  unsigned int rs2_reg = in->getOperand(0).getReg();

  for (auto thread : active_threads) {
    int rs2 = rf->get_register(warp->warp_id, thread, rs2_reg, warp->is_cpu);
    int rs1 = rf->get_register(warp->warp_id, thread, base, warp->is_cpu);
    // RISC-V 64-bit: zero-extend 32-bit register value to 64-bit address
    uint64_t rs1_64 = static_cast<uint32_t>(rs1);
    uint64_t addr = rs1_64 + static_cast<uint64_t>(static_cast<int64_t>(disp));
    addresses.push_back(addr);
    values.push_back(rs2);
    valid_threads.push_back(thread);
  }

  cu->store(warp, addresses, 1, values, valid_threads);

  // Advance PC only after successful memory request
  for (auto thread : valid_threads) {
    warp->pc[thread] += 4;
  }
  return !warp->suspended;
}

bool ExecutionUnit::amoadd_w(Warp *warp, std::vector<size_t> active_threads,
                              MCInst *in) {
  // AMOADD_W instruction: amoadd.w rd, rs2, (rs1)
  // Performs: old_value = *rs1; *rs1 = old_value + rs2; rd = old_value
  // Operands: rd (dest), rs2 (value to add), rs1 (base address), offset (usually 0)
  assert(in->getNumOperands() >= 3);

  // Matching SIMTight: check canPut before accepting memory request
  if (!cu->can_put()) {
    return false;  // Memory system busy, need to retry - PC stays unchanged
  }

  std::vector<uint64_t> addresses;
  std::vector<int> add_values;
  std::vector<size_t> valid_threads;
  unsigned int rd = in->getOperand(0).getReg();
  unsigned int rs2_reg = in->getOperand(1).getReg();
  unsigned int rs1_reg = in->getOperand(2).getReg();
  int64_t offset = 0;
  if (in->getNumOperands() >= 4) {
    offset = in->getOperand(3).getImm();
  }

  for (auto thread : active_threads) {
    int rs2 = rf->get_register(warp->warp_id, thread, rs2_reg, warp->is_cpu);
    int rs1 = rf->get_register(warp->warp_id, thread, rs1_reg, warp->is_cpu);
    // RISC-V 64-bit: zero-extend 32-bit register value to 64-bit address
    uint64_t rs1_64 = static_cast<uint32_t>(rs1);
    uint64_t addr = rs1_64 + static_cast<uint64_t>(static_cast<int64_t>(offset));
    addresses.push_back(addr);
    add_values.push_back(rs2);
    valid_threads.push_back(thread);
  }

  // Queue the atomic add request (warp will be suspended, old values written on resume)
  cu->atomic_add(warp, addresses, WORD_SIZE, rd, add_values, valid_threads);

  // Advance PC only after successful memory request
  for (auto thread : valid_threads) {
    warp->pc[thread] += 4;
  }

  // Return false (no immediate write required - old values written on resume)
  return false;
}
bool ExecutionUnit::jal(Warp *warp, std::vector<size_t> active_threads,
                        MCInst *in) {
  assert(in->getNumOperands() == 2);

  for (auto thread : active_threads) {
    int rd = in->getOperand(0).getReg();
    int64_t imm = in->getOperand(1).getImm();

    rf->set_register(warp->warp_id, thread, rd, warp->pc[thread] + 4, warp->is_cpu);
    warp->pc[thread] += imm;
  }
  return active_threads.size() > 0;
}
bool ExecutionUnit::jalr(Warp *warp, std::vector<size_t> active_threads,
                         MCInst *in) {
  assert(in->getNumOperands() == 3);

  for (auto thread : active_threads) {
    int rd = in->getOperand(0).getReg();
    int rs1 =
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg(), warp->is_cpu);
    int64_t imm = in->getOperand(2).getImm();

    rf->set_register(warp->warp_id, thread, rd, warp->pc[thread] + 4, warp->is_cpu);
    // RISC-V 64-bit: zero-extend 32-bit register value to 64-bit address
    // JALR: target address is (rs1 + imm) with LSB cleared (& ~1)
    uint64_t rs1_64 = static_cast<uint32_t>(rs1);
    uint64_t target = (rs1_64 + static_cast<uint64_t>(static_cast<int64_t>(imm))) & ~1ULL;
    if (target == 0) {
      warp->finished[thread] = true;
    } else {
      warp->pc[thread] = target;
    }
  }
  return active_threads.size() > 0;
}
bool ExecutionUnit::beq(Warp *warp, std::vector<size_t> active_threads,
                        MCInst *in) {
  assert(in->getNumOperands() == 3);

  for (auto thread : active_threads) {
    int rs1 =
        rf->get_register(warp->warp_id, thread, in->getOperand(0).getReg(), warp->is_cpu);
    int rs2 =
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg(), warp->is_cpu);
    int64_t imm = in->getOperand(2).getImm();

    if (rs1 == rs2) {
      warp->pc[thread] += imm;
    } else {
      warp->pc[thread] += 4;
    }
  }
  return active_threads.size() > 0;
}
bool ExecutionUnit::bne(Warp *warp, std::vector<size_t> active_threads,
                        MCInst *in) {
  assert(in->getNumOperands() == 3);

  for (auto thread : active_threads) {
    int rs1 =
        rf->get_register(warp->warp_id, thread, in->getOperand(0).getReg(), warp->is_cpu);
    int rs2 =
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg(), warp->is_cpu);
    int64_t imm = in->getOperand(2).getImm();

    if (rs1 != rs2) {
      warp->pc[thread] += imm;
    } else {
      warp->pc[thread] += 4;
    }
  }
  return active_threads.size() > 0;
}
bool ExecutionUnit::blt(Warp *warp, std::vector<size_t> active_threads,
                        MCInst *in) {
  assert(in->getNumOperands() == 3);

  for (auto thread : active_threads) {
    int rs1 =
        rf->get_register(warp->warp_id, thread, in->getOperand(0).getReg(), warp->is_cpu);
    int rs2 =
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg(), warp->is_cpu);
    int64_t imm = in->getOperand(2).getImm();

    if (rs1 < rs2) {
      warp->pc[thread] += imm;
    } else {
      warp->pc[thread] += 4;
    }
  }
  return active_threads.size() > 0;
}
bool ExecutionUnit::bltu(Warp *warp, std::vector<size_t> active_threads,
                         MCInst *in) {
  assert(in->getNumOperands() == 3);

  for (auto thread : active_threads) {
    int rs1 =
        rf->get_register(warp->warp_id, thread, in->getOperand(0).getReg(), warp->is_cpu);
    int rs2 =
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg(), warp->is_cpu);
    int64_t imm = in->getOperand(2).getImm();

    if (uint64_t(rs1) < uint64_t(rs2)) {
      warp->pc[thread] += imm;
    } else {
      warp->pc[thread] += 4;
    }
  }
  return active_threads.size() > 0;
}
bool ExecutionUnit::bge(Warp *warp, std::vector<size_t> active_threads,
                        MCInst *in) {
  assert(in->getNumOperands() == 3);

  for (auto thread : active_threads) {
    int rs1 =
        rf->get_register(warp->warp_id, thread, in->getOperand(0).getReg(), warp->is_cpu);
    int rs2 =
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg(), warp->is_cpu);
    int64_t imm = in->getOperand(2).getImm();

    if (rs1 >= rs2) {
      warp->pc[thread] += imm;
    } else {
      warp->pc[thread] += 4;
    }
  }
  return active_threads.size() > 0;
}
bool ExecutionUnit::bgeu(Warp *warp, std::vector<size_t> active_threads,
                         MCInst *in) {
  assert(in->getNumOperands() == 3);

  for (auto thread : active_threads) {
    int rs1 =
        rf->get_register(warp->warp_id, thread, in->getOperand(0).getReg(), warp->is_cpu);
    int rs2 =
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg(), warp->is_cpu);
    int64_t imm = in->getOperand(2).getImm();

    if (uint64_t(rs1) >= uint64_t(rs2)) {
      warp->pc[thread] += imm;
    } else {
      warp->pc[thread] += 4;
    }
  }
  return active_threads.size() > 0;
}
bool ExecutionUnit::slti(Warp *warp, std::vector<size_t> active_threads,
                         MCInst *in) {
  assert(in->getNumOperands() == 3);
  for (auto thread : active_threads) {
    unsigned int rd = in->getOperand(0).getReg();
    int rs1 =
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg(), warp->is_cpu);
    int64_t imm = in->getOperand(2).getImm();
    rf->set_register(warp->warp_id, thread, rd, (rs1 < imm) ? 1 : 0, warp->is_cpu);

    warp->pc[thread] += 4;
  }
  return active_threads.size() > 0;
}

bool ExecutionUnit::slt(Warp *warp, std::vector<size_t> active_threads,
                        MCInst *in) {
  assert(in->getNumOperands() == 3);
  for (auto thread : active_threads) {
    unsigned int rd = in->getOperand(0).getReg();
    int rs1 =
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg(), warp->is_cpu);
    int rs2 =
        rf->get_register(warp->warp_id, thread, in->getOperand(2).getReg(), warp->is_cpu);
    rf->set_register(warp->warp_id, thread, rd, (rs1 < rs2) ? 1 : 0, warp->is_cpu);

    warp->pc[thread] += 4;
  }
  return active_threads.size() > 0;
}

bool ExecutionUnit::sltiu(Warp *warp, std::vector<size_t> active_threads,
                          MCInst *in) {
  assert(in->getNumOperands() == 3);
  for (auto thread : active_threads) {
    unsigned int rd = in->getOperand(0).getReg();
    int rs1 =
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg(), warp->is_cpu);
    int64_t imm = in->getOperand(2).getImm();
    rf->set_register(
        warp->warp_id, thread, rd,
        (static_cast<uint32_t>(rs1) < static_cast<uint32_t>(imm)) ? 1 : 0, warp->is_cpu);

    warp->pc[thread] += 4;
  }
  return active_threads.size() > 0;
}

bool ExecutionUnit::sltu(Warp *warp, std::vector<size_t> active_threads,
                         MCInst *in) {
  assert(in->getNumOperands() == 3);
  for (auto thread : active_threads) {
    unsigned int rd = in->getOperand(0).getReg();
    int rs1 =
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg(), warp->is_cpu);
    int rs2 =
        rf->get_register(warp->warp_id, thread, in->getOperand(2).getReg(), warp->is_cpu);
    rf->set_register(
        warp->warp_id, thread, rd,
        (static_cast<uint32_t>(rs1) < static_cast<uint32_t>(rs2)) ? 1 : 0, warp->is_cpu);

    warp->pc[thread] += 4;
  }
  return active_threads.size() > 0;
}

bool ExecutionUnit::remu(Warp *warp, std::vector<size_t> active_threads,
                         MCInst *in) {
  assert(in->getNumOperands() == 3);
  for (auto thread : active_threads) {
    unsigned int rd = in->getOperand(0).getReg();
    int rs1 =
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg(), warp->is_cpu);
    int rs2 =
        rf->get_register(warp->warp_id, thread, in->getOperand(2).getReg(), warp->is_cpu);
    
    uint32_t u_rs1 = static_cast<uint32_t>(rs1);
    uint32_t u_rs2 = static_cast<uint32_t>(rs2);
    int result;
    
    if (u_rs2 == 0) {
      result = static_cast<int>(u_rs1);  // REMU: remainder is numerator
    } else {
      result = static_cast<int>(u_rs1 % u_rs2);
    }
    
    rf->set_register(warp->warp_id, thread, rd, result, warp->is_cpu);
    warp->pc[thread] += 4;
  }
  return active_threads.size() > 0;
}

bool ExecutionUnit::divu(Warp *warp, std::vector<size_t> active_threads,
                         MCInst *in) {
  assert(in->getNumOperands() == 3);
  for (auto thread : active_threads) {
    unsigned int rd = in->getOperand(0).getReg();
    int rs1 =
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg(), warp->is_cpu);
    int rs2 =
        rf->get_register(warp->warp_id, thread, in->getOperand(2).getReg(), warp->is_cpu);
    
    uint32_t u_rs1 = static_cast<uint32_t>(rs1);
    uint32_t u_rs2 = static_cast<uint32_t>(rs2);
    int result;
    
    if (u_rs2 == 0) {
      result = 0xFFFFFFFF;  // DIVU: quotient is all ones
    } else {
      result = static_cast<int>(u_rs1 / u_rs2);
    }
    
    rf->set_register(warp->warp_id, thread, rd, result, warp->is_cpu);
    warp->pc[thread] += 4;
  }
  return active_threads.size() > 0;
}

bool ExecutionUnit::div_(Warp *warp, std::vector<size_t> active_threads,
                         MCInst *in) {
  assert(in->getNumOperands() == 3);
  for (auto thread : active_threads) {
    unsigned int rd = in->getOperand(0).getReg();
    int rs1 =
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg(), warp->is_cpu);
    int rs2 =
        rf->get_register(warp->warp_id, thread, in->getOperand(2).getReg(), warp->is_cpu);
    
    int result;
    if (rs2 == 0) {
      result = -1;  // DIV: quotient is all ones
    } else if (rs1 == INT32_MIN && rs2 == -1) {
      result = INT32_MIN;  // DIV: result is -2^31 (overflow case)
    } else {
      result = rs1 / rs2;
    }
    
    rf->set_register(warp->warp_id, thread, rd, result, warp->is_cpu);
    warp->pc[thread] += 4;
  }
  return active_threads.size() > 0;
}

bool ExecutionUnit::rem_(Warp *warp, std::vector<size_t> active_threads,
                         MCInst *in) {
  assert(in->getNumOperands() == 3);
  for (auto thread : active_threads) {
    unsigned int rd = in->getOperand(0).getReg();
    int rs1 =
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg(), warp->is_cpu);
    int rs2 =
        rf->get_register(warp->warp_id, thread, in->getOperand(2).getReg(), warp->is_cpu);
    
    int result;
    if (rs2 == 0) {
      result = rs1;  // REM: remainder is numerator
    } else if (rs1 == INT32_MIN && rs2 == -1) {
      result = 0;  // REM: remainder is 0 (overflow case)
    } else {
      result = rs1 % rs2;
    }
    
    rf->set_register(warp->warp_id, thread, rd, result, warp->is_cpu);
    warp->pc[thread] += 4;
  }
  return active_threads.size() > 0;
}

bool ExecutionUnit::fence(Warp *warp, std::vector<size_t> active_threads,
                          MCInst *in) {
  // Matching SIMTight: check canPut before accepting memory fence request
  // If queue is full, return false to trigger retry (PC should NOT advance)
  if (!cu->can_put()) {
    return false;  // Memory system busy, need to retry - PC stays unchanged
  }

  // Queue the fence request (warp will be suspended, resumes when fence completes)
  cu->fence(warp);

  // Advance PC only after successful memory fence request (matching SIMTight)
  for (auto thread : active_threads) {
    warp->pc[thread] += 4;
  }

  // Return false (no immediate write required - fence completes on resume)
  return false;
}

bool ExecutionUnit::ecall(Warp *warp, std::vector<size_t> active_threads,
                          MCInst *in) {
  assert(in->getNumOperands() == 0);

  // This log call is inside ExecutionUnit, which is NOT a PipelineStage.
  // However, ExecutionUnit is owned by ExecuteSuspend.
  // We can't easily access ExecuteSuspend's log method.
  // But wait, ExecutionUnit calls global log.
  // If we want to mute this, we need to pass debug flag to ExecutionUnit too.
  // Or just let it log globally? No, user wants to mute CPU logs.
  // So ExecutionUnit needs to know if it's CPU or GPU.
  // Let's assume for now we leave it global, or update ExecutionUnit later.
  // Actually, ExecutionUnit has a lot of logs.

  // Let's stick to PipelineStage subclasses first.
  // ExecuteSuspend calls log.

  log("ExUn - Operating System", "Received an ecall");
  for (auto thread : active_threads) {
    warp->pc[thread] += 4;
  }
  return false;
}
bool ExecutionUnit::ebreak(Warp *warp, std::vector<size_t> active_threads,
                           MCInst *in) {
  assert(in->getNumOperands() == 0);

  log("ExUn - Debugger", "Received an ebreak");
  for (auto thread : active_threads) {
    warp->pc[thread] += 4;
  }
  return false;
}

bool ExecutionUnit::csrrw(Warp *warp, std::vector<size_t> active_threads,
                          MCInst *in) {
  assert(in->getNumOperands() == 3);

  for (auto thread : active_threads) {
    int csr = in->getOperand(1).getImm();
    int rd_reg = in->getOperand(0).getReg();
    int rs1_reg = in->getOperand(2).getReg();
    int rs1_val = rf->get_register(warp->warp_id, thread, rs1_reg, warp->is_cpu);

    bool handled = true;
    switch (csr) {
    case 0x800: {
      // SimEmit: Write-only CSR, emits word in simulation
      // In SIMTight: csrWrite = Just \x -> do display "0x" (formatHex 8 x)
      if (!Config::instance().isStatsOnly()) {
        std::cout << "[SimEmit] 0x" << std::hex << rs1_val << std::dec << std::endl;
      }
      // Write-only, so reads return undefined (we return 0)
      rf->set_register(warp->warp_id, thread, rd_reg, 0, warp->is_cpu);
    } break;
    case 0x801: {
      // SimFinish: Write-only CSR, terminates simulator
      // In SIMTight: csrWrite = Just \x -> do finish
      // For now, we'll just log it (actual termination handled elsewhere)
      if (!Config::instance().isStatsOnly()) {
        std::cout << "[SimFinish] Terminating simulator" << std::endl;
      }
      // Write-only, so reads return undefined (we return 0)
      rf->set_register(warp->warp_id, thread, rd_reg, 0, warp->is_cpu);
    } break;
    case 0x802:
      rf->set_register(warp->warp_id, thread, rd_reg, 1, warp->is_cpu);
      break;
    case 0x803: {
      // UART Put: Write byte to UART
      // Buffer the output for both CPU and GPU (printed later in main)
      char byte_val = static_cast<char>(rs1_val);
      gpu_controller->buffer_data(byte_val);
      // Write-only CSR, so reads return undefined (we return 0)
      rf->set_register(warp->warp_id, thread, rd_reg, 0, warp->is_cpu);
    } break;
    case 0x804:
      // Just treat as always ready
      rf->set_register(warp->warp_id, thread, rd_reg, 1, warp->is_cpu);
      break;
    case 0xF14: {
      // Hardware thread ID (hart ID) matching SIMTight's calculation:
      // hartId = zeroExtend (warpId # laneId) = (warpId << SIMTLogLanes) | laneId
      // See SIMTight/src/Core/SIMT.hs line 99: let hartId = zeroExtend (ins.execWarpId # ins.execLaneId)
      // SIMTLogLanes = 5 (since NUM_LANES = 32 = 2^5)
      constexpr unsigned SIMTLogLanes = 5;
      uint32_t mhartid_uint = (static_cast<uint32_t>(warp->warp_id) << SIMTLogLanes) | static_cast<uint32_t>(thread);
      int mhartid = static_cast<int>(mhartid_uint);
      rf->set_register(warp->warp_id, thread, rd_reg, mhartid, warp->is_cpu);
      
      // Debug logging for first few warps/threads to verify mhartid calculation
      if (warp->warp_id < 4 && thread < 2 && !warp->is_cpu && Config::instance().isDebug()) {
        std::ostringstream oss;
        oss << "Warp " << warp->warp_id << " Thread " << thread 
            << ": mhartid=0x" << std::hex << mhartid_uint 
            << " (" << std::dec << mhartid << ")";
        log("CSR", oss.str());
      }
    } break;
    case 0x805: {
      static std::string input_buffer = "16\n";
      static size_t input_index = 0;

      int input_char = 0;
      if (input_index < input_buffer.size()) {
        input_char = input_buffer[input_index++];
      } else {
        input_char = -1; // EOF
      }
      if (!Config::instance().isStatsOnly())
        std::cout << "[Input] Returning " << input_char << std::endl;
      rf->set_register(warp->warp_id, thread, rd_reg, input_char, warp->is_cpu);
    } break;
    case 0x806: {
      // InstrAddr: Write-only CSR, sets instruction mem address (for CPU)
      // In SIMTight: csrWrite = Just \x -> do addrReg <== x
      // This is CPU-only, but we should handle it to avoid errors
      // Write-only, so reads return undefined (we return 0)
      rf->set_register(warp->warp_id, thread, rd_reg, 0, warp->is_cpu);
    } break;
    case 0x807: {
      // WriteInstr: Write-only CSR, writes to instruction mem (for CPU)
      // In SIMTight: csrWrite = Just \x -> do writeInstr (addrReg.val) x
      // This is CPU-only, but we should handle it to avoid errors
      // Write-only, so reads return undefined (we return 0)
      rf->set_register(warp->warp_id, thread, rd_reg, 0, warp->is_cpu);
    } break;
    case 0x820: {
      // Read-only CSR: returns 1 if can put (queue not full), 0 if can't put
      // In SIMTight: csrRead = Just do return (zeroExtend reqs.notFull)
      bool active = gpu_controller->is_gpu_active();
      int can_put = active ? 0 : 1;  // Can put if GPU is not active
      rf->set_register(warp->warp_id, thread, rd_reg, can_put, warp->is_cpu);
      // Writes to CSR 0x820 are ignored (read-only)
    } break;
    case 0x821: {
      // SIMTInstrAddr: Write-only CSR, sets instruction mem address (for SIMT)
      // In SIMTight: csrWrite = Just \x -> do addrReg <== x
      // This is CPU-only (used to write instructions to SIMT instruction memory)
      // Write-only, so reads return undefined (we return 0)
      rf->set_register(warp->warp_id, thread, rd_reg, 0, warp->is_cpu);
    } break;
    case 0x822: {
      // SIMTWriteInstr: Write-only CSR, writes to instruction mem (for SIMT)
      // In SIMTight: csrWrite = Just \x -> do enq reqs (simtCmd_WriteInstr, addrReg.val, x)
      // This is CPU-only (used to write instructions to SIMT instruction memory)
      // Write-only, so reads return undefined (we return 0)
      rf->set_register(warp->warp_id, thread, rd_reg, 0, warp->is_cpu);
    } break;
    case 0x823: {
      // Write-only CSR: writing PC starts kernel (if rs1_val != 0)
      // In SIMTight: csrWrite = Just \x -> do enq reqs (simtCmd_StartPipeline, x, ...)
      if (rs1_val != 0) {
        gpu_controller->set_pc(rs1_val);
        gpu_controller->launch_kernel();
      }
      // CSR 0x823 is write-only, so reads return undefined (we return 0)
      rf->set_register(warp->warp_id, thread, rd_reg, 0, warp->is_cpu);
    } break;
    case 0x824: {
      bool active = gpu_controller->is_gpu_active();
      int status = active ? 0 : 1;
      rf->set_register(warp->warp_id, thread, rd_reg, status, warp->is_cpu);
    } break;
    case 0x825: {
      // SIMTGet: Read-only CSR, gets SIMT response (stat value after SIMTAskStats)
      // In SIMTight: csrRead = Just do return (zeroExtend resp.val)
      // This is a global CSR read by CPU, stored in HostGPUControl
      // Return as unsigned to match pebblesSIMTGet() return type
      // IMPORTANT: CPU has warp_id=0, same as GPU warp 0, but HostRegisterFile
      // isolates CPU registers from GPU registers, so this should write to CPU's
      // own register file, not GPU warp 0's register file
      unsigned val = gpu_controller->get_stat_value();
      // For CPU (is_cpu=true), HostRegisterFile::set_register() ignores warp_id/thread
      // and writes to CPU's own registers vector, so this is safe
      int reg_val = static_cast<int>(val);
      rf->set_register(warp->warp_id, thread, rd_reg, reg_val, warp->is_cpu);
      // Writes to CSR 0x825 are ignored (read-only)
    } break;
    case 0x826: {
      // RISC-V 64-bit: addresses are 32-bit, zero-extended to 64-bit (not sign-extended)
      // rs1_val contains a 32-bit address value, we need to zero-extend it
      uint64_t arg_addr = static_cast<uint32_t>(rs1_val);
      gpu_controller->set_arg_ptr(arg_addr);
      break;
    }
    case 0x827: {
      // SIMTSetWarpsPerBlock: Write-only CSR, sets number of warps per block
      // (A block is a group of threads that synchronise on a barrier)
      // (A value of 0 indicates all warps form one block)
      // Matching SIMTight: simtCmd_SetWarpsPerBlock command
      // In SIMTight: warpsPerBlock = n, barrierMask = (n == 0) ? all_ones : (1 << n) - 1
      unsigned warps_per_block = static_cast<unsigned>(rs1_val);
      gpu_controller->set_warps_per_block(warps_per_block);
      // Write-only, so reads return undefined (we return 0)
      rf->set_register(warp->warp_id, thread, rd_reg, 0, warp->is_cpu);
    } break;
    case 0x828: {
      // SIMTAskStats: Write-only CSR, requests a stat counter
      // In SIMTight: writes request to queue, response comes via SIMTGet (0x825)
      // For simplicity, we directly compute and store the stat value in CSR 0x825
      // (matching the behavior but not the exact mechanism)
      uint64_t val = 0;
      switch (rs1_val) {
      case 0:  // STAT_SIMT_CYCLES
        val = GPUStatisticsManager::instance().get_gpu_cycles();
        break;
      case 1:  // STAT_SIMT_INSTRS
        val = GPUStatisticsManager::instance().get_gpu_instrs();
        break;
      case 5:  // STAT_SIMT_RETRIES
        val = GPUStatisticsManager::instance().get_gpu_retries();
        break;
      case 6:  // STAT_SIMT_SUSP_BUBBLES
        val = GPUStatisticsManager::instance().get_gpu_susps();
        break;
      case 9:  // STAT_SIMT_DRAM_ACCESSES
        val = GPUStatisticsManager::instance().get_gpu_dram_accs();
        break;
      default:
        val = 0;
        break;
      }
      // Store result in HostGPUControl (SIMTGet CSR 0x825 will read it)
      // This is a global CSR, not per-warp
      // Truncate to 32 bits (matching SIMTight's 32-bit response)
      unsigned stat_val = static_cast<unsigned>(val & 0xFFFFFFFFU);
      gpu_controller->set_stat_value(stat_val);
      // Write-only, so reads return undefined (we return 0)
      rf->set_register(warp->warp_id, thread, rd_reg, 0, warp->is_cpu);
    } break;
    case 0x830: {
      std::optional<int> val = rf->get_csr(warp->warp_id, thread, 0x830);
      if (val.has_value()) {
        rf->set_register(warp->warp_id, thread, rd_reg, val.value(), warp->is_cpu);
      } else {
        rf->set_register(warp->warp_id, thread, rd_reg, 0, warp->is_cpu);
      }
      if (rs1_val != 0 || in->getOperand(0).getReg() == 0) {
        rf->set_csr(warp->warp_id, thread, 0x830, rs1_val);
        // Matching SIMTight: writing 0 to CSR 0x830 is a barrier command
        // Writing non-zero is termination (handled separately if needed)
        if (rs1_val == 0) {
          // Barrier: mark warp as in barrier (matching SIMTight: barrierBits!warpId5 <== true)
          // SIMTight asserts that warp must be converged before entering barrier:
          // dynamicAssert (inv excGlobal.val .==>. activeMask5 .==. ones)
          //   "SIMT pipeline: warp command issued by diverged warp"
          // This means all threads should be active (converged) before barrier entry.
          // We check this by verifying all non-finished threads have the same PC and nesting level.
          bool all_converged = true;
          uint64_t leader_pc = 0;
          uint64_t leader_nesting = 0;
          bool found_leader = false;
          
          for (size_t t = 0; t < warp->size; t++) {
            if (warp->finished[t]) continue;
            if (!found_leader) {
              leader_pc = warp->pc[t];
              leader_nesting = warp->nesting_level[t];
              found_leader = true;
            } else {
              if (warp->pc[t] != leader_pc || warp->nesting_level[t] != leader_nesting) {
                all_converged = false;
                break;
              }
            }
          }
          
          if (!all_converged && found_leader) {
            // Log warning: barrier entered by diverged warp (shouldn't happen if noclPop works correctly)
            std::string name = warp->is_cpu ? "CPU" : "Warp " + std::to_string(warp->warp_id);
            log("CSRRW", "WARNING: Barrier entered by diverged " + name + 
                " (PCs/nesting levels differ). This may indicate a convergence issue.");
          }
          
          warp->in_barrier = true;
          // Warp is not suspended for barrier - it's just marked as in barrier
          // The scheduler will skip warps in barrier, and barrier release will clear the flag
        } else {
          // Termination: mark warp as finished (matching SIMTight: completedWarps <== completedWarps.val + 1)
          // For now, just mark all threads as finished
          for (size_t t = 0; t < warp->size; t++) {
            warp->finished[t] = true;
          }
        }
      }
    } break;
    case 0x831: {
      uint64_t args = gpu_controller->get_arg_ptr();
      // CSR 0x831 returns 32-bit address (as per SIMTight)
      // The address is 32-bit, but we need to preserve it correctly when storing in int register
      // Cast to uint32_t first to get the lower 32 bits, then to int (which will preserve the bit pattern)
      uint32_t args_u32 = static_cast<uint32_t>(args);
      int args_32 = static_cast<int>(args_u32);
      rf->set_register(warp->warp_id, thread, rd_reg, args_32, warp->is_cpu);
    } break;
    case 0xc00: {
      // Cycle: Read-only CSR, cycle count (lower 32 bits)
      // In SIMTight: csrRead = Just do return (lower count.val)
      uint64_t cycles = GPUStatisticsManager::instance().get_gpu_cycles();
      rf->set_register(warp->warp_id, thread, rd_reg, cycles & 0xFFFFFFFF, warp->is_cpu);
    } break;
    case 0xc80: {
      // CycleH: Read-only CSR, cycle count (upper 32 bits)
      // In SIMTight: csrRead = Just do return (upper count.val)
      uint64_t cycles = GPUStatisticsManager::instance().get_gpu_cycles();
      rf->set_register(warp->warp_id, thread, rd_reg, (cycles >> 32) & 0xFFFFFFFF, warp->is_cpu);
    } break;
    default:
      handled = false;
      break;
    }

    if (handled) {
      warp->pc[thread] += 4;
      continue;
    }

    std::optional<int> csrr = rf->get_csr(warp->warp_id, thread, csr);
    if (!csrr.has_value()) {
      std::string name = warp->is_cpu ? "CPU" : "Warp " + std::to_string(warp->warp_id);
      log("CSRRW", "Control/Status Register " + std::to_string(csr) +
                       " is undefined for " + name + " and thread " +
                       std::to_string(thread) +
                       " -> trapping (skipping for now)");
      continue;
    }
    rf->set_register(warp->warp_id, thread, rd_reg, csrr.value(), warp->is_cpu);
    rf->set_csr(warp->warp_id, thread, csr, rs1_val);

    warp->pc[thread] += 4;
  }
  return false;
}
bool ExecutionUnit::noclpush(Warp *warp, std::vector<size_t> active_threads,
                             MCInst *in) {
  for (auto thread : active_threads) {
    warp->nesting_level[thread]++;
    warp->pc[thread] += 4;
  }
  return false;
}
bool ExecutionUnit::noclpop(Warp *warp, std::vector<size_t> active_threads,
                            MCInst *in) {
  // noclPop() semantics: causes threads to converge by waiting for all threads
  // that were active at the previous noclPush(). To ensure convergence, we need
  // to update ALL threads in the warp that have nesting_level >= 1, not just
  // the currently active threads. This ensures that threads that have diverged
  // will eventually converge.
  //
  // Matching SIMTight behavior: noclPop() decrements nesting level for threads
  // that execute it. However, if only active threads (those with max nesting
  // level) execute it, threads with lower nesting levels won't advance their PC
  // and won't converge. By updating all threads with nesting_level >= 1, we
  // ensure they all advance their PC and will converge in subsequent cycles.
  for (size_t thread = 0; thread < warp->size; thread++) {
    if (!warp->finished[thread] && warp->nesting_level[thread] >= 1) {
      warp->nesting_level[thread]--;
      warp->pc[thread] += 4;
    }
  }
  return false;
}
bool ExecutionUnit::cache_line_flush(Warp *warp,
                                     std::vector<size_t> active_threads,
                                     MCInst *in) {
  for (auto thread : active_threads) {
    // TODO: Implement this function
    warp->pc[thread] += 4;
  }
  return false;
}

ExecuteSuspend::ExecuteSuspend(CoalescingUnit *cu, RegisterFile *rf,
                               uint64_t max_addr, LLVMDisassembler *disasm,
                               HostGPUControl *gpu_controller)
    : max_addr(max_addr), cu(cu), disasm(disasm) {
  eu = new ExecutionUnit(cu, rf, disasm, gpu_controller);
  log("Execute/Suspend", "Initializing execute/suspend pipeline stage");
}

void ExecuteSuspend::execute() {
  // Check if we have a warp to process (either new or retrying)
  // Matching SIMTight: warps stay in execute stage when retrying
  if (!PipelineStage::input_latch->updated)
    return;
  
  Warp *warp = PipelineStage::input_latch->warp;
  MCInst inst = PipelineStage::input_latch->inst;
  std::vector<size_t> active_threads =
      PipelineStage::input_latch->active_threads;

  // Matching SIMTight: count suspension bubble when a suspended warp enters execute stage
  // This happens when the scheduler chooses a suspended warp (creating a pipeline bubble)
  if (warp->suspended && !warp->is_cpu) {
    GPUStatisticsManager::instance().increment_gpu_susps();
  }

  execute_result result = eu->execute(warp, active_threads, inst);

  // Handle retry logic (matching SIMTight behavior):
  // - When retry happens, warp STAYS in execute stage (does NOT advance)
  // - PC is NOT updated (stays the same)
  // - Warp keeps retrying the same instruction until canPut becomes true
  // - Instruction is NOT counted
  // - Retries are counted every cycle the warp is retrying
  // Matching SIMTight: retry is per-thread (per-lane), stored in thread state
  // Matching SIMTight: retry counter increments once per cycle when any retry is active
  // Check if warp was already retrying (before processing this cycle)
  bool was_retrying = false;
  for (auto thread : active_threads) {
    if (warp->retrying[thread]) {
      was_retrying = true;
      break;
    }
  }
  
  // Matching SIMTight: count retry every cycle that retryWire.val is true
  // If warp was already retrying, count it for this cycle
  if (was_retrying && !warp->is_cpu) {
    GPUStatisticsManager::instance().increment_gpu_retries();
  }
  
  if (!result.success && !warp->suspended && !warp->is_cpu) {
    // Retry needed: stay in execute stage, count retry, don't count instruction
    // PC was NOT updated in instruction function, so it stays the same
    // Count retry for this cycle (if not already counted above)
    if (!was_retrying) {
      GPUStatisticsManager::instance().increment_gpu_retries();
    }
    // Set retry flag for all active threads (matching SIMTight: retryWire.val per-lane)
    for (auto thread : active_threads) {
      warp->retrying[thread] = true;
    }
    // Warp stays in execute stage - keep input_latch updated so it's processed again next cycle (matches SIMTight: retryWire.val keeps warp in execute with same PC)
    PipelineStage::input_latch->updated = true;
    PipelineStage::output_latch->updated = false;  // No output on retry
    return;  // Don't process further - warp stays in execute and will retry next cycle
  } else {
    // Instruction succeeded or warp was suspended - clear retry flag for all active threads
    for (auto thread : active_threads) {
      warp->retrying[thread] = false;
    }
  }

  // Count instructions only if successful and not retried (matching SIMTight:
  // "when (inv retryWire.val) do incInstrCount <== true")
  if (result.success && result.counted) {
    if (!warp->is_cpu) {
      GPUStatisticsManager::instance().increment_gpu_instrs(
          active_threads.size());
    } else {
      GPUStatisticsManager::instance().increment_cpu_instrs();
    }
  }

  // Reinsert warp logic (matching SIMTight): if instruction succeeded and warp not suspended, reinsert; if suspended, don't reinsert yet (will resume later)
  if (!warp->suspended) {
    // Warp not suspended: reinsert it (instruction succeeded)
    for (int i = 0; i < warp->size; i++) {
      if (!warp->finished[i] && warp->pc[i] <= max_addr) {
        insert_warp(warp);
        break;
      }
    }
    PipelineStage::input_latch->updated = false;
  } else {
    // Warp is suspended (by functional unit or memory), will be resumed later
    PipelineStage::input_latch->updated = false;
  }
  // We use the updated flag to tell the writeback/resume stage
  // whether or not to "perform a writeback" or to check for memory
  // responses or functional unit completions
  PipelineStage::output_latch->updated = result.write_required;
  PipelineStage::output_latch->warp = warp;
  PipelineStage::output_latch->active_threads =
      PipelineStage::input_latch->active_threads;
  PipelineStage::output_latch->inst = PipelineStage::input_latch->inst;

  std::string inst_name = disasm->getOpcodeName(inst.getOpcode());
  std::stringstream op_stream;
  for (llvm::MCOperand op : inst.getOperands()) {
    op_stream << operandToString(op) << " ";
  }

  std::string name = warp->is_cpu ? "CPU" : "Warp " + std::to_string(warp->warp_id);
  if (!result.success) {
    log("Execute/Suspend", name +
                               " could not perform instruction " + inst_name);
    return;
  }

  log("Execute/Suspend", name +
                             " executed " + inst_name + "\t" + op_stream.str());
}

bool ExecuteSuspend::is_active() { return PipelineStage::input_latch->updated; }

ExecuteSuspend::~ExecuteSuspend() { delete eu; }