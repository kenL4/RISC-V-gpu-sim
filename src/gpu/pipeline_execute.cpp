#include "pipeline_execute.hpp"
#include "../disassembler/llvm_disasm.hpp"
#include "../stats/stats.hpp"
#include "../config.hpp"

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
    bool was_suspended_before = warp->suspended;
    res.write_required = mul(warp, active_threads, &inst);
    // If mul() returns false AND warp is not suspended, unit was busy - need to retry
    // If mul() returns false AND warp is suspended, operation succeeded (writeback happens later)
    // Matching SIMTight: retry when unit busy (warp not suspended), suspend when operation accepted
    if (!res.write_required && !warp->suspended) {
      res.success = false;
      res.counted = false;
    }
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
      static size_t retry_count = 0;
      retry_count++;
      if (retry_count <= 10) {  // Only print first 10
        std::cout << "[DEBUG] Memory retry detected: LW, warp_id=" << warp->warp_id 
                  << ", write_required=" << res.write_required 
                  << ", suspended=" << warp->suspended << std::endl;
      }
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
    bool was_suspended_before = warp->suspended;
    res.write_required = remu(warp, active_threads, &inst);
    // If remu() returns false AND warp is not suspended, unit was busy - need to retry
    // If remu() returns false AND warp is suspended, operation succeeded (writeback happens later)
    if (!res.write_required && !warp->suspended) {
      res.success = false;
      res.counted = false;
    }
  } else if (mnemonic == "DIVU") {
    bool was_suspended_before = warp->suspended;
    res.write_required = divu(warp, active_threads, &inst);
    // If divu() returns false AND warp is not suspended, unit was busy - need to retry
    // If divu() returns false AND warp is suspended, operation succeeded (writeback happens later)
    if (!res.write_required && !warp->suspended) {
      res.success = false;
      res.counted = false;
    }
  } else if (mnemonic == "DIV") {
    bool was_suspended_before = warp->suspended;
    res.write_required = div_(warp, active_threads, &inst);
    // If div_() returns false AND warp is not suspended, unit was busy - need to retry
    // If div_() returns false AND warp is suspended, operation succeeded (writeback happens later)
    if (!res.write_required && !warp->suspended) {
      res.success = false;
      res.counted = false;
    }
  } else if (mnemonic == "REM") {
    bool was_suspended_before = warp->suspended;
    res.write_required = rem_(warp, active_threads, &inst);
    // If rem_() returns false AND warp is not suspended, unit was busy - need to retry
    // If rem_() returns false AND warp is suspended, operation succeeded (writeback happens later)
    if (!res.write_required && !warp->suspended) {
      res.success = false;
      res.counted = false;
    }
  } else if (mnemonic == "FENCE") {
    res.write_required = fence(warp, active_threads, &inst);
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
    int rs1 =
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg());
    int rs2 =
        rf->get_register(warp->warp_id, thread, in->getOperand(2).getReg());
    rf->set_register(warp->warp_id, thread, rd, rs1 + rs2);

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
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg());
    int64_t imm = in->getOperand(2).getImm();
    rf->set_register(warp->warp_id, thread, rd, rs1 + imm);

    warp->pc[thread] += 4;
  }
  return active_threads.size() > 0;
}
bool ExecutionUnit::sub(Warp *warp, std::vector<size_t> active_threads,
                        MCInst *in) {
  assert(in->getNumOperands() == 3);
  for (auto thread : active_threads) {
    unsigned int rd = in->getOperand(0).getReg();
    int rs1 =
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg());
    int rs2 =
        rf->get_register(warp->warp_id, thread, in->getOperand(2).getReg());
    rf->set_register(warp->warp_id, thread, rd, rs1 - rs2);

    warp->pc[thread] += 4;
  }
  return active_threads.size() > 0;
}

bool ExecutionUnit::mul(Warp *warp, std::vector<size_t> active_threads,
                        MCInst *in) {
  assert(in->getNumOperands() == 3);
  
  unsigned int rd = in->getOperand(0).getReg();
  unsigned int rs1_reg = in->getOperand(1).getReg();
  unsigned int rs2_reg = in->getOperand(2).getReg();
  
  // Collect register values for all active threads
  std::map<size_t, int> rs1_vals, rs2_vals;
  for (auto thread : active_threads) {
    rs1_vals[thread] = rf->get_register(warp->warp_id, thread, rs1_reg);
    rs2_vals[thread] = rf->get_register(warp->warp_id, thread, rs2_reg);
  }
  
  // Issue to multiplier unit (will suspend warp)
  if (!mul_unit.issue(warp, active_threads, rs1_vals, rs2_vals, rd)) {
    return false;  // Unit is busy, need to retry
  }
  
  // Advance PC (warp is suspended, will resume when operation completes)
  for (auto thread : active_threads) {
    warp->pc[thread] += 4;
  }
  
  // Don't write yet - will write when operation completes
  return false;  // write_required = false, we'll handle writeback on resume
}
bool ExecutionUnit::and_(Warp *warp, std::vector<size_t> active_threads,
                         MCInst *in) {
  assert(in->getNumOperands() == 3);
  for (auto thread : active_threads) {
    unsigned int rd = in->getOperand(0).getReg();
    int rs1 =
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg());
    int rs2 =
        rf->get_register(warp->warp_id, thread, in->getOperand(2).getReg());
    rf->set_register(warp->warp_id, thread, rd, rs1 & rs2);

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
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg());
    int64_t imm = in->getOperand(2).getImm();
    rf->set_register(warp->warp_id, thread, rd, rs1 & imm);

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
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg());
    int rs2 =
        rf->get_register(warp->warp_id, thread, in->getOperand(2).getReg());
    rf->set_register(warp->warp_id, thread, rd, rs1 | rs2);

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
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg());
    int64_t imm = in->getOperand(2).getImm();
    rf->set_register(warp->warp_id, thread, rd, rs1 | imm);

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
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg());
    int rs2 =
        rf->get_register(warp->warp_id, thread, in->getOperand(2).getReg());
    rf->set_register(warp->warp_id, thread, rd, rs1 ^ rs2);

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
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg());
    int64_t imm = in->getOperand(2).getImm();
    rf->set_register(warp->warp_id, thread, rd, rs1 ^ imm);

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
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg());
    int rs2 =
        rf->get_register(warp->warp_id, thread, in->getOperand(2).getReg());
    rf->set_register(warp->warp_id, thread, rd,
                     static_cast<uint64_t>(rs1) << rs2);

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
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg());
    int64_t imm = in->getOperand(2).getImm();
    rf->set_register(warp->warp_id, thread, rd,
                     static_cast<uint64_t>(rs1) << imm);

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
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg());
    int rs2 =
        rf->get_register(warp->warp_id, thread, in->getOperand(2).getReg());
    rf->set_register(warp->warp_id, thread, rd,
                     static_cast<uint64_t>(rs1) >> rs2);

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
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg());
    int64_t imm = in->getOperand(2).getImm();
    rf->set_register(warp->warp_id, thread, rd,
                     static_cast<uint64_t>(rs1) >> imm);

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
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg());
    int rs2 =
        rf->get_register(warp->warp_id, thread, in->getOperand(2).getReg());
    rf->set_register(warp->warp_id, thread, rd, rs1 >> rs2);

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
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg());
    int64_t imm = in->getOperand(2).getImm();
    rf->set_register(warp->warp_id, thread, rd, rs1 >> imm);

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
  // If queue is full, return false to trigger retry
  if (!cu->can_put()) {
    return false;  // Memory system busy, need to retry
  }

  std::vector<uint64_t> addresses;
  std::vector<size_t> valid_threads;
  unsigned int rd = in->getOperand(0).getReg();
  unsigned int base = in->getOperand(1).getReg();
  int64_t disp = in->getOperand(2).getImm();

  for (auto thread : active_threads) {
    int rs1 = rf->get_register(warp->warp_id, thread, base);
    addresses.push_back(rs1 + disp);
    valid_threads.push_back(thread);
  }

  // Queue the load request (warp will be suspended, results written on resume)
  cu->load(warp, addresses, WORD_SIZE, rd, valid_threads);

  // Advance PC before returning (warp is suspended, but PC should advance)
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
    return false;  // Memory system busy, need to retry
  }

  std::vector<uint64_t> addresses;
  std::vector<size_t> valid_threads;
  unsigned int rd = in->getOperand(0).getReg();
  unsigned int base = in->getOperand(1).getReg();
  int64_t disp = in->getOperand(2).getImm();

  for (auto thread : active_threads) {
    int rs1 = rf->get_register(warp->warp_id, thread, base);
    addresses.push_back(rs1 + disp);
    valid_threads.push_back(thread);
  }

  // Queue the load request (warp will be suspended, results written on resume)
  cu->load(warp, addresses, WORD_SIZE / 2, rd, valid_threads);

  // Advance PC before returning (warp is suspended, but PC should advance)
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
    return false;  // Memory system busy, need to retry
  }

  std::vector<uint64_t> addresses;
  std::vector<size_t> valid_threads;
  unsigned int rd = in->getOperand(0).getReg();
  unsigned int base = in->getOperand(1).getReg();
  int64_t disp = in->getOperand(2).getImm();

  for (auto thread : active_threads) {
    int rs1 = rf->get_register(warp->warp_id, thread, base);
    addresses.push_back(rs1 + disp);
    valid_threads.push_back(thread);
  }

  // Queue the load request (warp will be suspended, results written on resume)
  cu->load(warp, addresses, WORD_SIZE / 2, rd, valid_threads);

  // Advance PC before returning (warp is suspended, but PC should advance)
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
    return false;  // Memory system busy, need to retry
  }

  std::vector<uint64_t> addresses;
  std::vector<size_t> valid_threads;
  unsigned int rd = in->getOperand(0).getReg();
  unsigned int base = in->getOperand(1).getReg();
  int64_t disp = in->getOperand(2).getImm();

  for (auto thread : active_threads) {
    int rs1 = rf->get_register(warp->warp_id, thread, base);
    addresses.push_back(rs1 + disp);
    valid_threads.push_back(thread);
  }

  // Queue the load request (warp will be suspended, results written on resume)
  cu->load(warp, addresses, 1, rd, valid_threads);

  // Advance PC before returning (warp is suspended, but PC should advance)
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
    return false;  // Memory system busy, need to retry
  }

  std::vector<uint64_t> addresses;
  std::vector<size_t> valid_threads;
  unsigned int rd = in->getOperand(0).getReg();
  unsigned int base = in->getOperand(1).getReg();
  int64_t disp = in->getOperand(2).getImm();

  for (auto thread : active_threads) {
    int rs1 = rf->get_register(warp->warp_id, thread, base);
    addresses.push_back(rs1 + disp);
    valid_threads.push_back(thread);
  }

  // Queue the load request (warp will be suspended, results written on resume)
  cu->load(warp, addresses, 1, rd, valid_threads);

  // Advance PC before returning (warp is suspended, but PC should advance)
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
    return false;  // Memory system busy, need to retry
  }

  std::vector<uint64_t> addresses;
  std::vector<int> values;
  std::vector<size_t> valid_threads;
  unsigned int base = in->getOperand(1).getReg();
  int64_t disp = in->getOperand(2).getImm();
  unsigned int rs2_reg = in->getOperand(0).getReg();

  for (auto thread : active_threads) {
    int rs2 = rf->get_register(warp->warp_id, thread, rs2_reg);
    int rs1 = rf->get_register(warp->warp_id, thread, base);
    addresses.push_back(rs1 + disp);
    values.push_back(rs2);
    valid_threads.push_back(thread);
  }

  cu->store(warp, addresses, WORD_SIZE, values);

  // Advance PC before returning (even if warp is suspended)
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
    return false;  // Memory system busy, need to retry
  }

  std::vector<uint64_t> addresses;
  std::vector<int> values;
  std::vector<size_t> valid_threads;
  unsigned int base = in->getOperand(1).getReg();
  int64_t disp = in->getOperand(2).getImm();
  unsigned int rs2_reg = in->getOperand(0).getReg();

  for (auto thread : active_threads) {
    int rs2 = rf->get_register(warp->warp_id, thread, rs2_reg);
    int rs1 = rf->get_register(warp->warp_id, thread, base);
    addresses.push_back(rs1 + disp);
    values.push_back(rs2);
    valid_threads.push_back(thread);
  }

  cu->store(warp, addresses, WORD_SIZE / 2, values);

  // Advance PC before returning (even if warp is suspended)
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
    return false;  // Memory system busy, need to retry
  }

  std::vector<uint64_t> addresses;
  std::vector<int> values;
  std::vector<size_t> valid_threads;
  unsigned int base = in->getOperand(1).getReg();
  int64_t disp = in->getOperand(2).getImm();
  unsigned int rs2_reg = in->getOperand(0).getReg();

  for (auto thread : active_threads) {
    int rs2 = rf->get_register(warp->warp_id, thread, rs2_reg);
    int rs1 = rf->get_register(warp->warp_id, thread, base);
    addresses.push_back(rs1 + disp);
    values.push_back(rs2);
    valid_threads.push_back(thread);
  }

  cu->store(warp, addresses, 1, values);

  // Advance PC before returning (even if warp is suspended)
  for (auto thread : valid_threads) {
    warp->pc[thread] += 4;
  }
  return !warp->suspended;
}
bool ExecutionUnit::jal(Warp *warp, std::vector<size_t> active_threads,
                        MCInst *in) {
  assert(in->getNumOperands() == 2);

  for (auto thread : active_threads) {
    int rd = in->getOperand(0).getReg();
    int64_t imm = in->getOperand(1).getImm();

    rf->set_register(warp->warp_id, thread, rd, warp->pc[thread] + 4);
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
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg());
    int64_t imm = in->getOperand(2).getImm();

    rf->set_register(warp->warp_id, thread, rd, warp->pc[thread] + 4);
    uint64_t target = rs1 + imm;
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
        rf->get_register(warp->warp_id, thread, in->getOperand(0).getReg());
    int rs2 =
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg());
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
        rf->get_register(warp->warp_id, thread, in->getOperand(0).getReg());
    int rs2 =
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg());
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
        rf->get_register(warp->warp_id, thread, in->getOperand(0).getReg());
    int rs2 =
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg());
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
        rf->get_register(warp->warp_id, thread, in->getOperand(0).getReg());
    int rs2 =
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg());
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
        rf->get_register(warp->warp_id, thread, in->getOperand(0).getReg());
    int rs2 =
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg());
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
        rf->get_register(warp->warp_id, thread, in->getOperand(0).getReg());
    int rs2 =
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg());
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
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg());
    int64_t imm = in->getOperand(2).getImm();
    rf->set_register(warp->warp_id, thread, rd, (rs1 < imm) ? 1 : 0);

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
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg());
    int rs2 =
        rf->get_register(warp->warp_id, thread, in->getOperand(2).getReg());
    rf->set_register(warp->warp_id, thread, rd, (rs1 < rs2) ? 1 : 0);

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
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg());
    int64_t imm = in->getOperand(2).getImm();
    rf->set_register(
        warp->warp_id, thread, rd,
        (static_cast<uint32_t>(rs1) < static_cast<uint32_t>(imm)) ? 1 : 0);

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
        rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg());
    int rs2 =
        rf->get_register(warp->warp_id, thread, in->getOperand(2).getReg());
    rf->set_register(
        warp->warp_id, thread, rd,
        (static_cast<uint32_t>(rs1) < static_cast<uint32_t>(rs2)) ? 1 : 0);

    warp->pc[thread] += 4;
  }
  return active_threads.size() > 0;
}

bool ExecutionUnit::remu(Warp *warp, std::vector<size_t> active_threads,
                         MCInst *in) {
  assert(in->getNumOperands() == 3);
  
  unsigned int rd = in->getOperand(0).getReg();
  unsigned int rs1_reg = in->getOperand(1).getReg();
  unsigned int rs2_reg = in->getOperand(2).getReg();
  
  // Collect register values for all active threads
  std::map<size_t, int> rs1_vals, rs2_vals;
  for (auto thread : active_threads) {
    rs1_vals[thread] = rf->get_register(warp->warp_id, thread, rs1_reg);
    rs2_vals[thread] = rf->get_register(warp->warp_id, thread, rs2_reg);
  }
  
  // Issue to divider unit (unsigned remainder)
  if (!div_unit.issue(warp, active_threads, rs1_vals, rs2_vals, rd, false, true)) {
    return false;  // Unit is busy, need to retry
  }
  
  // Advance PC
  for (auto thread : active_threads) {
    warp->pc[thread] += 4;
  }
  
  return false;  // write_required = false, we'll handle writeback on resume
}

bool ExecutionUnit::divu(Warp *warp, std::vector<size_t> active_threads,
                         MCInst *in) {
  assert(in->getNumOperands() == 3);
  
  unsigned int rd = in->getOperand(0).getReg();
  unsigned int rs1_reg = in->getOperand(1).getReg();
  unsigned int rs2_reg = in->getOperand(2).getReg();
  
  // Collect register values for all active threads
  std::map<size_t, int> rs1_vals, rs2_vals;
  for (auto thread : active_threads) {
    rs1_vals[thread] = rf->get_register(warp->warp_id, thread, rs1_reg);
    rs2_vals[thread] = rf->get_register(warp->warp_id, thread, rs2_reg);
  }
  
  // Issue to divider unit (unsigned division)
  if (!div_unit.issue(warp, active_threads, rs1_vals, rs2_vals, rd, false, false)) {
    return false;  // Unit is busy, need to retry
  }
  
  // Advance PC
  for (auto thread : active_threads) {
    warp->pc[thread] += 4;
  }
  
  return false;  // write_required = false, we'll handle writeback on resume
}

bool ExecutionUnit::div_(Warp *warp, std::vector<size_t> active_threads,
                         MCInst *in) {
  assert(in->getNumOperands() == 3);
  
  unsigned int rd = in->getOperand(0).getReg();
  unsigned int rs1_reg = in->getOperand(1).getReg();
  unsigned int rs2_reg = in->getOperand(2).getReg();
  
  // Collect register values for all active threads
  std::map<size_t, int> rs1_vals, rs2_vals;
  for (auto thread : active_threads) {
    rs1_vals[thread] = rf->get_register(warp->warp_id, thread, rs1_reg);
    rs2_vals[thread] = rf->get_register(warp->warp_id, thread, rs2_reg);
  }
  
  // Issue to divider unit (signed division)
  if (!div_unit.issue(warp, active_threads, rs1_vals, rs2_vals, rd, true, false)) {
    return false;  // Unit is busy, need to retry
  }
  
  // Advance PC
  for (auto thread : active_threads) {
    warp->pc[thread] += 4;
  }
  
  return false;  // write_required = false, we'll handle writeback on resume
}

bool ExecutionUnit::rem_(Warp *warp, std::vector<size_t> active_threads,
                         MCInst *in) {
  assert(in->getNumOperands() == 3);
  
  unsigned int rd = in->getOperand(0).getReg();
  unsigned int rs1_reg = in->getOperand(1).getReg();
  unsigned int rs2_reg = in->getOperand(2).getReg();
  
  // Collect register values for all active threads
  std::map<size_t, int> rs1_vals, rs2_vals;
  for (auto thread : active_threads) {
    rs1_vals[thread] = rf->get_register(warp->warp_id, thread, rs1_reg);
    rs2_vals[thread] = rf->get_register(warp->warp_id, thread, rs2_reg);
  }
  
  // Issue to divider unit (signed remainder)
  if (!div_unit.issue(warp, active_threads, rs1_vals, rs2_vals, rd, true, true)) {
    return false;  // Unit is busy, need to retry
  }
  
  // Advance PC
  for (auto thread : active_threads) {
    warp->pc[thread] += 4;
  }
  
  return false;  // write_required = false, we'll handle writeback on resume
}

bool ExecutionUnit::fence(Warp *warp, std::vector<size_t> active_threads,
                          MCInst *in) {
  // FENCE is currently a no-op in this simulator as we don't model
  // complex memory consistency or out-of-order execution that requires it.
  // We just advance the PC.
  for (auto thread : active_threads) {
    warp->pc[thread] += 4;
  }
  return active_threads.size() > 0;
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
    int rs1_val =
        rf->get_register(warp->warp_id, thread, in->getOperand(2).getReg());

    bool handled = true;
    switch (csr) {
    case 0x802:
      rf->set_register(warp->warp_id, thread, rd_reg, 1);
      break;
    case 0x803:
      gpu_controller->buffer_data(static_cast<char>(rs1_val));
      break;
    case 0x804:
      // Just treat as always ready
      rf->set_register(warp->warp_id, thread, rd_reg, 1);
      break;
    case 0xF14: {
      int mhartid = warp->warp_id * 32 + thread; // Assuming 32 threads per warp
      rf->set_register(warp->warp_id, thread, rd_reg, mhartid);
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
      rf->set_register(warp->warp_id, thread, rd_reg, input_char);
    } break;
    case 0x820: {
      // Read-only CSR: returns 1 if can put (queue not full), 0 if can't put
      // In SIMTight: csrRead = Just do return (zeroExtend reqs.notFull)
      bool active = gpu_controller->is_gpu_active();
      int can_put = active ? 0 : 1;  // Can put if GPU is not active
      rf->set_register(warp->warp_id, thread, rd_reg, can_put);
      // Writes to CSR 0x820 are ignored (read-only)
    } break;
    case 0x823: {
      // Write-only CSR: writing PC starts kernel (if rs1_val != 0)
      // In SIMTight: csrWrite = Just \x -> do enq reqs (simtCmd_StartPipeline, x, ...)
      if (rs1_val != 0) {
        gpu_controller->set_pc(rs1_val);
        gpu_controller->launch_kernel();
      }
      // CSR 0x823 is write-only, so reads return undefined (we return 0)
      rf->set_register(warp->warp_id, thread, rd_reg, 0);
    } break;
    case 0x824: {
      bool active = gpu_controller->is_gpu_active();
      int status = active ? 0 : 1;
      rf->set_register(warp->warp_id, thread, rd_reg, status);
    } break;
    case 0x825:
      rf->set_register(warp->warp_id, thread, rd_reg, 0);
      break;
    case 0x826:
      gpu_controller->set_arg_ptr(rs1_val);
      break;
    case 0x827:
      gpu_controller->set_dims(rs1_val);
      break;
    case 0x828: {
      uint64_t val = 0;
      switch (rs1_val) {
      case 0:
        val = GPUStatisticsManager::instance().get_gpu_cycles();
        break;
      case 1:
        val = GPUStatisticsManager::instance().get_gpu_instrs();
        break;
      case 9:
        val = GPUStatisticsManager::instance().get_gpu_dram_accs();
        break;
      default:
        val = 0;
        break;
      }
      rf->set_csr(warp->warp_id, thread, 0x825, val);
    } break;
    case 0x830: {
      std::optional<int> val = rf->get_csr(warp->warp_id, thread, 0x830);
      if (val.has_value()) {
        rf->set_register(warp->warp_id, thread, rd_reg, val.value());
      } else {
        rf->set_register(warp->warp_id, thread, rd_reg, 0);
      }
      if (rs1_val != 0 || in->getOperand(0).getReg() == 0) {
        rf->set_csr(warp->warp_id, thread, 0x830, rs1_val);
      }
    } break;
    case 0x831: {
      uint64_t args = gpu_controller->get_arg_ptr();
      rf->set_register(warp->warp_id, thread, rd_reg, args);
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
    rf->set_register(warp->warp_id, thread, rd_reg, csrr.value());
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
  for (auto thread : active_threads) {
    warp->nesting_level[thread]--;
    warp->pc[thread] += 4;
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
  // Tick functional units every cycle
  eu->get_mul_unit().tick();
  eu->get_div_unit().tick();
  
  // Check if we have a warp to process (either new or retrying)
  // Matching SIMTight: warps stay in execute stage when retrying
  if (!PipelineStage::input_latch->updated)
    return;
  
  Warp *warp = PipelineStage::input_latch->warp;
  MCInst inst = PipelineStage::input_latch->inst;
  std::vector<size_t> active_threads =
      PipelineStage::input_latch->active_threads;

  execute_result result = eu->execute(warp, active_threads, inst);

  // Count retries when instruction needs to retry (matching SIMTight: "when retryWire.val do incRetryCount <== true")
  // In SIMTight, retryWire is set when execute stage calls retry, and retry count is incremented every cycle
  // Since warps stay in execute stage when retrying, we count retries every cycle the instruction needs to retry
  if (!result.success && !warp->suspended && !warp->is_cpu) {
    GPUStatisticsManager::instance().increment_gpu_retries();
    warp->retrying = true;
  } else {
    // Instruction succeeded or warp was suspended - clear retry flag
    warp->retrying = false;
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

  // Reinsert warp logic (matching SIMTight behavior):
  // - If instruction succeeded and warp not suspended: reinsert
  // - If instruction needs retry (result.success == false): reinsert without advancing PC
  //   Note: We reinsert to avoid blocking the pipeline, but count retries when the warp
  //   is in execute stage and needs to retry (matching SIMTight's per-cycle retry counting)
  // - If warp was suspended (by functional unit or memory): don't reinsert yet (will resume later)
  if (!warp->suspended) {
    // Warp not suspended: reinsert it
    // Note: If result.success == false (retry needed), PC was NOT advanced,
    // so reinserting will retry the same instruction - correct behavior
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