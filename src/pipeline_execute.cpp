#include "pipeline_execute.hpp"

// In RISC-V, Word is always 32-bit (4 bytes)
#define WORD_SIZE 4

ExecutionUnit::ExecutionUnit(CoalescingUnit *cu, RegisterFile *rf, LLVMDisassembler *disasm): 
                    cu(cu), rf(rf), disasm(disasm) {}

execute_result ExecutionUnit::execute(Warp *warp, std::vector<size_t> active_threads, MCInst &inst) {
    execute_result res { true, false };

    std::string mnemonic = disasm->getOpcodeName(inst.getOpcode());
    if (mnemonic == "ADDI") {
        res.write_required = addi(warp, active_threads, &inst);
    } else if (mnemonic == "ADD") {
        res.write_required = add(warp, active_threads, &inst);
    } else if (mnemonic == "SUB") {
        res.write_required = sub(warp, active_threads, &inst);
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
    } else if (mnemonic == "LH") {
        res.write_required = lh(warp, active_threads, &inst);
    } else if (mnemonic == "LHU") {
        res.write_required = lhu(warp, active_threads, &inst);
    } else if (mnemonic == "LB") {
        res.write_required = lb(warp, active_threads, &inst);
    } else if (mnemonic == "SW") {
        res.write_required = sw(warp, active_threads, &inst);
    } else if (mnemonic == "SH") {
        res.write_required = sh(warp, active_threads, &inst);
    } else if (mnemonic == "SB") {
        res.write_required = sb(warp, active_threads, &inst);
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
    } else if (mnemonic == "ECALL") {
        res.write_required = ecall(warp, active_threads, &inst);
    } else if (mnemonic == "EBREAK") {
        res.write_required = ebreak(warp, active_threads, &inst);
    } else if (mnemonic == "CSRRW") {
        res.write_required = csrrw(warp, active_threads, &inst);
    } else {
        // Default to skip instruction
        for (auto thread : active_threads) {
            warp->pc[thread] += 4;
        }
        res.success = false;
        std::cout << "[WARNING] Unknown instruction " << mnemonic << std::endl;
    }
    return res;
}

bool ExecutionUnit::add(Warp *warp, std::vector<size_t> active_threads, MCInst* in) {
    assert(in->getNumOperands() == 3);
    for (auto thread : active_threads) {
        unsigned int rd = in->getOperand(0).getReg();
        int rs1 = rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg());
        int rs2 = rf->get_register(warp->warp_id, thread, in->getOperand(2).getReg());
        rf->set_register(warp->warp_id, thread, rd, rs1 + rs2);

        warp->pc[thread] += 4;
    }
    return active_threads.size() > 0;
}
bool ExecutionUnit::addi(Warp *warp, std::vector<size_t> active_threads, MCInst* in) {
    assert(in->getNumOperands() == 3);
    for (auto thread : active_threads) {
        unsigned int rd = in->getOperand(0).getReg();
        int rs1 = rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg());
        int64_t imm = in->getOperand(2).getImm();
        rf->set_register(warp->warp_id, thread, rd, rs1 + imm);
        
        warp->pc[thread] += 4;
    }
    return active_threads.size() > 0;
}
bool ExecutionUnit::sub(Warp *warp, std::vector<size_t> active_threads, MCInst* in) {
    assert(in->getNumOperands() == 3);
    for (auto thread : active_threads) {
        unsigned int rd = in->getOperand(0).getReg();
        int rs1 = rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg());
        int rs2 = rf->get_register(warp->warp_id, thread, in->getOperand(2).getReg());
        rf->set_register(warp->warp_id, thread, rd, rs1 - rs2);

        warp->pc[thread] += 4;
    }
    return active_threads.size() > 0;
}
bool ExecutionUnit::and_(Warp *warp, std::vector<size_t> active_threads, MCInst* in) {
    assert(in->getNumOperands() == 3);
    for (auto thread : active_threads) {
        unsigned int rd = in->getOperand(0).getReg();
        int rs1 = rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg());
        int rs2 = rf->get_register(warp->warp_id, thread, in->getOperand(2).getReg());
        rf->set_register(warp->warp_id, thread, rd, rs1 & rs2);

        warp->pc[thread] += 4;
    }
    return active_threads.size() > 0;
}
bool ExecutionUnit::andi(Warp *warp, std::vector<size_t> active_threads, MCInst* in) {
    assert(in->getNumOperands() == 3);
    for (auto thread : active_threads) {
        unsigned int rd = in->getOperand(0).getReg();
        int rs1 = rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg());
        int64_t imm = in->getOperand(2).getImm();
        rf->set_register(warp->warp_id, thread, rd, rs1 & imm);

        warp->pc[thread] += 4;
    }
    return active_threads.size() > 0;
}
bool ExecutionUnit::or_(Warp *warp, std::vector<size_t> active_threads, MCInst* in) {
    assert(in->getNumOperands() == 3);
    for (auto thread : active_threads) {
        unsigned int rd = in->getOperand(0).getReg();
        int rs1 = rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg());
        int rs2 = rf->get_register(warp->warp_id, thread, in->getOperand(2).getReg());
        rf->set_register(warp->warp_id, thread, rd, rs1 | rs2);

        warp->pc[thread] += 4;
    }
    return active_threads.size() > 0;
}
bool ExecutionUnit::ori(Warp *warp, std::vector<size_t> active_threads, MCInst* in) {
    assert(in->getNumOperands() == 3);
    for (auto thread : active_threads) {
        unsigned int rd = in->getOperand(0).getReg();
        int rs1 = rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg());
        int64_t imm = in->getOperand(2).getImm();
        rf->set_register(warp->warp_id, thread, rd, rs1 | imm);

        warp->pc[thread] += 4;
    }
    return active_threads.size() > 0;
}
bool ExecutionUnit::xor_(Warp *warp, std::vector<size_t> active_threads, MCInst* in) {
    assert(in->getNumOperands() == 3);
    for (auto thread : active_threads) {
        unsigned int rd = in->getOperand(0).getReg();
        int rs1 = rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg());
        int rs2 = rf->get_register(warp->warp_id, thread, in->getOperand(2).getReg());
        rf->set_register(warp->warp_id, thread, rd, rs1 ^ rs2);

        warp->pc[thread] += 4;
    }
    return active_threads.size() > 0;
}
bool ExecutionUnit::xori(Warp *warp, std::vector<size_t> active_threads, MCInst* in) {
    assert(in->getNumOperands() == 3);
    for (auto thread : active_threads) {
        unsigned int rd = in->getOperand(0).getReg();
        int rs1 = rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg());
        int64_t imm = in->getOperand(2).getImm();
        rf->set_register(warp->warp_id, thread, rd, rs1 ^ imm);

        warp->pc[thread] += 4;
    }
    return active_threads.size() > 0;
}
bool ExecutionUnit::sll(Warp *warp, std::vector<size_t> active_threads, MCInst* in) {
    assert(in->getNumOperands() == 3);
    for (auto thread : active_threads) {
        unsigned int rd = in->getOperand(0).getReg();
        int rs1 = rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg());
        int rs2 = rf->get_register(warp->warp_id, thread, in->getOperand(2).getReg());
        rf->set_register(warp->warp_id, thread, rd, static_cast<uint64_t>(rs1) << rs2);

        warp->pc[thread] += 4;
    }
    return active_threads.size() > 0;
}
bool ExecutionUnit::slli(Warp *warp, std::vector<size_t> active_threads, MCInst* in) {
    assert(in->getNumOperands() == 3);
    for (auto thread : active_threads) {
        unsigned int rd = in->getOperand(0).getReg();
        int rs1 = rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg());
        int64_t imm = in->getOperand(2).getImm();
        rf->set_register(warp->warp_id, thread, rd, static_cast<uint64_t>(rs1) << imm);

        warp->pc[thread] += 4;
    }
    return active_threads.size() > 0;
}
bool ExecutionUnit::srl(Warp *warp, std::vector<size_t> active_threads, MCInst* in) {
    assert(in->getNumOperands() == 3);
    for (auto thread : active_threads) {
        unsigned int rd = in->getOperand(0).getReg();
        int rs1 = rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg());
        int rs2 = rf->get_register(warp->warp_id, thread, in->getOperand(2).getReg());
        rf->set_register(warp->warp_id, thread, rd, static_cast<uint64_t>(rs1) >> rs2);

        warp->pc[thread] += 4;
    }
    return active_threads.size() > 0;
}
bool ExecutionUnit::srli(Warp *warp, std::vector<size_t> active_threads, MCInst* in) {
    assert(in->getNumOperands() == 3);
    for (auto thread : active_threads) {
        unsigned int rd = in->getOperand(0).getReg();
        int rs1 = rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg());
        int64_t imm = in->getOperand(2).getImm();
        rf->set_register(warp->warp_id, thread, rd, static_cast<uint64_t>(rs1) >> imm);

        warp->pc[thread] += 4;
    }
    return active_threads.size() > 0;
}
bool ExecutionUnit::sra(Warp *warp, std::vector<size_t> active_threads, MCInst* in) {
    assert(in->getNumOperands() == 3);
    for (auto thread : active_threads) {
        unsigned int rd = in->getOperand(0).getReg();
        int rs1 = rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg());
        int rs2 = rf->get_register(warp->warp_id, thread, in->getOperand(2).getReg());
        rf->set_register(warp->warp_id, thread, rd, rs1 >> rs2);

        warp->pc[thread] += 4;
    }
    return active_threads.size() > 0;
}
bool ExecutionUnit::srai(Warp *warp, std::vector<size_t> active_threads, MCInst* in) {
    assert(in->getNumOperands() == 3);
    for (auto thread : active_threads) {
        unsigned int rd = in->getOperand(0).getReg();
        int rs1 = rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg());
        int64_t imm = in->getOperand(2).getImm();
        rf->set_register(warp->warp_id, thread, rd, rs1 >> imm);

        warp->pc[thread] += 4;
    }
    return active_threads.size() > 0;
}
bool ExecutionUnit::lui(Warp *warp, std::vector<size_t> active_threads, MCInst* in) {
    assert(in->getNumOperands() == 2);
    for (auto thread : active_threads) {
        unsigned int rd = in->getOperand(0).getReg();
        int64_t imm = in->getOperand(1).getImm();
        rf->set_register(warp->warp_id, thread, rd, static_cast<uint64_t>(imm) << 12);

        warp->pc[thread] += 4;
    }
    return active_threads.size() > 0;
}
bool ExecutionUnit::auipc(Warp *warp, std::vector<size_t> active_threads, MCInst* in) {
    assert(in->getNumOperands() == 2);
    for (auto thread : active_threads) {
        unsigned int rd = in->getOperand(0).getReg();
        int64_t imm = in->getOperand(1).getImm();
        rf->set_register(warp->warp_id, thread, rd, warp->pc[thread] + (static_cast<uint64_t>(imm) << 12));

        warp->pc[thread] += 4;
    }
    return active_threads.size() > 0;
}
bool ExecutionUnit::lw(Warp *warp, std::vector<size_t> active_threads, MCInst* in) {
    assert(in->getNumOperands() == 3);

    for (auto thread : active_threads) {
        unsigned int rd = in->getOperand(0).getReg();
        unsigned int base = in->getOperand(1).getReg();
        int64_t disp = in->getOperand(2).getImm();
        int rs1 = rf->get_register(warp->warp_id, thread, base);
        int res = cu->load(warp, rs1 + disp, WORD_SIZE);
        
        // Eventhough we update the register values here
        // The warp will be suspended so the updates won't be visible
        // till after it resumes
        rf->set_register(warp->warp_id, thread, rd, res);
        warp->pc[thread] += 4;
    }
    // After a load instruction, you don't need to writeback unless
    // the warp was never actually suspended
    return !warp->suspended;
}
bool ExecutionUnit::lh(Warp *warp, std::vector<size_t> active_threads, MCInst* in) {
    assert(in->getNumOperands() == 3);

    for (auto thread : active_threads) {
        unsigned int rd = in->getOperand(0).getReg();
        unsigned int base = in->getOperand(1).getReg();
        int64_t disp = in->getOperand(2).getImm();
        int rs1 = rf->get_register(warp->warp_id, thread, base);
        int res = cu->load(warp, rs1 + disp, WORD_SIZE / 2);
        
        // Eventhough we update the register values here
        // The warp will be suspended so the updates won't be visible
        // till after it resumes
        rf->set_register(warp->warp_id, thread, rd, res);
        warp->pc[thread] += 4;
    }
    // After a load instruction, you don't need to writeback unless
    // the warp was never actually suspended
    return !warp->suspended;
}
bool ExecutionUnit::lhu(Warp *warp, std::vector<size_t> active_threads, MCInst* in) {
    assert(in->getNumOperands() == 3);

    for (auto thread : active_threads) {
        unsigned int rd = in->getOperand(0).getReg();
        unsigned int base = in->getOperand(1).getReg();
        int64_t disp = in->getOperand(2).getImm();;
        int rs1 = rf->get_register(warp->warp_id, thread, base);
        int res = uint64_t(cu->load(warp, rs1 + disp, WORD_SIZE / 2));
        
        // Eventhough we update the register values here
        // The warp will be suspended so the updates won't be visible
        // till after it resumes
        rf->set_register(warp->warp_id, thread, rd, res);
        warp->pc[thread] += 4;
    }
    // After a load instruction, you don't need to writeback unless
    // the warp was never actually suspended
    return !warp->suspended;
}
bool ExecutionUnit::lb(Warp *warp, std::vector<size_t> active_threads, MCInst* in) {
    assert(in->getNumOperands() == 3);

    for (auto thread : active_threads) {
        unsigned int rd = in->getOperand(0).getReg();
        unsigned int base = in->getOperand(1).getReg();
        int64_t disp = in->getOperand(2).getImm();;
        int rs1 = rf->get_register(warp->warp_id, thread, base);
        int res = cu->load(warp, rs1 + disp, 1);
        
        // Eventhough we update the register values here
        // The warp will be suspended so the updates won't be visible
        // till after it resumes
        rf->set_register(warp->warp_id, thread, rd, res);
        warp->pc[thread] += 4;
    }
    // After a load instruction, you don't need to writeback unless
    // the warp was never actually suspended
    return !warp->suspended;
}
bool ExecutionUnit::lbu(Warp *warp, std::vector<size_t> active_threads, MCInst* in) {
    assert(in->getNumOperands() == 3);

    for (auto thread : active_threads) {
        unsigned int rd = in->getOperand(0).getReg();
        unsigned int base = in->getOperand(1).getReg();
        int64_t disp = in->getOperand(2).getImm();;
        int rs1 = rf->get_register(warp->warp_id, thread, base);
        int res = uint64_t(cu->load(warp, rs1 + disp, 1));
        
        // Eventhough we update the register values here
        // The warp will be suspended so the updates won't be visible
        // till after it resumes
        rf->set_register(warp->warp_id, thread, rd, res);
        warp->pc[thread] += 4;
    }
    // After a load instruction, you don't need to writeback unless
    // the warp was never actually suspended
    return !warp->suspended;
}
bool ExecutionUnit::sw(Warp *warp, std::vector<size_t> active_threads, MCInst* in) {
    assert(in->getNumOperands() == 3);

    for (auto thread : active_threads) {
        int rs2 = rf->get_register(warp->warp_id, thread, in->getOperand(0).getReg());
        unsigned int base = in->getOperand(1).getReg();
        int64_t disp = in->getOperand(2).getImm();;
        int rs1 = rf->get_register(warp->warp_id, thread, base);
        cu->store(warp, rs1 + disp, WORD_SIZE, rs2);
        
        warp->pc[thread] += 4;
    }
    // After a store instruction, you don't need to writeback unless
    // the warp was never actually suspended
    return !warp->suspended;
}
bool ExecutionUnit::sh(Warp *warp, std::vector<size_t> active_threads, MCInst* in) {
    assert(in->getNumOperands() == 3);

    for (auto thread : active_threads) {
        int rs2 = rf->get_register(warp->warp_id, thread, in->getOperand(0).getReg());
        unsigned int base = in->getOperand(1).getReg();
        int64_t disp = in->getOperand(2).getImm();;
        int rs1 = rf->get_register(warp->warp_id, thread, base);
        cu->store(warp, rs1 + disp, WORD_SIZE / 2, rs2);
        
        warp->pc[thread] += 4;
    }
    // After a store instruction, you don't need to writeback unless
    // the warp was never actually suspended
    return !warp->suspended;
}
bool ExecutionUnit::sb(Warp *warp, std::vector<size_t> active_threads, MCInst* in) {
    assert(in->getNumOperands() == 3);

    for (auto thread : active_threads) {
        int rs2 = rf->get_register(warp->warp_id, thread, in->getOperand(0).getReg());
        unsigned int base = in->getOperand(1).getReg();
        int64_t disp = in->getOperand(2).getImm();;
        int rs1 = rf->get_register(warp->warp_id, thread, base);
        cu->store(warp, rs1 + disp, 1, rs2);
        
        warp->pc[thread] += 4;
    }
    // After a store instruction, you don't need to writeback unless
    // the warp was never actually suspended
    return !warp->suspended;
}
bool ExecutionUnit::jal(Warp *warp, std::vector<size_t> active_threads, MCInst* in) {
    assert(in->getNumOperands() == 2);

    for (auto thread : active_threads) {
        int rd = in->getOperand(0).getReg();
        int64_t imm = in->getOperand(1).getImm();

        rf->set_register(warp->warp_id, thread, rd, warp->pc[thread] + 4);
        warp->pc[thread] += imm;
    }
    return active_threads.size() > 0;
}
bool ExecutionUnit::jalr(Warp *warp, std::vector<size_t> active_threads, MCInst* in) {
    assert(in->getNumOperands() == 3);

    for (auto thread : active_threads) {
        int rd = in->getOperand(0).getReg();
        int rs1 = rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg());
        int64_t imm = in->getOperand(2).getImm();

        rf->set_register(warp->warp_id, thread, rd, warp->pc[thread] + 4);
        warp->pc[thread] = rs1 + imm;
    }
    return active_threads.size() > 0;
}
bool ExecutionUnit::beq(Warp *warp, std::vector<size_t> active_threads, MCInst* in) {
    assert(in->getNumOperands() == 3);

    for (auto thread : active_threads) {
        int rs1 = rf->get_register(warp->warp_id, thread, in->getOperand(0).getReg());
        int rs2 = rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg());
        int64_t imm = in->getOperand(2).getImm();

        if (rs1 == rs2) {
            warp->pc[thread] += imm;
        } else {
            warp->pc[thread] += 4;
        }
    }
    return active_threads.size() > 0;
}
bool ExecutionUnit::bne(Warp *warp, std::vector<size_t> active_threads, MCInst* in) {
    assert(in->getNumOperands() == 3);

    for (auto thread : active_threads) {
        int rs1 = rf->get_register(warp->warp_id, thread, in->getOperand(0).getReg());
        int rs2 = rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg());
        int64_t imm = in->getOperand(2).getImm();
        std::cout << rs1 << " == " << rs2 << " -> " << imm << std::endl;

        if (rs1 != rs2) {
            warp->pc[thread] += imm;
        } else {
            warp->pc[thread] += 4;
        }
    }
    return active_threads.size() > 0;
}
bool ExecutionUnit::blt(Warp *warp, std::vector<size_t> active_threads, MCInst* in) {
    assert(in->getNumOperands() == 3);

    for (auto thread : active_threads) {
        int rs1 = rf->get_register(warp->warp_id, thread, in->getOperand(0).getReg());
        int rs2 = rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg());
        int64_t imm = in->getOperand(2).getImm();

        if (rs1 < rs2) {
            warp->pc[thread] += imm;
        } else {
            warp->pc[thread] += 4;
        }
    }
    return active_threads.size() > 0;
}
bool ExecutionUnit::bltu(Warp *warp, std::vector<size_t> active_threads, MCInst* in) {
    assert(in->getNumOperands() == 3);

    for (auto thread : active_threads) {
        int rs1 = rf->get_register(warp->warp_id, thread, in->getOperand(0).getReg());
        int rs2 = rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg());
        int64_t imm = in->getOperand(2).getImm();

        if (uint64_t(rs1) < uint64_t(rs2)) {
            warp->pc[thread] += imm;
        } else {
            warp->pc[thread] += 4;
        }
    }
    return active_threads.size() > 0;
}
bool ExecutionUnit::bge(Warp *warp, std::vector<size_t> active_threads, MCInst* in) {
    assert(in->getNumOperands() == 3);

    for (auto thread : active_threads) {
        int rs1 = rf->get_register(warp->warp_id, thread, in->getOperand(0).getReg());
        int rs2 = rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg());
        int64_t imm = in->getOperand(2).getImm();

        if (rs1 >= rs2) {
            warp->pc[thread] += imm;
        } else {
            warp->pc[thread] += 4;
        }
    }
    return active_threads.size() > 0;
}
bool ExecutionUnit::bgeu(Warp *warp, std::vector<size_t> active_threads, MCInst* in) {
    assert(in->getNumOperands() == 3);

    for (auto thread : active_threads) {
        int rs1 = rf->get_register(warp->warp_id, thread, in->getOperand(0).getReg());
        int rs2 = rf->get_register(warp->warp_id, thread, in->getOperand(1).getReg());
        int64_t imm = in->getOperand(2).getImm();

        if (uint64_t(rs1) >= uint64_t(rs2)) {
            warp->pc[thread] += imm;
        } else {
            warp->pc[thread] += 4;
        }
    }
    return active_threads.size() > 0;
}
bool ExecutionUnit::ecall(Warp *warp, std::vector<size_t> active_threads, MCInst* in) {
    assert(in->getNumOperands() == 0);

    log("ExUn - Operating System", "Received an ecall");
    for (auto thread : active_threads) {
        warp->pc[thread] += 4;
    }
    return false;
}
bool ExecutionUnit::ebreak(Warp *warp, std::vector<size_t> active_threads, MCInst* in) {
    assert(in->getNumOperands() == 0);

    log("ExUn - Debugger", "Received an ebreak");
    for (auto thread : active_threads) {
        warp->pc[thread] += 4;
    }
    return false;
}
bool ExecutionUnit::csrrw(Warp *warp, std::vector<size_t> active_threads, MCInst* in) {
    std::cout << "TODO: Implement CSRRW and GPU initialization!! [and noclpush and noclpop :)]" << std::endl;
    for (auto thread : active_threads) {
        warp->pc[thread] += 4;
    }
    return false;
}

ExecuteSuspend::ExecuteSuspend(CoalescingUnit *cu, RegisterFile *rf, uint64_t max_addr, LLVMDisassembler *disasm): 
    max_addr(max_addr), cu(cu), disasm(disasm) {
    eu = new ExecutionUnit(cu, rf, disasm);
    log("Execute/Suspend", "Initializing execute/suspend pipeline stage");
}

void ExecuteSuspend::execute() {
    if (!PipelineStage::input_latch->updated) return;
    
    Warp *warp = PipelineStage::input_latch->warp;
    MCInst inst = PipelineStage::input_latch->inst;
    std::vector<size_t> active_threads = PipelineStage::input_latch->active_threads;

    execute_result result = eu->execute(warp, active_threads, inst);
    for (int i = 0; i < warp->size; i++) {
        if (warp->pc[i] <= max_addr) {
            insert_warp(warp);
            break;
        }
    }

    PipelineStage::input_latch->updated = false;
    // We use the updated flag to tell the writeback/resume stage
    // whether or not to "perform a writeback" or to check for memory
    // responses
    PipelineStage::output_latch->updated = result.write_required;
    PipelineStage::output_latch->warp = warp;
    PipelineStage::output_latch->active_threads = PipelineStage::input_latch->active_threads;
    PipelineStage::output_latch->inst = PipelineStage::input_latch->inst;

    std::string inst_name = disasm->getOpcodeName(inst.getOpcode());
    std::stringstream op_stream;
    for (llvm::MCOperand op : inst.getOperands()) {
        op_stream << operandToString(op) << " ";
    }

    if (!result.success) {
        log("Execute/Suspend", "Warp " + std::to_string(warp->warp_id) + " could not perform instruction " + inst_name);
        return;
    }
    
    log("Execute/Suspend", "Warp " + std::to_string(warp->warp_id) + " executed " + inst_name + "\t" + op_stream.str());
}

bool ExecuteSuspend::is_active() {
    return PipelineStage::input_latch->updated;
}

ExecuteSuspend::~ExecuteSuspend() {
    delete eu;
}