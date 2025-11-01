#include "pipeline_execute.hpp"

// Assuming RISC-V with 64 bit registers
#define WORD_SIZE 8

ExecutionUnit::ExecutionUnit(CoalescingUnit *cu, RegisterFile *rf): cu(cu), rf(rf) {}

execute_result ExecutionUnit::execute(Warp *warp, std::vector<size_t> active_threads, cs_insn *insn) {
    std::string mnemonic = insn->mnemonic;
    cs_riscv *riscv = &(insn->detail->riscv);
    execute_result res { true, false };

    if (mnemonic == "addi") {
        res.write_required = addi(warp, active_threads, riscv);
    } else if (mnemonic == "add") {
        res.write_required = add(warp, active_threads, riscv);
    } else if (mnemonic == "neg") {
        res.write_required = neg(warp, active_threads, riscv);
    } else if (mnemonic == "sub") {
        res.write_required = sub(warp, active_threads, riscv);
    } else if (mnemonic == "sub") {
        res.write_required = sub(warp, active_threads, riscv);
    } else if (mnemonic == "and") {
        // Name followed by an underscore as and is a reserved keyword
        res.write_required = and_(warp, active_threads, riscv);
    } else if (mnemonic == "andi") {
        res.write_required = andi(warp, active_threads, riscv);
    } else if (mnemonic == "not") {
        res.write_required = not_(warp, active_threads, riscv);
    } else if (mnemonic == "or") {
        res.write_required = or_(warp, active_threads, riscv);
    } else if (mnemonic == "ori") {
        res.write_required = ori(warp, active_threads, riscv);
    } else if (mnemonic == "xor") {
        res.write_required = xor_(warp, active_threads, riscv);
    } else if (mnemonic == "xori") {
        res.write_required = xori(warp, active_threads, riscv);
    } else if (mnemonic == "sll") {
        res.write_required = sll(warp, active_threads, riscv);
    } else if (mnemonic == "slli") {
        res.write_required = slli(warp, active_threads, riscv);
    } else if (mnemonic == "srl") {
        res.write_required = srl(warp, active_threads, riscv);
    } else if (mnemonic == "srli") {
        res.write_required = srli(warp, active_threads, riscv);
    } else if (mnemonic == "sra") {
        res.write_required = sra(warp, active_threads, riscv);
    } else if (mnemonic == "srai") {
        res.write_required = srai(warp, active_threads, riscv);
    } else if (mnemonic == "li") {
        res.write_required = li(warp, active_threads, riscv);
    } else if (mnemonic == "lui") {
        res.write_required = lui(warp, active_threads, riscv);
    } else if (mnemonic == "auipc") {
        res.write_required = auipc(warp, active_threads, riscv);
    } else if (mnemonic == "lw") {
        res.write_required = lw(warp, active_threads, riscv);
    // } else if (mnemonic == "lh") {
    //     res.write_required = lh(warp, active_threads, riscv);
    // } else if (mnemonic == "lhu") {
    //     res.write_required = lhu(warp, active_threads, riscv);
    // } else if (mnemonic == "lb") {
    //     res.write_required = lb(warp, active_threads, riscv);
    // } else if (mnemonic == "la") {
    //     res.write_required = la(warp, active_threads, riscv);
    // } else if (mnemonic == "sw") {
    //     res.write_required = sw(warp, active_threads, riscv);
    // } else if (mnemonic == "sh") {
    //     res.write_required = sh(warp, active_threads, riscv);
    // } else if (mnemonic == "sb") {
    //     res.write_required = sb(warp, active_threads, riscv);
    } else {
        // Default to skip instruction
        // Bit of a hard-coded way to get next instruction
        for (auto thread : active_threads) {
            warp->pc[thread] += 4;
        }
        res.success = false;
    }
    return res;
}

bool ExecutionUnit::add(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv) {
    assert(riscv->op_count == 3);
    for (auto thread : active_threads) {
        unsigned int rd = riscv->operands[0].reg;
        int rs1 = rf->get_register(warp->warp_id, thread, riscv->operands[1].reg);
        int rs2 = rf->get_register(warp->warp_id, thread, riscv->operands[2].reg);
        rf->set_register(warp->warp_id, thread, rd, rs1 + rs2);

        warp->pc[thread] += 4;
    }
    return active_threads.size() > 0;
}
bool ExecutionUnit::addi(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv) {
    assert(riscv->op_count == 3);
    for (auto thread : active_threads) {
        unsigned int rd = riscv->operands[0].reg;
        int rs1 = rf->get_register(warp->warp_id, thread, riscv->operands[1].reg);
        int64_t imm = riscv->operands[2].imm;
        rf->set_register(warp->warp_id, thread, rd, rs1 + imm);

        warp->pc[thread] += 4;
    }
    return active_threads.size() > 0;
}
bool ExecutionUnit::neg(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv) {
    assert(riscv->op_count == 2);
    for (auto thread : active_threads) {
        unsigned int rd = riscv->operands[0].reg;
        int rs2 = rf->get_register(warp->warp_id, thread, riscv->operands[1].reg);
        rf->set_register(warp->warp_id, thread, rd, -rs2);

        warp->pc[thread] += 4;
    }
    return active_threads.size() > 0;
}
bool ExecutionUnit::sub(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv) {
    assert(riscv->op_count == 3);
    for (auto thread : active_threads) {
        unsigned int rd = riscv->operands[0].reg;
        int rs1 = rf->get_register(warp->warp_id, thread, riscv->operands[1].reg);
        int rs2 = rf->get_register(warp->warp_id, thread, riscv->operands[2].reg);
        rf->set_register(warp->warp_id, thread, rd, rs1 - rs2);

        warp->pc[thread] += 4;
    }
    return active_threads.size() > 0;
}
bool ExecutionUnit::and_(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv) {
    assert(riscv->op_count == 3);
    for (auto thread : active_threads) {
        unsigned int rd = riscv->operands[0].reg;
        int rs1 = rf->get_register(warp->warp_id, thread, riscv->operands[1].reg);
        int rs2 = rf->get_register(warp->warp_id, thread, riscv->operands[2].reg);
        rf->set_register(warp->warp_id, thread, rd, rs1 & rs2);

        warp->pc[thread] += 4;
    }
    return active_threads.size() > 0;
}
bool ExecutionUnit::andi(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv) {
    assert(riscv->op_count == 3);
    for (auto thread : active_threads) {
        unsigned int rd = riscv->operands[0].reg;
        int rs1 = rf->get_register(warp->warp_id, thread, riscv->operands[1].reg);
        int64_t imm = riscv->operands[2].imm;
        rf->set_register(warp->warp_id, thread, rd, rs1 & imm);

        warp->pc[thread] += 4;
    }
    return active_threads.size() > 0;
}
bool ExecutionUnit::not_(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv) {
    assert(riscv->op_count == 2);
    for (auto thread : active_threads) {
        unsigned int rd = riscv->operands[0].reg;
        int rs1 = rf->get_register(warp->warp_id, thread, riscv->operands[1].reg);
        rf->set_register(warp->warp_id, thread, rd, ~rs1);

        warp->pc[thread] += 4;
    }
    return active_threads.size() > 0;
}
bool ExecutionUnit::or_(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv) {
    assert(riscv->op_count == 3);
    for (auto thread : active_threads) {
        unsigned int rd = riscv->operands[0].reg;
        int rs1 = rf->get_register(warp->warp_id, thread, riscv->operands[1].reg);
        int rs2 = rf->get_register(warp->warp_id, thread, riscv->operands[2].reg);
        rf->set_register(warp->warp_id, thread, rd, rs1 | rs2);

        warp->pc[thread] += 4;
    }
    return active_threads.size() > 0;
}
bool ExecutionUnit::ori(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv) {
    assert(riscv->op_count == 3);
    for (auto thread : active_threads) {
        unsigned int rd = riscv->operands[0].reg;
        int rs1 = rf->get_register(warp->warp_id, thread, riscv->operands[1].reg);
        int64_t imm = riscv->operands[2].imm;
        rf->set_register(warp->warp_id, thread, rd, rs1 | imm);

        warp->pc[thread] += 4;
    }
    return active_threads.size() > 0;
}
bool ExecutionUnit::xor_(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv) {
    assert(riscv->op_count == 3);
    for (auto thread : active_threads) {
        unsigned int rd = riscv->operands[0].reg;
        int rs1 = rf->get_register(warp->warp_id, thread, riscv->operands[1].reg);
        int rs2 = rf->get_register(warp->warp_id, thread, riscv->operands[2].reg);
        rf->set_register(warp->warp_id, thread, rd, rs1 ^ rs2);

        warp->pc[thread] += 4;
    }
    return active_threads.size() > 0;
}
bool ExecutionUnit::xori(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv) {
    assert(riscv->op_count == 3);
    for (auto thread : active_threads) {
        unsigned int rd = riscv->operands[0].reg;
        int rs1 = rf->get_register(warp->warp_id, thread, riscv->operands[1].reg);
        int64_t imm = riscv->operands[2].imm;
        rf->set_register(warp->warp_id, thread, rd, rs1 ^ imm);

        warp->pc[thread] += 4;
    }
    return active_threads.size() > 0;
}
bool ExecutionUnit::sll(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv) {
    assert(riscv->op_count == 3);
    for (auto thread : active_threads) {
        unsigned int rd = riscv->operands[0].reg;
        int rs1 = rf->get_register(warp->warp_id, thread, riscv->operands[1].reg);
        int rs2 = rf->get_register(warp->warp_id, thread, riscv->operands[2].reg);
        rf->set_register(warp->warp_id, thread, rd, static_cast<uint64_t>(rs1) << rs2);

        warp->pc[thread] += 4;
    }
    return active_threads.size() > 0;
}
bool ExecutionUnit::slli(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv) {
    assert(riscv->op_count == 3);
    for (auto thread : active_threads) {
        unsigned int rd = riscv->operands[0].reg;
        int rs1 = rf->get_register(warp->warp_id, thread, riscv->operands[1].reg);
        int64_t imm = riscv->operands[2].imm;
        rf->set_register(warp->warp_id, thread, rd, static_cast<uint64_t>(rs1) << imm);

        warp->pc[thread] += 4;
    }
    return active_threads.size() > 0;
}
bool ExecutionUnit::srl(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv) {
    assert(riscv->op_count == 3);
    for (auto thread : active_threads) {
        unsigned int rd = riscv->operands[0].reg;
        int rs1 = rf->get_register(warp->warp_id, thread, riscv->operands[1].reg);
        int rs2 = rf->get_register(warp->warp_id, thread, riscv->operands[2].reg);
        rf->set_register(warp->warp_id, thread, rd, static_cast<uint64_t>(rs1) >> rs2);

        warp->pc[thread] += 4;
    }
    return active_threads.size() > 0;
}
bool ExecutionUnit::srli(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv) {
    assert(riscv->op_count == 3);
    for (auto thread : active_threads) {
        unsigned int rd = riscv->operands[0].reg;
        int rs1 = rf->get_register(warp->warp_id, thread, riscv->operands[1].reg);
        int64_t imm = riscv->operands[2].imm;
        rf->set_register(warp->warp_id, thread, rd, static_cast<uint64_t>(rs1) >> imm);

        warp->pc[thread] += 4;
    }
    return active_threads.size() > 0;
}
bool ExecutionUnit::sra(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv) {
    assert(riscv->op_count == 3);
    for (auto thread : active_threads) {
        unsigned int rd = riscv->operands[0].reg;
        int rs1 = rf->get_register(warp->warp_id, thread, riscv->operands[1].reg);
        int rs2 = rf->get_register(warp->warp_id, thread, riscv->operands[2].reg);
        rf->set_register(warp->warp_id, thread, rd, rs1 >> rs2);

        warp->pc[thread] += 4;
    }
    return active_threads.size() > 0;
}
bool ExecutionUnit::srai(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv) {
    assert(riscv->op_count == 3);
    for (auto thread : active_threads) {
        unsigned int rd = riscv->operands[0].reg;
        int rs1 = rf->get_register(warp->warp_id, thread, riscv->operands[1].reg);
        int64_t imm = riscv->operands[2].imm;
        rf->set_register(warp->warp_id, thread, rd, rs1 >> imm);

        warp->pc[thread] += 4;
    }
    return active_threads.size() > 0;
}
bool ExecutionUnit::li(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv) {
    assert(riscv->op_count == 2);
    for (auto thread : active_threads) {
        unsigned int rd = riscv->operands[0].reg;
        int64_t imm = riscv->operands[1].imm;
        rf->set_register(warp->warp_id, thread, rd, imm);

        warp->pc[thread] += 4;
    }
    return active_threads.size() > 0;
}
bool ExecutionUnit::lui(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv) {
    assert(riscv->op_count == 2);
    for (auto thread : active_threads) {
        unsigned int rd = riscv->operands[0].reg;
        int64_t imm = riscv->operands[1].imm;
        rf->set_register(warp->warp_id, thread, rd, static_cast<uint64_t>(imm) << 12);

        warp->pc[thread] += 4;
    }
    return active_threads.size() > 0;
}
bool ExecutionUnit::auipc(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv) {
    assert(riscv->op_count == 2);
    for (auto thread : active_threads) {
        unsigned int rd = riscv->operands[0].reg;
        int64_t imm = riscv->operands[1].imm;
        rf->set_register(warp->warp_id, thread, rd, warp->pc[thread] + (static_cast<uint64_t>(imm) << 12));

        warp->pc[thread] += 4;
    }
    return active_threads.size() > 0;
}
bool ExecutionUnit::lw(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv) {
    assert(riscv->op_count == 2);

    for (auto thread : active_threads) {
        unsigned int rd = riscv->operands[0].reg;
        riscv_op_mem mem = riscv->operands[1].mem;
        int rs1 = rf->get_register(warp->warp_id, thread, mem.base);
        int res = cu->load(warp, rs1 + mem.disp, WORD_SIZE);
        
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

ExecuteSuspend::ExecuteSuspend(CoalescingUnit *cu, RegisterFile *rf, uint64_t max_addr): 
    max_addr(max_addr), cu(cu) {
    eu = new ExecutionUnit(cu, rf);
    log("Execute/Suspend", "Initializing execute/suspend pipeline stage");
}

void ExecuteSuspend::execute() {
    if (!PipelineStage::input_latch->updated) return;
    
    Warp *warp = PipelineStage::input_latch->warp;
    cs_insn *insn = PipelineStage::input_latch->instruction;
    std::vector<size_t> active_threads = PipelineStage::input_latch->active_threads;

    execute_result result = eu->execute(warp, active_threads, insn);
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
    PipelineStage::output_latch->instruction = PipelineStage::input_latch->instruction;

    if (!result.success) {
        log("Execute/Suspend", "Warp " + std::to_string(warp->warp_id) + " could not perform instruction " + insn->mnemonic);
        return;
    }
    
    log("Execute/Suspend", "Warp " + std::to_string(warp->warp_id) + " executed " + insn->mnemonic + "\t" + insn->op_str);
}

bool ExecuteSuspend::is_active() {
    return PipelineStage::input_latch->updated;
}

ExecuteSuspend::~ExecuteSuspend() {
    delete eu;
}