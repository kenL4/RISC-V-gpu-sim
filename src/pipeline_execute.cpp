#include "pipeline_execute.hpp"

/*
 * The Execution Unit is the unit that handles the actual
 * computation and production of side-effects of instructions
 * in the pipeline.
 */
ExecutionUnit::ExecutionUnit(RegisterFile *rf): rf(rf) {}
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
    assert(riscv->op_count == 3);
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

ExecuteSuspend::ExecuteSuspend(RegisterFile *rf, uint64_t max_addr): max_addr(max_addr) {
    eu = new ExecutionUnit(rf);
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