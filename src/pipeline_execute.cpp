#include "utils.hpp"
#include "pipeline.hpp"
#include "pipeline_warp_scheduler.hpp"
#include "register_file.hpp"

typedef struct execute_result {
    bool success;
    bool write_required;
} execute_result;

/*
 * The Execution Unit is the unit that handles the actual
 * computation and production of side-effects of instructions
 * in the pipeline.
 */
class ExecutionUnit {
public:
    ExecutionUnit(RegisterFile *rf): rf(rf) {}
    execute_result execute(Warp *warp, std::vector<size_t> active_threads, cs_insn *insn) {
        std::string mnemonic = insn->mnemonic;
        cs_riscv *riscv = &(insn->detail->riscv);
        execute_result res { false, false };

        if (mnemonic == "addi") {
            res.success = true;
            res.write_required = addi(warp, active_threads, riscv);
        }

        // Default to skip instruction
        // Bit of a hard-coded way to get next instruction
        for (auto thread : active_threads) {
            warp->pc[thread] += 4;
        }
        return res;
    }
private:
    RegisterFile *rf;
    bool addi(Warp *warp, std::vector<size_t> active_threads, cs_riscv *riscv) {
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
};

/*
 * The Execute/Suspend unit executes the instruction and reinserts
 * the warp ID into the warp queue. It also performs the memory access
 * request.
 * 
 * We don't need to handle hazards as there will only be one
 * instruction per warp in the pipeline at any given time.
 */
class ExecuteSuspend: public PipelineStage {
public:
    std::function<void(Warp *warp)> insert_warp;
    ExecuteSuspend(RegisterFile *rf, uint64_t max_addr): max_addr(max_addr) {
        eu = new ExecutionUnit(rf);
        log("Execute/Suspend", "Initializing execute/suspend pipeline stage");
    }
    void execute() override {
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
    };

    bool is_active() override {
        return PipelineStage::input_latch->updated;
    }

    ~ExecuteSuspend() {
        delete eu;
    };
private:
    ExecutionUnit *eu;
    uint64_t max_addr;
};