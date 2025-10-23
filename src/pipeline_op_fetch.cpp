#include "utils.hpp"
#include "pipeline.hpp"
#include "register_file.hpp"

/*
 * The Operand Fetch unit performs a lookup on the register file
 * for a given warp ID and the source register IDs
 * 
 * For now, I am deferring resolution of operands to
 * the execute stage so this is essentially a no-op.
 */
class OperandFetch: public PipelineStage {
public:
    OperandFetch() {
        log("Operand Fetch", "Initializing operand fetch pipeline stage");
    }
    void execute() override {
        if (!PipelineStage::input_latch->updated) return;
        
        Warp *warp = PipelineStage::input_latch->warp;
        cs_insn *inst = PipelineStage::input_latch->instruction;

        PipelineStage::input_latch->updated = false;
        PipelineStage::output_latch->updated = true;
        PipelineStage::output_latch->warp = warp;
        PipelineStage::output_latch->active_threads = PipelineStage::input_latch->active_threads;
        PipelineStage::output_latch->instruction = PipelineStage::input_latch->instruction;
        
        log("Operand Fetch", "Warp " + std::to_string(warp->warp_id) + " using operands " + inst->op_str);
    };

    bool is_active() override {
        return PipelineStage::input_latch->updated;
    }

    ~OperandFetch() {};
};