#include "pipeline_op_fetch.hpp"

OperandFetch::OperandFetch() {
    log("Operand Fetch", "Initializing operand fetch pipeline stage");
}

void OperandFetch::execute() {
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

bool OperandFetch::is_active() {
    return PipelineStage::input_latch->updated;
}