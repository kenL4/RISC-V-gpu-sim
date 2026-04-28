#include "pipeline_op_latch.hpp"

OperandLatch::OperandLatch() {
    log("Operand Latch", "Initializing operand latch pipeline stage");
}

void OperandLatch::execute() {
    if (!PipelineStage::input_latch->updated) return;
    if (PipelineStage::output_latch->updated) return;

    Warp *warp = PipelineStage::input_latch->warp;
    
    // Just a pass through stage in the SIMTight GPU
    
    PipelineStage::input_latch->updated = false;
    PipelineStage::output_latch->updated = true;
    PipelineStage::output_latch->warp = warp;
    PipelineStage::output_latch->active_threads = PipelineStage::input_latch->active_threads;
    PipelineStage::output_latch->inst = PipelineStage::input_latch->inst;
    
    std::string name = warp->is_cpu ? "CPU" : "Warp " + std::to_string(warp->warp_id);
    log("Operand Latch", name + " operands latched");
}

bool OperandLatch::is_active() {
    return PipelineStage::input_latch->updated || PipelineStage::output_latch->updated;
}
