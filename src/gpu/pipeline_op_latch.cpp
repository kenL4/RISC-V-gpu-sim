#include "pipeline_op_latch.hpp"

OperandLatch::OperandLatch() {
    log("Operand Latch", "Initializing operand latch pipeline stage");
}

void OperandLatch::execute() {
    if (!PipelineStage::input_latch->updated) return;
    
    Warp *warp = PipelineStage::input_latch->warp;
    
    // Latch operands (with register file load latency accounted for)
    // With default loadLatency=1, this is a pass-through stage that provides
    // the pipeline boundary matching SIMTight's Stage 4 (Operand Latch)
    
    PipelineStage::input_latch->updated = false;
    PipelineStage::output_latch->updated = true;
    PipelineStage::output_latch->warp = warp;
    PipelineStage::output_latch->active_threads = PipelineStage::input_latch->active_threads;
    PipelineStage::output_latch->inst = PipelineStage::input_latch->inst;
    
    log("Operand Latch", "Warp " + std::to_string(warp->warp_id) + " operands latched");
}

bool OperandLatch::is_active() {
    return PipelineStage::input_latch->updated;
}
