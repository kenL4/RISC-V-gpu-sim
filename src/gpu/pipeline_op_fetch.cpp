#include "pipeline_op_fetch.hpp"

OperandFetch::OperandFetch() {
    log("Operand Fetch", "Initializing operand fetch pipeline stage");
}

void OperandFetch::execute() {
    if (!PipelineStage::input_latch->updated) return;
    
    Warp *warp = PipelineStage::input_latch->warp;
    llvm::MCInst *inst = &(PipelineStage::input_latch->inst);

    PipelineStage::input_latch->updated = false;
    PipelineStage::output_latch->updated = true;
    PipelineStage::output_latch->warp = warp;
    PipelineStage::output_latch->active_threads = PipelineStage::input_latch->active_threads;
    PipelineStage::output_latch->inst = PipelineStage::input_latch->inst;
    
    std::stringstream op_stream;
    for (llvm::MCOperand op : inst->getOperands()) {
        op_stream << operandToString(op) << " ";
    }
    std::string name = warp->is_cpu ? "CPU" : "Warp " + std::to_string(warp->warp_id);
    log("Operand Fetch", name + " using operands " + op_stream.str());
};

bool OperandFetch::is_active() {
    return PipelineStage::input_latch->updated;
}