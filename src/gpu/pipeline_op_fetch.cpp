#include "pipeline_op_fetch.hpp"

OperandFetch::OperandFetch() {
    log("Operand Fetch", "Initializing operand fetch pipeline stage");
}

void OperandFetch::execute() {
    if (!input_latch->updated) return;
    
    Warp *warp = input_latch->warp;
    const llvm::MCInst &inst = input_latch->inst;

    input_latch->updated = false;
    output_latch->updated = true;
    output_latch->warp = warp;
    output_latch->active_threads = input_latch->active_threads;
    output_latch->inst = inst;
    
    if (!warp->is_cpu) {
      std::stringstream op_stream;
      for (const auto &op : inst.getOperands()) {
          op_stream << operandToString(op) << " ";
      }
      log("Operand Fetch", "Warp " + std::to_string(warp->warp_id) + 
          " using operands " + op_stream.str());
    }
};

bool OperandFetch::is_active() {
    return input_latch->updated;
}