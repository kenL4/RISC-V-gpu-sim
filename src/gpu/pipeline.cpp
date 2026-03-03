#include "pipeline.hpp"

void Pipeline::execute() {
    // Execute backwards to avoid overwriting latches prematurely
    for (auto it = stages.rbegin(); it != stages.rend(); it++) {
        (*it)->execute();
    }
}

bool Pipeline::has_active_stages() {
    for (auto &stage: stages) {
        if (stage->is_active()) return true;
    }
    return false;
}

std::shared_ptr<PipelineStage> Pipeline::get_stage(int index) {
    return stages[index];
}

void MockPipelineStage::execute() {
    if (!input_latch->updated) return;
    if (output_latch->updated) return;

    Warp *warp = input_latch->warp;
    if (!warp->is_cpu) {
      log("MockPipelineStage", "Warp " + std::to_string(warp->warp_id) + " executing");
    }

    input_latch->updated = false;
    output_latch->updated = true;
    output_latch->warp = warp;
    output_latch->active_threads = input_latch->active_threads;
    output_latch->inst = input_latch->inst;
    output_latch->has_result = input_latch->has_result;
}

bool MockPipelineStage::is_active() {
    return input_latch->updated || output_latch->updated;
}