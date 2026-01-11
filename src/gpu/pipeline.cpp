#include "pipeline.hpp"

void Pipeline::execute() {
    // Execute backwards to avoid overwriting latches prematurely
    for (auto it = stages.rbegin(); it != stages.rend(); it++) {
        (*it)->execute();
    }
}

bool Pipeline::has_active_stages() {
    int i = 0;
    for (auto &stage: stages) {
        if (stage->is_active()) return true;
    }
    return false;
}

std::shared_ptr<PipelineStage> Pipeline::get_stage(int index) {
    return stages[index];
}

void MockPipelineStage::execute() {
    if (!PipelineStage::input_latch->updated) return;
    
    Warp *warp = PipelineStage::input_latch->warp;
    std::string name = warp->is_cpu ? "CPU" : "Warp " + std::to_string(warp->warp_id);
    log("MockPipelineStage", name + " executing");

    // Update pipeline latches
    PipelineStage::input_latch->updated = false;
    PipelineStage::output_latch->updated = true;
    PipelineStage::output_latch->warp = PipelineStage::input_latch->warp;
};

bool MockPipelineStage::is_active()  {
    // A pipeline stage will do something if there is data passed to it
    return PipelineStage::input_latch->updated;
}