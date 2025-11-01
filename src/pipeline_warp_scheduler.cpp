#include "pipeline_warp_scheduler.hpp"

WarpScheduler::WarpScheduler(int warp_size, int warp_count, uint64_t start_pc): warp_size(warp_size), warp_count(warp_count) {
    log("Warp Scheduler", "Initializing warp scheduling pipeline stage");
    for (int i = 0; i < warp_count; i++) {
        Warp *warp = new Warp(i, warp_size, start_pc);
        warp_queue.push(warp);
    }
}

void WarpScheduler::execute() {
    if (warp_queue.empty()) {
        return;
    }

    Warp *top = warp_queue.front();
    warp_queue.pop();

    // Find the first live warp
    while (top->suspended) {
        warp_queue.push(top);

        top = warp_queue.front();
        warp_queue.pop();
    }

    // Update pipeline latch
    PipelineStage::output_latch->updated = true;
    PipelineStage::output_latch->warp = top;

    log("Warp Scheduler", "Warp " + std::to_string(top->warp_id) + " scheduled to run");
}

bool WarpScheduler::is_active() {
    return warp_queue.size() > 0; // || PipelineStage::input_latch->updated;
}

void WarpScheduler::insert_warp(Warp *warp) {
    warp_queue.push(warp);
}

WarpScheduler::~WarpScheduler() {
    while (warp_queue.size() > 0) {
        delete warp_queue.front();
        warp_queue.pop();
    }

    log("Warp Scheduler", "Destroyed pipeline stage");
}