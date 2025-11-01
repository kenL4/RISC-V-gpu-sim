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

    std::queue<Warp *> suspended_queue;

    Warp *scheduled_warp = nullptr;
    while (!warp_queue.empty()) {
        if (!warp_queue.front()->suspended) {
            scheduled_warp = warp_queue.front();
            warp_queue.pop();
            break;
        }

        suspended_queue.push(warp_queue.front());
        warp_queue.pop();
    }

    // Reinsert suspended threads into main warp queue
    while (!suspended_queue.empty()) {
        Warp *suspended_warp = suspended_queue.front();
        suspended_queue.pop();
        warp_queue.push(suspended_warp);
    }

    // Update pipeline latch
    PipelineStage::output_latch->updated = scheduled_warp != nullptr;
    PipelineStage::output_latch->warp = scheduled_warp;

    if (scheduled_warp == nullptr) {
        log("Warp Scheduler", "No warp ready to be scheduled");    
        return;
    }
    log("Warp Scheduler", "Warp " + std::to_string(scheduled_warp->warp_id) + " scheduled to run");
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