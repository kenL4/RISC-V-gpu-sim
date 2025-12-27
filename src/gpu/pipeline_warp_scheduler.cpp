#include "pipeline_warp_scheduler.hpp"

WarpScheduler::WarpScheduler(int warp_size, int warp_count, uint64_t start_pc,
                             bool start_active)
    : warp_size(warp_size), warp_count(warp_count) {
  log("Warp Scheduler", "Initializing warp scheduling pipeline stage");
  if (start_active) {
    for (int i = 0; i < warp_count; i++) {
      // Only CPU is "start_active" so we can assume warp is true
      Warp *warp = new Warp(i, warp_size, start_pc, true);
      warp_queue.push(warp);
    }
  }
}

void WarpScheduler::flush_new_warps() {
  while (new_warp_queue.size() > 0) {
    warp_queue.push(new_warp_queue.front());
    new_warp_queue.pop();
  }
}

void WarpScheduler::execute() {
  if (warp_queue.empty()) {
    // Ensure new warps are brought in next cycle
    flush_new_warps();
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
  log("Warp Scheduler",
      "Warp " + std::to_string(scheduled_warp->warp_id) + " scheduled to run");

  // Ensure new warps are brought in next cycle
  flush_new_warps();
}

bool WarpScheduler::is_active() {
  return warp_queue.size() > 0 || new_warp_queue.size() > 0;
}

void WarpScheduler::insert_warp(Warp *warp) { new_warp_queue.push(warp); }

WarpScheduler::~WarpScheduler() {
  flush_new_warps();

  while (warp_queue.size() > 0) {
    delete warp_queue.front();
    warp_queue.pop();
  }

  log("Warp Scheduler", "Destroyed pipeline stage");
}