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

// firstHot: Isolate first (lowest) set bit
// Returns a bitmask with only the lowest set bit of x
uint64_t WarpScheduler::firstHot(uint64_t x) {
  return x & (~x + 1);
}

// Fair scheduler: matches SIMTight's fairScheduler function
// Returns (new_history, chosen_bitmask)
std::pair<uint64_t, uint64_t> WarpScheduler::fair_scheduler(uint64_t hist, uint64_t avail) {
  // First choice: available warp that's NOT in history
  uint64_t first = firstHot(avail & ~hist);
  
  if (first != 0) {
    // Found an available warp not in history: add to history and choose it
    return std::make_pair(hist | first, first);
  } else {
    // All available warps are in history: choose any available and reset history
    uint64_t second = firstHot(avail);
    return std::make_pair(second, second);
  }
}


void WarpScheduler::execute() {
  warp_issued_this_cycle = false;

  // 2nd substage: Output the chosen warp from buffer (matching SIMTight's 2nd substage)
  if (chosen_warp_buffer != nullptr) {
    // Output the warp that was chosen in the previous cycle
    PipelineStage::output_latch->updated = true;
    PipelineStage::output_latch->warp = chosen_warp_buffer;
    warp_issued_this_cycle = true;
    if (!chosen_warp_buffer->is_cpu) {
      std::string name = "Warp " + std::to_string(chosen_warp_buffer->warp_id);
      log("Warp Scheduler",
          name + " scheduled to run (substage 2)");
    }
    chosen_warp_buffer = nullptr;  // Clear buffer
  } else {
    // No warp in buffer, output nothing
    PipelineStage::output_latch->updated = false;
    PipelineStage::output_latch->warp = nullptr;
  }

  // 1st substage: Choose a warp for next cycle (matching SIMTight's 1st substage)
  flush_new_warps();

  if (warp_queue.empty()) {
    return;  // No warps available
  }

  // Barrier release logic (matching SIMTight's barrier release unit)
  // Check if all warps in queue are in barrier, and if so, release them all
  bool all_in_barrier = true;
  size_t warps_in_barrier = 0;
  std::queue<Warp *> temp_check_queue;
  while (!warp_queue.empty()) {
    Warp *w = warp_queue.front();
    warp_queue.pop();
    temp_check_queue.push(w);
    
    if (!w->is_cpu && !w->finished[0]) {  // Only check active GPU warps
      if (w->in_barrier) {
        warps_in_barrier++;
      } else {
        all_in_barrier = false;
      }
    }
  }
  
  // Restore queue
  warp_queue = std::move(temp_check_queue);
  
  // If all warps are in barrier, release them all (matching SIMTight: barrier release unit)
  if (all_in_barrier && warps_in_barrier > 0) {
    std::queue<Warp *> temp_release_queue;
    while (!warp_queue.empty()) {
      Warp *w = warp_queue.front();
      warp_queue.pop();
      if (w->in_barrier && !w->is_cpu) {
        w->in_barrier = false;  // Release from barrier
      }
      temp_release_queue.push(w);
    }
    warp_queue = std::move(temp_release_queue);
  }
  
  // Build available bitmask: bit i set if warp i is available (not suspended and not in barrier)
  // Match SIMTight: avail = warpQueue & ~suspended & ~inBarrier
  uint64_t avail = 0;
  
  // First pass: build available bitmask
  std::queue<Warp *> temp_queue;
  while (!warp_queue.empty()) {
    Warp *w = warp_queue.front();
    warp_queue.pop();
    temp_queue.push(w);
    
    if (!w->suspended && !w->in_barrier) {
      // Set bit at position warp_id
      if (w->warp_id < 64) {
        avail |= (1ULL << w->warp_id);
      }
    }
  }
  
  // Restore queue
  warp_queue = std::move(temp_queue);

  // Apply fair scheduler
  uint64_t chosen_bitmask = 0;
  if (avail != 0) {
    std::pair<uint64_t, uint64_t> result = fair_scheduler(sched_history, avail);
    sched_history = result.first;
    chosen_bitmask = result.second;
  }

  // Find and remove warp with matching warp_id from chosen bitmask
  Warp *chosen_warp = nullptr;
  if (chosen_bitmask != 0) {
    // Find the warp_id from the bitmask (lowest set bit)
    uint64_t chosen_warp_id = __builtin_ctzll(chosen_bitmask);  // Count trailing zeros = bit position
    
    // Find and remove the chosen warp from queue
    std::queue<Warp *> new_queue;
    while (!warp_queue.empty()) {
      Warp *w = warp_queue.front();
      warp_queue.pop();
      
      if (w->warp_id == chosen_warp_id && !w->suspended && chosen_warp == nullptr) {
        // Found the chosen warp (first match)
        chosen_warp = w;
        // Don't reinsert chosen warp
      } else {
        // Keep this warp in the queue
        new_queue.push(w);
      }
    }
    
    // Update queue
    warp_queue = std::move(new_queue);
  }

  // Store chosen warp in buffer for next cycle (2nd substage)
  if (chosen_warp != nullptr) {
    chosen_warp_buffer = chosen_warp;
    log("Warp Scheduler",
        "Warp " + std::to_string(chosen_warp->warp_id) + " chosen (substage 1, fair scheduler)");
  }
}

bool WarpScheduler::is_active() {
  // Must also check chosen_warp_buffer because of 2-cycle latency
  // If a warp is buffered, we're still active (will output on next cycle)
  return warp_queue.size() > 0 || new_warp_queue.size() > 0 || chosen_warp_buffer != nullptr;
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