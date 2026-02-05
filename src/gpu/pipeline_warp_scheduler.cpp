#include "pipeline_warp_scheduler.hpp"
#include "config.hpp"
#include "mem/mem_coalesce.hpp"
#include <map>
#include <climits>

WarpScheduler::WarpScheduler(int warp_size, int warp_count, uint64_t start_pc,
                             CoalescingUnit *cu, bool start_active)
    : warp_size(warp_size), warp_count(warp_count), cu(cu), warps_per_block(0),
      barrier_release_state(0), barrier_shift_reg(0), release_warp_id(0),
      release_warp_count(0), release_success(false), barrier_bits(0) {
  log("Warp Scheduler", "Initializing warp scheduling pipeline stage");
  if (start_active) {
    for (int i = 0; i < warp_count; i++) {
      // Only CPU is "start_active" so we can assume warp is true
      Warp *warp = new Warp(i, warp_size, start_pc, true);
      warp_queue.push(warp);
      all_warps[i] = warp;
    }
  }
}

void WarpScheduler::flush_new_warps() {
  // Warps that completed last cycle are in reinsert_ready; merge them first (1-cycle delay, matching SIMTight).
  while (!reinsert_ready.empty()) {
    new_warp_queue.push(reinsert_ready.front());
    reinsert_ready.pop();
  }
  while (new_warp_queue.size() > 0) {
    warp_queue.push(new_warp_queue.front());
    new_warp_queue.pop();
  }
}

uint64_t WarpScheduler::firstHot(uint64_t x) {
  return x & (~x + 1);
}

std::pair<uint64_t, uint64_t> WarpScheduler::fair_scheduler(uint64_t hist, uint64_t avail) {
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
    chosen_warp_buffer = nullptr;
  } else {
    PipelineStage::output_latch->updated = false;
    PipelineStage::output_latch->warp = nullptr;
  }

  // 1st substage: Choose a warp for next cycle (matching SIMTight's 1st substage)
  // Swap delay queue with ready: warps that completed last cycle (now in reinsert_ready) become available this cycle.
  reinsert_ready.swap(reinsert_delay_queue);
  flush_new_warps();

  barrier_bits = 0;
  for (const auto& [warp_id, warp] : all_warps) {
    if (!warp->is_cpu && !warp->finished[0] && warp->in_barrier) {
      if (warp_id < 64) {
        barrier_bits |= (1ULL << warp_id);
      }
    }
  }
  
  barrier_release_unit();

  if (warp_queue.empty()) {
    return;
  }
  
  // Build available bitmask
  uint64_t avail = 0;
  std::queue<Warp *> temp_queue;
  while (!warp_queue.empty()) {
    Warp *w = warp_queue.front();
    warp_queue.pop();
    temp_queue.push(w);
    
    if (!w->suspended && !w->in_barrier) {
      if (w->warp_id < 64) {
        avail |= (1ULL << w->warp_id);
      }
    }
  }
  
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
    uint64_t chosen_warp_id = __builtin_ctzll(chosen_bitmask);
    
    // Find and remove the chosen warp from queue
    std::queue<Warp *> new_queue;
    while (!warp_queue.empty()) {
      Warp *w = warp_queue.front();
      warp_queue.pop();
      
      if (w->warp_id == chosen_warp_id && !w->suspended && chosen_warp == nullptr) {
        chosen_warp = w;
      } else {
        new_queue.push(w);
      }
    }
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
  return warp_queue.size() > 0 || new_warp_queue.size() > 0 ||
         reinsert_delay_queue.size() > 0 || reinsert_ready.size() > 0 ||
         chosen_warp_buffer != nullptr;
}

void WarpScheduler::insert_warp(Warp *warp) {
  // Delay re-insertion by 1 cycle so warp becomes available next cycle (match SIMTight pipeline latency).
  reinsert_delay_queue.push(warp);
  if (warp->warp_id < 64) {
    all_warps[warp->warp_id] = warp;
  }
}

void WarpScheduler::barrier_release_unit() {
  uint64_t barrier_mask;
  if (warps_per_block == 0) {
    barrier_mask = UINT64_MAX;  // All warps
  } else if (warps_per_block >= 64) {
    barrier_mask = UINT64_MAX;  // All 64 warps (or more, but we only have 64)
  } else {
    barrier_mask = (1ULL << warps_per_block) - 1;
  }
  
  // Barrier release state machine (matching SIMTight's state machine)
  if (barrier_release_state == 0) {
    // State 0: Wait for a barrier
    if (barrier_bits != 0) {
      barrier_shift_reg = barrier_bits;
      release_warp_id = 0;
      barrier_release_state = 1;
    }
    // If barrier_bits == 0, stay in state 0 (no warps in barrier
  } else if (barrier_release_state == 1) {
    // State 1: Check on current barrier status
    bool all_in_barrier = ((barrier_shift_reg & barrier_mask) == barrier_mask);
    release_success = all_in_barrier;
    release_warp_count = 1;
    
    if (barrier_shift_reg == 0) {
      barrier_release_state = 0;
    } else {
      barrier_release_state = 2;
    }
  } else if (barrier_release_state == 2) {
    // State 2: Shift and release
    unsigned warps_per_block_check = (warps_per_block == 0) ? 64 : warps_per_block;
    unsigned block_start_warp = (warps_per_block == 0) ? 0 : ((release_warp_id / warps_per_block_check) * warps_per_block_check);
    unsigned block_end_warp = block_start_warp + warps_per_block_check - 1;
    
    if (release_success) {
      // Release all warps in the block at once for simplicity
      for (unsigned warp_id = block_start_warp; warp_id <= block_end_warp && warp_id < 64; warp_id++) {
        auto it = all_warps.find(warp_id);
        if (it != all_warps.end()) {
          Warp *w = it->second;
          if (w->in_barrier && !w->is_cpu) {
            w->in_barrier = false;
            // Update barrier_bits to reflect the release
            barrier_bits &= ~(1ULL << warp_id);
          }
        }
      }
      
      // Shift barrier_shift_reg by warps_per_block to skip all warps in the block
      for (unsigned i = 0; i < warps_per_block_check && release_warp_id < 64; i++) {
        barrier_shift_reg = barrier_shift_reg >> 1;
        release_warp_id++;
        release_warp_count++;
      }
    } else {
      // If release_success is false, process warps
      // This handles the case where not all warps are ready yet
      if (release_warp_id < 64) {
        auto it = all_warps.find(release_warp_id);
        if (it != all_warps.end()) {
          Warp *w = it->second;
          if (w->in_barrier && !w->is_cpu) {
            // Don't release - not all warps are ready
          }
        }
      }
      
      barrier_shift_reg = barrier_shift_reg >> 1;
      release_warp_id++;
      release_warp_count++;
    }
    
    // Move back to state 1 when finished with block
    if (warps_per_block == 0) {
      if (release_warp_id >= 64) {
        barrier_release_state = 1;
        // If we've processed all warps, go back to state 0
        if (barrier_shift_reg == 0) {
          barrier_release_state = 0;
        }
      }
    } else {
      // Process warps_per_block warps before going back to state 1
      if (release_warp_count > warps_per_block || release_warp_id >= 64) {
        barrier_release_state = 1;
        // If we've processed all warps, go back to state 0
        if (release_warp_id >= 64) {
          barrier_release_state = 0;
        }
      }
    }
  }
}

void WarpScheduler::set_warps_per_block(unsigned n) {
  warps_per_block = n;
  log("Warp Scheduler", "Set warps per block to " + std::to_string(n) + 
      (n == 0 ? " (all warps)" : ""));
}

WarpScheduler::~WarpScheduler() {
  flush_new_warps();

  while (warp_queue.size() > 0) {
    delete warp_queue.front();
    warp_queue.pop();
  }
  while (reinsert_delay_queue.size() > 0) reinsert_delay_queue.pop();
  while (reinsert_ready.size() > 0) reinsert_ready.pop();

  log("Warp Scheduler", "Destroyed pipeline stage");
}