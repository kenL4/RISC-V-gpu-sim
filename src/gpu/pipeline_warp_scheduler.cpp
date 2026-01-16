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
      all_warps[i] = warp;  // Track all warps for barrier release
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

  // Update barrier bits from all warps (matching SIMTight's relBarrierVec)
  barrier_bits = 0;
  for (const auto& [warp_id, warp] : all_warps) {
    if (!warp->is_cpu && !warp->finished[0] && warp->in_barrier) {
      if (warp_id < 64) {
        barrier_bits |= (1ULL << warp_id);
      }
    }
  }
  

  // Barrier release unit (matching SIMTight's makeBarrierReleaseUnit)
  barrier_release_unit();

  if (warp_queue.empty()) {
    return;  // No warps available
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

void WarpScheduler::insert_warp(Warp *warp) { 
  new_warp_queue.push(warp);
  // Track all warps for barrier release
  if (warp->warp_id < 64) {
    all_warps[warp->warp_id] = warp;
  }
}

// Barrier release unit (matching SIMTight's makeBarrierReleaseUnit)
// This logic looks for threads in a block that have all entered a barrier, and then releases them.
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
    // State 0: Load barrier bits into shift register
    if (barrier_bits != 0) {
      barrier_shift_reg = barrier_bits;
      release_warp_id = 0;
      barrier_release_state = 1;
    }
    // If barrier_bits == 0, stay in state 0 (no warps in barrier)
  } else if (barrier_release_state == 1) {
    bool all_in_barrier = ((barrier_shift_reg & barrier_mask) == barrier_mask);
    release_success = all_in_barrier;
    release_warp_count = 1;
    
    // Move to next state
    if (barrier_shift_reg == 0) {
      barrier_release_state = 0;  // No warps in barrier, go back to state 0 to reload
    } else {
      barrier_release_state = 2;  // Some warps in barrier, process them
    }
  } else if (barrier_release_state == 2) {
    // State 2: Shift and release
    // Matching SIMTight: when (releaseState.val .==. 2 .&&. inv ins.relDisable) do
    unsigned warps_per_block_check = (warps_per_block == 0) ? 64 : warps_per_block;
    unsigned block_start_warp = (warps_per_block == 0) ? 0 : ((release_warp_id / warps_per_block_check) * warps_per_block_check);
    unsigned block_end_warp = block_start_warp + warps_per_block_check - 1;
    
    if (release_success) {
      // Release all warps in the block at once
      for (unsigned warp_id = block_start_warp; warp_id <= block_end_warp && warp_id < 64; warp_id++) {
        auto it = all_warps.find(warp_id);
        if (it != all_warps.end()) {
          Warp *w = it->second;
          if (w->in_barrier && !w->is_cpu) {
            w->in_barrier = false;  // Clear barrier bit
            // Update barrier_bits to reflect the release
            barrier_bits &= ~(1ULL << warp_id);
          }
        }
      }
      
      // Shift barrier_shift_reg by warps_per_block to skip all warps in the block
      // This is equivalent to processing all warps in the block sequentially
      for (unsigned i = 0; i < warps_per_block_check && release_warp_id < 64; i++) {
        barrier_shift_reg = barrier_shift_reg >> 1;
        release_warp_id++;
        release_warp_count++;
      }
    } else {
      // If release_success is false, process warps one by one (matching SIMTight's behavior)
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
      
      // Shift barrier shift register right by 1
      barrier_shift_reg = barrier_shift_reg >> 1;
      
      // Move to next warp
      release_warp_id++;
      release_warp_count++;
    }
    
    // Move back to state 1 when finished with block
    // Matching SIMTight: when (releaseWarpCount.val .==. ins.relWarpsPerBlock) do releaseState <== 1
    if (warps_per_block == 0) {
      // Process all warps (64) before going back to state 1
      if (release_warp_id >= 64) {
        barrier_release_state = 1;
        // If we've processed all warps, go back to state 0
        if (barrier_shift_reg == 0) {
          barrier_release_state = 0;
        }
      }
    } else {
      // Process warps_per_block warps before going back to state 1
      // Note: release_warp_count starts at 1 in State 1, so we match when release_warp_count == warps_per_block
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
  // Matching SIMTight: warpsPerBlock = n, barrierMask = (n == 0) ? all_ones : (1 << n) - 1
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

  log("Warp Scheduler", "Destroyed pipeline stage");
}