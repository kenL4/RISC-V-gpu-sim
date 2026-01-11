#include "pipeline_ats.hpp"
#include <string>
#include <vector>

ActiveThreadSelection::ActiveThreadSelection() {
  log("Active Thread Selection", "Initializing Active Thread Selection Stage");
}

bool ActiveThreadSelection::is_active() {
  // Must also check stage_buffer because of 2-cycle latency
  // If data is buffered, we're still active (will output on next cycle)
  return PipelineStage::input_latch->updated || stage_buffer.valid;
}

void ActiveThreadSelection::execute() {
  // 2nd substage: Output buffered results (matching SIMTight's 2nd substage)
  if (stage_buffer.valid) {
    PipelineStage::output_latch->updated = true;
    PipelineStage::output_latch->warp = stage_buffer.warp;
    PipelineStage::output_latch->active_threads = stage_buffer.active_threads;
    log("Active Thread Selection",
        "Warp " + std::to_string(stage_buffer.warp->warp_id) + " has " +
            std::to_string(stage_buffer.active_threads.size()) + " active threads (substage 2)");
    stage_buffer.valid = false;  // Clear buffer
  } else {
    PipelineStage::output_latch->updated = false;
    PipelineStage::output_latch->warp = nullptr;
  }

  // 1st substage: Compute active threads (matching SIMTight's 1st substage)
  if (!PipelineStage::input_latch->updated) {
    return;
  }

  Warp *warp = PipelineStage::input_latch->warp;
  uint64_t max_nesting = 0;
  uint64_t warp_pc = 0;
  bool found_active = false;
  for (int i = 0; i < warp->size; i++) {
    if (warp->finished[i])
      continue;
    if (!found_active || warp->nesting_level[i] > max_nesting) {
      max_nesting = warp->nesting_level[i];
      warp_pc = warp->pc[i];
      found_active = true;
    }
  }

  if (!found_active) {
    // All threads finished?
    // We should probably handle this case.
    // For now, just return empty active threads.
    PipelineStage::input_latch->updated = false;
    // Store in buffer for next cycle
    stage_buffer.warp = warp;
    stage_buffer.active_threads = {};
    stage_buffer.valid = true;
    log("Active Thread Selection", "Warp " + std::to_string(warp->warp_id) +
                                       " has 0 active threads (all finished) (substage 1)");
    return;
  }
  
  std::vector<uint64_t> active_threads;
  for (int i = 0; i < warp->size; i++) {
    if (warp->finished[i])
      continue;
    bool same_nesting = max_nesting == warp->nesting_level[i];
    bool same_pc = warp_pc == warp->pc[i];
    if (same_nesting && same_pc) {
      active_threads.emplace_back(i);
    }
  }

  // Store results in buffer for next cycle (2nd substage)
  PipelineStage::input_latch->updated = false;
  stage_buffer.warp = warp;
  stage_buffer.active_threads = active_threads;
  stage_buffer.valid = true;
  log("Active Thread Selection",
      "Warp " + std::to_string(warp->warp_id) + " computed " +
          std::to_string(active_threads.size()) + " active threads (substage 1)");
}