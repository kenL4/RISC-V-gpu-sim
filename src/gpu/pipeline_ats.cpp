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
    std::string name = stage_buffer.warp->is_cpu ? "CPU" : "Warp " + std::to_string(stage_buffer.warp->warp_id);
    log("Active Thread Selection",
        name + " has " +
            std::to_string(stage_buffer.active_threads.size()) + " active threads (substage 2)");
    stage_buffer.valid = false;  // Clear buffer
  } else {
    PipelineStage::output_latch->updated = false;
    PipelineStage::output_latch->warp = nullptr;
  }

  // 1st substage: Compute active threads (matching SIMTight's 1st substage)
  // Matching SIMTight: Active threads are those with the max nesting level
  // On a tie, favour instructions undergoing a retry
  // Active threads match the leader's state exactly (PC + nesting_level + retry)
  if (!PipelineStage::input_latch->updated) {
    return;
  }

  Warp *warp = PipelineStage::input_latch->warp;
  
  // Find leader thread: max of (nesting_level # retry)
  // Matching SIMTight: maxOf compares (nestLevel # retry) as a combined value
  // On tie, favors retry (retry=1 is treated as higher)
  int leader_idx = -1;
  uint64_t leader_value = 0;  // (nesting_level << 1) | retry
  for (int i = 0; i < warp->size; i++) {
    if (warp->finished[i])
      continue;
    uint64_t value = (warp->nesting_level[i] << 1) | (warp->retrying[i] ? 1 : 0);
    if (leader_idx == -1 || value > leader_value) {
      leader_idx = i;
      leader_value = value;
    }
  }

  if (leader_idx == -1) {
    // All threads finished
    PipelineStage::input_latch->updated = false;
    stage_buffer.warp = warp;
    stage_buffer.active_threads = {};
    stage_buffer.valid = true;
    std::string name = stage_buffer.warp->is_cpu ? "CPU" : "Warp " + std::to_string(stage_buffer.warp->warp_id);
    log("Active Thread Selection", name + " has 0 active threads (all finished) (substage 1)");
    return;
  }
  
  // Leader's state (matching SIMTight's state2)
  uint64_t leader_pc = warp->pc[leader_idx];
  uint64_t leader_nesting = warp->nesting_level[leader_idx];
  bool leader_retry = warp->retrying[leader_idx];
  
  // Active threads: those matching leader's state exactly (state2 === s)
  // Matching SIMTight: activeList = [state2 === s | s <- stateMemOuts2]
  std::vector<uint64_t> active_threads;
  for (int i = 0; i < warp->size; i++) {
    if (warp->finished[i])
      continue;
    // Compare full state: PC, nesting_level, and retry must all match
    bool state_matches = (warp->pc[i] == leader_pc) &&
                         (warp->nesting_level[i] == leader_nesting) &&
                         (warp->retrying[i] == leader_retry);
    if (state_matches) {
      active_threads.emplace_back(i);
    }
  }

  // Store results in buffer for next cycle (2nd substage)
  PipelineStage::input_latch->updated = false;
  stage_buffer.warp = warp;
  stage_buffer.active_threads = active_threads;
  stage_buffer.valid = true;
  std::string name = stage_buffer.warp->is_cpu ? "CPU" : "Warp " + std::to_string(stage_buffer.warp->warp_id);
  log("Active Thread Selection",
      name + " computed " +
          std::to_string(active_threads.size()) + " active threads (substage 1)");
}