#pragma once

#include "pipeline.hpp"
#include "utils.hpp"

/*
 * Represents the warp scheduler unit in the pipeline
 * It will use a barrel scheduler to fairly pick between the warps
 * that have no suspended threads
 */
class WarpScheduler : public PipelineStage {
public:
  WarpScheduler(int warp_size, int warp_count, uint64_t start_pc,
                bool start_active = true);
  void execute() override;
  bool is_active() override;
  void set_active(bool active) { this->active = active; }
  void insert_warp(Warp *warp);
  bool did_issue_warp() const { return warp_issued_this_cycle; }

  ~WarpScheduler();

private:
  int warp_size;
  int warp_count;
  std::queue<Warp *> warp_queue;
  std::queue<Warp *> new_warp_queue;
  bool active = true;
  bool warp_issued_this_cycle = false;

  // 2-cycle latency modeling (matching SIMTight's 2 substages)
  // 1st substage: Choose warp
  // 2nd substage: Output chosen warp
  Warp *chosen_warp_buffer = nullptr;  // Buffer between substage 1 and 2

  // Fair scheduler state (matching SIMTight)
  // History bitmask: bit i set means warp i was recently scheduled
  uint64_t sched_history = 0;

  // Helper functions for fair scheduling
  static uint64_t firstHot(uint64_t x);
  std::pair<uint64_t, uint64_t> fair_scheduler(uint64_t hist, uint64_t avail);

  void flush_new_warps();
};