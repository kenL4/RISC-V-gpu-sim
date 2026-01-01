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

  ~WarpScheduler();

private:
  int warp_size;
  int warp_count;
  std::queue<Warp *> warp_queue;
  std::queue<Warp *> new_warp_queue;
  bool active = true;

  void flush_new_warps();
};