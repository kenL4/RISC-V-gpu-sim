#pragma once

#include "gpu/pipeline.hpp"
#include "mem_data.hpp"
#include "utils.hpp"

class CoalescingUnit {
public:
  CoalescingUnit(DataMemory *scratchpad_mem)
      : scratchpad_mem(scratchpad_mem) {};
  int load(Warp *warp, uint64_t addr, size_t bytes);
  void store(Warp *warp, uint64_t addr, size_t bytes, int val);
  bool is_busy();
  Warp *get_resumable_warp();
  void tick();

private:
  std::map<Warp *, size_t> blocked_warps;
  DataMemory *scratchpad_mem;

  void suspend_warp(Warp *warp);
};