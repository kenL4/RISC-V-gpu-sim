#pragma once

#include "gpu/pipeline.hpp"
#include "mem_data.hpp"
#include "utils.hpp"

class CoalescingUnit {
public:
  CoalescingUnit(DataMemory *scratchpad_mem)
      : scratchpad_mem(scratchpad_mem) {};
  std::vector<int> load(Warp *warp, const std::vector<uint64_t> &addrs,
                        size_t bytes);
  void store(Warp *warp, const std::vector<uint64_t> &addrs, size_t bytes,
             const std::vector<int> &vals);
  bool is_busy();
  Warp *get_resumable_warp();
  void tick();

private:
  std::map<Warp *, size_t> blocked_warps;
  DataMemory *scratchpad_mem;

  void suspend_warp(Warp *warp, const std::vector<uint64_t> &addrs,
                    size_t access_size);
  int calculate_bursts(const std::vector<uint64_t> &addrs, size_t access_size);
};