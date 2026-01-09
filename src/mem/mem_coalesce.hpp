#pragma once

#include "gpu/pipeline.hpp"
#include "mem_data.hpp"
#include "utils.hpp"

class CoalescingUnit {
public:
  CoalescingUnit(DataMemory *scratchpad_mem);
  std::vector<int> load(Warp *warp, const std::vector<uint64_t> &addrs,
                        size_t bytes);
  void store(Warp *warp, const std::vector<uint64_t> &addrs, size_t bytes,
             const std::vector<int> &vals);
  bool is_busy();
  bool is_busy_for_pipeline(bool is_cpu_pipeline);
  Warp *get_resumable_warp();
  Warp *get_resumable_warp_for_pipeline(bool is_cpu_pipeline);
  void tick();
  
  // Suspend warp for a given number of cycles (for functional unit latencies)
  void suspend_warp_latency(Warp *warp, size_t latency);

private:
  std::map<Warp *, size_t> blocked_warps;
  DataMemory *scratchpad_mem;

  void suspend_warp(Warp *warp, const std::vector<uint64_t> &addrs,
                    size_t access_size, bool is_store);
  int calculate_bursts(const std::vector<uint64_t> &addrs, size_t access_size,
                       bool is_store);
};