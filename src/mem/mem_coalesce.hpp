#pragma once

#include "data_cache.hpp"
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
  Warp *get_resumable_warp();
  void tick();

  // Cache statistics accessors
  uint64_t get_cache_hits() { return cache.get_hits(); }
  uint64_t get_cache_misses() { return cache.get_misses(); }
  double get_cache_hit_rate() { return cache.get_hit_rate(); }

  // Flush cache (for kernel completion)
  void flush_cache() { cache.flush(); }

private:
  std::map<Warp *, size_t> blocked_warps;
  DataMemory *scratchpad_mem;
  DataCache cache;

  void suspend_warp(Warp *warp, const std::vector<uint64_t> &addrs,
                    size_t access_size, bool is_store);
  int calculate_cache_misses(const std::vector<uint64_t> &addrs,
                             size_t access_size, bool is_store);
  int calculate_bursts(const std::vector<uint64_t> &addrs, size_t access_size,
                       bool is_store);
};