#include "mem_coalesce.hpp"

// Use centralized config values from config.hpp (included via data_cache.hpp)
// SIM_DRAM_LATENCY = 30 (matching SIMTight)
// SIM_CACHE_HIT_LATENCY = 2
// SIM_CACHE_LINE_SIZE = 64 (DRAM beat size)

CoalescingUnit::CoalescingUnit(DataMemory *scratchpad_mem)
    : scratchpad_mem(scratchpad_mem) {
  cache.set_backing_memory(scratchpad_mem);
}

int CoalescingUnit::calculate_bursts(const std::vector<uint64_t> &addrs,
                                     size_t access_size) {
  std::set<uint64_t> active_blocks;
  for (uint64_t addr : addrs) {
    uint64_t start_block = addr / SIM_CACHE_LINE_SIZE;
    uint64_t end_block = (addr + access_size - 1) / SIM_CACHE_LINE_SIZE;

    for (uint64_t block = start_block; block <= end_block; block++) {
      active_blocks.insert(block);
    }
  }
  return active_blocks.size();
}

int CoalescingUnit::calculate_cache_misses(const std::vector<uint64_t> &addrs,
                                           size_t access_size, bool is_store) {
  int misses = 0;
  std::set<uint64_t> checked_lines;

  for (uint64_t addr : addrs) {
    // Check each cache line this access touches
    uint64_t start_line = addr / SIM_CACHE_LINE_SIZE;
    uint64_t end_line = (addr + access_size - 1) / SIM_CACHE_LINE_SIZE;

    for (uint64_t line = start_line; line <= end_line; line++) {
      // Only count each unique cache line once per coalesced access
      if (checked_lines.find(line) != checked_lines.end()) {
        continue;
      }
      checked_lines.insert(line);

      uint64_t line_addr = line * SIM_CACHE_LINE_SIZE;
      // access() returns true on hit, false on miss
      if (!cache.access(line_addr, is_store)) {
        misses++;
      }
    }
  }
  return misses;
}

void CoalescingUnit::suspend_warp(Warp *warp,
                                  const std::vector<uint64_t> &addrs,
                                  size_t access_size, bool is_store) {
  warp->suspended = true;

  // Calculate cache misses (this also populates the cache)
  int cache_misses = calculate_cache_misses(addrs, access_size, is_store);

  // Only count DRAM accesses for cache misses
  for (int i = 0; i < cache_misses; i++) {
    if (warp->is_cpu) {
      GPUStatisticsManager::instance().increment_cpu_dram_accs();
    } else {
      GPUStatisticsManager::instance().increment_gpu_dram_accs();
    }
  }

  // Calculate latency based on hits vs misses
  int total_blocks = calculate_bursts(addrs, access_size);
  int cache_hits = total_blocks - cache_misses;

  // Latency model:
  // - Cache hits: fast (CACHE_HIT_LATENCY per hit)
  // - Cache misses: slow (DRAM_LATENCY + DRAM_BURST_LATENCY per miss)
  int latency;
  if (cache_misses > 0) {
    // On any miss, we pay DRAM latency plus burst latency per miss
    latency = SIM_DRAM_LATENCY + cache_misses;
  } else {
    // All hits - just cache latency
    latency = SIM_CACHE_HIT_LATENCY;
  }

  blocked_warps[warp] = latency;
}

std::vector<int> CoalescingUnit::load(Warp *warp,
                                      const std::vector<uint64_t> &addrs,
                                      size_t bytes) {
  suspend_warp(warp, addrs, bytes, false);
  std::vector<int> results;

  // Read from cache (all lines are now guaranteed to be present)
  for (uint64_t addr : addrs) {
    int64_t value = 0;
    for (size_t i = 0; i < bytes; i++) {
      uint8_t byte = cache.load_byte(addr + i);
      value |= (static_cast<int64_t>(byte) << (8 * i));
    }
    // Sign extend if needed
    if (bytes < 8) {
      int shift = 64 - (bytes * 8);
      value = (value << shift) >> shift;
    }
    results.push_back(static_cast<int>(value));
  }
  return results;
}

void CoalescingUnit::store(Warp *warp, const std::vector<uint64_t> &addrs,
                           size_t bytes, const std::vector<int> &vals) {
  suspend_warp(warp, addrs, bytes, true);

  // Write to cache
  for (size_t i = 0; i < addrs.size(); i++) {
    uint64_t addr = addrs[i];
    uint64_t val = static_cast<uint64_t>(vals[i]);
    for (size_t j = 0; j < bytes; j++) {
      cache.store_byte(addr + j, (val >> (8 * j)) & 0xFF);
    }
  }
}

bool CoalescingUnit::is_busy() { return !blocked_warps.empty(); }

Warp *CoalescingUnit::get_resumable_warp() {
  Warp *resumable_warp = nullptr;

  for (auto &[key, val] : blocked_warps) {
    if (val == 0) {
      resumable_warp = key;
      break;
    }
  }

  if (resumable_warp == nullptr)
    return nullptr;

  blocked_warps.erase(resumable_warp);
  return resumable_warp;
}

void CoalescingUnit::tick() {
  for (auto &[key, val] : blocked_warps) {
    if (val > 0)
      val--;
  }
}