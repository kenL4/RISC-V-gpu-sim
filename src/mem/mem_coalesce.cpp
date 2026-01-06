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
  // SIMTight coalescing strategy implementation
  // SIMTight has 32 lanes and 64-byte (512-bit) DRAM beats
  constexpr size_t NUM_LANES = 32;
  constexpr size_t LOG_LANES = 5;

  if (addrs.empty()) {
    return 0;
  }

  // Filter out SRAM requests first
  std::vector<uint64_t> dram_addrs;
  for (uint64_t addr : addrs) {
    uint64_t addr_32 = 0xFFFFFFFF & addr;
    // Skip SRAM region (shared SRAM)
    if (SIM_SHARED_SRAM_BASE <= addr_32 && addr_32 < SIM_SIMT_STACK_BASE) {
      continue;
    }
    dram_addrs.push_back(addr);
  }

  if (dram_addrs.empty()) {
    return 0;
  }

  // Strategy 1: SameAddress - all lanes access the same address
  // This is the most efficient: 1 DRAM access serves all lanes
  bool all_same_addr = true;
  uint64_t first_addr = dram_addrs[0];
  for (size_t i = 1; i < dram_addrs.size(); i++) {
    if (dram_addrs[i] != first_addr) {
      all_same_addr = false;
      break;
    }
  }
  if (all_same_addr) {
    // One DRAM access serves all lanes
    return 1;
  }

  // Strategy 2: SameBlock - lanes access same block with lane-ID-aligned
  // addresses In SIMTight, this works when:
  // - Upper bits (block address) are the same for all lanes
  // - Lower bits match the lane ID pattern
  //
  // For word accesses (4 bytes), the pattern is:
  //   addr[1:0] are same across all lanes (sub-word offset)
  //   addr[LOG_LANES+1:2] equals lane ID
  //
  // When SameBlock succeeds:
  // - Byte/Half accesses: 1 beat
  // - Word accesses: 2 beats (since 32 lanes * 4 bytes = 128 bytes > 64 byte
  // beat)

  // Check if all addresses are in the same 64*NUM_LANES byte block
  // (For SameBlock to work, upper bits must match)
  uint64_t block_addr = first_addr >> (LOG_LANES + 2); // Upper bits above lane
                                                       // addressing
  bool same_block = true;
  for (const auto &addr : dram_addrs) {
    if ((addr >> (LOG_LANES + 2)) != block_addr) {
      same_block = false;
      break;
    }
  }

  if (same_block) {
    // Check for lane-ID-aligned pattern based on access size
    bool is_word_access = (access_size >= 4);

    if (is_word_access) {
      // Word mode: addr[1:0] same, addr[LOG_LANES+1:2] = lane_id
      // This means we can pack 16 words per beat (64 bytes / 4 bytes)
      // With 32 lanes, we need 2 beats
      uint64_t sub_word_bits = first_addr & 0x3; // bits [1:0]
      bool valid_pattern = true;

      for (size_t lane = 0; lane < dram_addrs.size(); lane++) {
        uint64_t addr = dram_addrs[lane];
        // Check sub-word bits are consistent
        if ((addr & 0x3) != sub_word_bits) {
          valid_pattern = false;
          break;
        }
        // Check lane ID pattern: bits [LOG_LANES+1:2] should equal lane ID
        uint64_t lane_bits = (addr >> 2) & (NUM_LANES - 1);
        if (lane_bits != lane) {
          valid_pattern = false;
          break;
        }
      }

      if (valid_pattern) {
        // Word mode SameBlock: 2 beats
        return 2;
      }
    } else {
      // Byte/Half mode checking
      bool is_half_access = (access_size == 2);

      if (is_half_access) {
        // Half mode: addr[LOG_LANES:1] = lane_id, addr[LOG_LANES+1] same
        uint64_t upper_bit = (first_addr >> (LOG_LANES + 1)) & 0x1;
        bool valid_pattern = true;

        for (size_t lane = 0; lane < dram_addrs.size(); lane++) {
          uint64_t addr = dram_addrs[lane];
          if (((addr >> (LOG_LANES + 1)) & 0x1) != upper_bit) {
            valid_pattern = false;
            break;
          }
          uint64_t lane_bits = (addr >> 1) & (NUM_LANES - 1);
          if (lane_bits != lane) {
            valid_pattern = false;
            break;
          }
        }

        if (valid_pattern) {
          return 1;
        }
      } else {
        // Byte mode: addr[LOG_LANES-1:0] = lane_id, addr[LOG_LANES+1:LOG_LANES]
        // same
        uint64_t upper_bits = (first_addr >> LOG_LANES) & 0x3;
        bool valid_pattern = true;

        for (size_t lane = 0; lane < dram_addrs.size(); lane++) {
          uint64_t addr = dram_addrs[lane];
          if (((addr >> LOG_LANES) & 0x3) != upper_bits) {
            valid_pattern = false;
            break;
          }
          uint64_t lane_bits = addr & (NUM_LANES - 1);
          if (lane_bits != lane) {
            valid_pattern = false;
            break;
          }
        }

        if (valid_pattern) {
          return 1;
        }
      }
    }
  }

  // Fallback: Neither SameAddress nor SameBlock applies fully
  // SIMTight uses SameAddress as fallback, grouping lanes by address
  // Each unique address needs one DRAM access
  std::set<uint64_t> unique_addrs;
  for (uint64_t addr : dram_addrs) {
    unique_addrs.insert(addr);
  }
  return unique_addrs.size();
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

  // Calculate number of coalesced DRAM bursts (unique cache-line-sized blocks)
  // This is the number of DRAM transactions that would be issued
  int dram_bursts = calculate_bursts(addrs, access_size);
  for (int i = 0; i < dram_bursts; i++) {
    if (warp->is_cpu) {
      GPUStatisticsManager::instance().increment_cpu_dram_accs();
    } else {
      GPUStatisticsManager::instance().increment_gpu_dram_accs();
    }
  }

  // Still use cache for latency modeling (not sure if this is correct)
  int cache_misses = calculate_cache_misses(addrs, access_size, is_store);

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