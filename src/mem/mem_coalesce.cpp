#include "mem_coalesce.hpp"

#define DRAM_LATENCY 100
#define DRAM_BURST_LATENCY 4
#define DRAM_BLOCK_SIZE 64

int CoalescingUnit::calculate_bursts(const std::vector<uint64_t> &addrs,
                                     size_t access_size) {
  std::set<uint64_t> active_blocks;
  for (uint64_t addr : addrs) {
    uint64_t start_block = addr / DRAM_BLOCK_SIZE;
    uint64_t end_block = (addr + access_size - 1) / DRAM_BLOCK_SIZE;

    for (uint64_t block = start_block; block <= end_block; block++) {
      active_blocks.insert(block);
    }
  }
  return active_blocks.size();
}

void CoalescingUnit::suspend_warp(Warp *warp,
                                  const std::vector<uint64_t> &addrs,
                                  size_t access_size) {
  warp->suspended = true;
  int bursts = calculate_bursts(addrs, access_size);

  for (int i = 0; i < bursts; i++) {
    if (warp->is_cpu) {
        GPUStatisticsManager::instance().increment_cpu_dram_accs();
    } else {
        GPUStatisticsManager::instance().increment_gpu_dram_accs();
    }
  }

  int latency = DRAM_LATENCY + (bursts * DRAM_BURST_LATENCY);
  blocked_warps[warp] = latency;
}

std::vector<int> CoalescingUnit::load(Warp *warp,
                                      const std::vector<uint64_t> &addrs,
                                      size_t bytes) {
  suspend_warp(warp, addrs, bytes);
  std::vector<int> results;
  for (uint64_t addr : addrs) {
    results.push_back(scratchpad_mem->load(addr, bytes));
  }
  return results;
}

void CoalescingUnit::store(Warp *warp, const std::vector<uint64_t> &addrs,
                           size_t bytes, const std::vector<int> &vals) {
  suspend_warp(warp, addrs, bytes);
  for (size_t i = 0; i < addrs.size(); i++) {
    scratchpad_mem->store(addrs[i], bytes, vals[i]);
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

  if (resumable_warp == nullptr) return nullptr;

  blocked_warps.erase(resumable_warp);
  return resumable_warp;
}

void CoalescingUnit::tick() {
  for (auto &[key, val] : blocked_warps) {
    if (val > 0) val--;
  }
}