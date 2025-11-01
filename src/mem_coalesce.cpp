#include "mem_coalesce.hpp"

#define L1_CACHE_LATENCY 1
#define DRAM_LATENCY 2

void CoalescingUnit::suspend_warp(Warp *warp) {
    warp->suspended = true;
    blocked_warps[warp] = DRAM_LATENCY;
}

int CoalescingUnit::load(Warp *warp, uint64_t addr, size_t size) {
    suspend_warp(warp);
    return 123;
}

bool CoalescingUnit::is_busy() {
    return !blocked_warps.empty();
}

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
        val--;
    }
}