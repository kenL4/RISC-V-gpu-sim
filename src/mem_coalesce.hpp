#pragma once

#include "utils.hpp"
#include "pipeline.hpp"
#include "mem_data.hpp"

class CoalescingUnit {
public:
    CoalescingUnit(DataMemory *scratchpad_mem): scratchpad_mem(scratchpad_mem) {};
    int load(Warp *warp, uint64_t addr, size_t size);
    void store(Warp *warp, uint64_t addr, size_t size, int val);
    bool is_busy();
    Warp *get_resumable_warp();
    void tick();
private:
    std::map<Warp*, size_t> blocked_warps;
    DataMemory *scratchpad_mem;

    void suspend_warp(Warp *warp);
};