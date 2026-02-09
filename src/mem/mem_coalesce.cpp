#include "mem_coalesce.hpp"
#include "mem_data.hpp"
#include "config.hpp"
#include "gen/llvm_riscv_registers.h"
#include "stats/stats.hpp"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <set>
#include <sstream>
#include <iomanip>
#include <cstdint>

CoalescingUnit::CoalescingUnit(DataMemory *scratchpad_mem, const std::string *trace_file)
    : scratchpad_mem(scratchpad_mem) {
  if (trace_file != nullptr) {
    tracer = std::make_unique<Tracer>(*trace_file);
  }
}

bool CoalescingUnit::can_put() {
  return pending_request_queue.size() < MEM_REQ_QUEUE_CAPACITY;
}

// Uses the SameAddress and SameBlock strategies
int CoalescingUnit::calculate_bursts(const std::vector<uint64_t> &addrs,
                                     size_t access_size, bool is_store) {
  constexpr size_t LOG_LANES = 5;

  if (addrs.empty()) {
    return 0;
  }

  // Build list of (lane_id, addr) pairs for DRAM requests only
  std::vector<std::pair<size_t, uint64_t>> pending;
  for (size_t lane = 0; lane < addrs.size(); lane++) {
    uint64_t addr = addrs[lane];
    uint64_t addr_32 = 0xFFFFFFFF & addr;

    // Skip SRAM region (shared SRAM)
    // In the real GPU, these would be rerouted via a switching network
    if (SIM_SHARED_SRAM_BASE <= addr_32 && addr_32 < SIM_SIMT_STACK_BASE) {
      continue;
    }

    pending.push_back({lane, addr});
  }

  if (pending.empty()) return 0;

  int total_dram_accesses = 0;

  // Iterative coalescing: process lanes until all are served
  while (!pending.empty()) {
    // Pick the first pending lane as the leader (matching SIMTight)
    size_t leader_lane = pending[0].first;
    uint64_t leader_addr = pending[0].second;

    // Compute coalescing masks based on leader
    std::vector<size_t> same_block_lanes;
    std::vector<size_t> same_addr_lanes;

    uint64_t leader_block = leader_addr >> (LOG_LANES + 2);
    uint64_t leader_low_bits = leader_addr & ((1ULL << (LOG_LANES + 2)) - 1);

    for (const auto &[lane, addr] : pending) {
      // Check if in same block (required for BOTH SameAddress and SameBlock)
      uint64_t block = addr >> (LOG_LANES + 2);
      bool in_same_block = (block == leader_block);

      // Check SameAddress: sameBlock AND lower LOG_LANES+2 bits match
      uint64_t low_bits = addr & ((1ULL << (LOG_LANES + 2)) - 1);
      if (in_same_block && low_bits == leader_low_bits) {
        same_addr_lanes.push_back(lane);
      }

      // Check SameBlock based on access size
      if (in_same_block) {
        bool matches = false;

        if (access_size >= 4) {
          // Word mode: addr[1:0] must match leader, addr[LOG_LANES+1:2] must
          // equal lane_id
          uint64_t sub_word = addr & 0x3;
          uint64_t leader_sub_word = leader_addr & 0x3;
          uint64_t lane_bits = (addr >> 2) & (NUM_LANES - 1);
          if (sub_word == leader_sub_word && lane_bits == lane) {
            matches = true;
          }
        } else if (access_size == 2) {
          // Half mode: addr[LOG_LANES:1] must equal lane_id, addr[LOG_LANES+1]
          // must match leader
          uint64_t upper_bit = (addr >> (LOG_LANES + 1)) & 0x1;
          uint64_t leader_upper = (leader_addr >> (LOG_LANES + 1)) & 0x1;
          uint64_t lane_bits = (addr >> 1) & (NUM_LANES - 1);
          if (upper_bit == leader_upper && lane_bits == lane) {
            matches = true;
          }
        } else {
          // Byte mode: addr[LOG_LANES-1:0] must equal lane_id,
          // addr[LOG_LANES+1:LOG_LANES] must match leader
          uint64_t upper_bits = (addr >> LOG_LANES) & 0x3;
          uint64_t leader_upper = (leader_addr >> LOG_LANES) & 0x3;
          uint64_t lane_bits = addr & (NUM_LANES - 1);
          if (upper_bits == leader_upper && lane_bits == lane) {
            matches = true;
          }
        }

        if (matches) {
          same_block_lanes.push_back(lane);
        }
      }
    }

    // Use SameBlock if it satisfies leader AND at least one
    // other lane
    bool use_same_block =
        (same_block_lanes.size() > 1) &&
        (std::find(same_block_lanes.begin(), same_block_lanes.end(),
                   leader_lane) != same_block_lanes.end());

    std::vector<size_t> *served_lanes;
    int bursts_for_this_access;

    if (use_same_block) {
      served_lanes = &same_block_lanes;
      // Word accesses need 2 beats, byte/half need 1
      bursts_for_this_access = (access_size >= 4) ? 2 : 1;
    } else {
      // Use SameAddress strategy - always 1 transaction
      served_lanes = &same_addr_lanes;
      bursts_for_this_access = 1;
    }

    total_dram_accesses += bursts_for_this_access;

    // Remove served lanes from pending
    std::set<size_t> served_set(served_lanes->begin(), served_lanes->end());
    std::vector<std::pair<size_t, uint64_t>> remaining;
    for (const auto &p : pending) {
      if (served_set.find(p.first) == served_set.end()) {
        remaining.push_back(p);
      }
    }
    pending = std::move(remaining);
  }

  return total_dram_accesses;
}

int CoalescingUnit::calculate_request_count(const std::vector<uint64_t> &addrs,
                                           size_t access_size) {
  constexpr size_t LOG_LANES = 5;
  if (addrs.empty()) return 0;

  // Build list of (lane_id, addr) pairs for DRAM requests only
  std::vector<std::pair<size_t, uint64_t>> pending;
  for (size_t lane = 0; lane < addrs.size(); lane++) {
    uint64_t addr = addrs[lane];
    uint64_t addr_32 = 0xFFFFFFFF & addr;
    // Skip SRAM region (shared SRAM)
    if (SIM_SHARED_SRAM_BASE <= addr_32 && addr_32 < SIM_SIMT_STACK_BASE) {
      continue;
    }
    pending.push_back({lane, addr});
  }

  if (pending.empty()) return 0;
  int request_count = 0;

  // Iterative coalescing: process lanes until all are served
  while (!pending.empty()) {
    request_count++;  // Count each coalesced request

    // Pick the first pending lane as the leader
    size_t leader_lane = pending[0].first;
    uint64_t leader_addr = pending[0].second;

    // Compute coalescing masks based on leader
    std::vector<size_t> same_block_lanes;
    std::vector<size_t> same_addr_lanes;

    // SameBlock and SameAddress are computed relative to leader
    uint64_t leader_block = leader_addr >> (LOG_LANES + 2);
    uint64_t leader_low_bits = leader_addr & ((1ULL << (LOG_LANES + 2)) - 1);

    for (const auto &[lane, addr] : pending) {
      uint64_t block = addr >> (LOG_LANES + 2);
      bool in_same_block = (block == leader_block);

      uint64_t low_bits = addr & ((1ULL << (LOG_LANES + 2)) - 1);
      if (in_same_block && low_bits == leader_low_bits) {
        same_addr_lanes.push_back(lane);
      }

      if (in_same_block) {
        bool matches = false;

        if (access_size >= 4) {
          uint64_t sub_word = addr & 0x3;
          uint64_t leader_sub_word = leader_addr & 0x3;
          uint64_t lane_bits = (addr >> 2) & (NUM_LANES - 1);
          if (sub_word == leader_sub_word && lane_bits == lane) {
            matches = true;
          }
        } else if (access_size == 2) {
          uint64_t upper_bit = (addr >> (LOG_LANES + 1)) & 0x1;
          uint64_t leader_upper = (leader_addr >> (LOG_LANES + 1)) & 0x1;
          uint64_t lane_bits = (addr >> 1) & (NUM_LANES - 1);
          if (upper_bit == leader_upper && lane_bits == lane) {
            matches = true;
          }
        } else {
          uint64_t upper_bits = (addr >> LOG_LANES) & 0x3;
          uint64_t leader_upper = (leader_addr >> LOG_LANES) & 0x3;
          uint64_t lane_bits = addr & (NUM_LANES - 1);
          if (upper_bits == leader_upper && lane_bits == lane) {
            matches = true;
          }
        }

        if (matches) {
          same_block_lanes.push_back(lane);
        }
      }
    }

    bool use_same_block =
        (same_block_lanes.size() > 1) &&
        (std::find(same_block_lanes.begin(), same_block_lanes.end(),
                   leader_lane) != same_block_lanes.end());

    std::vector<size_t> *served_lanes;
    if (use_same_block) {
      served_lanes = &same_block_lanes;
    } else {
      served_lanes = &same_addr_lanes;
    }

    // Remove served lanes from pending
    std::set<size_t> served_set(served_lanes->begin(), served_lanes->end());
    std::vector<std::pair<size_t, uint64_t>> remaining;
    for (const auto &p : pending) {
      if (served_set.find(p.first) == served_set.end()) {
        remaining.push_back(p);
      }
    }
    pending = std::move(remaining);
  }

  return request_count;
}

std::vector<uint64_t> CoalescingUnit::compute_coalesced_addresses(const std::vector<uint64_t> &addrs,
                                                                   size_t access_size) {
  constexpr size_t LOG_LANES = 5;
  std::vector<uint64_t> coalesced_addrs;
  
  if (addrs.empty()) return coalesced_addrs;

  // Build list of (lane_id, addr) pairs for DRAM requests only
  std::vector<std::pair<size_t, uint64_t>> pending;
  for (size_t lane = 0; lane < addrs.size(); lane++) {
    uint64_t addr = addrs[lane];
    uint64_t addr_32 = 0xFFFFFFFF & addr;
    // Skip SRAM region (shared SRAM)
    if (SIM_SHARED_SRAM_BASE <= addr_32 && addr_32 < SIM_SIMT_STACK_BASE) {
      continue;
    }
    pending.push_back({lane, addr});
  }

  if (pending.empty()) return coalesced_addrs;

  // Iterative coalescing: process lanes until all are served
  while (!pending.empty()) {
    // Pick the first pending lane as the leader
    size_t leader_lane = pending[0].first;
    uint64_t leader_addr = pending[0].second;

    // Compute coalescing masks based on leader
    std::vector<size_t> same_block_lanes;
    std::vector<size_t> same_addr_lanes;

    // SameBlock and SameAddress are computed relative to leader
    uint64_t leader_block = leader_addr >> (LOG_LANES + 2);
    uint64_t leader_low_bits = leader_addr & ((1ULL << (LOG_LANES + 2)) - 1);

    for (const auto &[lane, addr] : pending) {
      uint64_t block = addr >> (LOG_LANES + 2);
      bool in_same_block = (block == leader_block);

      uint64_t low_bits = addr & ((1ULL << (LOG_LANES + 2)) - 1);
      if (in_same_block && low_bits == leader_low_bits) {
        same_addr_lanes.push_back(lane);
      }

      if (in_same_block) {
        bool matches = false;

        if (access_size >= 4) {
          uint64_t sub_word = addr & 0x3;
          uint64_t leader_sub_word = leader_addr & 0x3;
          uint64_t lane_bits = (addr >> 2) & (NUM_LANES - 1);
          if (sub_word == leader_sub_word && lane_bits == lane) {
            matches = true;
          }
        } else if (access_size == 2) {
          uint64_t upper_bit = (addr >> (LOG_LANES + 1)) & 0x1;
          uint64_t leader_upper = (leader_addr >> (LOG_LANES + 1)) & 0x1;
          uint64_t lane_bits = (addr >> 1) & (NUM_LANES - 1);
          if (upper_bit == leader_upper && lane_bits == lane) {
            matches = true;
          }
        } else {
          uint64_t upper_bits = (addr >> LOG_LANES) & 0x3;
          uint64_t leader_upper = (leader_addr >> LOG_LANES) & 0x3;
          uint64_t lane_bits = addr & (NUM_LANES - 1);
          if (upper_bits == leader_upper && lane_bits == lane) {
            matches = true;
          }
        }

        if (matches) {
          same_block_lanes.push_back(lane);
        }
      }
    }

    bool use_same_block =
        (same_block_lanes.size() > 1) &&
        (std::find(same_block_lanes.begin(), same_block_lanes.end(),
                   leader_lane) != same_block_lanes.end());

    std::vector<size_t> *served_lanes;
    if (use_same_block) {
      served_lanes = &same_block_lanes;
    } else {
      served_lanes = &same_addr_lanes;
    }

    // Add the leader address to coalesced addresses (this represents the coalesced group)
    coalesced_addrs.push_back(leader_addr);

    // Remove served lanes from pending
    std::set<size_t> served_set(served_lanes->begin(), served_lanes->end());
    std::vector<std::pair<size_t, uint64_t>> remaining;
    for (const auto &p : pending) {
      if (served_set.find(p.first) == served_set.end()) {
        remaining.push_back(p);
      }
    }
    pending = std::move(remaining);
  }

  return coalesced_addrs;
}

/*
 * Stack address translation for data operations (stores/loads in scratchpad_mem).
 * Uses a simple per-thread layout that ensures each thread has unique physical addresses.
 * This must be self-consistent: store(V) and load(V) for the same thread must
 * map to the same physical address.
 */
uint64_t CoalescingUnit::translate_stack_address(uint64_t virtual_addr, Warp *warp, size_t thread_id) {
  uint64_t addr_32 = virtual_addr & 0xFFFFFFFF;
  if (addr_32 < SIM_SIMT_STACK_BASE) return virtual_addr;
  uint64_t offset = addr_32 - SIM_SIMT_STACK_BASE;

  if (warp->is_cpu) {
    return (virtual_addr & 0xFFFFFFFF00000000ULL) | (SIM_CPU_STACK_BASE + offset);
  } else {
    uint64_t warp_offset = static_cast<uint64_t>(warp->warp_id) << (SIMT_LOG_LANES + SIMT_LOG_BYTES_PER_STACK);
    uint64_t thread_offset = static_cast<uint64_t>(thread_id) << SIMT_LOG_BYTES_PER_STACK;
    uint64_t physical_addr = SIM_SIMT_STACK_BASE + warp_offset + thread_offset + offset;
    return (virtual_addr & 0xFFFFFFFF00000000ULL) | physical_addr;
  }
}

/*
 * SIMTight-matching interleaved address for coalescing/DRAM counting purposes.
 * From SIMTight src/Core/SIMT.hs (interleaveAddr):
 *   If vaddr[31:19] == all 1s (stack region):
 *     paddr = 0b11 # vaddr[18:2] # warp_id[5:0] # lane_id[4:0] # vaddr[1:0]
 *   Else:
 *     paddr = vaddr (unchanged)
 *
 * This interleaving ensures that when all lanes in a warp access the same
 * stack offset, their physical addresses differ only in bits [6:2] (= lane_id),
 * enabling SameBlock coalescing — matching how SIMTight's hardware counts DRAMAccs.
 */
uint64_t CoalescingUnit::interleave_addr_simtight(uint64_t virtual_addr, Warp *warp, size_t thread_id) {
  uint64_t addr_32 = virtual_addr & 0xFFFFFFFF;

  // SIMTight stack detection: bits [31:19] must all be 1
  uint32_t top_bits = addr_32 >> SIMT_LOG_BYTES_PER_STACK;
  uint32_t all_ones = (1U << (32 - SIMT_LOG_BYTES_PER_STACK)) - 1;
  if (top_bits != all_ones) return virtual_addr;

  uint32_t stackOffset = (addr_32 >> 2) & ((1U << (SIMT_LOG_BYTES_PER_STACK - 2)) - 1);
  uint32_t wordOffset = addr_32 & 0x3;
  uint32_t warp_id = static_cast<uint32_t>(warp->warp_id) & ((1U << SIMT_LOG_WARPS) - 1);
  uint32_t lane_id = static_cast<uint32_t>(thread_id) & ((1U << SIMT_LOG_LANES) - 1);

  uint32_t paddr = (0x3U << 30)
                 | (stackOffset << (2 + SIMT_LOG_LANES + SIMT_LOG_WARPS))
                 | (warp_id << (2 + SIMT_LOG_LANES))
                 | (lane_id << 2)
                 | wordOffset;

  return (virtual_addr & 0xFFFFFFFF00000000ULL) | paddr;
}

std::vector<uint64_t> CoalescingUnit::build_translated_lane_addrs(
    Warp *warp, const std::vector<uint64_t> &addrs,
    const std::vector<size_t> &active_threads) {
  // Build a NUM_LANES-sized vector indexed by lane_id with translated physical addresses.
  // Inactive lanes get SIM_SHARED_SRAM_BASE (in the SRAM region, filtered by calculate_bursts).
  // This ensures that the vector index == lane_id, which is required for correct
  // SameBlock matching (addr[6:2] must equal lane_id).
  std::vector<uint64_t> lane_addrs(NUM_LANES, SIM_SHARED_SRAM_BASE);
  for (size_t i = 0; i < addrs.size(); i++) {
    size_t lane_id = active_threads[i];
    lane_addrs[lane_id] = interleave_addr_simtight(addrs[i], warp, lane_id);
  }
  return lane_addrs;
}

void CoalescingUnit::suspend_warp(Warp *warp,
                                  const std::vector<uint64_t> &addrs,
                                  size_t access_size, bool is_store) {
  warp->suspended = true;

  // Calculate number of coalesced DRAM bursts (unique cache-line-sized blocks)
  // This is the number of DRAM transactions that would be issued
  int dram_bursts = calculate_bursts(addrs, access_size, is_store);
  
  // Count DRAM accesses matching SIMTight behavior:
  // Both loads and stores count by burst length. SIMTight's coalescing unit
  // issues burstLen separate DRAM requests for stores, each adding 1 to
  // dramStoreSig; for loads, dramLoadSig = burstLen fires once.
  int dram_access_count = dram_bursts;
  
  for (int i = 0; i < dram_access_count; i++) {
    if (warp->is_cpu) {
      GPUStatisticsManager::instance().increment_cpu_dram_accs();
    } else {
      GPUStatisticsManager::instance().increment_gpu_dram_accs();
    }
  }

  // Latency model (matching SIMTight - no cache, direct DRAM access):
  // Each DRAM transaction has SIM_DRAM_LATENCY cycles of latency
  // Multiple transactions are pipelined, so total latency = base latency + additional bursts
  // For simplicity, use: SIM_DRAM_LATENCY + (bursts - 1) if bursts > 0
  int latency;
  if (dram_bursts == 0) {
    // If no DRAM accesses (e.g., all accesses to SRAM), use minimal latency
    latency = 1;
  } else if (dram_bursts == 1) {
    latency = SIM_DRAM_LATENCY;
  } else {
    // Multiple bursts: first has full latency, additional ones add 1 cycle each (pipelined)
    latency = SIM_DRAM_LATENCY + (dram_bursts - 1);
  }

  blocked_warps[warp] = latency;
}

void CoalescingUnit::load(Warp *warp, const std::vector<uint64_t> &addrs,
                          size_t bytes, unsigned int rd_reg,
                          const std::vector<size_t> &active_threads,
                          bool is_zero_extend) {
  // Note: Caller should check can_put() before calling this function. If queue is full, caller should retry.
  
  // Trace addresses coming into coalescing unit (skip CPU warps)
  if (tracer && !warp->is_cpu) {
    TraceEvent event;
    event.cycle = GPUStatisticsManager::instance().get_gpu_cycles();
    event.warp_id = warp->warp_id;
    // Use PC of first active thread, or thread 0 if no active threads
    if (!active_threads.empty() && active_threads[0] < warp->pc.size()) {
      event.pc = warp->pc[active_threads[0]];
    } else if (!warp->pc.empty()) {
      event.pc = warp->pc[0];
    } else {
      event.pc = 0;
    }
    event.event_type = MEM_REQ_ISSUE;
    event.addrs = addrs;
    tracer->trace_event(event);
  }
  
  MemRequest req;
  req.warp = warp;
  req.addrs = addrs;
  req.bytes = bytes;
  req.is_store = false;
  req.is_atomic = false;
  req.is_fence = false;
  req.is_zero_extend = is_zero_extend;
  req.rd_reg = rd_reg;
  req.active_threads = active_threads;
  pending_request_queue.push(req);
  
  warp->suspended = true;

  // Latency: use original addresses for simulation correctness
  int sim_bursts = calculate_bursts(addrs, bytes, false);
  int latency;
  if (sim_bursts == 0) {
    latency = COALESCING_PIPELINE_DEPTH + 1;
  } else if (sim_bursts == 1) {
    latency = COALESCING_PIPELINE_DEPTH + SIM_DRAM_LATENCY;
  } else {
    latency = COALESCING_PIPELINE_DEPTH + SIM_DRAM_LATENCY + (sim_bursts - 1);
  }
  blocked_warps[warp] = latency;

  // Count DRAM accesses on interleaved physical addresses (matching SIMTight's
  // hardware coalescing which operates on interleaved physical addresses)
  std::vector<uint64_t> phys_addrs = build_translated_lane_addrs(warp, addrs, active_threads);
  int dram_access_count = calculate_bursts(phys_addrs, bytes, false);
  for (int i = 0; i < dram_access_count; i++) {
    if (warp->is_cpu) {
      GPUStatisticsManager::instance().increment_cpu_dram_accs();
      if (GPUStatisticsManager::instance().is_gpu_pipeline_active()) {
        GPUStatisticsManager::instance().increment_gpu_active_cpu_dram_accs();
      }
    } else {
      GPUStatisticsManager::instance().increment_gpu_dram_accs();
    }
  }
}

void CoalescingUnit::store(Warp *warp, const std::vector<uint64_t> &addrs,
                           size_t bytes, const std::vector<int> &vals,
                           const std::vector<size_t> &active_threads) {
  // Note: Caller should check can_put() before calling this function. If queue is full, caller should retry.
  
  // Trace addresses coming into coalescing unit (skip CPU warps)
  if (tracer && !warp->is_cpu) {
    TraceEvent event;
    event.cycle = GPUStatisticsManager::instance().get_gpu_cycles();
    event.warp_id = warp->warp_id;
    // Use PC of first active thread, or thread 0 if no active threads
    if (!active_threads.empty() && active_threads[0] < warp->pc.size()) {
      event.pc = warp->pc[active_threads[0]];
    } else if (!warp->pc.empty()) {
      event.pc = warp->pc[0];
    } else {
      event.pc = 0;
    }
    event.event_type = MEM_REQ_ISSUE;
    event.addrs = addrs;
    tracer->trace_event(event);
  }
  
  MemRequest req;
  req.warp = warp;
  req.addrs = addrs;
  req.bytes = bytes;
  req.is_store = true;
  req.is_atomic = false;
  req.is_fence = false;
  req.store_values = vals;
  req.active_threads = active_threads;
  pending_request_queue.push(req);
  
  warp->suspended = true;

  // Latency: use original addresses for simulation correctness
  int sim_bursts = calculate_bursts(addrs, bytes, true);
  int latency;
  if (sim_bursts == 0) {
    latency = COALESCING_PIPELINE_DEPTH + 1;
  } else {
    latency = COALESCING_PIPELINE_DEPTH + SIM_DRAM_LATENCY;
  }
  blocked_warps[warp] = latency;

  // Count DRAM accesses on interleaved physical addresses
  // SIMTight's coalescing unit issues burstLen separate DRAM requests for stores
  // (one per beat), and each fires dramStoreSig=1 in the DRAM wrapper, so stores
  // should be counted by burst length (same as loads), not 1 per coalesced group.
  std::vector<uint64_t> phys_addrs = build_translated_lane_addrs(warp, addrs, active_threads);
  int dram_access_count = calculate_bursts(phys_addrs, bytes, true);
  for (int i = 0; i < dram_access_count; i++) {
    if (warp->is_cpu) {
      GPUStatisticsManager::instance().increment_cpu_dram_accs();
      if (GPUStatisticsManager::instance().is_gpu_pipeline_active()) {
        GPUStatisticsManager::instance().increment_gpu_active_cpu_dram_accs();
      }
    } else {
      GPUStatisticsManager::instance().increment_gpu_dram_accs();
    }
  }
}

void CoalescingUnit::fence(Warp *warp) {
  // Queue the fence request
  MemRequest req;
  req.warp = warp;
  req.addrs = {};
  req.bytes = 0;
  req.is_store = false;
  req.is_atomic = false;
  req.is_fence = true;
  req.active_threads = {};
  pending_request_queue.push(req);
  
  warp->suspended = true;
  // A bit of a hack
  blocked_warps[warp] = MEM_REQ_QUEUE_CAPACITY;
}

void CoalescingUnit::atomic_add(Warp *warp, const std::vector<uint64_t> &addrs,
                                 size_t bytes, unsigned int rd_reg,
                                 const std::vector<int> &add_values,
                                 const std::vector<size_t> &active_threads) {
  // Queue the atomic add request
  
  // Trace addresses coming into coalescing unit (skip CPU warps)
  if (tracer && !warp->is_cpu) {
    TraceEvent event;
    event.cycle = GPUStatisticsManager::instance().get_gpu_cycles();
    event.warp_id = warp->warp_id;
    // Use PC of first active thread, or thread 0 if no active threads
    if (!active_threads.empty() && active_threads[0] < warp->pc.size()) {
      event.pc = warp->pc[active_threads[0]];
    } else if (!warp->pc.empty()) {
      event.pc = warp->pc[0];
    } else {
      event.pc = 0;
    }
    event.event_type = MEM_REQ_ISSUE;
    event.addrs = addrs;
    tracer->trace_event(event);
  }
  
  MemRequest req;
  req.warp = warp;
  req.addrs = addrs;
  req.bytes = bytes;
  req.is_store = false;
  req.is_atomic = true;
  req.is_fence = false;
  req.atomic_add_values = add_values;
  req.rd_reg = rd_reg;
  req.active_threads = active_threads;
  pending_request_queue.push(req);
  
  warp->suspended = true;

  // Latency: use original addresses for simulation correctness
  int sim_bursts = calculate_bursts(addrs, bytes, true);
  int latency;
  if (sim_bursts == 0) {
    latency = COALESCING_PIPELINE_DEPTH + 1;
  } else {
    latency = COALESCING_PIPELINE_DEPTH + SIM_DRAM_LATENCY;
  }
  blocked_warps[warp] = latency;

  // Count DRAM accesses on interleaved physical addresses
  // Atomics use store path in SIMTight, which issues burstLen DRAM requests per
  // coalesced group (same reasoning as stores above).
  std::vector<uint64_t> phys_addrs = build_translated_lane_addrs(warp, addrs, active_threads);
  int dram_access_count = calculate_bursts(phys_addrs, bytes, true);
  for (int i = 0; i < dram_access_count; i++) {
    if (warp->is_cpu) {
      GPUStatisticsManager::instance().increment_cpu_dram_accs();
      if (GPUStatisticsManager::instance().is_gpu_pipeline_active()) {
        GPUStatisticsManager::instance().increment_gpu_active_cpu_dram_accs();
      }
    } else {
      GPUStatisticsManager::instance().increment_gpu_dram_accs();
    }
  }
}

bool CoalescingUnit::is_busy() { 
  return !pending_request_queue.empty() || !blocked_warps.empty();
}

bool CoalescingUnit::is_busy_for_pipeline(bool is_cpu_pipeline) {
  for (auto &[key, val] : blocked_warps) {
    if (key->is_cpu == is_cpu_pipeline) {
      return true;
    }
  }
  return false;
}

Warp *CoalescingUnit::get_resumable_warp_for_pipeline(bool is_cpu_pipeline) {
  Warp *resumable_warp = nullptr;

  for (auto &[key, val] : blocked_warps) {
    if (val == 0 && key->is_cpu == is_cpu_pipeline) {
      // Check if this is a fence that's trying to complete
      // If so, verify that all pending operations from this warp have completed
      bool is_fence_completing = false;
      
      // Check if there's a fence request for this warp in the pipeline
      std::queue<PipelineRequest> temp_pipeline = pipeline_queue;
      while (!temp_pipeline.empty()) {
        if (temp_pipeline.front().req.warp == key && temp_pipeline.front().req.is_fence) {
          is_fence_completing = true;
          break;
        }
        temp_pipeline.pop();
      }
      
      // If this is a fence, check for pending operations from the same warp
      if (is_fence_completing) {
        bool has_pending = false;
        
        // Check pending queue
        std::queue<MemRequest> temp_queue = pending_request_queue;
        while (!temp_queue.empty()) {
          if (temp_queue.front().warp == key && !temp_queue.front().is_fence) {
            has_pending = true;
            break;
          }
          temp_queue.pop();
        }
        
        // Check pipeline queue
        if (!has_pending) {
          std::queue<PipelineRequest> temp_pipeline2 = pipeline_queue;
          while (!temp_pipeline2.empty()) {
            if (temp_pipeline2.front().req.warp == key && 
                !temp_pipeline2.front().req.is_fence) {
              has_pending = true;
              break;
            }
            temp_pipeline2.pop();
          }
        }
        
        // If there are pending operations, don't resume the fence yet
        if (has_pending) {
          blocked_warps[key] = 1;  // Add one more cycle of delay
          continue;
        }
      }
      
      resumable_warp = key;
      break;
    }
  }

  if (resumable_warp == nullptr)
    return nullptr;

  blocked_warps.erase(resumable_warp);
  return resumable_warp;
}

void CoalescingUnit::suspend_warp_latency(Warp *warp, size_t latency) {
  warp->suspended = true;
  blocked_warps[warp] = latency;
}

void CoalescingUnit::tick() {
  // Pipeline model: Move requests through pipeline stages
  // 1. Advance requests in pipeline_queue (increment cycles_in_pipeline)
  // 2. Process requests that have completed pipeline (cycles_in_pipeline >= DEPTH)
  // 3. Move requests from pending_queue to pipeline_queue (if pipeline has space)
  
  // Step 1 & 2: Advance pipeline and process completed requests
  bool requests_processed_this_cycle = false;
  size_t pipeline_size = pipeline_queue.size();
  for (size_t i = 0; i < pipeline_size; i++) {
    PipelineRequest pipe_req = pipeline_queue.front();
    pipeline_queue.pop();
    
    pipe_req.cycles_in_pipeline++;
    
    if (pipe_req.cycles_in_pipeline >= COALESCING_PIPELINE_DEPTH) {
      // Request has completed pipeline, process it
      process_mem_request(pipe_req.req);
      requests_processed_this_cycle = true;
    } else {
      // Request still in pipeline, put it back
      pipeline_queue.push(pipe_req);
    }
  }
  
  // Step 3: Move requests from pending to pipeline
  bool go1_active = !pipeline_queue.empty();
  bool stalled = false;
  bool can_consume = (!stalled || !go1_active);
  
  if (can_consume && !pending_request_queue.empty() && 
      pipeline_queue.size() < COALESCING_PIPELINE_DEPTH &&
      (pipeline_queue.size() + pending_request_queue.size()) < MEM_REQ_QUEUE_CAPACITY) {
    PipelineRequest pipe_req;
    pipe_req.req = pending_request_queue.front();
    pipe_req.cycles_in_pipeline = 0;
    pending_request_queue.pop();
    pipeline_queue.push(pipe_req);
  }
  
  // Decrement latency counters for blocked warps
  for (auto &[key, val] : blocked_warps) {
    if (val > 0)
      val--;
  }
}

void CoalescingUnit::process_mem_request(const MemRequest &req) {
  // Trace addresses going out of coalescing unit (after translation and coalescing, skip CPU warps)
  if (tracer && !req.is_fence && !req.warp->is_cpu) {
    // First translate all addresses
    std::vector<uint64_t> translated_addrs;
    for (size_t i = 0; i < req.addrs.size(); i++) {
      uint64_t virtual_addr = req.addrs[i];
      uint64_t addr = translate_stack_address(virtual_addr, req.warp, req.active_threads[i]);
      translated_addrs.push_back(addr);
    }
    
    // Then compute coalesced addresses (one per coalesced group)
    std::vector<uint64_t> coalesced_addrs = compute_coalesced_addresses(translated_addrs, req.bytes);
    
    if (!coalesced_addrs.empty()) {
      TraceEvent event;
      event.cycle = GPUStatisticsManager::instance().get_gpu_cycles();
      event.warp_id = req.warp->warp_id;
      // Use PC of first active thread, or thread 0 if no active threads
      if (!req.active_threads.empty() && req.active_threads[0] < req.warp->pc.size()) {
        event.pc = req.warp->pc[req.active_threads[0]];
      } else if (!req.warp->pc.empty()) {
        event.pc = req.warp->pc[0];
      } else {
        event.pc = 0;
      }
      event.event_type = DRAM_REQ_ISSUE;
      event.addrs = coalesced_addrs;
      tracer->trace_event(event);
    }
  }
  
  if (req.is_fence) {
    bool has_pending_ops = false;
    
    // Check pending_request_queue (operations queued after the fence)
    std::queue<MemRequest> temp_queue = pending_request_queue;
    while (!temp_queue.empty()) {
      if (temp_queue.front().warp == req.warp && !temp_queue.front().is_fence) {
        has_pending_ops = true;
        break;
      }
      temp_queue.pop();
    }
    
    // Check pipeline_queue (operations currently in pipeline)
    if (!has_pending_ops) {
      std::queue<PipelineRequest> temp_pipeline = pipeline_queue;
      while (!temp_pipeline.empty()) {
        if (temp_pipeline.front().req.warp == req.warp && !temp_pipeline.front().req.is_fence) {
          has_pending_ops = true;
          break;
        }
        temp_pipeline.pop();
      }
    }
    
    // If there are pending operations, delay fence completion (latency choice is a hack)
    if (has_pending_ops && blocked_warps.find(req.warp) != blocked_warps.end()) {
      blocked_warps[req.warp] = COALESCING_PIPELINE_DEPTH + SIM_DRAM_LATENCY + 5;
    }
  } else if (req.is_atomic) {
    // Process atomic add: read old value, add to it, write back, return old value
    std::map<size_t, int> results;
    for (size_t i = 0; i < req.addrs.size(); i++) {
      uint64_t virtual_addr = req.addrs[i];
      uint64_t addr = translate_stack_address(virtual_addr, req.warp, req.active_threads[i]);

      int64_t old_value = scratchpad_mem->load(addr, req.bytes);
      int64_t new_value = old_value + req.atomic_add_values[i];

      scratchpad_mem->store(addr, req.bytes, static_cast<uint64_t>(new_value));
      results[req.active_threads[i]] = static_cast<int>(old_value);
    }
    
    load_results_map[req.warp] = std::make_pair(req.rd_reg, results);
  } else if (req.is_store) {
    // Process store: write to memory
    assert(req.addrs.size() == req.store_values.size() && "Store request: addresses and values must have same size");
    
    for (size_t i = 0; i < req.addrs.size(); i++) {
      uint64_t virtual_addr = req.addrs[i];
      uint64_t addr = translate_stack_address(virtual_addr, req.warp, req.active_threads[i]);
      uint64_t val = static_cast<uint64_t>(req.store_values[i]);
      scratchpad_mem->store(addr, req.bytes, val);
    }
  } else {
    // Process load: read from memory and store results
    std::map<size_t, int> results;
    for (size_t i = 0; i < req.addrs.size(); i++) {
      uint64_t virtual_addr = req.addrs[i];
      uint64_t base_addr = translate_stack_address(virtual_addr, req.warp, req.active_threads[i]);
      
      uint64_t raw = 0;
      for (int j = 0; j < req.bytes; j++) {
        uint64_t addr = base_addr + j;
        if (scratchpad_mem->get_raw_memory().find(addr) != scratchpad_mem->get_raw_memory().end()) {
          raw += (uint64_t)scratchpad_mem->get_raw_memory().at(addr) << (8 * j);
        }
      }
      
      // Apply sign-extension or zero-extension based on load type
      int64_t value;
      if (req.is_zero_extend) {
        value = zero_extend(raw, req.bytes);
      } else {
        value = sign_extend(raw, req.bytes);
      }
      
      results[req.active_threads[i]] = static_cast<int>(value);
    }
    
    load_results_map[req.warp] = std::make_pair(req.rd_reg, results);
  }
}

std::pair<unsigned int, std::map<size_t, int>> CoalescingUnit::get_load_results(Warp *warp) {
  auto it = load_results_map.find(warp);
  if (it != load_results_map.end()) {
    std::pair<unsigned int, std::map<size_t, int>> results = it->second;
    load_results_map.erase(it);  // Remove after retrieving
    return results;
  }
  return std::make_pair(0, std::map<size_t, int>());
}

bool CoalescingUnit::has_pending_memory_ops(Warp *warp) {
  // If warp is suspended, it has pending memory operations (can't enter barrier while suspended)
  if (warp->suspended) return true;
  
  // Check if warp is blocked with pending memory operations
  auto blocked_it = blocked_warps.find(warp);
  if (blocked_it != blocked_warps.end() && blocked_it->second > 0) return true;
  
  // Check pending_request_queue for this warp
  std::queue<MemRequest> temp_queue = pending_request_queue;
  while (!temp_queue.empty()) {
    if (temp_queue.front().warp == warp) return true;
    temp_queue.pop();
  }
  
  // Check pipeline_queue for this warp
  std::queue<PipelineRequest> temp_pipeline = pipeline_queue;
  while (!temp_pipeline.empty()) {
    if (temp_pipeline.front().req.warp == warp) return true;
    temp_pipeline.pop();
  }
  
  return false;
}