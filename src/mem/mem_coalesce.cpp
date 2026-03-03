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

CoalescingUnit::~CoalescingUnit() {
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

bool CoalescingUnit::is_sram_access(const MemRequest &req) const {
  if (req.is_fence || req.addrs.empty()) return false;
  for (const auto &addr : req.addrs) {
    uint64_t addr_32 = addr & 0xFFFFFFFF;
    if (!(SIM_SHARED_SRAM_BASE <= addr_32 && addr_32 < SIM_SIMT_STACK_BASE)) {
      return false;
    }
  }
  return true;
}

int CoalescingUnit::calculate_sram_bank_conflicts(const MemRequest &req) const {
  if (!req.is_store && !req.is_atomic && req.addrs.size() > 1) {
    bool all_same = true;
    for (size_t i = 1; i < req.addrs.size(); i++) {
      if (req.addrs[i] != req.addrs[0]) {
        all_same = false;
        break;
      }
    }
    if (all_same) return 2;
  }

  int bank_count[SRAM_BANKS] = {};
  for (const auto &addr : req.addrs) {
    int bank = (addr >> 2) & (SRAM_BANKS - 1);
    bank_count[bank]++;
  }
  int max_per_bank = 0;
  for (size_t i = 0; i < SRAM_BANKS; i++) {
    max_per_bank = std::max(max_per_bank, bank_count[i]);
  }
  return std::max(max_per_bank, 2);
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

  if (instr_tracer && !warp->is_cpu) {
    TraceEvent event;
    event.cycle = GPUStatisticsManager::instance().get_gpu_cycles();
    event.pc = warp->pc[0];
    event.warp_id = warp->warp_id;
    event.lane_id = -1;
    event.event_type = WARP_SUSPEND;
    instr_tracer->trace_event(event);
  }

  std::vector<uint64_t> phys_addrs = build_translated_lane_addrs(warp, addrs, active_threads);
  int sim_bursts = calculate_bursts(phys_addrs, bytes, false);
  int sim_groups = calculate_request_count(phys_addrs, bytes);

  int latency;
  if (sim_bursts == 0) {
    latency = COALESCING_PIPELINE_DEPTH + 1;
  } else {
    int feedback_cycles = (sim_groups > 1) ? 3 * (sim_groups - 1) : 0;
    int bursts_extra = (sim_bursts > sim_groups) ? sim_bursts - sim_groups : 0;
    latency = feedback_cycles + COALESCING_PIPELINE_DEPTH + SIM_DRAM_LATENCY
            + SIM_DRAM_RESP_OVERHEAD
            + bursts_extra;
  }
  blocked_warps[warp] = latency;

  int dram_access_count = sim_bursts;
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

  if (!warp->is_cpu) {
    GPUStatisticsManager::instance().increment_gpu_dram_accs();
  }

  if (instr_tracer && !warp->is_cpu) {
    TraceEvent event;
    event.cycle = GPUStatisticsManager::instance().get_gpu_cycles();
    event.pc = warp->pc[0];
    event.warp_id = warp->warp_id;
    event.lane_id = -1;
    event.event_type = WARP_SUSPEND;
    instr_tracer->trace_event(event);
  }

  int latency = COALESCING_PIPELINE_DEPTH + SIM_DRAM_LATENCY + SIM_DRAM_RESP_OVERHEAD;
  blocked_warps[warp] = latency;
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

  if (instr_tracer && !warp->is_cpu) {
    TraceEvent event;
    event.cycle = GPUStatisticsManager::instance().get_gpu_cycles();
    event.pc = warp->pc[0];
    event.warp_id = warp->warp_id;
    event.lane_id = -1;
    event.event_type = WARP_SUSPEND;
    instr_tracer->trace_event(event);
  }

  std::vector<uint64_t> phys_addrs = build_translated_lane_addrs(warp, addrs, active_threads);
  int sim_bursts = calculate_bursts(phys_addrs, bytes, true);
  int sim_groups = calculate_request_count(phys_addrs, bytes);

  int latency;
  if (sim_bursts == 0) {
    latency = COALESCING_PIPELINE_DEPTH + 1;
  } else {
    int feedback_cycles = (sim_groups > 1) ? 4 * (sim_groups - 1) : 0;
    latency = feedback_cycles + COALESCING_PIPELINE_DEPTH + SIM_DRAM_LATENCY
            + SIM_DRAM_RESP_OVERHEAD;
  }
  blocked_warps[warp] = latency;

  // Count DRAM accesses on interleaved physical addresses
  // Atomics use store path in SIMTight, which issues burstLen DRAM requests per
  // coalesced group (same reasoning as stores above).
  int dram_access_count = sim_bursts;
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
  return !pending_request_queue.empty() || !blocked_warps.empty()
      || coalescing_remaining > 0 || coalescing_waiting;
}

void CoalescingUnit::reset_dram_state() {
  coalescing_remaining = 0;
  coalescing_waiting = false;
  go5_busy_remaining = 0;
  inflight_count_reg = 0;
  for (size_t s = 0; s < COALESCING_PIPELINE_DEPTH; s++)
    pipeline_stages[s] = std::nullopt;
  dram_queue_depth = 0;
  dram_inflight = 0;
  while (!dram_response_schedule.empty()) dram_response_schedule.pop();
  while (!sram_queue.empty()) sram_queue.pop();
  sram_processing_remaining = 0;
  next_dram_resp_available = 0;
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
      bool still_in_queues = false;
      {
        std::queue<MemRequest> tmp = pending_request_queue;
        while (!tmp.empty()) {
          if (tmp.front().warp == key) { still_in_queues = true; break; }
          tmp.pop();
        }
      }
      if (!still_in_queues) {
        for (size_t s = 0; s < COALESCING_PIPELINE_DEPTH; s++) {
          if (pipeline_stages[s] && pipeline_stages[s]->req.warp == key) {
            still_in_queues = true; break;
          }
        }
      }
      if (still_in_queues) {
        blocked_warps[key] = 1;
        continue;
      }

      // Check if this is a fence that's trying to complete
      // If so, verify that all pending operations from this warp have completed
      bool is_fence_completing = false;

      // Check if there's a fence request for this warp in the pipeline
      for (size_t s = 0; s < COALESCING_PIPELINE_DEPTH; s++) {
        if (pipeline_stages[s] && pipeline_stages[s]->req.warp == key &&
            pipeline_stages[s]->req.is_fence) {
          is_fence_completing = true;
          break;
        }
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

        if (!has_pending) {
          for (size_t s = 0; s < COALESCING_PIPELINE_DEPTH; s++) {
            if (pipeline_stages[s] && pipeline_stages[s]->req.warp == key &&
                !pipeline_stages[s]->req.is_fence) {
              has_pending = true;
              break;
            }
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

  if (resumable_warp == divider_warp) {
    divider_warp = nullptr;
  }

  mul_pipeline_warps.erase(resumable_warp);

  blocked_warps.erase(resumable_warp);
  return resumable_warp;
}

void CoalescingUnit::suspend_warp_latency(Warp *warp, size_t latency) {
  warp->suspended = true;
  blocked_warps[warp] = latency;
}

void CoalescingUnit::suspend_for_func_unit(Warp *warp, size_t latency,
                                           unsigned int rd_reg,
                                           const std::map<size_t, int> &results) {
  warp->suspended = true;
  blocked_warps[warp] = latency;
  load_results_map[warp] = {rd_reg, results};

  if (instr_tracer) {
    TraceEvent event;
    event.cycle = tick_counter;
    event.warp_id = warp->warp_id;
    event.event_type = WARP_SUSPEND;
    instr_tracer->trace_event(event);
  }
}

void CoalescingUnit::tick() {

  tick_counter++;

  size_t old_dram_inflight = dram_inflight;

  while (!dram_response_schedule.empty() &&
         dram_response_schedule.front().first <= tick_counter) {
    dram_inflight -= dram_response_schedule.front().second;
    dram_response_schedule.pop();
  }

  if (sram_processing_remaining > 0) {
    sram_processing_remaining--;
    if (sram_processing_remaining == 0 && !sram_queue.empty()) {
      sram_processing_remaining = sram_queue.front();
      sram_queue.pop();
    }
  }

  int go5_current = go5_busy_remaining;

  if (go5_busy_remaining > 0) {
    go5_busy_remaining--;
  }

  bool stalling = false;
  constexpr size_t EXIT_STAGE = COALESCING_PIPELINE_DEPTH - 1;

  if (pipeline_stages[EXIT_STAGE]) {
    bool is_sram = is_sram_access(pipeline_stages[EXIT_STAGE]->req);
    if (is_sram) {
      if (sram_queue.size() >= SRAM_QUEUE_CAPACITY) {
        stalling = true;
      }
    } else {
      if (go5_current > 0 || old_dram_inflight >= DRAM_MAX_INFLIGHT) {
        stalling = true;
      }
    }
  }

  if (!stalling && coalescing_remaining > 0) {
    coalescing_remaining--;
    if (coalescing_remaining == 0 && !pending_request_queue.empty()) {
      coalescing_waiting = true;
    }
  }

  bool old_occupied[COALESCING_PIPELINE_DEPTH];
  for (size_t s = 0; s < COALESCING_PIPELINE_DEPTH; s++) {
    old_occupied[s] = pipeline_stages[s].has_value();
  }

  int inflight_old = inflight_count_reg;
  bool space_for_two = (inflight_old <= 2);
  int inflight_incr = 0;
  int inflight_decr = 0;

  if (!stalling) {
    bool exit_happened = false;
    int exit_burst_len = 0;
    bool exit_is_store = false;

    if (pipeline_stages[EXIT_STAGE]) {
      auto &pipe_req = *pipeline_stages[EXIT_STAGE];
      bool is_sram = is_sram_access(pipe_req.req);

      if (is_sram) {
        int bank_cycles = calculate_sram_bank_conflicts(pipe_req.req);
        sram_queue.push(bank_cycles);
        if (sram_processing_remaining == 0) {
          sram_processing_remaining = sram_queue.front();
          sram_queue.pop();
        }
      } else {
        exit_is_store = pipe_req.req.is_store;
        if (!pipe_req.req.addrs.empty()) {
          std::vector<uint64_t> phys = build_translated_lane_addrs(
              pipe_req.req.warp, pipe_req.req.addrs, pipe_req.req.active_threads);
          int bursts = calculate_bursts(phys, pipe_req.req.bytes, pipe_req.req.is_store);
          int groups = calculate_request_count(phys, pipe_req.req.bytes);
          exit_burst_len = (groups > 0) ? (bursts + groups - 1) / groups : 1;
        } else {
          exit_burst_len = 1;
        }
      }

      process_mem_request(pipe_req.req);
      pipeline_stages[EXIT_STAGE] = std::nullopt;
      exit_happened = true;
      inflight_decr = 1;
    }

    for (int s = (int)EXIT_STAGE - 1; s >= 0; s--) {
      if (pipeline_stages[s] && !pipeline_stages[s + 1]) {
        pipeline_stages[s + 1] = std::move(pipeline_stages[s]);
        pipeline_stages[s] = std::nullopt;
      }
    }

    if (exit_happened && exit_burst_len > 0) {
      go5_busy_remaining = exit_is_store ? exit_burst_len : 1;
    }
  } else {
    constexpr size_t STALL_ADVANCE_LIMIT = COALESCING_PIPELINE_DEPTH - 2;
    for (int s = (int)STALL_ADVANCE_LIMIT; s >= 0; s--) {
      if (old_occupied[s] && !old_occupied[s + 1]) {
        pipeline_stages[s + 1] = std::move(pipeline_stages[s]);
        pipeline_stages[s] = std::nullopt;
      }
    }
  }

  bool can_consume = space_for_two
                   && (!stalling || !old_occupied[1])
                   && go5_busy_remaining == 0
                   && coalescing_remaining == 0;

  if (can_consume) {
    bool consumed = false;

    if (coalescing_waiting && !pipeline_stages[0]) {
      if (!pending_request_queue.empty()) {
        PipelineRequest pipe_req;
        pipe_req.req = pending_request_queue.front();
        pending_request_queue.pop();
        pipeline_stages[0] = pipe_req;
        coalescing_waiting = false;
        consumed = true;
        inflight_incr = 1;
      }
    }

    if (!consumed && !coalescing_waiting && !pending_request_queue.empty()
        && !pipeline_stages[0]) {
      MemRequest &front = pending_request_queue.front();
      int groups = 0;
      if (!front.is_fence && !front.addrs.empty()) {
        std::vector<uint64_t> phys = build_translated_lane_addrs(
            front.warp, front.addrs, front.active_threads);
        groups = calculate_request_count(phys, front.bytes);
      }
      if (groups <= 1) {
        PipelineRequest pipe_req;
        pipe_req.req = pending_request_queue.front();
        pending_request_queue.pop();
        pipeline_stages[0] = pipe_req;
        inflight_incr = 1;
      } else {
        coalescing_remaining = groups - 1;
      }
    }
  }

  inflight_count_reg = inflight_count_reg + inflight_incr - inflight_decr;

  if (dram_queue_depth > 0)
    dram_queue_depth--;

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

  if (!req.is_fence && !req.addrs.empty() && is_sram_access(req)) {
    if (!req.is_store) {
      auto it = blocked_warps.find(req.warp);
      if (it != blocked_warps.end()) {
        size_t queue_wait = sram_processing_remaining;
        std::queue<int> tmp = sram_queue;
        while (!tmp.empty()) {
          queue_wait += tmp.front();
          tmp.pop();
        }
        size_t sram_latency = queue_wait + BANKED_SRAM_LATENCY;
        if (sram_latency > it->second) {
          it->second = sram_latency;
        }
      }
    }
  }

  if (req.is_fence) {
    int beats = 1;
    int groups = 1;

    if (dram_trace && !req.warp->is_cpu) {
      uint64_t cycle = GPUStatisticsManager::instance().get_gpu_cycles();
      *dram_trace << cycle << "," << req.warp->warp_id << ","
                  << "F" << "," << beats << "," << groups << ","
                  << "DRAM" << "," << "0x0"
                  << "\n";
    }

    size_t first_resp_arrival = tick_counter + 2 + dram_queue_depth + SIM_DRAM_LATENCY;
    size_t total_resp_processing = static_cast<size_t>(beats + groups);
    size_t resp_start = std::max(first_resp_arrival, next_dram_resp_available);
    next_dram_resp_available = resp_start + total_resp_processing;
    size_t group_complete_tick = next_dram_resp_available;
    dram_response_schedule.push({group_complete_tick, groups});

    size_t resume_tick = next_dram_resp_available + 3;
    size_t dram_latency = resume_tick - tick_counter;

    auto it = blocked_warps.find(req.warp);
    if (it != blocked_warps.end()) {
      if (dram_latency > static_cast<size_t>(it->second)) {
        it->second = dram_latency;
      }
    }

    dram_queue_depth += beats;
    dram_inflight += groups;
  }

  if (!req.is_fence && !req.addrs.empty()) {
    std::vector<uint64_t> phys_addrs = build_translated_lane_addrs(
        req.warp, req.addrs, req.active_threads);
    int beats = calculate_bursts(phys_addrs, req.bytes, req.is_store);
    bool is_sram = is_sram_access(req);
    int groups = calculate_request_count(phys_addrs, req.bytes);

    if (dram_trace && !req.warp->is_cpu) {
      uint64_t cycle = GPUStatisticsManager::instance().get_gpu_cycles();
      char type = req.is_atomic ? 'A' : (req.is_store ? 'S' : 'L');
      uint64_t addr = 0;
      if (!phys_addrs.empty()) {
        for (auto a : phys_addrs) { if (a != SIM_SHARED_SRAM_BASE) { addr = a; break; } }
      }
      *dram_trace << cycle << "," << req.warp->warp_id << ","
                  << type << "," << beats << "," << groups << ","
                  << (is_sram ? "SRAM" : "DRAM") << ","
                  << "0x" << std::hex << (addr >> DRAM_BEAT_LOG_BYTES) << std::dec
                  << "\n";
    }

    if (beats > 0) {
      if (!req.is_store) {
        size_t first_resp_arrival = tick_counter + 2 + dram_queue_depth + SIM_DRAM_LATENCY;

        size_t total_resp_processing = static_cast<size_t>(beats + groups);

        size_t resp_start = std::max(first_resp_arrival, next_dram_resp_available);
        next_dram_resp_available = resp_start + total_resp_processing;

        size_t group_complete_tick = next_dram_resp_available;
        dram_response_schedule.push({group_complete_tick, groups});

        size_t resume_tick = next_dram_resp_available + 3;
        size_t dram_latency = resume_tick - tick_counter;

        auto it = blocked_warps.find(req.warp);
        if (it != blocked_warps.end()) {
          if (dram_latency > static_cast<size_t>(it->second)) {
            it->second = dram_latency;
          }
        }

        dram_inflight += groups;
      }

      dram_queue_depth += beats;
    }
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
  
  for (size_t s = 0; s < COALESCING_PIPELINE_DEPTH; s++) {
    if (pipeline_stages[s] && pipeline_stages[s]->req.warp == warp) return true;
  }
  
  return false;
}