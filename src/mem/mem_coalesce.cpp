#include "mem_coalesce.hpp"
#include "config.hpp"
#include <algorithm>
#include <iostream>
#include <set>

// Use centralized config values from config.hpp
// SIM_DRAM_LATENCY = 30 (matching SIMTight)
// DRAM_BEAT_BYTES = 64 (DRAM beat size)

CoalescingUnit::CoalescingUnit(DataMemory *scratchpad_mem)
    : scratchpad_mem(scratchpad_mem) {
}

bool CoalescingUnit::can_put() {
  // Matching SIMTight: memReqs.canPut checks only the INPUT queue capacity (32)
  // The pipeline capacity (inflightCount = 4) is checked separately when consuming from input queue
  // In SIMTight, canPut only checks memReqsQueue.notFull (input queue), not the pipeline capacity
  // This matches the behavior where retries occur when the input queue is full
  return pending_request_queue.size() < MEM_REQ_QUEUE_CAPACITY;
}

int CoalescingUnit::calculate_bursts(const std::vector<uint64_t> &addrs,
                                     size_t access_size, bool is_store) {
  // SIMTight coalescing strategy implementation with iterative processing
  // SIMTight has 32 lanes and 64-byte (512-bit) DRAM beats
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
    if (SIM_SHARED_SRAM_BASE <= addr_32 && addr_32 < SIM_SIMT_STACK_BASE) {
      continue;
    }
    pending.push_back({lane, addr});
  }

  if (pending.empty()) {
    return 0;
  }

  int total_dram_accesses = 0;

  // Iterative coalescing: process lanes until all are served
  while (!pending.empty()) {
    // Pick the first pending lane as the leader (matching SIMTight)
    size_t leader_lane = pending[0].first;
    uint64_t leader_addr = pending[0].second;

    // Compute coalescing masks based on leader
    std::vector<size_t> same_block_lanes;
    std::vector<size_t> same_addr_lanes;

    // SameBlock and SameAddress are computed relative to leader
    uint64_t leader_block = leader_addr >> (LOG_LANES + 2);
    // For SameAddress: lower LOG_LANES+2 bits must match
    uint64_t leader_low_bits = leader_addr & ((1ULL << (LOG_LANES + 2)) - 1);

    for (const auto &[lane, addr] : pending) {
      // Check if in same block (required for BOTH SameAddress and SameBlock)
      uint64_t block = addr >> (LOG_LANES + 2);
      bool in_same_block = (block == leader_block);

      // Check SameAddress: sameBlock AND lower LOG_LANES+2 bits match
      // In SIMTight: sameAddr = sameBlock && (a1[6:0] == a2[6:0])
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

    // Choose strategy: use SameBlock if it satisfies leader AND at least one
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
  // Count the number of coalesced requests (not bursts)
  // This is used for store access counting in SIMTight (stores count 1 per request)
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
    if (SIM_SHARED_SRAM_BASE <= addr_32 && addr_32 < SIM_SIMT_STACK_BASE) {
      continue;
    }
    pending.push_back({lane, addr});
  }

  if (pending.empty()) {
    return 0;
  }

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

void CoalescingUnit::suspend_warp(Warp *warp,
                                  const std::vector<uint64_t> &addrs,
                                  size_t access_size, bool is_store) {
  warp->suspended = true;

  // Calculate number of coalesced DRAM bursts (unique cache-line-sized blocks)
  // This is the number of DRAM transactions that would be issued
  int dram_bursts = calculate_bursts(addrs, access_size, is_store);
  
  // Count DRAM accesses matching SIMTight behavior:
  // - For loads: count the burst length (number of beats in the burst)
  // - For stores: count 1 per store request (regardless of burst length)
  int dram_access_count;
  if (is_store) {
    dram_access_count = calculate_request_count(addrs, access_size);
  } else {
    dram_access_count = dram_bursts;  // For loads, count burst length
  }
  
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
  // This models pipelined DRAM access where first access has full latency, 
  // subsequent ones are pipelined with minimal additional latency
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
                          const std::vector<size_t> &active_threads) {
  // Note: Caller should check can_put() before calling this function. If queue is full, caller should retry.
  
  // Queue the request (matching SIMTight: requests go into queue before processing)
  MemRequest req;
  req.warp = warp;
  req.addrs = addrs;
  req.bytes = bytes;
  req.is_store = false;
  req.is_atomic = false;
  req.is_fence = false;
  req.rd_reg = rd_reg;
  req.active_threads = active_threads;
  pending_request_queue.push(req);
  
  // Suspend warp immediately (request will be processed in tick(), results stored and written when warp resumes)
  warp->suspended = true;
  
  // Calculate DRAM bursts and count DRAM accesses (matching suspend_warp logic)
  int dram_bursts = calculate_bursts(addrs, bytes, false);
  int dram_access_count = dram_bursts;  // For loads, count burst length
  
  // Count DRAM accesses
  for (int i = 0; i < dram_access_count; i++) {
    if (warp->is_cpu) {
      GPUStatisticsManager::instance().increment_cpu_dram_accs();
    } else {
      GPUStatisticsManager::instance().increment_gpu_dram_accs();
    }
  }
  
  // Add warp to blocked_warps immediately with full latency (pipeline + DRAM: COALESCING_PIPELINE_DEPTH cycles, then DRAM latency)
  int latency;
  if (dram_bursts == 0) {
    latency = COALESCING_PIPELINE_DEPTH + 1;  // Pipeline + minimal latency
  } else if (dram_bursts == 1) {
    latency = COALESCING_PIPELINE_DEPTH + SIM_DRAM_LATENCY;
  } else {
    latency = COALESCING_PIPELINE_DEPTH + SIM_DRAM_LATENCY + (dram_bursts - 1);
  }
  blocked_warps[warp] = latency;
}

void CoalescingUnit::store(Warp *warp, const std::vector<uint64_t> &addrs,
                           size_t bytes, const std::vector<int> &vals) {
  // Note: Caller should check can_put() before calling this function. If queue is full, caller should retry.
  
  // Queue the request (matching SIMTight: requests go into queue before processing)
  MemRequest req;
  req.warp = warp;
  req.addrs = addrs;
  req.bytes = bytes;
  req.is_store = true;
  req.is_atomic = false;
  req.is_fence = false;
  req.store_values = vals;
  pending_request_queue.push(req);
  
  // Suspend warp immediately (request will be processed in tick())
  warp->suspended = true;
  
  // Calculate DRAM bursts and count DRAM accesses (matching suspend_warp logic)
  int dram_bursts = calculate_bursts(addrs, bytes, true);
  int dram_access_count = calculate_request_count(addrs, bytes);  // For stores, count request count
  
  // Count DRAM accesses
  for (int i = 0; i < dram_access_count; i++) {
    if (warp->is_cpu) {
      GPUStatisticsManager::instance().increment_cpu_dram_accs();
    } else {
      GPUStatisticsManager::instance().increment_gpu_dram_accs();
    }
  }
  
  // Add warp to blocked_warps immediately with full latency (pipeline + DRAM: COALESCING_PIPELINE_DEPTH cycles, then DRAM latency)
  int latency;
  if (dram_bursts == 0) {
    latency = COALESCING_PIPELINE_DEPTH + 1;  // Pipeline + minimal latency
  } else {
    // For stores, latency is pipeline + DRAM latency
    latency = COALESCING_PIPELINE_DEPTH + SIM_DRAM_LATENCY;
  }
  blocked_warps[warp] = latency;
}

void CoalescingUnit::fence(Warp *warp) {
  // Note: Caller should check can_put() before calling this function. If queue is full, caller should retry.
  // Matching SIMTight: FENCE sends memGlobalFenceOp to memory unit and suspends warp
  
  // Queue the fence request (matching SIMTight: requests go into queue before processing)
  MemRequest req;
  req.warp = warp;
  req.addrs = {};  // Fence doesn't need addresses
  req.bytes = 0;
  req.is_store = false;
  req.is_atomic = false;
  req.is_fence = true;
  req.active_threads = {};
  pending_request_queue.push(req);
  
  // Suspend warp immediately (fence will be processed in tick(), warp resumes when complete)
  warp->suspended = true;
  
  // Fence latency: need to account for time request spends in pending queue + pipeline
  // The latency counter starts decrementing immediately when set, so we need to be conservative
  // Worst case: request waits in pending queue (could be many cycles if pipeline is full),
  // then goes through pipeline (COALESCING_PIPELINE_DEPTH cycles)
  // To be safe, use a latency that accounts for worst-case queue delay
  // Using MEM_REQ_QUEUE_CAPACITY as worst-case queue wait time seems excessive,
  // but we need to ensure the warp doesn't resume before the fence actually completes
  // For now, use 2 * COALESCING_PIPELINE_DEPTH to account for queue delay + pipeline
  blocked_warps[warp] = 2 * COALESCING_PIPELINE_DEPTH;
}

void CoalescingUnit::atomic_add(Warp *warp, const std::vector<uint64_t> &addrs,
                                 size_t bytes, unsigned int rd_reg,
                                 const std::vector<int> &add_values,
                                 const std::vector<size_t> &active_threads) {
  // Note: Caller should check can_put() before calling this function
  // If queue is full, caller should retry (this function should not be called)
  
  // Queue the atomic add request (matching SIMTight: requests go into queue before processing)
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
  
  // Suspend warp immediately (request will be processed in tick())
  // Results will be stored and written when warp resumes
  warp->suspended = true;
  
  // Calculate DRAM bursts and count DRAM accesses (matching suspend_warp logic)
  // Atomic operations have same latency as stores
  int dram_bursts = calculate_bursts(addrs, bytes, true);
  int dram_access_count = calculate_request_count(addrs, bytes);  // For stores/atomics, count request count
  
  // Count DRAM accesses
  for (int i = 0; i < dram_access_count; i++) {
    if (warp->is_cpu) {
      GPUStatisticsManager::instance().increment_cpu_dram_accs();
    } else {
      GPUStatisticsManager::instance().increment_gpu_dram_accs();
    }
  }
  
  // Add warp to blocked_warps immediately with full latency (pipeline + DRAM: COALESCING_PIPELINE_DEPTH cycles, then DRAM latency)
  int latency;
  if (dram_bursts == 0) {
    latency = COALESCING_PIPELINE_DEPTH + 1;  // Pipeline + minimal latency
  } else {
    latency = COALESCING_PIPELINE_DEPTH + SIM_DRAM_LATENCY;
  }
  blocked_warps[warp] = latency;
}

bool CoalescingUnit::is_busy() { 
  // Unit is busy if there are pending requests in queue OR blocked warps
  // This matches the test expectation: after load() is called, the unit should be busy
  // even though the request hasn't been processed yet (it's in the queue)
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

Warp *CoalescingUnit::get_resumable_warp_for_pipeline(bool is_cpu_pipeline) {
  Warp *resumable_warp = nullptr;

  for (auto &[key, val] : blocked_warps) {
    if (val == 0 && key->is_cpu == is_cpu_pipeline) {
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
  // Note: Order matters - we advance existing pipeline requests BEFORE adding new ones
  // so new requests don't get advanced in the same cycle they're added
  
  // Step 1 & 2: Advance pipeline and process completed requests
  // Track if we processed any requests this cycle (pipeline busy condition)
  // In SIMTight, when stage 5 is busy (go5.val), the pipeline stalls and doesn't consume from input queue
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
  
  // Step 3: Move requests from pending to pipeline (matching SIMTight: consume when (NOT stalled OR NOT go1) AND NOT in feedback)
  // Since we don't model DRAM queue stalls, we can always consume when there's space
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
  if (req.is_fence) {
    // Process memory fence: ensure all previous memory operations complete
    // Matching SIMTight: memGlobalFenceOp ensures memory ordering
    // In our simple model, fence just needs to go through pipeline
    // No actual memory access needed - just ensures ordering
    // Warp was already added to blocked_warps when request was queued
    // The latency counter will reach 0 and warp will be resumed via get_resumable_warp()
    // No additional processing needed - fence is complete when it exits pipeline
  } else if (req.is_atomic) {
    // Process atomic add: read old value, add to it, write back, return old value
    std::map<size_t, int> results;
    for (size_t i = 0; i < req.addrs.size(); i++) {
      uint64_t addr = req.addrs[i];
      // Read old value
      int64_t old_value = scratchpad_mem->load(addr, req.bytes);
      // Add the increment value
      int64_t new_value = old_value + req.atomic_add_values[i];
      // Write new value back
      scratchpad_mem->store(addr, req.bytes, static_cast<uint64_t>(new_value));
      // Return old value (will be written to rd_reg when warp resumes)
      results[req.active_threads[i]] = static_cast<int>(old_value);
    }
    
    // Store results with rd_reg (will be written to registers when warp resumes)
    load_results_map[req.warp] = std::make_pair(req.rd_reg, results);
    
    // Warp was already added to blocked_warps when request was queued
    // Just need to count DRAM accesses (suspend_warp does this, but we already did it)
    // No need to call suspend_warp again - latency was already set correctly
  } else if (req.is_store) {
    // Process store: write to memory
    for (size_t i = 0; i < req.addrs.size(); i++) {
      uint64_t addr = req.addrs[i];
      uint64_t val = static_cast<uint64_t>(req.store_values[i]);
      scratchpad_mem->store(addr, req.bytes, val);
    }
    
    // Warp was already added to blocked_warps when request was queued
    // Just need to count DRAM accesses (suspend_warp does this, but we already did it)
    // No need to call suspend_warp again - latency was already set correctly
  } else {
    // Process load: read from memory and store results
    std::map<size_t, int> results;
    for (size_t i = 0; i < req.addrs.size(); i++) {
      int64_t value = scratchpad_mem->load(req.addrs[i], req.bytes);
      results[req.active_threads[i]] = static_cast<int>(value);
    }
    
    // Store results with rd_reg (will be written to registers when warp resumes)
    load_results_map[req.warp] = std::make_pair(req.rd_reg, results);
    
    // Warp was already added to blocked_warps when request was queued
    // Just need to count DRAM accesses (suspend_warp does this, but we already did it)
    // No need to call suspend_warp again - latency was already set correctly
  }
}

std::pair<unsigned int, std::map<size_t, int>> CoalescingUnit::get_load_results(Warp *warp) {
  auto it = load_results_map.find(warp);
  if (it != load_results_map.end()) {
    std::pair<unsigned int, std::map<size_t, int>> results = it->second;
    load_results_map.erase(it);  // Remove after retrieving
    return results;
  }
  return std::make_pair(0, std::map<size_t, int>());  // Empty if no results
}