#pragma once

#include "gpu/pipeline.hpp"
#include "mem_data.hpp"
#include "utils.hpp"
#include <queue>
#include <map>

// Memory request structure for queueing (matching SIMTight's queue model)
struct MemRequest {
  Warp *warp;
  std::vector<uint64_t> addrs;
  size_t bytes;
  bool is_store;
  std::vector<int> store_values;  // Only used for stores
  unsigned int rd_reg;  // Destination register for loads
  std::vector<size_t> active_threads;  // Active threads for this request
};

class CoalescingUnit {
public:
  CoalescingUnit(DataMemory *scratchpad_mem);
  
  // Check if memory request queue can accept new requests (matching SIMTight canPut)
  // Returns false when queue is full, causing retries
  bool can_put();
  
  // Queue a load request (returns immediately, results stored and written on resume)
  void load(Warp *warp, const std::vector<uint64_t> &addrs, size_t bytes,
            unsigned int rd_reg, const std::vector<size_t> &active_threads);
  
  // Queue a store request (returns immediately)
  void store(Warp *warp, const std::vector<uint64_t> &addrs, size_t bytes,
             const std::vector<int> &vals);
  
  bool is_busy();
  bool is_busy_for_pipeline(bool is_cpu_pipeline);
  Warp *get_resumable_warp();
  Warp *get_resumable_warp_for_pipeline(bool is_cpu_pipeline);
  void tick();
  
  // Suspend warp for a given number of cycles (for functional unit latencies)
  void suspend_warp_latency(Warp *warp, size_t latency);
  
  // Get stored load results for a warp (called when warp resumes)
  // Returns pair of (rd_reg, thread_id -> value map), or empty pair if no results
  std::pair<unsigned int, std::map<size_t, int>> get_load_results(Warp *warp);

private:
  std::map<Warp *, size_t> blocked_warps;
  DataMemory *scratchpad_mem;
  
  // Queue of pending memory requests (matching SIMTight's memReqsQueue)
  // Requests are queued before processing, and processed one per cycle in tick()
  std::queue<MemRequest> pending_request_queue;
  
  // Store load results by warp (thread_id -> value)
  // Results are written to registers when warp resumes
  std::map<Warp *, std::pair<unsigned int, std::map<size_t, int>>> load_results_map;  // rd_reg -> (thread_id -> value)

  void suspend_warp(Warp *warp, const std::vector<uint64_t> &addrs,
                    size_t access_size, bool is_store);
  int calculate_bursts(const std::vector<uint64_t> &addrs, size_t access_size,
                       bool is_store);
  int calculate_request_count(const std::vector<uint64_t> &addrs, size_t access_size);
  
  // Process a queued memory request (called from tick())
  void process_mem_request(const MemRequest &req);
};