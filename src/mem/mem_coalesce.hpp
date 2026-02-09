#pragma once

#include "gpu/pipeline.hpp"
#include "mem_data.hpp"
#include "utils.hpp"
#include "trace/trace.hpp"
#include <queue>
#include <map>
#include <memory>

struct MemRequest {
  Warp *warp;
  std::vector<uint64_t> addrs;
  size_t bytes;
  bool is_store;
  bool is_atomic;
  bool is_fence;
  bool is_zero_extend;
  std::vector<int> store_values;  // Only used for stores
  std::vector<int> atomic_add_values;  // Only used for atomic add operations
  unsigned int rd_reg;
  std::vector<size_t> active_threads;
};

class CoalescingUnit {
public:
  CoalescingUnit(DataMemory *scratchpad_mem, const std::string *trace_file = nullptr);
  
  bool can_put();
  void load(Warp *warp, const std::vector<uint64_t> &addrs, size_t bytes,
            unsigned int rd_reg, const std::vector<size_t> &active_threads,
            bool is_zero_extend = false);
  void store(Warp *warp, const std::vector<uint64_t> &addrs, size_t bytes,
             const std::vector<int> &vals, const std::vector<size_t> &active_threads);
  void atomic_add(Warp *warp, const std::vector<uint64_t> &addrs, size_t bytes,
                  unsigned int rd_reg, const std::vector<int> &add_values,
                  const std::vector<size_t> &active_threads);
  void fence(Warp *warp);
  
  bool is_busy();
  bool is_busy_for_pipeline(bool is_cpu_pipeline);
  Warp *get_resumable_warp_for_pipeline(bool is_cpu_pipeline);
  void tick();
  
  void suspend_warp_latency(Warp *warp, size_t latency);
  std::pair<unsigned int, std::map<size_t, int>> get_load_results(Warp *warp);
  bool has_pending_memory_ops(Warp *warp);

private:
  std::map<Warp *, size_t> blocked_warps;
  DataMemory *scratchpad_mem;
  std::queue<MemRequest> pending_request_queue;
  
  struct PipelineRequest {
    MemRequest req;
    size_t cycles_in_pipeline;
  };
  std::queue<PipelineRequest> pipeline_queue;
  static constexpr size_t COALESCING_PIPELINE_DEPTH = 5;  // Matching SIMTight's 5-stage pipeline
  
  std::map<Warp *, std::pair<unsigned int, std::map<size_t, int>>> load_results_map;
  std::unique_ptr<Tracer> tracer;
  
  void suspend_warp(Warp *warp, const std::vector<uint64_t> &addrs,
                    size_t access_size, bool is_store);
  int calculate_bursts(const std::vector<uint64_t> &addrs, size_t access_size,
                       bool is_store);
  int calculate_request_count(const std::vector<uint64_t> &addrs, size_t access_size);
  
  // Compute coalesced addresses (one per coalesced group) for DRAM requests
  std::vector<uint64_t> compute_coalesced_addresses(const std::vector<uint64_t> &addrs,
                                                      size_t access_size);
  
  // Translate virtual stack address to physical per-thread stack address
  // Uses contiguous per-thread layout for actual data operations
  uint64_t translate_stack_address(uint64_t virtual_addr, Warp *warp, size_t thread_id);

  // SIMTight-matching interleaved address for coalescing/DRAM counting only
  uint64_t interleave_addr_simtight(uint64_t virtual_addr, Warp *warp, size_t thread_id);

  // Build a NUM_LANES-sized vector of translated physical addresses indexed by lane_id.
  // Inactive lanes get an SRAM sentinel address (filtered out by calculate_bursts).
  std::vector<uint64_t> build_translated_lane_addrs(
      Warp *warp, const std::vector<uint64_t> &addrs,
      const std::vector<size_t> &active_threads);

  void process_mem_request(const MemRequest &req);
};