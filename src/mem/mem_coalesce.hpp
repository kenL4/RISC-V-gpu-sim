#pragma once

#include "gpu/pipeline.hpp"
#include "mem_data.hpp"
#include "utils.hpp"
#include "trace/trace.hpp"
#include <queue>
#include <map>
#include <memory>
#include <optional>
#include <unordered_set>

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
  ~CoalescingUnit();
  
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
  void reset_dram_state();
  size_t pending_size() const { return pending_request_queue.size(); }
  size_t pipeline_size() const {
    size_t count = 0;
    for (size_t s = 0; s < COALESCING_PIPELINE_DEPTH; s++)
      if (pipeline_stages[s]) count++;
    return count;
  }
  size_t blocked_size() const { return blocked_warps.size(); }
  size_t get_coalescing_remaining() const { return coalescing_remaining; }
  bool get_coalescing_waiting() const { return coalescing_waiting; }
  Warp *get_resumable_warp_for_pipeline(bool is_cpu_pipeline);
  void tick();
  
  void suspend_warp_latency(Warp *warp, size_t latency);
  void suspend_for_func_unit(Warp *warp, size_t latency, unsigned int rd_reg,
                             const std::map<size_t, int> &results);
  std::pair<unsigned int, std::map<size_t, int>> get_load_results(Warp *warp);
  bool has_pending_memory_ops(Warp *warp);

  bool can_use_divider() const { return divider_warp == nullptr; }
  void acquire_divider(Warp *warp) { divider_warp = warp; }

  static constexpr size_t MUL_PIPELINE_CAPACITY = 4;
  bool can_use_multiplier() const { return mul_pipeline_warps.size() < MUL_PIPELINE_CAPACITY; }
  void acquire_multiplier(Warp *warp) { mul_pipeline_warps.insert(warp); }

  void set_instr_tracer(Tracer *t) { instr_tracer = t; }
  void set_dram_trace(std::ofstream *f) { dram_trace = f; }

private:
  std::map<Warp *, size_t> blocked_warps;
  Warp *divider_warp = nullptr;
  std::unordered_set<Warp *> mul_pipeline_warps;
  DataMemory *scratchpad_mem;
  std::queue<MemRequest> pending_request_queue;
  
  struct PipelineRequest {
    MemRequest req;
  };
  static constexpr size_t COALESCING_PIPELINE_DEPTH = 5;
  std::optional<PipelineRequest> pipeline_stages[COALESCING_PIPELINE_DEPTH] = {};
  
  std::map<Warp *, std::pair<unsigned int, std::map<size_t, int>>> load_results_map;
  std::unique_ptr<Tracer> tracer;
  Tracer *instr_tracer = nullptr;
  std::ofstream *dram_trace = nullptr;

  size_t coalescing_remaining = 0;

  bool coalescing_waiting = false;

  int go5_busy_remaining = 0;

  int inflight_count_reg = 0;

  static constexpr size_t SRAM_QUEUE_CAPACITY = 2;
  static constexpr size_t SRAM_BANKS = 16;
  static constexpr size_t BANKED_SRAM_LATENCY = 10;
  std::queue<int> sram_queue;
  int sram_processing_remaining = 0;

public:
  size_t dram_queue_depth = 0;

  static constexpr size_t DRAM_MAX_INFLIGHT = 32;
  size_t dram_inflight = 0;
  size_t tick_counter = 0;
  std::queue<std::pair<size_t, size_t>> dram_response_schedule;

  size_t next_dram_resp_available = 0;

  bool is_sram_access(const MemRequest &req) const;

  int calculate_sram_bank_conflicts(const MemRequest &req) const;

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