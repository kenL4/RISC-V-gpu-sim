#pragma once

#include "config.hpp"
#include "gpu/pipeline.hpp"
#include <map>
#include <queue>

// Multiplier unit with 3-cycle pipeline latency
class MulUnit {
public:
  MulUnit();
  
  // Issue a multiplication operation for a warp
  // rs1_vals and rs2_vals should be indexed by thread ID
  // Returns true if the request was accepted
  bool issue(Warp *warp, std::vector<size_t> active_threads, 
             const std::map<size_t, int> &rs1_vals, 
             const std::map<size_t, int> &rs2_vals, 
             unsigned int rd);
  
  // Check if unit is busy (has pending operations)
  bool is_busy();
  
  // Get a warp that has completed its multiplication (removes from completed_operations)
  Warp *get_completed_warp();

  // Peek at a completed warp without removing it
  Warp *peek_completed_warp();

  // Advance time by one cycle
  void tick();

  // Get the result value for a completed operation
  int get_result(Warp *warp, size_t thread);

  // Get the destination register for a completed operation
  unsigned int get_rd(Warp *warp);

  // Get active threads for a completed operation
  std::vector<size_t> get_active_threads(Warp *warp);

private:
  // Structure to track an in-flight multiplication operation
  struct MulOperation {
    Warp *warp;
    std::vector<size_t> active_threads;
    std::map<std::pair<Warp *, size_t>, int> results;  // (warp, thread) -> result
    unsigned int rd;
    size_t cycles_remaining;
  };
  
  std::queue<MulOperation> pipeline;
  std::map<Warp *, MulOperation> completed_operations;
};

// Divider/Remainder unit with 32-cycle latency (sequential)
class DivUnit {
public:
  DivUnit();
  
  // Issue a division/remainder operation for a warp
  // rs1_vals and rs2_vals should be indexed by thread ID
  // is_signed: true for signed operations (DIV, REM), false for unsigned (DIVU, REMU)
  // get_remainder: true for REM/REMU, false for DIV/DIVU
  // Returns true if the request was accepted
  bool issue(Warp *warp, std::vector<size_t> active_threads, 
             const std::map<size_t, int> &rs1_vals,
             const std::map<size_t, int> &rs2_vals,
             unsigned int rd, bool is_signed, bool get_remainder);
  
  // Check if unit is busy (has pending operations)
  bool is_busy();
  
  // Get a warp that has completed its division/remainder (removes from completed_operations)
  Warp *get_completed_warp();

  // Peek at a completed warp without removing it
  Warp *peek_completed_warp();

  // Advance time by one cycle
  void tick();

  // Get the result value for a completed operation
  int get_result(Warp *warp, size_t thread);

  // Get the destination register for a completed operation
  unsigned int get_rd(Warp *warp);

  // Get active threads for a completed operation
  std::vector<size_t> get_active_threads(Warp *warp);

private:
  // Structure to track an in-flight division/remainder operation
  struct DivOperation {
    Warp *warp;
    std::vector<size_t> active_threads;
    std::map<std::pair<Warp *, size_t>, int> results;  // (warp, thread) -> result
    unsigned int rd;
    bool is_signed;
    bool get_remainder;
    size_t cycles_remaining;
  };
  
  // Sequential divider - only one operation at a time
  DivOperation *current_operation;
  std::map<Warp *, DivOperation> completed_operations;
};
