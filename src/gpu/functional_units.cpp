#include "functional_units.hpp"
#include <algorithm>
#include <climits>

MulUnit::MulUnit() {}

bool MulUnit::issue(Warp *warp, std::vector<size_t> active_threads, 
                    const std::map<size_t, int> &rs1_vals,
                    const std::map<size_t, int> &rs2_vals,
                    unsigned int rd) {
  // Matching SIMTight: canPut returns false when result queue is full
  // In SIMTight, canPut = inv stall.val, and stall is set when stage 3 tries to enqueue
  // but the result queue is full. Since we check canPut BEFORE entering the pipeline,
  // we need to be conservative: reject if result queue is full.
  // Note: We could also account for operations in pipeline that will complete soon,
  // but being conservative (only checking current queue size) should still cause retries
  // when the queue is full, which is the main case we need to handle.
  if (result_queue.size() >= RESULT_QUEUE_CAPACITY) {
    return false;  // Result queue is full, need to retry
  }
  
  MulOperation op;
  op.warp = warp;
  op.active_threads = active_threads;
  op.rd = rd;
  op.cycles_remaining = SIM_MUL_LATENCY;
  
  // Compute results for all active threads (each thread has its own values)
  for (size_t thread : active_threads) {
    auto rs1_it = rs1_vals.find(thread);
    auto rs2_it = rs2_vals.find(thread);
    
    if (rs1_it != rs1_vals.end() && rs2_it != rs2_vals.end()) {
      int result = rs1_it->second * rs2_it->second;
      op.results[{warp, thread}] = result;
    }
  }
  
  // Suspend the warp
  warp->suspended = true;
  
  // Add to pipeline
  pipeline.push(op);
  
  return true;
}

bool MulUnit::is_busy() {
  // Busy if there's an operation in the pipeline OR result queue has operations
  return !pipeline.empty() || !result_queue.empty();
}

Warp *MulUnit::peek_completed_warp() {
  if (result_queue.empty()) {
    return nullptr;
  }
  
  return result_queue.front().warp;
}

Warp *MulUnit::get_completed_warp() {
  if (result_queue.empty()) {
    return nullptr;
  }
  
  Warp *warp = result_queue.front().warp;
  result_queue.pop();
  return warp;
}

int MulUnit::get_result(Warp *warp, size_t thread) {
  // The warp should be at the front of the queue (from peek_completed_warp())
  if (result_queue.empty() || result_queue.front().warp != warp) {
    return 0;  // Should not happen
  }
  
  auto result_it = result_queue.front().results.find({warp, thread});
  if (result_it != result_queue.front().results.end()) {
    return result_it->second;
  }
  return 0;  // Should not happen
}

unsigned int MulUnit::get_rd(Warp *warp) {
  // The warp should be at the front of the queue (from peek_completed_warp())
  if (result_queue.empty() || result_queue.front().warp != warp) {
    return 0;
  }
  return result_queue.front().rd;
}

std::vector<size_t> MulUnit::get_active_threads(Warp *warp) {
  // The warp should be at the front of the queue (from peek_completed_warp())
  if (result_queue.empty() || result_queue.front().warp != warp) {
    return {};
  }
  return result_queue.front().active_threads;
}

void MulUnit::tick() {
  // Advance all operations in pipeline
  size_t pipeline_size = pipeline.size();
  for (size_t i = 0; i < pipeline_size; i++) {
    MulOperation &op = pipeline.front();
    if (op.cycles_remaining > 0) {
      op.cycles_remaining--;
    }
    
    // If operation is complete, move to result queue (matching SIMTight behavior)
    // Only move if result queue has space (this should always be true here since
    // we check capacity in issue(), but check to be safe)
    if (op.cycles_remaining == 0) {
      if (result_queue.size() < RESULT_QUEUE_CAPACITY) {
        result_queue.push(op);
      }
      // If result queue is full, the operation stays in pipeline (this shouldn't happen
      // if issue() is working correctly, but handle it gracefully)
    } else {
      // Still processing, keep in pipeline
      pipeline.push(op);
    }
    pipeline.pop();
  }
}

DivUnit::DivUnit() : current_operation(nullptr) {}

bool DivUnit::issue(Warp *warp, std::vector<size_t> active_threads,
                    const std::map<size_t, int> &rs1_vals,
                    const std::map<size_t, int> &rs2_vals,
                    unsigned int rd, bool is_signed, bool get_remainder) {
  // Sequential divider - can only handle one operation at a time
  if (current_operation != nullptr) {
    return false;  // Unit is busy
  }
  
  DivOperation op;
  op.warp = warp;
  op.active_threads = active_threads;
  op.rd = rd;
  op.is_signed = is_signed;
  op.get_remainder = get_remainder;
  op.cycles_remaining = SIM_DIV_LATENCY;
  
  // Compute results for all active threads (each thread has its own values)
  for (size_t thread : active_threads) {
    auto rs1_it = rs1_vals.find(thread);
    auto rs2_it = rs2_vals.find(thread);
    
    if (rs1_it == rs1_vals.end() || rs2_it == rs2_vals.end()) {
      continue;
    }
    
    int rs1_val = rs1_it->second;
    int rs2_val = rs2_it->second;
    int result;
    
    if (is_signed) {
      // Signed division/remainder
      if (rs2_val == 0) {
        // Division by zero
        if (get_remainder) {
          result = rs1_val;  // REM: remainder is numerator
        } else {
          result = -1;  // DIV: quotient is all ones
        }
      } else if (rs1_val == INT32_MIN && rs2_val == -1) {
        // Overflow case: -2^31 / -1 = 2^31 (which overflows signed int)
        if (get_remainder) {
          result = 0;  // REM: remainder is 0
        } else {
          result = INT32_MIN;  // DIV: result is -2^31
        }
      } else {
        if (get_remainder) {
          result = rs1_val % rs2_val;  // REM
        } else {
          result = rs1_val / rs2_val;  // DIV
        }
      }
    } else {
      // Unsigned division/remainder
      uint32_t u_rs1 = static_cast<uint32_t>(rs1_val);
      uint32_t u_rs2 = static_cast<uint32_t>(rs2_val);
      
      if (u_rs2 == 0) {
        if (get_remainder) {
          result = static_cast<int>(u_rs1);  // REMU: remainder is numerator
        } else {
          result = 0xFFFFFFFF;  // DIVU: quotient is all ones
        }
      } else {
        if (get_remainder) {
          result = static_cast<int>(u_rs1 % u_rs2);  // REMU
        } else {
          result = static_cast<int>(u_rs1 / u_rs2);  // DIVU
        }
      }
    }
    
    op.results[{warp, thread}] = result;
  }
  
  // Suspend the warp
  warp->suspended = true;
  
  // Set as current operation (sequential unit - only one at a time)
  current_operation = new DivOperation(std::move(op));
  
  return true;
}

bool DivUnit::is_busy() {
  // Busy if there's a current operation OR completed operations waiting
  return current_operation != nullptr || !completed_operations.empty();
}

Warp *DivUnit::peek_completed_warp() {
  if (completed_operations.empty()) {
    return nullptr;
  }
  
  auto it = completed_operations.begin();
  return it->first;
}

Warp *DivUnit::get_completed_warp() {
  if (completed_operations.empty()) {
    return nullptr;
  }
  
  auto it = completed_operations.begin();
  Warp *warp = it->first;
  completed_operations.erase(it);
  return warp;
}

int DivUnit::get_result(Warp *warp, size_t thread) {
  auto it = completed_operations.find(warp);
  if (it == completed_operations.end()) {
    return 0;  // Should not happen
  }
  
  auto result_it = it->second.results.find({warp, thread});
  if (result_it == it->second.results.end()) {
    return 0;  // Should not happen
  }
  
  return result_it->second;
}

unsigned int DivUnit::get_rd(Warp *warp) {
  auto it = completed_operations.find(warp);
  if (it == completed_operations.end()) {
    return 0;  // Should not happen
  }
  return it->second.rd;
}

std::vector<size_t> DivUnit::get_active_threads(Warp *warp) {
  auto it = completed_operations.find(warp);
  if (it == completed_operations.end()) {
    return {};
  }
  return it->second.active_threads;
}

void DivUnit::tick() {
  if (current_operation != nullptr) {
    if (current_operation->cycles_remaining > 0) {
      current_operation->cycles_remaining--;
    }
    
    // If operation is complete, move to completed and free current
    if (current_operation->cycles_remaining == 0) {
      completed_operations[current_operation->warp] = *current_operation;
      delete current_operation;
      current_operation = nullptr;
    }
  }
}
