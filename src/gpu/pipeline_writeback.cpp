#include "pipeline_writeback.hpp"
#include "pipeline_execute.hpp"
#include "functional_units.hpp"

WritebackResume::WritebackResume(CoalescingUnit *cu, RegisterFile *rf, bool is_cpu_pipeline)
    : cu(cu), rf(rf), is_cpu_pipeline(is_cpu_pipeline) {
  log("Writeback/Resume", "Initializing Writeback/Resume pipeline stage");
}

void WritebackResume::set_execution_unit(ExecutionUnit *eu) {
  execution_unit = eu;
}

void WritebackResume::execute() {
  // Check for completed functional unit operations first
  bool handled_func_unit = false;
  if (execution_unit) {
    // Check multiplier unit - peek first to get data before removing
    Warp *mul_warp = execution_unit->get_mul_unit().peek_completed_warp();
    if (mul_warp != nullptr) {
      // Only handle warps that belong to this pipeline
      if (mul_warp->is_cpu != is_cpu_pipeline) {
        // Wrong pipeline - don't handle it
        // Skip removal in this case
      } else {
        // Get all data before removing the warp
        unsigned int rd = execution_unit->get_mul_unit().get_rd(mul_warp);
        std::vector<size_t> active_threads = execution_unit->get_mul_unit().get_active_threads(mul_warp);
        
        // Write back results before removing from completed_operations
        for (size_t thread : active_threads) {
          int result = execution_unit->get_mul_unit().get_result(mul_warp, thread);
          rf->set_register(mul_warp->warp_id, thread, rd, result, mul_warp->is_cpu);
        }
        
        // Now remove from completed_operations
        execution_unit->get_mul_unit().get_completed_warp();
        
        mul_warp->suspended = false;
        
        // Reinsert warp to scheduler
        if (insert_warp) {
          insert_warp(mul_warp);
        }
        handled_func_unit = true;
      }
    }
    
    // Check divider unit - peek first to get data before removing
    Warp *div_warp = execution_unit->get_div_unit().peek_completed_warp();
    if (div_warp != nullptr) {
      // Only handle warps that belong to this pipeline
      if (div_warp->is_cpu != is_cpu_pipeline) {
        // Wrong pipeline - don't handle it (but don't remove it either)
        // It will be handled by the other pipeline
      } else {
        // Get all data before removing the warp
        unsigned int rd = execution_unit->get_div_unit().get_rd(div_warp);
        std::vector<size_t> active_threads = execution_unit->get_div_unit().get_active_threads(div_warp);
        
        // Write back results before removing from completed_operations
        for (size_t thread : active_threads) {
          int result = execution_unit->get_div_unit().get_result(div_warp, thread);
          rf->set_register(div_warp->warp_id, thread, rd, result, div_warp->is_cpu);
        }
        
        // Now remove from completed_operations
        execution_unit->get_div_unit().get_completed_warp();
        
        div_warp->suspended = false;
        std::string name = div_warp->is_cpu ? "CPU" : "Warp " + std::to_string(div_warp->warp_id);
        log("Writeback/Resume",
            name + " completed DIV/REM operation");
        
        // Reinsert warp to scheduler
        if (insert_warp) {
          insert_warp(div_warp);
        }
        handled_func_unit = true;
      }
    }
  }
  
  // Handle normal pipeline writeback (non-suspended instructions)
  if (PipelineStage::input_latch->updated) {
    Warp *warp = PipelineStage::input_latch->warp;
    PipelineStage::input_latch->updated = false;
    PipelineStage::output_latch->updated = true;
    PipelineStage::output_latch->warp = warp;
    PipelineStage::output_latch->active_threads =
        PipelineStage::input_latch->active_threads;
    PipelineStage::output_latch->inst = PipelineStage::input_latch->inst;

    std::string name = warp->is_cpu ? "CPU" : "Warp " + std::to_string(warp->warp_id);
    log("Writeback/Resume",
        name + " values were written back");

    if (Config::instance().isRegisterDump())
      rf->pretty_print(warp->warp_id);
    return;
  }
  
  // Handle memory operation resumption
  // Use pipeline-specific method to only get warps for this pipeline
  Warp *warp = cu->get_resumable_warp_for_pipeline(is_cpu_pipeline);
  
  if (warp != nullptr) {
    warp->suspended = false;
    
    // Check if there are load results to write back
    auto load_results = cu->get_load_results(warp);
    if (!load_results.second.empty()) {
      // Write load results to registers
      unsigned int rd_reg = load_results.first;
      for (const auto &[thread_id, value] : load_results.second) {
        rf->set_register(warp->warp_id, thread_id, rd_reg, value, warp->is_cpu);
      }
    }
    
    PipelineStage::input_latch->updated = false;
    PipelineStage::output_latch->updated = true;  // Signal that warp was resumed
    PipelineStage::output_latch->warp = warp;
    PipelineStage::output_latch->active_threads = {};  // Will be set by next stage
    PipelineStage::output_latch->inst = llvm::MCInst();  // No instruction for resume

    std::string name = warp->is_cpu ? "CPU" : "Warp " + std::to_string(warp->warp_id);
    log("Writeback/Resume",
        name + " resumed from memory operation");
    
    // Reinsert warp to scheduler so it can continue
    if (insert_warp) {
      insert_warp(warp);
    }

    if (Config::instance().isRegisterDump())
      rf->pretty_print(warp->warp_id);
    return;
  }
  
  // If we handled a functional unit completion but nothing else, that's fine
  // The warp was already reinserted, so we don't need to do anything else
  if (handled_func_unit) {
    return;
  }
};

bool WritebackResume::is_active() {
  bool func_units_busy = false;
  if (execution_unit) {
    func_units_busy = execution_unit->get_mul_unit().is_busy() || 
                      execution_unit->get_div_unit().is_busy();
  }
  
  // Check if there are warps for this pipeline in the coalescing unit
  // Since CoalescingUnit is shared, we need to check if it has warps for our pipeline
  bool mem_busy_for_this_pipeline = cu->is_busy_for_pipeline(is_cpu_pipeline);
  
  return PipelineStage::input_latch->updated || mem_busy_for_this_pipeline || func_units_busy;
}