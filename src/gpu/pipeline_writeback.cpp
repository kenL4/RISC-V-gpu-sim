#include "pipeline_writeback.hpp"
#include "../config.hpp"

WritebackResume::WritebackResume(CoalescingUnit *cu, RegisterFile *rf, bool is_cpu_pipeline)
    : cu(cu), rf(rf), is_cpu_pipeline(is_cpu_pipeline) {
  log("Writeback/Resume", "Initializing Writeback/Resume pipeline stage");
}

void WritebackResume::execute() {
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
    PipelineStage::output_latch->updated = true;
    PipelineStage::output_latch->warp = warp;
    PipelineStage::output_latch->active_threads = {};
    PipelineStage::output_latch->inst = llvm::MCInst();

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
};

bool WritebackResume::is_active() {
  // Check if there are warps for this pipeline in the coalescing unit
  // Since CoalescingUnit is shared, we need to check if it has warps for our pipeline
  bool mem_busy_for_this_pipeline = cu->is_busy_for_pipeline(is_cpu_pipeline);
  
  return PipelineStage::input_latch->updated || mem_busy_for_this_pipeline;
}