#include "pipeline_writeback.hpp"
#include "../config.hpp"
#include "../stats/stats.hpp"

WritebackResume::WritebackResume(CoalescingUnit *cu, RegisterFile *rf, bool is_cpu_pipeline)
    : cu(cu), rf(rf), is_cpu_pipeline(is_cpu_pipeline) {
  log("Writeback/Resume", "Initializing Writeback/Resume pipeline stage");
}

void WritebackResume::execute() {

  bool handle_execute = false;
  if (PipelineStage::input_latch->updated) {
    Warp *warp = PipelineStage::input_latch->warp;
    handle_execute = PipelineStage::input_latch->has_result;
    PipelineStage::input_latch->updated = false;
    PipelineStage::output_latch->updated = true;
    PipelineStage::output_latch->warp = warp;
    PipelineStage::output_latch->active_threads =
        PipelineStage::input_latch->active_threads;
    PipelineStage::output_latch->inst = PipelineStage::input_latch->inst;

    if (!warp->suspended) {
      for (size_t i = 0; i < warp->size; i++) {
        if (!warp->finished[i]) {
          if (handle_execute && insert_warp_with_susp_delay) {
            insert_warp_with_susp_delay(warp);
          } else if (insert_warp) {
            insert_warp(warp);
          }
          break;
        }
      }
    }

    std::string name = warp->is_cpu ? "CPU" : "Warp " + std::to_string(warp->warp_id);
    log("Writeback/Resume",
        name + " values were written back");

    if (Config::instance().isRegisterDump())
      rf->pretty_print(warp->warp_id);

    if (handle_execute) {
      return;
    }
  }

  Warp *warp = cu->get_resumable_warp_for_pipeline(is_cpu_pipeline);

  if (warp != nullptr) {
    warp->suspended = false;

    if (instr_tracer && !warp->is_cpu) {
      TraceEvent event;
      event.cycle = GPUStatisticsManager::instance().get_gpu_cycles();
      event.pc = warp->pc[0];
      event.warp_id = warp->warp_id;
      event.lane_id = -1;
      event.event_type = WARP_RESUME;
      instr_tracer->trace_event(event);
    }

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
