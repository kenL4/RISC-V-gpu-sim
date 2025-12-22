#include "pipeline_writeback.hpp"

WritebackResume::WritebackResume(CoalescingUnit *cu, RegisterFile *rf)
    : cu(cu), rf(rf) {
  log("Writeback/Resume", "Initializing Writeback/Resume pipeline stage");
}

void WritebackResume::execute() {
  cu->tick();

  // Take the thread from execute unless not busy then
  // resume from suspended
  Warp *warp = PipelineStage::input_latch->warp;
  if (!PipelineStage::input_latch->updated) {
    warp = cu->get_resumable_warp();
    if (warp == nullptr)
      return;
    // Resume the warp
    warp->suspended = false;
  }

  // At this point, we know an instructions has successfully
  // executed OR it has been resumed
  GPUStatisticsManager::instance().increment_gpu_instrs();

  PipelineStage::input_latch->updated = false;
  PipelineStage::output_latch->updated = true;
  PipelineStage::output_latch->warp = warp;
  PipelineStage::output_latch->active_threads =
      PipelineStage::input_latch->active_threads;
  PipelineStage::output_latch->inst = PipelineStage::input_latch->inst;

  log("Writeback/Resume",
      "Warp " + std::to_string(warp->warp_id) + " values were written back");

  if (Config::instance().isRegisterDump())
    rf->pretty_print(warp->warp_id);
};

bool WritebackResume::is_active() {
  return PipelineStage::input_latch->updated || cu->is_busy();
}