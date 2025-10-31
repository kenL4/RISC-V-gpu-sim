#include "pipeline_writeback.hpp"

WritebackResume::WritebackResume(RegisterFile *rf): rf(rf) {
    log("Writeback/Resume", "Initializing Writeback/Resume pipeline stage");
}

void WritebackResume::execute() {
    // TODO: Check memory responses to unsuspend threads
    if (!PipelineStage::input_latch->updated) return;
    
    Warp *warp = PipelineStage::input_latch->warp;
    cs_insn *inst = PipelineStage::input_latch->instruction;

    PipelineStage::input_latch->updated = false;
    PipelineStage::output_latch->updated = true;
    PipelineStage::output_latch->warp = warp;
    PipelineStage::output_latch->active_threads = PipelineStage::input_latch->active_threads;
    PipelineStage::output_latch->instruction = PipelineStage::input_latch->instruction;
    
    log("Writeback/Resume", "Warp " + std::to_string(warp->warp_id) + " values were written back");

    if (Config::instance().isRegisterDump()) rf->pretty_print(warp->warp_id);
};

bool WritebackResume::is_active() {
    return PipelineStage::input_latch->updated;
}