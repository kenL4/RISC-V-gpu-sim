#include "pipeline_instr_fetch.hpp"

InstructionFetch::InstructionFetch(InstructionMemory *im): im(im) {
    log("Instruction Fetch", "Initializing instruction fetch pipeline stage");
}

void InstructionFetch::execute() {
    if (!PipelineStage::input_latch->updated) return;
    
    Warp *warp = PipelineStage::input_latch->warp;
    uint64_t thread_id = PipelineStage::input_latch->active_threads[0];
    uint64_t warp_pc = warp->pc[thread_id];
    cs_insn *fetched_instr = im->get_instruction(warp_pc);

    PipelineStage::input_latch->updated = false;
    PipelineStage::output_latch->updated = true;
    PipelineStage::output_latch->warp = warp;
    PipelineStage::output_latch->active_threads = PipelineStage::input_latch->active_threads;
    PipelineStage::output_latch->instruction = fetched_instr;

    log("Instruction Fetch", "Warp " + std::to_string(warp->warp_id) + 
        " will execute instruction " + fetched_instr->mnemonic);
};

bool InstructionFetch::is_active() {
    return PipelineStage::input_latch->updated;
}