#include "utils.hpp"
#include "pipeline.hpp"
#include "mem_instr.hpp"

/*
 * The Instruction Fetch unit looks up the instruction
 * associated with the active threads.
 */
class InstructionFetch: public PipelineStage {
public:
    InstructionFetch(InstructionMemory *im): im(im) {
        log("Instruction Fetch", "Initializing instruction fetch pipeline stage");
    }
    void execute() override {
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
            " executing instruction " + fetched_instr->mnemonic + 
            "\t" + fetched_instr->op_str);
    };

    bool is_active() override {
        return PipelineStage::input_latch->updated;
    }

    ~InstructionFetch() {};
private:
    InstructionMemory *im;
};