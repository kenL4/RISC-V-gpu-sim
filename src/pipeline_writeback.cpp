#include "utils.hpp"
#include "pipeline.hpp"
#include "register_file.hpp"

/*
 * The Writeback/Resume unit writes back the per-lane
 * results to the register file for each active thread.
 * It also handles the clearing of the suspension bit,
 * if there are no writes to be done.
 */
class WritebackResume: public PipelineStage {
public:
    WritebackResume (RegisterFile *rf): rf(rf) {
        log("Writeback/Resume", "Initializing Writeback/Resume pipeline stage");
    }
    void execute() override {
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

    bool is_active() override {
        return PipelineStage::input_latch->updated;
    }

    ~WritebackResume () {};
private:
    RegisterFile *rf;
};