#include "utils.hpp"
#include "pipeline.hpp"

/*
 * The Active Thread Selection unit finds the threads in a warp with
 * the deepest nesting level and the same PC and 
 */
class ActiveThreadSelection : public PipelineStage {
public:
    ActiveThreadSelection() {
        log("Active Thread Selection", "Initializing Active Thread Selection Stage");
    }

    /*
     * Computes the vector of active threads based on nesting level
     */
    void execute() override {
        if (!PipelineStage::input_latch->updated) {
            return;
        }

        Warp *warp = PipelineStage::input_latch->warp;
        uint64_t max_nesting = warp->nesting_level[0];
        uint64_t warp_pc = warp->pc[0];
        for (int i = 0; i < warp->size; i++) {
            if (warp->nesting_level[i] > max_nesting) {
                max_nesting = warp->nesting_level[i];
                warp_pc = warp->pc[i];
            }
        }
        
        std::vector<uint64_t> active_threads;
        for (int i = 0; i < warp->size; i++) {
            bool same_nesting = max_nesting == warp->nesting_level[i];
            bool same_pc = warp_pc == warp->pc[i];
            if (same_nesting && same_pc) {
                active_threads.emplace_back(i);
            }
        }

        // Update pipeline latches
        PipelineStage::input_latch->updated = false;
        PipelineStage::output_latch->updated = true;
        PipelineStage::output_latch->warp = warp;
        // Could avoid copy constructor but okay for now
        PipelineStage::output_latch->active_threads = active_threads;

        log("Active Thread Selection", "Warp " + std::to_string(warp->warp_id) + " has " + std::to_string(active_threads.size()) + " active threads");
    }

    bool is_active() override {
        return PipelineStage::input_latch->updated;
    }

    ~ActiveThreadSelection() {
        
    }
};