#include "utils.hpp"
#include "pipeline.hpp"

/*
 * This unit finds the threads in a warp with
 * the deepest nesting level and the same PC and 
 */
class ActiveThreadSelection : public PipelineStage {
public:
    ActiveThreadSelection() {
        log("Active Thread Selection", "Initializing Active Thread Selection Stage");
    }

    /*
     * Computes the vector of active threads based on nesting level
     * TODO: handle checking for common PC
     */
    void execute() override {
        if (!PipelineStage::input_latch->updated) {
            return;
        }

        Warp *warp = PipelineStage::input_latch->warp;
        uint64_t max_nesting = 0;
        for (int i = 0; i < warp->size; i++) {
            max_nesting = std::max(max_nesting, warp->nesting_level[i]);
        }

        std::vector<uint64_t> active_threads;
        for (int i = 0; i < warp->size; i++) {
            if (max_nesting == warp->nesting_level[i]) {
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