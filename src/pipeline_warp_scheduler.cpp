#include "utils.hpp"
#include "pipeline.hpp"

/*
 * Represents the warp scheduler unit in the pipeline
 * It will use a barrel scheduler to fairly pick between the warps
 * that have no suspended threads
 */
class WarpScheduler : public PipelineStage {
public:
    WarpScheduler(int warp_size, int warp_count, uint64_t start_pc): warp_size(warp_size), warp_count(warp_count) {
        log("Warp Scheduler", "Initializing warp scheduling pipeline stage");
        for (int i = 0; i < warp_count; i++) {
            Warp *warp = new Warp(i, warp_size, start_pc);
            warp_queue.push(warp);
        }
    }

    void execute() override {
        // if (PipelineStage::input_latch->updated) {
        //     warp_queue.push(PipelineStage::input_latch->warp);
        //     PipelineStage::input_latch->updated = false;
        // }
        if (warp_queue.empty()) {
            return;
        }

        Warp *top = warp_queue.front();
        warp_queue.pop();

        // Update pipeline latch
        PipelineStage::output_latch->updated = true;
        PipelineStage::output_latch->warp = top;

        log("Warp Scheduler", "Warp " + std::to_string(top->warp_id) + " scheduled to run");
    }

    bool is_active() override {
        return warp_queue.size() > 0; // || PipelineStage::input_latch->updated;
    }

    ~WarpScheduler() {
        while (warp_queue.size() > 0) {
            delete warp_queue.front();
            warp_queue.pop();
        }

        log("Warp Scheduler", "Destroyed pipeline stage");
    }
private:
    int warp_size;
    int warp_count;
    std::queue<Warp*> warp_queue;
};