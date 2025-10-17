#include "utils.hpp"
#include "pipeline.hpp"

/*
 * Represents the warp scheduler unit in the pipeline
 * It will use a barrel scheduler to fairly pick between the warps
 * that have no suspended threads
 */
class WarpScheduler : public PipelineStage {
public:
    WarpScheduler(int warp_size, int warp_count): warp_size(warp_size), warp_count(warp_count) {
        log("WARP_SCHEDULER", "Initializing warp scheduling pipeline stage");
        for (int i = 0; i < warp_size; i++) {
            Warp *warp = new Warp(i, warp_size);
            warp_queue.push(warp);
        }
    }

    void execute() override {
        if (warp_queue.empty()) {
            return;
        }

        Warp *top = warp_queue.front();
        warp_queue.pop();

        // Update pipeline latch
        PipelineStage::output_latch->updated = true;
        PipelineStage::output_latch->warp_id = top->warp_id;

        log("WARP_SCHEDULER", "warp " + std::to_string(top->warp_id));
    }

    bool is_active() override {
        return warp_queue.size() > 0;
    }

    ~WarpScheduler() {
        while (warp_queue.size() > 0) {
            delete warp_queue.front();
            warp_queue.pop();
        }

        log("WARP_SCHEDULER", "Destroyed warp scheduling pipeline stage");
    }
private:
    int warp_size;
    int warp_count;
    std::queue<Warp*> warp_queue;
};