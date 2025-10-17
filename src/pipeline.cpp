#include "pipeline.hpp"

/*
 * The Pipeline class represents a series of computation stages
 * that each warp will pass through.
 */
class Pipeline {
public:
    /*
     * Insert a stage in our polymorphic stage container
     */
    template <typename T, typename... Args>
    void add_stage(Args... args) {
        stages.emplace_back(std::make_shared<T>(args...));
    }

    /*
     * Execute one cycle of the pipeline
     */
    void execute() {
        // Execute backwards to avoid overwriting latches prematurely
        for (auto it = stages.rbegin(); it != stages.rend(); it++) {
            (*it)->execute();
        }
    }

    /*
     * Returns true if any of the associated pipeline stages
     * are still active
     */
    bool has_active_stages() {
        for (auto &stage: stages) {
            if (stage->is_active()) return true;
        }
        return false;
    }

    /*
     * Returns the pipeline stage with index "index"
     */
    std::shared_ptr<PipelineStage> get_stage(int index) {
        return stages[index];
    }

private:
    std::vector<std::shared_ptr<PipelineStage>> stages;
};

/*
 * A dummy implementation of a pipeline stage
 */
class MockPipelineStage: public PipelineStage {
public:
    std::string name;
    MockPipelineStage(std::string name): name(name) {};
    void execute() override {
        if (!PipelineStage::input_latch->updated) return;
        
        uint64_t warp_id = PipelineStage::input_latch->warp_id;
        log(name, "warp " + std::to_string(warp_id));

        // Update pipeline latches
        PipelineStage::input_latch->updated = false;
        PipelineStage::output_latch->updated = true;
        PipelineStage::output_latch->warp_id = warp_id;
    };

    bool is_active() override {
        // A pipeline stage will do something if there is data passed to it
        return PipelineStage::input_latch->updated;
    }

    ~MockPipelineStage() {};
};