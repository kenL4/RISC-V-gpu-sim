#include "utils.hpp"
#include "pipeline.hpp"

/*
 * The Active Thread Selection unit finds the threads in a warp with
 * the deepest nesting level and the same PC and 
 */
class ActiveThreadSelection : public PipelineStage {
public:
    ActiveThreadSelection();
    /*
     * Computes the vector of active threads based on nesting level
     */
    void execute() override;
    bool is_active() override;
    ~ActiveThreadSelection() {}

private:
    struct BufferData {
        Warp *warp;
        std::vector<uint64_t> active_threads;
        bool valid;
        BufferData() : warp(nullptr), valid(false) {}
    };
    BufferData stage_buffer;
};