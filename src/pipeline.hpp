#pragma once

#include "utils.hpp"

/*
 * An individual warp. This maintains the per-warp state
 */
class Warp {
public:
    uint64_t warp_id;
    size_t size;
    Warp(uint64_t warp_id, size_t size): warp_id(warp_id), size(size) {};
    ~Warp() {};
};

/*
 * A latch between each pipeline stage that defines
 * the input/output interface between stages
 */
class PipelineLatch {
public:
    bool updated;
    uint64_t warp_id;
};

/*
 * The PipelineStage interface defines the
 * required capabilities of the stages in the pipeline
 */
class PipelineStage {
public:
    virtual ~PipelineStage() = default;
    virtual void execute() {};
    virtual bool is_active() {
        return false;
    };
    virtual void set_latches(PipelineLatch *input, PipelineLatch* output) {
        input_latch = input;
        output_latch = output;
    };
protected:
    PipelineLatch *input_latch;
    PipelineLatch *output_latch;
};