#pragma once

#include "utils.hpp"

/*
 * An individual warp. This maintains the per-warp state
 */
class Warp {
public:
    uint64_t warp_id;
    size_t size;
    std::vector<uint64_t> pc;
    std::vector<uint64_t> nesting_level;
    Warp(uint64_t warp_id, size_t size, uint64_t start_pc): warp_id(warp_id), size(size) {
        pc.reserve(size);
        nesting_level.reserve(size);
        
        // Clear the vectors
        for (int i = 0; i < size; i++) {
            pc[i] = start_pc;
            nesting_level[i] = 0;
        }
    };
    ~Warp() {};
};

/*
 * A latch between each pipeline stage that defines
 * the input/output interface between stages
 * TODO: Refactor the latch so that we
 * can vary it between stages (union/interface)
 */
class PipelineLatch {
public:
    bool updated;
    Warp* warp;
    std::vector<uint64_t> active_threads;
    cs_insn* instruction;
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