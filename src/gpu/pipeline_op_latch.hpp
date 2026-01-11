#pragma once

#include "utils.hpp"
#include "pipeline.hpp"

/*
 * The Operand Latch stage matches SIMTight's Stage 4 (Operand Latch).
 * This stage accounts for register file load latency and latches operands
 * before they are passed to the Execute stage.
 * 
 * In SIMTight, this stage delays signals by (loadLatency - 1) cycles,
 * which is 0 cycles with the default loadLatency of 1. The stage primarily
 * serves as a pipeline boundary to match SIMTight's 7-stage pipeline structure.
 */
class OperandLatch: public PipelineStage {
public:
    OperandLatch();
    void execute() override;
    bool is_active() override;
    ~OperandLatch() {};
};
