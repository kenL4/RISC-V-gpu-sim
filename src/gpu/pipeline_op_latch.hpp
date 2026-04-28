#pragma once

#include "utils.hpp"
#include "pipeline.hpp"

/*
 * The Operand Latch stage matches SIMTight's Stage 4 (Operand Latch).
 * This stage accounts for register file load latency and latches operands
 * before they are passed to the Execute stage.
 */
class OperandLatch: public PipelineStage {
public:
    OperandLatch();
    void execute() override;
    bool is_active() override;
    ~OperandLatch() {};
};
