#include "utils.hpp"
#include "pipeline.hpp"
#include "mem_instr.hpp"

/*
 * The Instruction Fetch unit looks up the instruction
 * associated with the active threads.
 */
class InstructionFetch: public PipelineStage {
public:
    InstructionFetch(InstructionMemory *im);
    void execute() override;
    bool is_active() override;
    ~InstructionFetch() {};
private:
    InstructionMemory *im;
};