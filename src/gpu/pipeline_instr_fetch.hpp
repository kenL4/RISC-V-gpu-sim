#include "mem/mem_instr.hpp"
#include "pipeline.hpp"
#include "utils.hpp"

/*
 * The Instruction Fetch unit looks up the instruction
 * associated with the active threads.
 */
class InstructionFetch : public PipelineStage {
public:
  InstructionFetch(InstructionMemory *im, LLVMDisassembler *disasm);
  void execute() override;
  bool is_active() override;
  ~InstructionFetch() {};

private:
  LLVMDisassembler *disasm;
  InstructionMemory *im;
};