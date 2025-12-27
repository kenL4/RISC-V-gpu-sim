#include "pipeline_instr_fetch.hpp"
#include "../disassembler/llvm_disasm.hpp"

InstructionFetch::InstructionFetch(InstructionMemory *im,
                                   LLVMDisassembler *disasm)
    : im(im), disasm(disasm) {
  log("Instruction Fetch", "Initializing instruction fetch pipeline stage");
}

void InstructionFetch::execute() {
  if (!PipelineStage::input_latch->updated)
    return;

  Warp *warp = PipelineStage::input_latch->warp;
  uint64_t thread_id = PipelineStage::input_latch->active_threads[0];
  uint64_t warp_pc = warp->pc[thread_id];
  uint8_t *inst_bytes = im->get_instruction(warp_pc);

  uint64_t remaining_buffer = im->get_max_addr() + 4 - warp_pc;
  llvm::ArrayRef<uint8_t> code_ref(inst_bytes, remaining_buffer);
  llvm::MCInst inst = disasm->disasm_inst(0, code_ref);

  PipelineStage::input_latch->updated = false;
  PipelineStage::output_latch->updated = true;
  PipelineStage::output_latch->warp = warp;
  PipelineStage::output_latch->active_threads =
      PipelineStage::input_latch->active_threads;
  PipelineStage::output_latch->inst = inst;

  log("Instruction Fetch", "Warp " + std::to_string(warp->warp_id) +
                               " will execute instruction " +
                               disasm->getOpcodeName(inst.getOpcode()));
};

bool InstructionFetch::is_active() {
  return PipelineStage::input_latch->updated;
}