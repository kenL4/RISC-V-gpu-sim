#include "mem_instr.hpp"

InstructionMemory::InstructionMemory(parse_output *data) {
    insn = data->insn;
    max_addr = insn[0].address;
    for (int i = 0; i < data->count; i++) {
        addr_to_insn[insn[i].address] = insn + i;
        max_addr = std::max(max_addr, insn[i].address);
    }
    base_addr = data->base_addr;
}

cs_insn *InstructionMemory::get_instruction(uint64_t address) {
    return addr_to_insn[address];
}
uint64_t InstructionMemory::get_base_addr() {
    return base_addr;
}
uint64_t InstructionMemory::get_max_addr() {
    return max_addr;
}