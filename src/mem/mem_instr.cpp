#include "mem_instr.hpp"

InstructionMemory::InstructionMemory(parse_output *data) {
    base_addr = data->base_addr;
    max_addr = data->max_addr - 4;
    for (uint64_t i = base_addr; i <= max_addr; i+=4) {
        uint64_t offset = i - base_addr;
        addr_to_insn[i] = data->code.data() + offset;
    }
    debug_log("Instruction addresses range from " + std::to_string(base_addr) + " -> " + std::to_string(max_addr));
}

uint8_t *InstructionMemory::get_instruction(uint64_t address) {
    return addr_to_insn[address];
}
uint64_t InstructionMemory::get_base_addr() {
    return base_addr;
}
uint64_t InstructionMemory::get_max_addr() {
    return max_addr;
}