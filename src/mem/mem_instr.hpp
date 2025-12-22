#pragma once

#include "utils.hpp"

class InstructionMemory {
public:
    InstructionMemory(parse_output *data);
    uint8_t *get_instruction(uint64_t address);
    uint64_t get_base_addr();
    uint64_t get_max_addr();
private:
    std::map<uint64_t, uint8_t *> addr_to_insn;
    uint64_t base_addr;
    uint64_t max_addr;
};