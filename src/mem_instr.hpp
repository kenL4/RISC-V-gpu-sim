#pragma once

#include "utils.hpp"

class InstructionMemory {
public:
    cs_insn *insn;
    InstructionMemory(parse_output *data);
    cs_insn *get_instruction(uint64_t address);
    uint64_t get_base_addr();
private:
    std::map<uint64_t, cs_insn*> addr_to_insn;
    uint64_t base_addr;
};