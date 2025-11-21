#pragma once

#include <elfio/elfio.hpp>
#include "utils.hpp"
#include "disassembler/llvm_disasm.hpp"

enum parse_error {
    PARSE_LOAD_ERROR = 0,
    PARSE_SUCCESS = 1,
};

struct parse_output {
    std::vector<uint8_t> code;
    uint64_t base_addr;
    uint64_t max_addr;
};

/*
 * A helper function to parse a RISC-V ELF binary
 * to an intermediate format for the simulation.
 */
parse_error parse_binary(std::string file, LLVMDisassembler &disasm, parse_output *out);