#pragma once

#include <elfio/elfio.hpp>
#include "utils.hpp"

enum parse_error {
    PARSE_LOAD_ERROR = 0,
    PARSE_SUCCESS = 1,
};

struct parse_output {
    csh handle;
    cs_insn *insn;
    size_t count;
    uint64_t base_addr;
};

/*
 * A helper function to parse a RISC-V ELF binary
 * to an intermediate format for the simulation.
 */
parse_error parse_binary(std::string file, parse_output *out);