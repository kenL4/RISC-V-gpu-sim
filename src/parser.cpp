#include "parser.hpp"

parse_error parse_binary(std::string file, parse_output *out) {
    ELFIO::elfio reader;
    if (!reader.load(file)) {
        return PARSE_LOAD_ERROR;
    }

    const ELFIO::section* text_section = reader.sections[".text"];
    const uint8_t *code = (const uint8_t*)text_section->get_data();
    size_t code_size = text_section->get_size();
    out->base_addr = text_section->get_address();

    if (cs_open(CS_ARCH_RISCV, CS_MODE_RISCVC, &out->handle) != CS_ERR_OK) {
        return PARSE_LOAD_ERROR;
    }
    // Ask Capstone to retrieve the individual operands too
    cs_option(out->handle, CS_OPT_DETAIL, CS_OPT_ON);
    out->count = cs_disasm(out->handle, code, code_size, out->base_addr, 0, &out->insn);

    if (out->count > 0) return PARSE_SUCCESS;
    return PARSE_LOAD_ERROR;
}