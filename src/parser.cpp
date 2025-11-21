#include "parser.hpp"

parse_error parse_binary(std::string file, LLVMDisassembler &disasm, parse_output *out) {
    ELFIO::elfio reader;
    if (!reader.load(file)) {
        return PARSE_LOAD_ERROR;
    }

    const ELFIO::section* text_section = reader.sections[".text"];
    const char* raw_data = text_section->get_data();
    size_t code_size = text_section->get_size();

    out->code.resize(code_size);
    std::copy(
        (const uint8_t*) raw_data,
        (const uint8_t*) raw_data + code_size,
        out->code.begin()
    );
    out->base_addr = text_section->get_address();
    out->max_addr = text_section->get_size() + out->base_addr;

    return PARSE_SUCCESS;
}