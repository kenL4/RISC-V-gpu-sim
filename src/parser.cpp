#include "parser.hpp"

parse_error parse_binary(std::string file, LLVMDisassembler &disasm, parse_output *out) {
    ELFIO::elfio reader;
    if (!reader.load(file)) {
        return PARSE_LOAD_ERROR;
    }

    // Load .text section (code)
    const ELFIO::section* text_section = reader.sections[".text"];
    if (text_section == nullptr) {
        return PARSE_LOAD_ERROR;
    }
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

    // Load data sections (.rodata, .data, .sdata, etc.)
    out->data_sections.clear();
    ELFIO::Elf_Half sec_num = reader.sections.size();
    for (ELFIO::Elf_Half i = 0; i < sec_num; ++i) {
        ELFIO::section* sec = reader.sections[i];
        std::string sec_name = sec->get_name();
        
        // Load read-only and data sections (but not .text which is already loaded)
        if (sec_name == ".rodata" || sec_name == ".data" || sec_name == ".sdata" || 
            sec_name == ".bss" || sec_name == ".sbss") {
            uint64_t sec_addr = sec->get_address();
            size_t sec_size = sec->get_size();
            
            if (sec_size > 0) {
                std::vector<uint8_t> sec_data(sec_size);
                
                // For .bss and .sbss, data is zero-initialized
                if (sec_name == ".bss" || sec_name == ".sbss") {
                    std::fill(sec_data.begin(), sec_data.end(), 0);
                } else {
                    // For other sections, load actual data
                    const char* sec_raw_data = sec->get_data();
                    if (sec_raw_data != nullptr) {
                        std::copy(
                            (const uint8_t*) sec_raw_data,
                            (const uint8_t*) sec_raw_data + sec_size,
                            sec_data.begin()
                        );
                    } else {
                        std::fill(sec_data.begin(), sec_data.end(), 0);
                    }
                }
                
                out->data_sections.push_back({sec_addr, sec_data});
            }
        }
    }

    return PARSE_SUCCESS;
}