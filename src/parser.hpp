#include <capstone/capstone.h>
#include <elfio/elfio.hpp>
#include "utils.hpp"

enum parse_error {
    PARSE_LOAD_ERROR = 0,
    PARSE_SUCCESS = 1,
};

/*
 * A helper function to parse a RISC-V ELF binary
 * to an intermediate format for the simulation.
 */
parse_error parse_binary(std::string file) {
    ELFIO::elfio reader;
    if (!reader.load(file)) {
        return PARSE_LOAD_ERROR;
    }

    const ELFIO::section* text_section = reader.sections[".text"];
    const uint8_t *code = (const uint8_t*)text_section->get_data();
    size_t code_size = text_section->get_size();
    uint64_t addr = text_section->get_address();

    csh handle;
    cs_insn *insn;
    size_t count;

    if (cs_open(CS_ARCH_RISCV, CS_MODE_RISCVC, &handle) != CS_ERR_OK) {
        return PARSE_LOAD_ERROR;
    }
    count = cs_disasm(handle, code, code_size, addr, 0, &insn);

    for (int i = 0; i < count; i++) {
        std::cout << std::hex << insn[i].address << ":\t"
            << insn[i].mnemonic << "\t"
            << insn[i].op_str << std::endl;
    }

    if (count > 0) cs_free(insn, count);
    cs_close(&handle);
    return PARSE_SUCCESS;
}