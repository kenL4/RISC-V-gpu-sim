#include "register_file.hpp"

RegisterFile::RegisterFile(size_t register_count, size_t thread_count): 
    registers_per_warp(register_count), thread_count(thread_count) {
    log("Register File", "Initialised with " + std::to_string(register_count) + " registers for " +
        std::to_string(thread_count) + " threads a warp");
}

int RegisterFile::get_register(uint64_t warp_id, int thread, int reg) {
    if (warp_id_to_registers.count(warp_id) <= 0) {
        warp_id_to_registers[warp_id].resize(registers_per_warp);
        for (auto &reg_vec : warp_id_to_registers[warp_id]) {
            reg_vec.resize(thread_count, 0);
        }
    }

    // Note: Capstone maintains a REG_INVALID as reg 0 so
    // we subtract 1 to index out register file
    return warp_id_to_registers[warp_id][reg - 1][thread];
}

void RegisterFile::set_register(uint64_t warp_id, int thread, int reg, int value) {
    if (warp_id_to_registers.count(warp_id) <= 0) {
        warp_id_to_registers[warp_id].resize(registers_per_warp);
        for (auto &reg_vec : warp_id_to_registers[warp_id]) {
            reg_vec.resize(thread_count, 0);
        }
    }

    // Note: Capstone maintains a REG_INVALID as reg 0 so
    // we subtract 1 to index our register file
    warp_id_to_registers[warp_id][reg - 1][thread] = value;
}

RegisterFile::~RegisterFile() {

}

void RegisterFile::pretty_print(uint64_t warp_id) {
    if (warp_id_to_registers.count(warp_id) == 0) {
        std::cout << "No registers for warp " << warp_id << "\n";
        return;
    }

    const auto &regs = warp_id_to_registers[warp_id];
    size_t thread_count = regs.empty() ? 0 : regs[0].size();

    // Print header: Thread IDs
    std::cout << std::setw(4) << "Thread";
    for (size_t t = 0; t < thread_count; ++t) {
        std::cout << std::setw(4) << t;
    }
    std::cout << "\n";

    // Print separator
    std::cout << "----";
    for (size_t t = 0; t < thread_count; ++t) std::cout << "----";
    std::cout << "\n";

    // Print each register
    for (size_t reg_idx = 0; reg_idx < regs.size(); ++reg_idx) {
        std::cout << std::setw(4) << ("x" + std::to_string(reg_idx));
        for (size_t t = 0; t < thread_count; ++t) {
            std::cout << std::setw(4) << regs[reg_idx][t];
        }
        std::cout << "\n";
    }
}