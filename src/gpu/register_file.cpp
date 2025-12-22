#include "register_file.hpp"

RegisterFile::RegisterFile(size_t register_count, size_t thread_count): 
    registers_per_warp(register_count), thread_count(thread_count) {
    log("Register File", "Initialised with " + std::to_string(register_count) + " registers for " +
        std::to_string(thread_count) + " threads a warp");
}

int get_register_idx(llvm::MCRegister reg) {
    return reg - llvm::RISCV::X0;
}

int RegisterFile::get_register(uint64_t warp_id, int thread, int reg) {
    if (warp_id_to_registers.count(warp_id) <= 0) {
        warp_id_to_registers[warp_id].resize(registers_per_warp);
        for (auto &reg_vec : warp_id_to_registers[warp_id]) {
            reg_vec.resize(thread_count, 0);
            for (int i = 0 ; i< reg_vec.size(); i++) {
                reg_vec[i] = 0;
            }
        }
    }

    return warp_id_to_registers[warp_id][get_register_idx(reg)][thread];
}

void RegisterFile::set_register(uint64_t warp_id, int thread, int reg, int value) {
    if (warp_id_to_registers.count(warp_id) <= 0) {
        warp_id_to_registers[warp_id].resize(registers_per_warp);
        for (auto &reg_vec : warp_id_to_registers[warp_id]) {
            reg_vec.resize(thread_count, 0);
            for (int i = 0 ; i< reg_vec.size(); i++) {
                reg_vec[i] = 0;
            }
        }
    }

    if (reg != llvm::RISCV::X0) warp_id_to_registers[warp_id][get_register_idx(reg)][thread] = value;
}

std::optional<int> RegisterFile::get_csr(uint64_t warp_id, int thread, int csr) {
    if (warp_id_to_csr.count(warp_id) <= 0) {
        warp_id_to_csr[warp_id].resize(thread_count);
    }

    if (warp_id_to_csr[warp_id][thread].find(csr) == warp_id_to_csr[warp_id][thread].end()) {
        // We return a null optional type before definition
        return {};
    }

    return warp_id_to_csr[warp_id][thread][csr];
}

void RegisterFile::set_csr(uint64_t warp_id, int thread, int csr, int value) {
    if (warp_id_to_csr.count(warp_id) <= 0) {
        warp_id_to_csr[warp_id].resize(thread_count);
    }

    warp_id_to_csr[warp_id][thread][csr] = value;
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
    std::cout << std::setw(4) << "Thrd";
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