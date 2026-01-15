#include "register_file.hpp"
#include "config.hpp"

RegisterFile::RegisterFile(size_t register_count, size_t thread_count): 
    registers_per_warp(register_count), thread_count(thread_count) {
    log("Register File", "Initialised with " + std::to_string(register_count) + " registers for " +
        std::to_string(thread_count) + " threads a warp");
}

int get_register_idx(llvm::MCRegister reg) {
    return reg - llvm::RISCV::X0;
}

void RegisterFile::ensure_warp_initialized(uint64_t warp_id) {
    if (warp_id_to_registers.find(warp_id) == warp_id_to_registers.end()) {
        warp_id_to_registers[warp_id].resize(registers_per_warp);
        for (auto &reg_vec : warp_id_to_registers[warp_id]) {
            reg_vec.resize(thread_count, 0);  // resize with 0 already initializes
        }
    }
}

int RegisterFile::get_register(uint64_t warp_id, int thread, int reg, bool is_cpu) {
    // CPU should never use GPU RegisterFile directly - it should use HostRegisterFile
    // If CPU tries to access, return 0 (shouldn't happen, but handle gracefully)
    if (is_cpu) {
        return 0;
    }
    ensure_warp_initialized(warp_id);
    return warp_id_to_registers[warp_id][get_register_idx(reg)][thread];
}

void RegisterFile::set_register(uint64_t warp_id, int thread, int reg, int value, bool is_cpu) {
    // CPU should never use GPU RegisterFile directly - it should use HostRegisterFile
    // If CPU tries to access, ignore it (shouldn't happen, but handle gracefully)
    if (is_cpu) {
        return;
    }
    ensure_warp_initialized(warp_id);

    // Don't write to X0 (zero register) - it's always 0
    if (reg != llvm::RISCV::X0) {
        int reg_idx = get_register_idx(reg);
        // Bounds check to prevent segfault
        if (reg_idx >= 0 && reg_idx < static_cast<int>(registers_per_warp) && 
            thread >= 0 && thread < static_cast<int>(thread_count)) {
            warp_id_to_registers[warp_id][reg_idx][thread] = value;
        }
    }
}

std::optional<int> RegisterFile::get_csr(uint64_t warp_id, int thread, int csr) {
    if (warp_id_to_csr.find(warp_id) == warp_id_to_csr.end()) {
        warp_id_to_csr[warp_id].resize(thread_count);
    }

    auto &csr_map = warp_id_to_csr[warp_id][thread];
    auto it = csr_map.find(csr);
    if (it == csr_map.end()) {
        return {};
    }

    return it->second;
}

void RegisterFile::set_csr(uint64_t warp_id, int thread, int csr, int value) {
    if (warp_id_to_csr.find(warp_id) == warp_id_to_csr.end()) {
        warp_id_to_csr[warp_id].resize(thread_count);
    }

    warp_id_to_csr[warp_id][thread][csr] = value;
}

RegisterFile::~RegisterFile() {

}

void RegisterFile::pretty_print(uint64_t warp_id) {
    if (warp_id_to_registers.find(warp_id) == warp_id_to_registers.end()) {
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