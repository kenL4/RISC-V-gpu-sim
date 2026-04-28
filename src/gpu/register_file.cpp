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
            reg_vec.resize(thread_count, 0);
        }
    }
}

int RegisterFile::get_register(uint64_t warp_id, int thread, int reg, bool is_cpu) {
    if (is_cpu) return 0;

    if (reg == llvm::RISCV::X0) return 0;
    ensure_warp_initialized(warp_id);
    return warp_id_to_registers[warp_id][get_register_idx(reg)][thread];
}

void RegisterFile::set_register(uint64_t warp_id, int thread, int reg, int value, bool is_cpu) {
    if (is_cpu) return;
    ensure_warp_initialized(warp_id);

    if (reg == llvm::RISCV::X0) return;

    int reg_idx = get_register_idx(reg);
    if (reg_idx >= 0 && reg_idx < static_cast<int>(registers_per_warp) && 
        thread >= 0 && thread < static_cast<int>(thread_count)) {
        warp_id_to_registers[warp_id][reg_idx][thread] = value;
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

/*
 * TODO: Make this look good
 */
void RegisterFile::pretty_print(uint64_t warp_id) {
    // DO NOTHING (removed because honestly it looked awful
    // and was not super helpful for debugging)
}