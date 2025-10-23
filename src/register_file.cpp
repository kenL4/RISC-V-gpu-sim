#include "register_file.hpp"

RegisterFile::RegisterFile(int register_count) {
    registers_per_warp = register_count;
}

int RegisterFile::get_register(uint64_t warp_id, int reg) {
    if (warp_id_to_registers.count(warp_id) <= 0) {
        warp_id_to_registers[warp_id] = {};
        for (int i = 0; i < registers_per_warp; i++) {
            warp_id_to_registers[warp_id].emplace_back(0);
        }
    }

    return warp_id_to_registers[warp_id][reg];
}

void RegisterFile::set_register(uint64_t warp_id, int reg, int value) {
    if (warp_id_to_registers.count(warp_id) <= 0) {
        warp_id_to_registers[warp_id] = {};
        for (int i = 0; i < registers_per_warp; i++) {
            warp_id_to_registers[warp_id].emplace_back(0);
        }
    }

    warp_id_to_registers[warp_id][reg] = value;
}

RegisterFile::~RegisterFile() {

}