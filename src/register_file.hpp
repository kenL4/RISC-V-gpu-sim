#pragma once

#include "utils.hpp"

class RegisterFile {
public:
    std::map<uint64_t, std::vector<int>> warp_id_to_registers;
    RegisterFile(int register_count);
    int get_register(uint64_t warp_id, int reg);
    void set_register(uint64_t warp_id, int reg, int value);
    ~RegisterFile();
private:
    uint64_t registers_per_warp;
};