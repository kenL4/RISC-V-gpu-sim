#pragma once

#include "utils.hpp"

class RegisterFile {
public:
    // This data structure represents a mapping from warp ID
    // to a registers_per_warp wide vector of the register value in each thread
    std::map<uint64_t, std::vector<std::vector<int>>> warp_id_to_registers;
    RegisterFile(size_t register_count, size_t thread_count);
    int get_register(uint64_t warp_id, int thread, int reg);
    void set_register(uint64_t warp_id, int thread, int reg, int value);
    void pretty_print(uint64_t warp_id);
    ~RegisterFile();
private:
    uint64_t registers_per_warp;
    size_t thread_count;
};