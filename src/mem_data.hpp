#pragma once

#include "utils.hpp"

class DataMemory {
public:
    int load(uint64_t addr, size_t bytes);
    void store(uint64_t addr, size_t bytes, uint64_t val);
private:
    std::map<uint64_t, uint8_t> memory;
};