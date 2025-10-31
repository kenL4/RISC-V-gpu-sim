#include "mem_data.hpp"

int DataMemory::load(uint64_t addr, size_t size) {
    // No-op
    return 0xDEADBEEF;
}

void DataMemory::store(uint64_t addr, size_t size, int val) {
    // No-op
}