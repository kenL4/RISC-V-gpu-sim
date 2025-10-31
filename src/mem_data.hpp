#include "utils.hpp"

// TODO: Properly simulate data memory
class DataMemory {
public:
    int load(uint64_t addr, size_t size);
    void store(uint64_t addr, size_t size, int val);
};