#include "../utils.hpp"

uint64_t GPUStatisticsManager::get_gpu_cycles() { return gpu_cycles; }
uint64_t GPUStatisticsManager::get_gpu_instrs() { return gpu_instrs; }
uint64_t GPUStatisticsManager::get_dram_accs() { return dram_accs; }

void GPUStatisticsManager::increment_gpu_cycles() { gpu_cycles++; }
void GPUStatisticsManager::increment_gpu_instrs() { gpu_instrs++; }
void GPUStatisticsManager::increment_dram_accs() { dram_accs++; }