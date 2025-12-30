#include "../utils.hpp"

uint64_t GPUStatisticsManager::get_gpu_cycles() { return gpu_cycles; }
uint64_t GPUStatisticsManager::get_gpu_instrs() { return gpu_instrs; }
uint64_t GPUStatisticsManager::get_dram_accs() { return dram_accs; }

uint64_t GPUStatisticsManager::get_cpu_instrs() { return cpu_instrs; }
void GPUStatisticsManager::increment_gpu_cycles() { gpu_cycles++; }
void GPUStatisticsManager::increment_gpu_instrs(size_t warp_size) { gpu_instrs += warp_size; }
void GPUStatisticsManager::increment_cpu_instrs() { cpu_instrs++; }
void GPUStatisticsManager::increment_dram_accs() { dram_accs++; }