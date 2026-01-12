#include "../utils.hpp"

uint64_t GPUStatisticsManager::get_gpu_cycles() { return gpu_cycles; }
uint64_t GPUStatisticsManager::get_gpu_instrs() { return gpu_instrs; }
uint64_t GPUStatisticsManager::get_gpu_dram_accs() { return gpu_dram_accs; }
uint64_t GPUStatisticsManager::get_gpu_retries() { return gpu_retries; }
uint64_t GPUStatisticsManager::get_gpu_susps() { return gpu_susps; }

uint64_t GPUStatisticsManager::get_cpu_instrs() { return cpu_instrs; }
uint64_t GPUStatisticsManager::get_cpu_dram_accs() { return cpu_dram_accs; }

void GPUStatisticsManager::increment_gpu_cycles() { gpu_cycles++; }
void GPUStatisticsManager::increment_gpu_instrs(size_t warp_size) {
  gpu_instrs += warp_size;
}
void GPUStatisticsManager::increment_gpu_dram_accs() { gpu_dram_accs++; }
void GPUStatisticsManager::increment_gpu_retries() { gpu_retries++; }
void GPUStatisticsManager::increment_gpu_susps() { gpu_susps++; }
void GPUStatisticsManager::increment_cpu_instrs() { cpu_instrs++; }
void GPUStatisticsManager::increment_cpu_dram_accs() { cpu_dram_accs++; }