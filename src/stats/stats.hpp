#pragma once

#include <stdint.h>

class GPUStatisticsManager {
public:
  static GPUStatisticsManager &instance() {
    static GPUStatisticsManager inst;
    return inst;
  }

  uint64_t get_gpu_cycles();
  uint64_t get_gpu_instrs();
  uint64_t get_dram_accs();

  void increment_gpu_cycles();
  void increment_gpu_instrs();
  void increment_dram_accs();

  uint64_t get_cpu_instrs();
  void increment_cpu_instrs();

private:
  uint64_t gpu_cycles = 0;
  uint64_t gpu_instrs = 0;
  uint64_t cpu_instrs = 0;
  uint64_t dram_accs = 0;

  GPUStatisticsManager() = default;
};