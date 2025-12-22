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

private:
  uint64_t gpu_cycles;
  uint64_t gpu_instrs;
  uint64_t dram_accs;

  GPUStatisticsManager() = default;
};