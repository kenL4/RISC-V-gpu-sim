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
  uint64_t get_gpu_dram_accs();
  uint64_t get_gpu_retries();
  uint64_t get_gpu_susps();

  void increment_gpu_cycles();
  void increment_gpu_instrs(size_t warp_size);
  void increment_gpu_dram_accs();
  void increment_gpu_retries();
  void increment_gpu_susps();

  void reset_gpu_cycles();
  void reset_gpu_instrs();
  void reset_gpu_dram_accs();
  void reset_gpu_retries();
  void reset_gpu_susps();

  uint64_t get_cpu_instrs();
  uint64_t get_cpu_dram_accs();
  void increment_cpu_instrs();
  void increment_cpu_dram_accs();

  uint64_t get_gpu_active_cpu_dram_accs();
  void increment_gpu_active_cpu_dram_accs();
  void reset_gpu_active_cpu_dram_accs();

  void set_gpu_pipeline_active(bool active);
  bool is_gpu_pipeline_active();

private:
  uint64_t gpu_cycles = 0;
  uint64_t gpu_instrs = 0;
  uint64_t gpu_dram_accs = 0;
  uint64_t gpu_retries = 0;
  uint64_t gpu_susps = 0;
  uint64_t cpu_instrs = 0;
  uint64_t cpu_dram_accs = 0;
  uint64_t gpu_active_cpu_dram_accs = 0;
  bool gpu_pipeline_active_flag = false;

  GPUStatisticsManager() = default;
};