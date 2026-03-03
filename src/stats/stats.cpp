#include "../utils.hpp"

uint64_t GPUStatisticsManager::get_gpu_cycles() { return gpu_cycles; }
uint64_t GPUStatisticsManager::get_gpu_instrs() { return gpu_instrs; }
uint64_t GPUStatisticsManager::get_gpu_dram_accs() { return gpu_dram_accs; }
uint64_t GPUStatisticsManager::get_gpu_retries() { return gpu_retries; }
uint64_t GPUStatisticsManager::get_gpu_susps() { return gpu_susps; }

uint64_t GPUStatisticsManager::get_cpu_instrs() { return cpu_instrs; }
uint64_t GPUStatisticsManager::get_cpu_dram_accs() { return cpu_dram_accs; }

void GPUStatisticsManager::reset_gpu_cycles() { gpu_cycles = 0; }
void GPUStatisticsManager::reset_gpu_instrs() {
  gpu_instrs = 0;
  instr_delay_pipe.fill(0);
  instr_pipe_head = 0;
  instr_pending_this_cycle = 0;
}
void GPUStatisticsManager::reset_gpu_dram_accs() { gpu_dram_accs = 0; }
void GPUStatisticsManager::reset_gpu_retries() { gpu_retries = 0; }
void GPUStatisticsManager::reset_gpu_susps() { gpu_susps = 0; }

void GPUStatisticsManager::increment_gpu_cycles() { gpu_cycles++; }
void GPUStatisticsManager::increment_gpu_instrs(size_t warp_size) {
  instr_pending_this_cycle += warp_size;
}
void GPUStatisticsManager::increment_gpu_dram_accs() { gpu_dram_accs++; }
void GPUStatisticsManager::increment_gpu_retries() { gpu_retries++; }
void GPUStatisticsManager::increment_gpu_susps() { gpu_susps++; }
void GPUStatisticsManager::increment_cpu_instrs() { cpu_instrs++; }
void GPUStatisticsManager::increment_cpu_dram_accs() { cpu_dram_accs++; }

uint64_t GPUStatisticsManager::get_gpu_active_cpu_dram_accs() { return gpu_active_cpu_dram_accs; }
void GPUStatisticsManager::increment_gpu_active_cpu_dram_accs() { gpu_active_cpu_dram_accs++; }
void GPUStatisticsManager::reset_gpu_active_cpu_dram_accs() { gpu_active_cpu_dram_accs = 0; }

void GPUStatisticsManager::set_gpu_pipeline_active(bool active) { gpu_pipeline_active_flag = active; }
bool GPUStatisticsManager::is_gpu_pipeline_active() { return gpu_pipeline_active_flag; }

void GPUStatisticsManager::tick_instr_pipeline() {
  if (gpu_pipeline_active_flag) {
    gpu_instrs += instr_delay_pipe[instr_pipe_head];
  }
  instr_delay_pipe[instr_pipe_head] = instr_pending_this_cycle;
  instr_pending_this_cycle = 0;
  instr_pipe_head = (instr_pipe_head + 1) % INSTR_TREE_DEPTH;
}
