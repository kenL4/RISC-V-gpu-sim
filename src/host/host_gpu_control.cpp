#include "host_gpu_control.hpp"
#include "config.hpp"

HostGPUControl::HostGPUControl()
    : kernel_pc(0), arg_ptr(0), dims(0), gpu_active(false), buf(""), stat_value(0U) {}

void HostGPUControl::set_scheduler(std::shared_ptr<WarpScheduler> scheduler) {
  this->scheduler = scheduler;
}

void HostGPUControl::set_pc(uint64_t pc) { kernel_pc = pc; }
void HostGPUControl::set_arg_ptr(uint64_t arg_ptr) { this->arg_ptr = arg_ptr; }
void HostGPUControl::set_dims(uint64_t dims) { this->dims = dims; }
void HostGPUControl::set_warps_per_block(unsigned n) {
  if (scheduler) {
    scheduler->set_warps_per_block(n);
  }
}

uint64_t HostGPUControl::get_arg_ptr() { return arg_ptr; }

void HostGPUControl::launch_kernel() {
  for (int i = 0; i < NUM_WARPS; i++) {
    Warp *warp = new Warp(i, NUM_LANES, kernel_pc, false);
    scheduler->insert_warp(warp);
  }
  gpu_active = true;
  if (!Config::instance().isStatsOnly()) {
    std::cout << "[HostGPUControl] Launched kernel with " << NUM_WARPS << " warps" << std::endl;
  }
  scheduler->set_active(true);
  // Set pipeline_active = true when kernel launches (matching SIMTight)
  if (pipeline != nullptr) {
    pipeline->set_pipeline_active(true);
  }
}

bool HostGPUControl::is_gpu_active() {
  return gpu_active && scheduler->is_active();
}

void HostGPUControl::buffer_data(char val) { 
  if (val == '\0') return;
  if (Config::instance().isQuick()) {
    std::cout << val << std::flush;
  } else {
    buf += val;
  }
}
std::string HostGPUControl::get_buffer() { return buf; }

void HostGPUControl::set_stat_value(unsigned val) { stat_value = val; }
unsigned HostGPUControl::get_stat_value() { return stat_value; }