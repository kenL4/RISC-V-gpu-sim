#include "host_gpu_control.hpp"

#define NUM_LANES 32

HostGPUControl::HostGPUControl()
    : kernel_pc(0), arg_ptr(0), dims(0), gpu_active(false), buf("") {}
void HostGPUControl::set_scheduler(std::shared_ptr<WarpScheduler> scheduler) {
  this->scheduler = scheduler;
}

void HostGPUControl::set_pc(uint64_t pc) { this->kernel_pc = pc; }
void HostGPUControl::set_arg_ptr(uint64_t arg_ptr) { this->arg_ptr = arg_ptr; }
void HostGPUControl::set_dims(uint64_t dims) { this->dims = dims; }

uint64_t HostGPUControl::get_arg_ptr() { return arg_ptr; }

void HostGPUControl::launch_kernel() {
  int warp_size = NUM_LANES;
  int num_warps = (dims + warp_size - 1) / warp_size;

  for (int i = 0; i < num_warps; i++) {
    Warp *warp = new Warp(i, warp_size, kernel_pc, false);
    scheduler->insert_warp(warp);
  }
  gpu_active = true;
  std::cerr << "[HostGPUControl] Launched kernel with " +
                   std::to_string(num_warps) + " warps"
            << std::endl;
  scheduler->set_active(true);
}

bool HostGPUControl::is_gpu_active() {

  if (gpu_active) {
    if (!scheduler->is_active()) {
      return scheduler->is_active();
    }
    return true;
  }
  return false;
}

void HostGPUControl::buffer_data(char val) {
  buf += val;
}
std::string HostGPUControl::get_buffer() {
  return buf;
}