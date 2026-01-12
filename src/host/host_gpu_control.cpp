#include "host_gpu_control.hpp"
#include "config.hpp"

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
  int num_warps = NUM_WARPS;

  for (int i = 0; i < num_warps; i++) {
    Warp *warp = new Warp(i, warp_size, kernel_pc, false);
    scheduler->insert_warp(warp);
  }
  gpu_active = true;
  if (!Config::instance().isStatsOnly()) std::cout << "[HostGPUControl] Launched kernel with " +
                   std::to_string(num_warps) + " warps"
            << std::endl;
  scheduler->set_active(true);
  // Set pipeline_active = true when kernel launches (matching SIMTight)
  if (pipeline != nullptr) {
    pipeline->set_pipeline_active(true);
  }
}

bool HostGPUControl::is_gpu_active() {
  return gpu_active && scheduler->is_active();
}

void HostGPUControl::buffer_data(char val) { buf += val; }
std::string HostGPUControl::get_buffer() { return buf; }