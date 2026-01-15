#include "gpu/pipeline_warp_scheduler.hpp"
#include "utils.hpp"

class HostGPUControl {
public:
  HostGPUControl();
  void set_scheduler(std::shared_ptr<WarpScheduler> scheduler);

  // Kernel config
  void set_pc(uint64_t pc);
  void set_arg_ptr(uint64_t ptr);
  void set_dims(uint64_t dims);  // Deprecated/unused - kept for compatibility
  void set_warps_per_block(unsigned n);  // Set warps per block for barrier synchronization

  // GPU-side accessors
  uint64_t get_arg_ptr();

  // Control
  void launch_kernel();
  bool is_gpu_active();
  void set_pipeline(Pipeline *p) { pipeline = p; }

  // I/O
  void buffer_data(char val);
  std::string get_buffer();

  // Statistics
  void set_stat_value(unsigned val);
  unsigned get_stat_value();

private:
  std::shared_ptr<WarpScheduler> scheduler;
  Pipeline *pipeline = nullptr;
  uint64_t kernel_pc;
  uint64_t arg_ptr;
  uint64_t dims;
  bool gpu_active;

  std::string buf;
  unsigned stat_value = 0;  // Value for SIMTGet CSR (0x825)
};