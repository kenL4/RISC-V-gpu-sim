#include "gpu/pipeline_warp_scheduler.hpp"
#include "utils.hpp"

class HostGPUControl {
public:
  HostGPUControl();
  void set_scheduler(std::shared_ptr<WarpScheduler> scheduler);

  // Kerenl config
  void set_pc(uint64_t pc);
  void set_arg_ptr(uint64_t ptr);
  void set_dims(uint64_t dims);

  // GPU-side accessors
  uint64_t get_arg_ptr();

  // Control
  void launch_kernel();
  bool is_gpu_active();
  void set_pipeline(Pipeline *pipeline) { this->pipeline = pipeline; }

  // I/O
  void buffer_data(char val);
  std::string get_buffer();

private:
  std::shared_ptr<WarpScheduler> scheduler;
  Pipeline *pipeline = nullptr;
  uint64_t kernel_pc;
  uint64_t arg_ptr;
  uint64_t dims;
  bool gpu_active;

  std::string buf;
};