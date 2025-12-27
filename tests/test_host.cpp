#include "test_host.hpp"
#include "gpu/pipeline_warp_scheduler.hpp"
#include "host/host_gpu_control.hpp"
#include "host/host_register_file.hpp"
#include <cassert>
#include <iostream>
#include <memory>

void test_host_register_file() {
  std::cout << "Running test_host_register_file..." << std::endl;
  // Mock inner register file data
  // RegisterFile(size_t register_count, size_t thread_count)
  RegisterFile rf(32, 32);
  HostRegisterFile hrf(&rf, 32);

  // Test setting registers
  // HostRegisterFile ignores warp_id and thread arguments for its own storage
  // hrf expects LLVM register enums, not raw indices 0..31
  hrf.set_register(0, 0, llvm::RISCV::X1, 123); // x1 = 123
  assert(hrf.get_register(0, 0, llvm::RISCV::X1) == 123);

  // Verify it ignores warp/thread args by accessing with different ones
  assert(hrf.get_register(99, 99, llvm::RISCV::X1) == 123);

  // Test x0 is always 0
  hrf.set_register(0, 0, llvm::RISCV::X0, 999);
  assert(hrf.get_register(0, 0, llvm::RISCV::X0) == 0);

  // Test CSR delegation
  // CSRs should pass through to the underlying RF
  hrf.set_csr(0, 0, 0xABC, 555);
  assert(hrf.get_csr(0, 0, 0xABC) == 555);
  // Verify it actually went to the underlying RF
  assert(rf.get_csr(0, 0, 0xABC) == 555);

  std::cout << "test_host_register_file passed!" << std::endl;
}

void test_host_gpu_control() {
  std::cout << "Running test_host_gpu_control..." << std::endl;
  HostGPUControl ctrl;

  // Test setters/getters
  ctrl.set_dims(1024);
  ctrl.set_arg_ptr(0x8000);
  ctrl.set_pc(0x1000);

  assert(ctrl.get_arg_ptr() == 0x8000);

  // Test buffer
  ctrl.buffer_data('H');
  ctrl.buffer_data('i');
  assert(ctrl.get_buffer() == "Hi");

  // Test interaction with Scheduler
  // WarpScheduler(int warp_size, int warp_count, uint64_t start_pc, bool
  // start_active)
  auto scheduler = std::make_shared<WarpScheduler>(32, 8, 0x0, false);
  ctrl.set_scheduler(scheduler);

  // Launch kernel
  // Dims 1024, Warp size 32 -> 32 warps
  // WarpScheduler warp_count is 8. Wait, WarpScheduler warp_count limit is a
  // hardware constraints? host_gpu_control.cpp: launch_kernel calculates
  // num_warps based on dims/warp_size. loops num_warps and inserts into
  // scheduler.

  // We should mock logic? No, let's use real one.
  // If we launch 1024 threads (32 warps)
  ctrl.launch_kernel();

  assert(ctrl.is_gpu_active());
  assert(scheduler->is_active());

  std::cout << "test_host_gpu_control passed!" << std::endl;
}
