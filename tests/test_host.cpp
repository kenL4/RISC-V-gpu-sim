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
  RegisterFile rf(32, 32);
  HostRegisterFile hrf(&rf, 32);
  
  hrf.set_register(0, 0, llvm::RISCV::X1, 123);
  assert(hrf.get_register(0, 0, llvm::RISCV::X1) == 123);

  // CPU ignores!!
  assert(hrf.get_register(99, 99, llvm::RISCV::X1) == 123);

  // x0 is constant test
  hrf.set_register(0, 0, llvm::RISCV::X0, 999);
  assert(hrf.get_register(0, 0, llvm::RISCV::X0) == 0);

  // csrs should pass through if consistent
  hrf.set_csr(0, 0, 0xABC, 555);
  assert(hrf.get_csr(0, 0, 0xABC) == 555);
  assert(rf.get_csr(0, 0, 0xABC) == 555);

  std::cout << "test_host_register_file passed!" << std::endl;
}

void test_host_gpu_control() {
  std::cout << "Running test_host_gpu_control..." << std::endl;
  HostGPUControl ctrl;

  // Test setters and get the arg ptr back
  ctrl.set_dims(1024);
  ctrl.set_arg_ptr(0x8000);
  ctrl.set_pc(0x1000);

  assert(ctrl.get_arg_ptr() == 0x8000);

  // Test buffer
  ctrl.buffer_data('H');
  ctrl.buffer_data('i');
  assert(ctrl.get_buffer() == "Hi");

  // Launch kernel
  ctrl.launch_kernel();

  assert(ctrl.is_gpu_active());
  assert(scheduler->is_active());

  std::cout << "test_host_gpu_control passed!" << std::endl;
}
