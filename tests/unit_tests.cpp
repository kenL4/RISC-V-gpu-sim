#include "test_host.hpp"
#include "test_memory.hpp"
#include <iostream>

int main() {
  std::cout << "Starting Unit Tests..." << std::endl;

  test_data_memory_load_store();
  test_instr_memory();
  test_coalesce_latency();

  test_host_register_file();
  test_host_gpu_control();

  std::cout << "All tests passed!" << std::endl;
  return 0;
}
