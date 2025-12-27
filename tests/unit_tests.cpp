#include "test_memory.hpp"
#include <iostream>

int main() {
  std::cout << "Starting Unit Tests..." << std::endl;

  test_data_memory_load_store();
  test_instr_memory();
  test_coalesce_latency();

  std::cout << "All tests passed!" << std::endl;
  return 0;
}
