#pragma once

#include "gpu/register_file.hpp"

class HostRegisterFile : public RegisterFile {
  /*
   * Decorator class for the CPU view of the Register File
   */
public:
  HostRegisterFile(RegisterFile *rf, int num_registers);
  int get_register(uint64_t warp_id, int thread, int reg, bool is_cpu = false) override;
  void set_register(uint64_t warp_id, int thread, int reg, int value, bool is_cpu = false) override;
  std::optional<int> get_csr(uint64_t warp_id, int thread, int csr) override;
  void set_csr(uint64_t warp_id, int thread, int csr, int value) override;
  void pretty_print(uint64_t warp_id) override;

private:
  RegisterFile *rf;
  int num_registers;
  std::vector<int> registers;
};