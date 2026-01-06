#include "host_register_file.hpp"

HostRegisterFile::HostRegisterFile(RegisterFile *rf, int num_registers)
    : RegisterFile(0, 0), rf(rf), num_registers(num_registers) {
    // For simulation purposes, we set SP(x2) to the STACK_BASE - 8
    rf->set_register(0, 0, llvm::RISCV::X2, SIM_CPU_INITIAL_SP);
}

/*
 * This wrapper ignores the warp_id and thread arguments
 */
static int get_register_idx(llvm::MCRegister reg) {
  return reg - llvm::RISCV::X0;
}

int HostRegisterFile::get_register(uint64_t warp_id, int thread, int reg) {

  if (registers.size() <= 0) {
    registers.resize(num_registers);
    // Clear values
    for (int i = 0; i < num_registers; i++) {
      registers[i] = 0;
    }
  }

  int idx = get_register_idx(reg);
  if (idx == 0 && registers[idx] != 0) {
    std::cerr << "[HostRF] x0 is corrupted! value=" << registers[idx]
              << std::endl;
  }
  return registers[idx];
}

void HostRegisterFile::set_register(uint64_t warp_id, int thread, int reg,
                                    int value) {

  if (registers.size() <= 0) {
    registers.resize(num_registers);
    // Clear values
    for (int i = 0; i < num_registers; i++) {
      registers[i] = 0;
    }
  }

  if (reg == llvm::RISCV::X0) {
    return;
  }

  int idx = get_register_idx(reg);
  if (idx < 0 || idx >= registers.size()) {
    return;
  }
  registers[idx] = value;
}

std::optional<int> HostRegisterFile::get_csr(uint64_t warp_id, int thread,
                                             int csr) {
  return rf->get_csr(warp_id, thread, csr);
}
void HostRegisterFile::set_csr(uint64_t warp_id, int thread, int csr,
                               int value) {
  rf->set_csr(warp_id, thread, csr, value);
}
void HostRegisterFile::pretty_print(uint64_t warp_id) {
  if(!Config::instance().isCPUDebug()) {
    return;
  }
  
  if (warp_id_to_registers.count(warp_id) == 0) {
    std::cout << "No registers for host" << std::endl;
    return;
  }

  // Print header: Thread IDs
  std::cout << std::setw(4) << "Host";
  std::cout << "\n";

  // Print separator
  std::cout << "----";
  std::cout << "\n";

  // Print each register
  for (size_t reg_idx = 0; reg_idx < registers.size(); ++reg_idx) {
    std::cout << std::setw(4) << ("x" + std::to_string(reg_idx));
    std::cout << std::setw(4) << registers[reg_idx];
    std::cout << "\n";
  }
}