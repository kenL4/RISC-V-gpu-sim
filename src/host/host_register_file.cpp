#include "host_register_file.hpp"
#include "config.hpp"

HostRegisterFile::HostRegisterFile(RegisterFile *rf, int num_registers)
    : RegisterFile(0, 0), rf(rf), num_registers(num_registers) {
    int sp_value = static_cast<int>(SIM_CPU_INITIAL_SP);
    rf->set_register(0, 0, llvm::RISCV::X2, sp_value);
    
    if (registers.size() == 0) {
        registers.resize(num_registers);
        for (int i = 0; i < num_registers; i++) {
            registers[i] = 0;
        }
    }
    registers[2] = sp_value;
}

/*
 * This wrapper ignores the warp_id and thread arguments
 */
static int get_register_idx(llvm::MCRegister reg) {
  return reg - llvm::RISCV::X0;
}

int HostRegisterFile::get_register(uint64_t warp_id, int thread, int reg, bool is_cpu) {
  if (registers.size() <= 0) {
      registers.resize(num_registers);
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
                                    int value, bool is_cpu) {
  if (registers.size() <= 0) {
      registers.resize(num_registers);
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

std::optional<int> HostRegisterFile::get_csr(uint64_t warp_id, int thread, int csr) {
  return rf->get_csr(warp_id, thread, csr);
}
void HostRegisterFile::set_csr(uint64_t warp_id, int thread, int csr, int value) {
  rf->set_csr(warp_id, thread, csr, value);
}
void HostRegisterFile::pretty_print(uint64_t warp_id) {
  // also removed like the register file version
}