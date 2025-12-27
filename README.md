# Simulating a RISC-V GPU
This project is a software simulator for a RISC-V GPU model, intended to be used a substrate to test and develop GPU kernels.

## Dependencies
We need a RISC-V compiler, and LLVM for disassembly.<br>
On Ubuntu 24.04, we can do:
```
sudo apt install gcc-riscv64-unknown-elf
```
You must also build LLVM with ```RISCV``` as one of the targets and install it to ```PATH```.<br>
Statically included dependencies: ```cxxopts, ELFIO```

## Getting started
Recursively clone the repo:
```git clone --recursive https://github.com/kenL4/RISC-V-gpu-sim.git```

To build the simulator:
```bash
make
```
This will automatically create the `build/` directory, configure with CMake, and build the project.
Alternatively, you can build manually using CMake:
```bash
mkdir build && cd build
cmake ..
make
```

## Running Tests
To build and run the unit test suite:
```bash
make test
```

## Running the Simulator
To run an example binary on the simulator:
```bash
# 1. Build the example kernel (requires RISC-V compiler)
cd examples/example1
make all
cd ../..

# 2. Run the simulator
# Usage: ./RISCVGpuSim <path_to_kernel_binary>
build/RISCVGpuSim examples/example1/kernel
```