# Simulating a RISC-V GPU
This project is a software simulator for a RISC-V GPU model, intended to be used a substrate to test and develop GPU kernels.

## Dependencies
We need a RISC-V compiler, and LLVM for disassembly.<br>
On Ubuntu 24.04, we can do:
```
sudo apt install gcc-riscv64-unknown-elf
```
You must also install LLVM 22 with ```RISCV``` as one of the targets and install it to ```PATH```. On Ubuntu, we can do:
```
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 22
```
You can find alternative ways to install the latest stable releases at the <a href="https://apt.llvm.org/">LLVM website</a>.<br>
Statically included dependencies: ```cxxopts, ELFIO```

## Getting started
Recursively clone the repo:
```git clone --recursive https://github.com/kenL4/RISC-V-gpu-sim.git```

Ensure that the ```LLVM_DIR``` environment variable points to the correct version:
```
export LLVM_DIR=/usr/lib/llvm-22/lib/cmake/llvm
```

Then, the simplest way to run and observe the stats for all included SIMTight example test kernels is just to run:
```
./run-samples.sh
```

If you want to run the simulator directly on a particular kernel, follow these instructions:

1. Build the simulator:
```bash
make
```
This will automatically create the `build/` directory, configure with CMake, and build the project.

2. Run the built simulator binary on a NoCL kernel:
```bash
# e.g.
./build/RISCVGpuSim ./Samples/VecAdd/app.elf
```

Note: You can just run the binary with no arguments to see what arguments and options are supported

## Running Unit Tests
To build and run the unit test suite:
```bash
make test
```