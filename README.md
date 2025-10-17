# Simulating a RISC-V GPU
This project is a software simulator for a RISC-V GPU model, intended to be used a substrate to test and develop GPU kernels.

## Dependencies
We need a RISC-V compiler, and Capstone.<br>
On Ubuntu 24.04, we can do:
```
sudo apt install gcc-riscv64-unknown-elf \
                 libcapstone-dev
```
Older versions of Capstone might not support RISC-V so you may have to build from [source](https://github.com/capstone-engine/capstone.git).<br>
Statically included dependencies: ```cxxopts, ELFIO```

## Getting started
Recursively clone the repo:
```git clone --recursive https://github.com/kenL4/RISC-V-gpu-sim.git```

To build the simulator:
```
mkdir build && cd build
cmake ..
cmake --build .
```

To run an example binary on the simulator:
```
cd examples/example1
make all
cd ../../build
./RISCVGpuSim examples/example1/kernel
```