# Simulating a RISC-V GPU
This project is a software simulator for a RISC-V GPU model, intended to be used a substrate to test and develop GPU kernels.

## Dependencies
We need a RISC-V compiler.<br>
On Ubuntu 24.04, we can do:
```sudo apt install gcc-riscv64-unknown-elf```

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
cd examples/
make bin
cd ../build
./RISCVGpuSim examples/kernels/example1.o
```