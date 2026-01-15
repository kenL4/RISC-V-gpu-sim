# Framebuffer Gradient Renderer

This application renders a gradient to a framebuffer using the SIMT GPU, which can then be visualized in Python.

## Building

```bash
cd SIMTight/apps/InHouse/Framebuffer
make RunSim
```

This will create:
- `app.elf` - The compiled application
- `code.v` and `data.v` - Verilog memory files
- `RunSim` - Host program to run the simulation

## Running with RISC-V-gpu-sim

1. Build RISC-V-gpu-sim (if not already built):
```bash
cd RISC-V-gpu-sim/build
cmake ..
make
```

2. Note the framebuffer address from running the app (look for "Framebuffer address:" in the output)

3. Run the simulation with memory dump (specify the framebuffer region):
```bash
cd RISC-V-gpu-sim/build
# Calculate size: 64 * 64 * 4 = 16384 bytes (0x4000)
./RISCVGpuSim ../../SIMTight/apps/InHouse/Framebuffer/app.elf \
  --memdump framebuffer_memory.bin \
  --memdump-start 0x80020000 \
  --memdump-size 0x4000
```

Or use start and end addresses:
```bash
./RISCVGpuSim ../../SIMTight/apps/InHouse/Framebuffer/app.elf \
  --memdump framebuffer_memory.bin \
  --memdump-start 0x80020000 \
  --memdump-end 0x80024000
```

4. Visualize the framebuffer (address is read from dump header, so it's optional):
```bash
cd RISC-V-gpu-sim/python_utils
python visualize_framebuffer.py ../build/framebuffer_memory.bin 64 64
```

Or if you want to specify a different address:
```bash
python visualize_framebuffer.py ../build/framebuffer_memory.bin 64 64 0x80020000
```

This will create `framebuffer.png` with the rendered gradient.

## Framebuffer Format

- Each pixel is a 32-bit integer
- RGB format: `(r << 16) | (g << 8) | b`
- In memory (little-endian): bytes are stored as `[b, g, r, 0]`
- Default size: 64x64 pixels (for debugging)

## Gradient Pattern

The current kernel renders:
- Red: increases from 0 to 255 across the width (x-axis)
- Green: increases from 0 to 255 across the height (y-axis)  
- Blue: constant at 128

This creates a diagonal gradient from black (top-left) to yellow (bottom-right).
