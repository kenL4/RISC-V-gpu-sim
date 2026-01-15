/*
 Render a gradient to a framebuffer
 Each pixel is a 32-bit integer with RGB format:
 Bits 0-7:   Blue
 Bits 8-15:  Green
 Bits 16-23: Red
 Bits 24-31: Unused (can be used for alpha)
 The framebuffer will be 64x64 pixels.
*/

#include <NoCL.h>

// Kernel for rendering a gradient to framebuffer
struct GradientKernel : Kernel {
  int width, height;
  int *framebuffer;

  INLINE void kernel() {
    // Calculate pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Only process valid pixels
    if (x < width && y < height) {
      // Calculate pixel index
      int idx = y * width + x;
      
      // Create a gradient: red increases with x, green increases with y
      // Blue is constant (or can vary with both)
      int r = (x * 255) / (width - 1);   // Red: 0 to 255 across width
      int g = (y * 255) / (height - 1);   // Green: 0 to 255 across height
      int b = 128;                        // Blue: constant
      
      // Pack RGB into 32-bit integer: 0xRRGGBB00
      int color = (r << 16) | (g << 8) | b;
      
      // Write to framebuffer
      framebuffer[idx] = color;
    }
  }
};

bool check_output(int *out_buf, int width, int height) {
  for (int i = 0; i < width * height; ++i) {
    int r = (i % width) * 255 / (width - 1);
    int g = (i / width) * 255 / (height - 1);
    int b = 128;
    int color = (r << 16) | (g << 8) | b;
    if (out_buf[i] != color) {
      puts("Detected an error at index: "); puthex(i); putchar('\n');
      puts("Expected value: "); puthex(i); putchar('\n');
      puts("Computed value: "); puthex(out_buf[i]);    putchar('\n');
      return false;
    }
  }
  return true;
}

int main() {
  // Are we in simulation?
  bool isSim = getchar();

  // Framebuffer dimensions (64x64 for debugging)
  int width = 64;
  int height = 64;
  
  // Allocate framebuffer in global memory (aligned for SIMT)
  nocl_aligned int framebuffer[width * height];

  // Initialize framebuffer to black
  for (int i = 0; i < width * height; i++) {
    framebuffer[i] = 0;
  }

  // Instantiate kernel
  GradientKernel k;
  
  // Set block dimensions (must be powers of 2)
  k.blockDim.x = SIMTLanes;  // 32 threads in X
  k.blockDim.y = 1;           // 1 thread in Y
  
  // Calculate grid dimensions
  k.gridDim.x = (width + k.blockDim.x - 1) / k.blockDim.x;  // Round up
  k.gridDim.y = height;  // One block per row
  
  // Assign parameters
  k.width = width;
  k.height = height;
  k.framebuffer = framebuffer;

  // Invoke kernel
  noclRunKernelAndDumpStats(&k);

  // Print framebuffer base address for debugging/memory dump
  puts("Framebuffer address: ");
  puthex((uint32_t)framebuffer);
  putchar('\n');
  puts("Framebuffer size: ");
  puthex(width * height * sizeof(int));
  putchar('\n');

  bool ok = check_output(framebuffer, width, height);
  puts("Self test: ");
  puts(ok ? "PASSED" : "FAILED");
  putchar('\n');

  return 0;
}