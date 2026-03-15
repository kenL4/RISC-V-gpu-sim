#include <NoCL.h>
#include <Rand.h>

// Image thresholding without warp compaction
// Pixels above threshold undergo iterative decay (variable work),
// pixels below threshold are zeroed out.
// The divergent iteration counts cause warp inefficiency.
struct ThresholdNaive : Kernel {
  int len;
  int *input, *output;
  int threshold;

  INLINE void kernel() {
    for (int i = threadIdx.x; i < len; i += blockDim.x) {
      int val = input[i];
      if (val > threshold) {
        // Iterative decay: number of iterations depends on pixel value
        // This creates divergence within a warp since pixels have
        // different values, causing some threads to iterate longer
        int result = val;
        while (result > threshold) {
          result = (result * 200) >> 8;
        }
        output[i] = result;
      } else {
        output[i] = 0;
      }
    }
  }
};

int main()
{
  // Are we in simulation?
  bool isSim = getchar();

  // Image size for benchmarking
  int N = isSim ? 1024 : 100000;

  // Input and output images
  simt_aligned int input[N], output[N];

  // Initialise input with pseudo-random grayscale values (0-255)
  uint32_t seed = 42;
  for (int i = 0; i < N; i++) {
    input[i] = rand15(&seed) & 0xff;
  }

  int threshold = 128;

  // Instantiate kernel
  ThresholdNaive k;

  // Use single block of threads
  k.blockDim.x = SIMTLanes * SIMTWarps;

  // Assign parameters
  k.len = N;
  k.input = input;
  k.output = output;
  k.threshold = threshold;

  // Invoke kernel
  noclRunKernelAndDumpStats(&k);

  // Check result
  bool ok = true;
  for (int i = 0; i < N; i++) {
    int val = input[i];
    int expected;
    if (val > threshold) {
      expected = val;
      while (expected > threshold) {
        expected = (expected * 200) >> 8;
      }
    } else {
      expected = 0;
    }
    ok = ok && output[i] == expected;
  }

  // Display result
  puts("Self test: ");
  puts(ok ? "PASSED" : "FAILED");
  putchar('\n');

  return 0;
}
