#include <NoCL.h>
#include <Rand.h>

// Vector addition with uncoalesced (strided) memory access
#define STRIDE 32

struct VecAddUncoalesced : Kernel {
  int numThreads;
  int *a, *b, *result;

  INLINE void kernel() {
    int tid = threadIdx.x;
    for (int i = tid * STRIDE; i < numThreads * STRIDE; i += blockDim.x * STRIDE)
      result[i] = a[i] + b[i];
  }
};

int main()
{
  // Are we in simulation?
  bool isSim = getchar();

  // Number of threads worth of work
  int numThreads = isSim ? 1024 : 100000;

  // Total array size: numThreads * STRIDE to spread accesses out
  int totalSize = numThreads * STRIDE;

  // Input and output vectors
  simt_aligned int a[totalSize], b[totalSize], result[totalSize];

  // Initialise only the elements that will be accessed
  uint32_t seed = 1;
  for (int i = 0; i < numThreads; i++) {
    int idx = i * STRIDE;
    a[idx] = rand15(&seed) & 0xff;
    b[idx] = rand15(&seed) & 0xff;
    result[idx] = 0;
  }

  // Instantiate kernel
  VecAddUncoalesced k;

  // Use single block of threads
  k.blockDim.x = SIMTLanes * SIMTWarps;

  // Assign parameters
  k.numThreads = numThreads;
  k.a = a;
  k.b = b;
  k.result = result;

  // Invoke kernel
  noclRunKernelAndDumpStats(&k);

  // Check result
  bool ok = true;
  for (int i = 0; i < numThreads; i++) {
    int idx = i * STRIDE;
    ok = ok && result[idx] == (a[idx] + b[idx]);
  }

  // Display result
  puts("Self test: ");
  puts(ok ? "PASSED" : "FAILED");
  putchar('\n');

  return 0;
}
