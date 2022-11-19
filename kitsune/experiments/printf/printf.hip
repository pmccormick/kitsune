#include <stdio.h>
#include <stdlib.h>
#include "hip/hip_runtime.h"

#define HIPCHECK(error) 				\
   if (error != hipSuccess) {				\
     printf("error: '%s' (%d) at %s:%d\n", 		\
	hipGetErrorString(error), error, __FILE__,      \
        __LINE__);					\
     exit(1);						\
   }							\

__global__ void print(size_t N) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N)
    printf("@#@!@#$!\n");
}


int main(int argc, char *argv[]) {
  size_t size = 1024;
  if (argc > 1 )
    size = atol(argv[1]);
  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
  hipLaunchKernelGGL(print, blocksPerGrid, threadsPerBlock, 0, 0, size);
  HIPCHECK(hipDeviceSynchronize());
  return 0;
}
