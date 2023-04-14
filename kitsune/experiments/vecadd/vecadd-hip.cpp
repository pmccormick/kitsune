#include <float.h>
#include <fstream>
#include <iostream>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>
#include "hip/hip_runtime.h"
#include "kitsune/timer.h"

#define HIPCHECK(error) 				\
   if (error != hipSuccess) {				\
     printf("error: '%s' (%d) at %s:%d\n", 		\
	hipGetErrorString(error), error, __FILE__,      \
        __LINE__);					\
     exit(1);						\
   }							\
     
const size_t VEC_SIZE = 1024 * 1024 * 256;

void random_fill(float *data, size_t N) {
  for (size_t i = 0; i < N; ++i)
    data[i] = rand() / (float)RAND_MAX;
}

__global__ void VectorAdd(float *A, float *B, float *C, size_t N) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N)
    C[i] = A[i] + B[i];
}


int main(int argc, char *argv[]) {
  size_t size = VEC_SIZE;
  if (argc > 1 )
    size = atol(argv[1]);
  fprintf(stdout, "problem size: %ld\n", size);

  hipDeviceProp_t devProp;
  hipGetDeviceProperties(&devProp, 0);


  hipError_t err = hipSuccess;
  float *A, *B, *C;
  HIPCHECK(hipMallocManaged(&A, size * sizeof(float)));
  HIPCHECK(hipMallocManaged(&B, size * sizeof(float))); 
  HIPCHECK(hipMallocManaged(&C, size * sizeof(float)));

  random_fill(A, size);
  random_fill(B, size);

  // This is loosely for consistency with the launch parameters
  // from kitsune.
  kitsune::timer r;  
  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
  kitsune::timer k;
  hipMemPrefetchAsync(A, size * sizeof(float), 0, 0);
  hipMemPrefetchAsync(B, size * sizeof(float), 0, 0);
  hipMemPrefetchAsync(C, size * sizeof(float), 0, 0);      
  hipLaunchKernelGGL(VectorAdd, blocksPerGrid, threadsPerBlock, 0, 0, A, B, C, size);
  HIPCHECK(hipDeviceSynchronize());
  double ktime = k.seconds(); 
  fprintf(stdout, "kernel: %7lg\n", ktime);

  // Sanity check the results...
  size_t error_count = 0;
  for (size_t i = 0; i < size; i++) {
    float sum = A[i] + B[i];
    if (C[i] != sum)
      error_count++;
  }

  if (error_count != 0)
    printf("bad result!\n");
  else {
    double rtime = r.seconds();
    fprintf(stdout, "total runtime: %7lg\n", rtime);
  }

  return 0;
}
