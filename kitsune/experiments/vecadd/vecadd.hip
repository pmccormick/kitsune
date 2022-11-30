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
  std::cout << " System minor " << devProp.minor << std::endl;
  std::cout << " System major " << devProp.major << std::endl;
  std::cout << " agent prop name " << devProp.name << std::endl;

  kitsune::timer r;

  hipError_t err = hipSuccess;
  float *A, *B, *C;
  if (hipMallocManaged(&A, size * sizeof(float)) != hipSuccess) {
    fprintf(stderr, "failed to allocate managed memory for A!\n");
    return 1;
  }
  if (hipMallocManaged(&B, size * sizeof(float)) != hipSuccess) {
    fprintf(stderr, "failed to allocate managed memory for B!\n");
    return 1;
  }
  if (hipMallocManaged(&C, size * sizeof(float)) != hipSuccess) {
    fprintf(stderr, "failed to allocate managed memory for C!\n");
    return 1;
  }

  random_fill(A, size);
  random_fill(B, size);

  // This is loosely for consistency with the launch parameters
  // from kitsune.
  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
  kitsune::timer k;
  hipLaunchKernelGGL(VectorAdd, blocksPerGrid, threadsPerBlock, 0, 0, A, B, C, size);
  double ktime = k.seconds(); 

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
