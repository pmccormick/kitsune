#include "kitsune/timer.h"

#include <cuda_runtime.h>
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

const size_t VEC_SIZE = 1024 * 1024 * 256;

enum PrefetchKinds {
  EXPLICIT = 0,     // Use explicit async prefetch calls. 
  PRELAUNCH = 1,    // Prelaunch the kernel to move pages to device. 
  NONE = 2          // Do nothing, default to built-in page management. 
};

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
  PrefetchKinds PFKind = NONE;

  if (argc > 1 ) {
    size = atol(argv[1]);
    if (argc == 3) {
      if (std::string(argv[2]) == "explicit")
        PFKind = EXPLICIT;
      else if (std::string(argv[2]) == "pre-launch")
        PFKind = PRELAUNCH;
      else
        PFKind = NONE;
    }
  }

  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

  cudaEvent_t start;
  cudaEventCreate(&start);
  cudaEventRecord(start);
  cudaError_t err = cudaSuccess;
  float *A, *B, *C;
  err = cudaMallocManaged(&A, size * sizeof(float));
  if (err != cudaSuccess) {
    fprintf(stderr, "failed to allocate managed memory for A!\n");
    return 1;
  }
  err = cudaMallocManaged(&B, size * sizeof(float));
  if (err != cudaSuccess) {
    fprintf(stderr, "failed to allocate managed memory for B!\n");
    return 1;
  }
  err = cudaMallocManaged(&C, size * sizeof(float));
  if (err != cudaSuccess) {
    fprintf(stderr, "failed to allocate managed memory for C!\n");
    return 1;
  }

  if (PFKind == EXPLICIT)
    cudaMemPrefetchAsync(C, sizeof(float) * size, 0, nullptr);

  random_fill(A, size);
  if (PFKind == EXPLICIT)
    cudaMemPrefetchAsync(A, sizeof(float) * size, 0, nullptr);

  random_fill(B, size);
  if (PFKind == EXPLICIT)
    cudaMemPrefetchAsync(B, sizeof(float) * size, 0, nullptr);

  if (PFKind == PRELAUNCH) {
    // prime the GPU...  This will move all data to the device
    // prior to the timed launch below....
    VectorAdd<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, size);
    //cudaDeviceSynchronize();
  }

  cudaEvent_t kstart, kstop;
  cudaEventCreate(&kstart);
  cudaEventCreate(&kstop);


  cudaEventRecord(kstart);
  VectorAdd<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, size);
  cudaEventRecord(kstop);
  cudaEventSynchronize(kstop);

  float msecs = 0;
  cudaEventElapsedTime(&msecs, kstart, kstop);
  printf("%.8g\n", msecs / 1000.0);

  cudaEventElapsedTime(&msecs, start, kstop);
  fprintf(stderr, "%.8lg\n", msecs / 1000.0);

  // Sanity check the results...
  size_t error_count = 0;
  for (size_t i = 0; i < size; i++) {
    float sum = A[i] + B[i];
    if (C[i] != sum)
      error_count++;
  }

  if (error_count != 0)
    printf("bad result!\n");

  return 0;
}
