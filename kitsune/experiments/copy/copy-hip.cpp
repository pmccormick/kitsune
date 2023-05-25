#include <float.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>
#include "hip/hip_runtime.h"

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

__global__ void Fill(float *A, size_t N) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N)
    A[i] = i / 100.0f;
}

__global__ void Copy(float *A, float *B, size_t N) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N)
    A[i] = B[i];
}


int main(int argc, char *argv[]) {
  using namespace std;
  size_t array_size = 1024 * 1024 * 256;
  unsigned int iterations = 10;
  if (argc >= 2)
    array_size = atol(argv[1]);
  if (argc == 3)
    iterations = atoi(argv[2]);

  hipDeviceProp_t devProp;
  hipGetDeviceProperties(&devProp, 0);
  cout << setprecision(5);
    
  cout << "\n";
  cout << "---- Simple copy benchmark (forall) ----\n"
       << "  Array size: " << array_size << "\n"
       << "  Iterations: " << iterations << "\n\n";
  cout << "Allocating arrays and filling with random values..." << std::flush;
  hipError_t err = hipSuccess;
  float *A, *B, *C, *D;
  unsigned int mb_size = (sizeof(float) * array_size) / (1024 * 1024);
  int threadsPerBlock = 256;
  int blocksPerGrid = (array_size + threadsPerBlock - 1) / threadsPerBlock;  
  HIPCHECK(hipMallocManaged(&A, array_size * sizeof(float)));
  HIPCHECK(hipMallocManaged(&B, array_size * sizeof(float)));
  HIPCHECK(hipMallocManaged(&C, array_size * sizeof(float)));
  HIPCHECK(hipMallocManaged(&D, array_size * sizeof(float)));    
  random_fill(A, array_size);
  random_fill(C, array_size);  
  cout << endl;

  hipMemPrefetchAsync(A, array_size * sizeof(float), 0, 0);
  hipMemPrefetchAsync(C, array_size * sizeof(float), 0, 0);  
  hipLaunchKernelGGL(Copy, blocksPerGrid, threadsPerBlock, 0, 0, C, D, array_size);

  double total_copy_time = 0.0;  
  double min_time = 100000.0;
  double max_time = 0.0;  
  auto start_time = chrono::steady_clock::now();
  for(int i = 0; i < iterations; i++) {  
    auto copy_start_time = chrono::steady_clock::now();
    hipMemPrefetchAsync(A, array_size * sizeof(float), 0, 0);
    hipMemPrefetchAsync(B, array_size * sizeof(float), 0, 0);
    hipLaunchKernelGGL(Copy, blocksPerGrid, threadsPerBlock, 0, 0, A, B, array_size);
    HIPCHECK(hipDeviceSynchronize());
    auto copy_end_time = chrono::steady_clock::now();    
    auto elapsed_time =
      chrono::duration<double>(copy_end_time-copy_start_time).count();
    if (elapsed_time < min_time)
      min_time = elapsed_time;
    if (elapsed_time > max_time)
      max_time = elapsed_time;
    cout  << "\t" << i << ". copy time: "
          << elapsed_time 
          << " sec., " << mb_size / elapsed_time << " MB/sec.\n";
    total_copy_time += elapsed_time;    
  }
  auto end_time = chrono::steady_clock::now();
  
  // Sanity check the results...
  size_t error_count = 0;
  for (size_t i = 0; i < array_size; i++) {
    if (A[i] != B[i])
    error_count++;
  }

  if (error_count != 0) {
    printf("bad result!\n");
    return 1;
  } else {
    cout << "pass (copy identical)\n";
  }

  cout << "Total time: "
       << chrono::duration<double>(end_time-start_time).count()
       << endl;
  cout << "Average copy time: "
       << total_copy_time / iterations
       << endl;
  cout << "*** " << min_time << ", " << max_time << "\n";
  cout << "----\n\n";
  return 0;
}
