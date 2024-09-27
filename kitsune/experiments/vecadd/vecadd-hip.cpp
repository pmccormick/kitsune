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
#include <chrono>

#include "hip/hip_runtime.h"

using namespace std;

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
  unsigned int iterations = 10;
  
  if (argc >= 2 )
    size = atol(argv[1]);
  if (argc == 3)
    iterations = atoi(argv[2]);
  
  hipDeviceProp_t devProp;
  HIPCHECK(hipGetDeviceProperties(&devProp, 0));

  hipError_t err = hipSuccess;
  cout << setprecision(5);
  cout << "\n";
  cout << "---- vector addition benchmark (hip) ----\n"
       << "  Vector size: " << size << " elements.\n"
       << "  Interations: " << iterations << "\n\n";    
  cout << "  Allocating arrays and filling with random values...";
  
  float *A, *B, *C;
  HIPCHECK(hipMallocManaged(&A, size * sizeof(float)));
  HIPCHECK(hipMallocManaged(&B, size * sizeof(float))); 
  HIPCHECK(hipMallocManaged(&C, size * sizeof(float)));
  random_fill(A, size);
  random_fill(B, size);
  cout << "  done.\n\n";
  
  double elapsed_time;
  double min_time = 100000.0;
  double max_time = 0.0;
  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
  HIPCHECK(hipMemPrefetchAsync(A, size * sizeof(float), 0, 0));
  HIPCHECK(hipMemPrefetchAsync(B, size * sizeof(float), 0, 0));
  HIPCHECK(hipMemPrefetchAsync(C, size * sizeof(float), 0, 0));
  for(unsigned t = 0; t < iterations; t++) {
    auto start_time = chrono::steady_clock::now();
    hipLaunchKernelGGL(VectorAdd, blocksPerGrid, threadsPerBlock,
		       0, 0, A, B, C, size);
    HIPCHECK(hipDeviceSynchronize());
    auto end_time = chrono::steady_clock::now();
    elapsed_time = chrono::duration<double>(end_time-start_time).count();
    if (elapsed_time < min_time)
      min_time = elapsed_time;
    if (elapsed_time > max_time)
      max_time = elapsed_time;
    cout << "\t" << t << ". iteration time: " << elapsed_time << ".\n";
  }
  
  cout << "  Checking final result..." << std::flush;
  size_t error_count = 0;
  for (size_t i = 0; i < size; i++) {
    float sum = A[i] + B[i];
    if (C[i] != sum)
      error_count++;
  }

  if (error_count != 0) {
    cout << "incorrect result! (" << error_count << "errors).\n";
    return 1;
  } else {
    cout << "  pass (answers match).\n\n"
         << "  Total time: " << elapsed_time
         << " seconds. (" << size / elapsed_time << " elements/sec.)\n"
         << "*** min time: " << min_time << ", max time: "
	 << max_time << "\n"      
         << "----\n\n";
  }

  return 0;
}
