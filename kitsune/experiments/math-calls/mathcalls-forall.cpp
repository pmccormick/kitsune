// See the README file for details.
#include <iostream>
#include <fstream>
#include <chrono>
#include <kitsune.h>
#include <cmath>
#include "kitsune/timer.h"
#include "kitrt/cuda/cuda.h"
#include "kitrt/memory_map.h"

using namespace std;
using namespace kitsune;

const size_t DEFAULT_ARRAY_SIZE = 1024 * 1024 * 128;

template <typename T>
T* alloc(int N) {
  return (T*)__kitrt_cuMemAllocManaged(sizeof(T) * N);
}

template <typename T>
void dealloc(T* array) {
  __kitrt_cuMemFree((void*)array);
}

template <typename T>
void random_fill(T* data, size_t N) {
  for(size_t i = 0; i < N; ++i)
    data[i] = 3.14 * (rand() / (T)RAND_MAX);
  __kitrt_memNeedsPrefetch(data);
}

template <typename T>
void parallel_work(T* dst, const T* src, int N) {
  forall(int i = 0; i < N; i++) {
    dst[i] = cos(2.0f/3.14 - src[i]) * ((sin(src[i]) * sin(src[i])) + (cos(src[i]) * cos(src[i])));
  } 
}


int main(int argc, char** argv) {
  size_t array_size = DEFAULT_ARRAY_SIZE;
  if (argc >= 2)
    array_size = atol(argv[1]);
  fprintf(stdout, "array size: %ld\n", array_size);
  float *data0 = alloc<float>(array_size);
  float *data1 = alloc<float>(array_size);
  random_fill(data0, array_size);
  auto start = chrono::steady_clock::now();
  parallel_work(data1, data0, array_size);
  auto end = chrono::steady_clock::now();
  cout  << "Execution time: "
          << chrono::duration<double>(end-start).count()
          << endl;
  return 0;
}
