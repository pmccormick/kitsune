//
// Copyright(c) 2020 Triad National Security, LLC
// All rights reserved.
//
// This file is part of the kitsune / llvm project.  It is released under
// the LLVM license.
//
// Simple example of an element-wise vector sum.
// To enable kitsune+tapir compilation add the flags to a standard
// clang compilation:
//
//    * -ftapir=rt-target : the runtime ABI to target.
//
#include <cstdio>
#include <stdlib.h>
#include <string>
#include <kitsune.h>
#include "kitsune/llvm-gpu-abi/llvm-gpu.h"
#include "kitsune/llvm-gpu-abi/kitrt-cuda.h"

using namespace std;

const size_t VEC_SIZE = 1024 * 1024 * 256;

enum PrefetchKinds {
  EXPLICIT = 0,  // Use explicit async prefetch calls.
  PRELAUNCH = 1, // Prelaunch the kernel to move pages to device.
  NONE = 2       // Do nothing, default to built-in page management.
};

void random_fill(float *data, size_t N) {
  for(size_t i = 0; i < N; ++i)
    data[i] = rand() / (float)RAND_MAX;
}

void fill(float *data, size_t N) {
   for(size_t i = 0; i < N; ++i)
     data[i] = float(i);
}

int main (int argc, char* argv[]) {
  size_t size = VEC_SIZE;
  PrefetchKinds PFKind = NONE;

  if (argc > 1) {
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

  void* Estart = __kitrt_cuCreateEvent();
  void* Eend   = __kitrt_cuCreateEvent();
  __kitrt_cuRecordEvent(Estart);
  float *A = (float *)__kitrt_cuMemAllocManaged(sizeof(float) * size);
  float *B = (float *)__kitrt_cuMemAllocManaged(sizeof(float) * size);
  float *C = (float *)__kitrt_cuMemAllocManaged(sizeof(float) * size);


  if (PFKind == EXPLICIT)
    __kitrt_cuMemPrefetchAsync(C, sizeof(float)*size);
  random_fill(A, size);

  if (PFKind == EXPLICIT)
    __kitrt_cuMemPrefetchAsync(A, sizeof(float)*size);
  random_fill(B, size);

  if (PFKind == EXPLICIT)
    __kitrt_cuMemPrefetchAsync(B, sizeof(float)*size);

  if (PFKind == PRELAUNCH) {
    // prime the GPU...  This will move all data to the device
    // prior to the timed launch below....
    forall(size_t i = 0; i < size; i++)
        C[i] = A[i] + B[i];
  }

  __kitrt_cuEnableEventTiming();
  forall(size_t i = 0; i < size; i++)
      C[i] = A[i] + B[i];
  __kitrt_cuDisableEventTiming();
  //__kitrt_cuStreamSynchronize(nullptr);

  __kitrt_cuRecordEvent(Eend);
  __kitrt_cuSynchronizeEvent(Eend);
  double etime = __kitrt_cuElapsedEventTime(Estart, Eend);
  fprintf(stderr, "%7lg\n", etime);

  // Sanity check the results...
  size_t error_count = 0;
  for(size_t i = 0; i < size; i++) {
    float sum = A[i] + B[i];
    if (C[i] != sum)
      error_count++;
  }

  if (error_count > 0)
    printf("bad result!\n");

  return 0;
}
