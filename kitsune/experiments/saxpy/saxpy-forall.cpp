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
#include <math.h>
#include <kitsune.h>
#include "kitsune/llvm-gpu-abi/llvm-gpu.h"
#include "kitsune/llvm-gpu-abi/kitrt-cuda.h"

using namespace std;

const size_t DEFAULT_SIZE = 1 << 26;
const float DEFAULT_X_VALUE = rand() % 1000000;
const float DEFAULT_Y_VALUE = rand() % 1000000;
const float DEFAULT_A_VALUE = rand() % 1000000;

bool check_saxpy(const float *v, size_t N) {
  float err = 0.0f;
  for(size_t i = 0; i < N; i++) {
    err = err + fabs(v[i] - (DEFAULT_A_VALUE * DEFAULT_X_VALUE + DEFAULT_Y_VALUE));
  }
  fprintf(stderr, "Error: %f\n", err);
  return err == 0.0f;
}

int main(int argc, char *argv[]) {
  size_t N = DEFAULT_SIZE;
  if (argc > 1) 
    N = atol(argv[1]);
  
  float *x = (float*)__kitrt_cuMemAllocManaged(sizeof(float) * N);
  float *y = (float*)__kitrt_cuMemAllocManaged(sizeof(float) * N);

  __kitrt_cuEnableEventTiming();    
  forall(size_t i = 0; i < N; i++) {
    x[i] = DEFAULT_X_VALUE;
    y[i] = DEFAULT_Y_VALUE;
  }
  
  forall(size_t i = 0; i < N; i++) {
    y[i] = DEFAULT_A_VALUE * x[i] + y[i];
  }

  if (! check_saxpy(y, N)) 
    return 1;
  else
    return 0;
}

