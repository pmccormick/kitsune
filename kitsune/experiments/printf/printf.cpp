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
#include "kitsune/timer.h"
#include "kitrt/cuda/cuda.h"

using namespace std;
using namespace kitsune;

const size_t VEC_SIZE = 1024 * 1024 * 2;

int main (int argc, char* argv[]) {
  size_t size = VEC_SIZE;
  float *A = (float *)__kitrt_cuMemAllocManaged(sizeof(float) * size);

  forall(size_t i = 0; i < size; i++) {
    A[i] = 0.0;
    printf("i = %ld\n", i);
  }

  return 0;
}

