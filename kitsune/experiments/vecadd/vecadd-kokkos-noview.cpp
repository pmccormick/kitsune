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
//    * -fkokkos : enable specialized Kokkos recognition and
//                 compilation (lower to Tapir).
//    * -fkokkos-no-init : disable Kokkos initialization and
//                 finalization calls to avoid conflicts with
//                 target runtime operation.
//    * -ftapir=rt-target : the runtime ABI to target.
//
#include "Kokkos_Core.hpp"
#include "Kokkos_DualView.hpp"
#include <kitsune.h>
#include "kitsune/timer.h"
#include <cstdio>



using namespace std;
using namespace kitsune;

const size_t VEC_SIZE = 1024 * 1024 * 256;

void random_fill(float *data, size_t N) {
  for(size_t i = 0; i < N; ++i)
    data[i] = rand() / (float)RAND_MAX;
}

int main (int argc, char* argv[]) {
  size_t size = VEC_SIZE;
  if (argc > 1)
    size = atol(argv[1]);

  fprintf(stdout, "problem size: %ld\n", size);

  Kokkos::initialize(argc, argv); {

    float *A = alloc<float>(size);
    float *B = alloc<float>(size);
    float *C = alloc<float>(size);

    random_fill(A, size);
    random_fill(B, size);

    fprintf(stdout, "running...\n");
    timer t;
    Kokkos::parallel_for(size, KOKKOS_LAMBDA(const int i) {
	C[i] = A[i] + B[i];
    });
    double loop_secs = t.seconds();
    
    fprintf(stdout, "checking result... ");
    // Sanity check the results...
    size_t error_count = 0;
    for (size_t i = 0; i < size; i++) {
      float sum = A[i] + B[i];
      if (C[i] != sum)
        error_count++;
    }

    if (error_count > 0)
      printf("bad result! %ld errors\n", error_count);
    else 
      fprintf(stdout, "correct.\nruntime: %7lg\n", loop_secs);

  } Kokkos::finalize();
  return 0;
}

