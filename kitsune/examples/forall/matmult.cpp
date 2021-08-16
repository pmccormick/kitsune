//
// Copyright(c) 2020 Triad National Security, LLC
// All rights reserved.
//
// This file is part of the kitsune / llvm project.  It is released under
// the LLVM license.
//
// Non-square matrix multiplication example. To enable
// kitsune+tapir compilation add the flags to a standard
// clang compilation:
//
//    * -ftapir=rt-target : the runtime ABI to target.
//
#include <cstdio>
#include <stdlib.h>
#include <kitsune.h>
#include "kitsune/timer.h"

using namespace std;
using namespace kitsune;

const size_t N = 8192;
const size_t M = 4096;
const size_t K = 512;

void random_fill(float *data, size_t N) {
  for(size_t i = 0; i < N; ++i) 
    data[i] = rand() / (float)RAND_MAX;
}

void zero_fill(float *data, size_t N) {
  for(size_t i = 0; i < N; ++i) 
    data[i] = 0.0f;
}

int main (int argc, char* argv[]) {

  fprintf(stderr, "**** kitsune+tapir forall example: matrix multiply\n");

  float *A = new float[N*K];
  float *B = new float[K*M];
  float *C = new float[N*M];

  random_fill(A, N*K);
  random_fill(B, M*K);
  zero_fill(C, N*M);
  
  timer t;  
  forall(size_t i = 0; i < N; i++) {
    for(size_t k = 0; k < K; ++k) 
      for(size_t j = 0; j < M; ++j) 
        C[i*M + j] += A[i*K + k] * B[k*M +j];
  }
  double loop_secs = t.seconds();

  fprintf(stderr, "(%s) %lf, %lf, %lf, %lf\n", 
         argv[0], C[0], C[(N*M)/4], C[(N*M)/2], C[(N*M)-1]);     
  fprintf(stdout, "%lf\n", loop_secs);

  delete []A;
  delete []B;
  delete []C;

  return 0;
}
