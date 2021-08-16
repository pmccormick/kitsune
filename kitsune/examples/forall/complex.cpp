//
// Copyright(c) 2020 Triad National Security, LLC
// All rights reserved.
//
// This file is part of the kitsune / llvm project.  It is released under
// the LLVM license.
// 
// 
// Example of operations over an array of complex numbers. 
// 
// To enable kitsune+tapir compilation add the flags to a standard 
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

const size_t VEC_SIZE = 1024 * 1024 * 256;

struct my_complex {
  float real;
  float img;
};

void random_fill(my_complex *data, size_t N) {
  for(size_t i = 0; i < N; ++i) { 
    data[i].real = rand() / (float)RAND_MAX;
    data[i].img = rand() / (float)RAND_MAX;
  }
}

int main (int argc, char* argv[]) {

  fprintf(stderr, "**** kitsune+tapir forall example: complex\n");

  my_complex *A = new my_complex[VEC_SIZE];
  my_complex *B = new my_complex[VEC_SIZE];
  my_complex *C = new my_complex[VEC_SIZE];

  random_fill(A, VEC_SIZE);
  random_fill(B, VEC_SIZE);
  
  timer t;
  forall(size_t i = 0; i < VEC_SIZE; ++i) {
    C[i].real = (A[i].real * B[i].real) - (A[i].img * B[i].img);
    C[i].img  = (A[i].real * B[i].img) - (A[i].img * B[i].real);
  }
  double loop_secs = t.seconds();
  fprintf(stderr, "(%s) %lf, %lf, %lf, %lf\n", 
          argv[0], C[0].real, C[0].img,
	        C[VEC_SIZE/4].real, C[VEC_SIZE/4].img);
  fprintf(stdout, "%lf\n", loop_secs);

  delete []A;
  delete []B;
  delete []C;

  return 0;
}
