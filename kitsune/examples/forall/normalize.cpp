//
// Copyright(c) 2020 Triad National Security, LLC
// All rights reserved.
//
// This file is part of the kitsune / llvm project.  It is released under
// the LLVM license.
//
//
#include <cstdio>
#include <cmath>
#include <kitsune.h>
#include "kitsune/timer.h"

using namespace std;
using namespace kitsune;

const size_t VEC_SIZE = 1024 * 1024 * 1024;

void random_fill(double *data, size_t N) {
  for(size_t i = 0; i < N; ++i)
    data[i] = (rand() / (double)RAND_MAX) * 10.0;
}

__attribute__((const, noinline)) double norm(const double *in, size_t N) {
  double sum = 0.0;
  for(size_t i = 0; i < N; ++i)
    sum += in[i] * in[i];
  return sqrt(sum);
}

int main (int argc, char* argv[]) {

  fprintf(stderr, "**** kitsune+tapir forall example: normalize (tapir paper)\n");

  double *in = new double[VEC_SIZE];
  double *out = new double[VEC_SIZE];

  random_fill(in, VEC_SIZE);
  
  timer t;  
  forall(size_t i = 0; i < VEC_SIZE; ++i) 
    out[i] = in[i] / norm(in, VEC_SIZE);
  double loop_secs = t.seconds();

  fprintf(stderr, "(%s) %lf, %lf, %lf, %lf\n", 
          argv[0], out[0], out[VEC_SIZE/4], out[VEC_SIZE/2], out[VEC_SIZE-1]); 
  fprintf(stdout, "%lf\n", loop_secs);

  delete []in;
  delete []out;

  return 0;
}
