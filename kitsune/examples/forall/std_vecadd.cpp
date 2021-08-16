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
#include <cstddef>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <kitsune.h>
#include "kitsune/timer.h"

using namespace std;
using namespace kitsune;

const size_t VEC_SIZE = 1024 * 1024 * 256;

int main (int argc, char* argv[]) {

  vector<float> A(VEC_SIZE);
  vector<float> B(VEC_SIZE);
  vector<float> C(VEC_SIZE);

  for(auto i : A) {
    A[i] = rand() / (float)RAND_MAX;
    B[i] = rand() / (float)RAND_MAX;    
  }

  timer t;
  forall(auto i : C) {
    C[i] = A[i] + B[i];
  }
  double loop_secs = t.seconds();

  size_t ti = 0;
  for(; ti < VEC_SIZE; ++ti) {
    float sum = A[ti] + B[ti];
    if (fabs(C[ti] - sum) > 1e-7f) 
      break; // whoops...
  }
 
  fprintf(stdout, "Result = %s (%ld, %ld)\n",
	  (ti == VEC_SIZE) ? "PASS" : "FAIL",
	  ti, VEC_SIZE);
  fprintf(stdout, "%lf\n", loop_secs);  

  return 0;
}

