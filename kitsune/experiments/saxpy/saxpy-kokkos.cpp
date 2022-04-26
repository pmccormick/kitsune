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
#include "kitsune/timer.h"
#include "kitsune/llvm-gpu-abi/llvm-gpu.h"
#include "kitsune/llvm-gpu-abi/kitrt-cuda.h"

#include "Kokkos_DualView.hpp"

typedef Kokkos::DualView<float*, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace> SaxpyDualView;
using namespace std;


const size_t DEFAULT_SIZE = 1 << 26;

// Unlike kitsune+tapir Kokkos has no way to handle these values as
// global constants with a dynamic allocation/assignment.  As a result
// we have to stuff in some CUDA-centric pieces; perhaps Kokkos has
// some equivalent to this that is similar to KOKKOS_INLINE_FUNCTION?
// For now we will just use CUDA syntax to do this... 
__managed__ __device__ float DEFAULT_X_VALUE;
__managed__ __device__ float DEFAULT_Y_VALUE;
__managed__ __device__ float DEFAULT_A_VALUE;

bool check_saxpy(const SaxpyDualView &v, size_t N) {
  float err = 0.0f;
  for(size_t i = 0; i < N; i++) {
    err = err + fabs(v.h_view(i) - (DEFAULT_A_VALUE * DEFAULT_X_VALUE + DEFAULT_Y_VALUE));
  }
  fprintf(stderr, "Error: %f\n", err);
  return err == 0.0f;
}

int main(int argc, char *argv[]) {
  int retval;

  // We must initialize these here -- they will be paged to the
  // GPU given they are managed...
  DEFAULT_X_VALUE = rand() % 1000000;
  DEFAULT_Y_VALUE = rand() % 1000000;
  DEFAULT_A_VALUE = rand() % 1000000;
  
  size_t N = DEFAULT_SIZE;
  if (argc > 1) 
    N = atol(argv[1]);
  
  Kokkos::initialize(argc, argv); {
    SaxpyDualView x = SaxpyDualView("x", N);
    SaxpyDualView y = SaxpyDualView("y", N);

    x.modify_device();
    y.modify_device();
    kitsune::timer t;    
    Kokkos::parallel_for("init", N, KOKKOS_LAMBDA(const int &i) {
	x.d_view(i) = DEFAULT_X_VALUE;
	y.d_view(i) = DEFAULT_Y_VALUE;
      });
    Kokkos::fence();
    double loop_secs = t.seconds();
    std::cout << loop_secs << std::endl;

    t.reset();
    y.modify_device();
    Kokkos::parallel_for("saxpy", N, KOKKOS_LAMBDA(const int &i) {
	y.d_view(i) = DEFAULT_A_VALUE * x.d_view(i) + y.d_view(i);
      });
    Kokkos::fence();
    loop_secs = t.seconds();
    std::cout << loop_secs << std::endl;
    y.sync_host();
    
    if (! check_saxpy(y, N)) 
      retval = 1;
    else
      retval = 0;
  }
  Kokkos::finalize();
  return retval;
}

