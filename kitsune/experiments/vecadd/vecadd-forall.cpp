#include <cstdio>
#include <stdlib.h>
#include <string>
#include <kitsune.h>
#include <cmath>
#include "kitsune/timer.h"
#include "__clang_hip_runtime_wrapper.h"

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

  kitsune::timer r;
  r.reset();
  float *A = alloc<float>(size);
  float *B = alloc<float>(size);
  float *C = alloc<float>(size);
  random_fill(A, size);
  random_fill(B, size);
  timer k;
  k.reset();
  timer f;
  for(int t = 0; t < 20; t++) {
    f.reset();
    forall(size_t i = 0; i < size; i++)
      C[i] = sqrtf(A[i]) + B[i];
    double ftime = f.seconds();
    fprintf(stdout, "   time: %7lg\n", ftime);
  }
  double ktime = k.seconds();
  fprintf(stdout, "loop time: %7lg\n", ktime/20.0);

  size_t error_count = 0;
  for (size_t i = 0; i < size; i++) {
    float sum = A[i] + B[i];
    if (C[i] != sum)
      error_count++;
  }
  dealloc(A);
  dealloc(B);
  dealloc(C);
  
  if (error_count > 0)
    fprintf(stderr, "bad result! %ld positions had errors!\n", error_count);
  else {
    double rtime = r.seconds();
    fprintf(stdout, "total runtime: %7lg\n", rtime);
  }
  return 0;
}
