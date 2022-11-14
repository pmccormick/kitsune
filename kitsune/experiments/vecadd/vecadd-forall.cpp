#include <cstdio>
#include <stdlib.h>
#include <string>
#include <kitsune.h>
#include "kitsune/timer.h"

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

  float *A = alloc<float>(size);
  float *B = alloc<float>(size);
  float *C = alloc<float>(size);

  random_fill(A, size);
  random_fill(B, size);

  timer k;
  forall(size_t i = 0; i < size; i++)
    C[i] = A[i] + B[i];
  double ktime = k.seconds();

  // Sanity check the results...  We will take a hit here on 
  // page faults back on the CPU side.  This will show up on
  // the overall runtime of the program (e.g., the forall 
  // might be faster but the overhead of page faults will 
  // wipe that out). 
  // 
  
  size_t error_count = 0;
  for(size_t i = 0; i < size; i++) {
    float sum = A[i] + B[i];
    if (C[i] != sum) {
      error_count++;
    }
  }

  dealloc(A);
  dealloc(B);
  dealloc(C);
  if (error_count > 0)
    fprintf(stderr, "bad result!\n");
  else
    fprintf(stdout, "loop time: %7lg\n", ktime);
  return 0;
}
