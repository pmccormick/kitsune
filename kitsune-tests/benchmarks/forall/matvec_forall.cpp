// 
// Non-square matrix multiplication example. To enable 
// kitsune+tapir compilation add the flags to a standard 
// clang compilation: 
//
//    * -fkokkos : enable specialized Kokkos recognition and 
//                 compilation (lower to Tapir).
//    * -fkokkos-no-init : disable Kokkos initialization and 
//                 finalization calls to avoid conflicts with
//                 target runtime operation. 
//    * -ftapir=rt-target : the runtime ABI to target. 
// 
#include <cstdio>
#include <cstdlib>
#include <kitsune.h>

#include "timer.h"

using namespace std;
using namespace kitsune;

const size_t N = 8192;

void random_fill(float *data, size_t N) {
  for(size_t i = 0; i < N; ++i) 
    data[i] = rand() / (float)RAND_MAX;
}

void zero_fill(float *data, size_t N) {
  for(size_t i = 0; i < N; ++i) 
    data[i] = 0.0f;
}

int main (int argc, char* argv[]) {

  fprintf(stderr, "**** kitsune+tapir kokkos example: matrix-vector multiply\n");

  float *matrix = new float[N*N];
  float *vector = new float[N];
  float *result = new float[N*N];

  random_fill(matrix, N*N);
  random_fill(vector, N);
  zero_fill(result, N*N);
  
  double loop_secs = 0;
  for (int ii = 0; ii<4; ii++) {
      timer t;  
      {
        forall (int i = 0; i<N; i++) {
          forall (int j = 0; j<N; j++) {
            result[i*N + j] += matrix[i*N + j] * vector[i];
          }
        }
      }
      loop_secs += t.seconds();
  }
  loop_secs /= 4;

  fprintf(stderr, "(%s) %lf, %lf, %lf, %lf\n", 
         argv[0], result[0], result[(N*N)/4], result[(N*N)/2], result[(N*N)-1]);     
  fprintf(stdout, "Time: %lf\n", loop_secs);

  delete []matrix;
  delete []vector;
  delete []result;

  return 0;
}
