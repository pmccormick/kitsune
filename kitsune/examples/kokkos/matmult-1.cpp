// 
// 
#include <cstdio>
#include <kitsune/timer.h>
#include <Kokkos_Core.hpp>

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
  timer t;  
  float *A = new float[N*K];
  float *B = new float[K*M];
  float *C = new float[N*M];

  random_fill(A, N*K);
  random_fill(B, M*K);
  zero_fill(C, N*M);
  
  Kokkos::initialize (argc, argv);
  double secs = t.seconds();
  fprintf(stdout, "initialization time: %lf seconds.\n", secs);

  t.reset();
  {
    Kokkos::parallel_for(N, KOKKOS_LAMBDA(const int i) {
      for(size_t k = 0; k < K; ++k) 
        for(size_t j = 0; j < M; ++j) 
          C[i*M + j] += A[i*K + k] * B[k*M +j];
    });
  }
  double loop_secs = t.seconds();
  
  secs = t.seconds();

  fprintf(stdout, "seconds = %lf\n", loop_secs);
  delete []A;
  delete []B;
  delete []C;

  Kokkos::finalize();
  return 0;
}
