// 
// Very simple example of an element-wise vector sum using
// the Kokkos parallel_for construct.  To compile with 
// Kitsune use the -fkokkos -fkokkos-no-init and -ftapir=rt
// flags.  'rt' should be set to the target runtime you 
// want to use for parallel execution (e.g., opencilk).
// 
#include <cstdio>
#include <kitsune/timer.h>
#include <Kokkos_Core.hpp>

using namespace std;
using namespace kitsune;

const size_t VEC_SIZE = 1024 * 1024 * 256;

void random_fill(float *data, size_t N) {
  for(size_t i = 0; i < N; ++i) 
    data[i] = rand() / (float)RAND_MAX;
}

int main (int argc, char* argv[]) {
  timer t;  
  float *A = new float[VEC_SIZE];
  float *B = new float[VEC_SIZE];
  float *C = new float[VEC_SIZE];

  random_fill(A, VEC_SIZE);
  random_fill(B, VEC_SIZE);
  
  Kokkos::initialize (argc, argv);
  double secs = t.seconds();
  fprintf(stdout, "initialization time: %lf seconds.\n", secs);
  t.reset();
  {
    Kokkos::parallel_for(VEC_SIZE, KOKKOS_LAMBDA(const int i) {
        C[i] = A[i] + B[i];
    });
  }
  double loop_secs = t.seconds();
  
  t.reset();
  // Verify correct result...  
  size_t i = 0;
  for(; i < VEC_SIZE; ++i) {
    float sum = A[i] + B[i];
    if (fabs(C[i] - sum) > 1e-7f)
      break; // whoops...
  }
  secs = t.seconds();
  fprintf(stdout, "validation time: %lf seconds.\n", secs);  

  fprintf(stdout, "seconds = %lf\n", loop_secs);
  fprintf(stdout, "Result = %s\n", (i == VEC_SIZE) ? "PASS" : "FAIL");
  delete []A;
  delete []B;
  delete []C;

  Kokkos::finalize();
  return 0;
}
