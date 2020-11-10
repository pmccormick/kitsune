// Very simple test of kokkos with two common forms of the 
// parallel_for construct.  We should be able to transform 
// all constructs from lambda into simple loops... 
#include <cstdio>
#include <Kokkos_Core.hpp>

using namespace std;

const size_t VEC_SIZE = 1024 * 1024 * 256;

void random_fill(float *data, size_t N) {
  for(size_t i = 0; i < N; ++i) 
    data[i] = rand() / (float)RAND_MAX;
}


int main (int argc, char* argv[]) {

  float *A = new float[VEC_SIZE];
  float *B = new float[VEC_SIZE];
  float *C = new float[VEC_SIZE];

  random_fill(A, VEC_SIZE);
  random_fill(B, VEC_SIZE);
  
  Kokkos::initialize (argc, argv);
  {
    Kokkos::parallel_for(VEC_SIZE, KOKKOS_LAMBDA(const int i) {
        C[i] = A[i] + B[i];
    });
  }
  Kokkos::finalize ();
  
  // Verify correct result (taking some floating point nuances into
  // play)...
  size_t i = 0;
  for(; i < VEC_SIZE; ++i) {
    float sum = A[i] + B[i];
    if (fabs(C[i] - sum) > 1e-7f)
      break; // whoops...
  }
 
  fprintf(stdout, "Result = %s\n", (i == VEC_SIZE) ? "PASS" : "FAIL");
  delete []A;
  delete []B;
  delete []C;
  return 0;
}
