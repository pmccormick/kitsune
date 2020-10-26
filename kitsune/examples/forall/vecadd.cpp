// Very simple test of kokkos with two common forms of the 
// parallel_for construct.  We should be able to transform 
// all constructs from lambda into simple loops... 
#include <cstdio>
#include <cstddef>
#include <cstdlib>
#include <cmath>
#include <kitsune.h>

using namespace std;

const size_t VEC_SIZE = 1024 * 1024 * 256;

int main (int argc, char* argv[]) {

  float *A = new float[VEC_SIZE];
  float *B = new float[VEC_SIZE];
  float *C = new float[VEC_SIZE];

  for(size_t i = 0; i < VEC_SIZE; ++i) {
      A[i] = rand() / (float)RAND_MAX;
      B[i] = rand() / (float)RAND_MAX;
  } 

  forall(size_t i = 0; i < VEC_SIZE; ++i) { 
    C[i] = A[i] + B[i];
  }

  size_t ti = 0;
  for(; ti < VEC_SIZE; ++ti) {
    float sum = A[ti] + B[ti];
    if (fabs(C[ti] - sum) > 1e-7f)
      break; // whoops...
  }
 
  fprintf(stdout, "Result = %s\n", (ti == VEC_SIZE) ? "PASS" : "FAIL");

  delete []A;
  delete []B;
  delete []C;
  
  return 0;
}

