// Very simple test of kokkos with two common forms of the 
// parallel_for construct.  We should be able to transform 
// all constructs from lambda into simple loops... 
#include <cstdio>
#include <cstddef>
#include <cstdlib>
#include <cmath>
#include <vector>

using namespace std;

const size_t VEC_SIZE = 1024 * 1024 * 256;

int main (int argc, char* argv[]) {

  vector<float> A(VEC_SIZE);
  vector<float> B(VEC_SIZE);
  vector<float> C(VEC_SIZE);

  for(auto i : A) {
    A[i] = rand() / (float)RAND_MAX;
  } 

  for(auto i : B) {
    B[i] = rand() / (float)RAND_MAX;
  } 

  for(auto i : C) {
    C[i] = A[i] + B[i];
  }

  size_t ti = 0;
  for(; ti < VEC_SIZE; ++ti) {
    float sum = A[ti] + B[ti];
    float delta = fabs(C[ti] - sum);
    if (delta > 1e-7f) {
      printf("delta error: %f\n", delta);
      break; // whoops...
    }
  }
 
  fprintf(stdout, "Result = %s (%ld, %ld)\n",
	  (ti == VEC_SIZE) ? "PASS" : "FAIL",
	  ti, VEC_SIZE);

  return 0;
}

