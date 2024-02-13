#include <cstdlib>
#include <kitsune.h>

extern "C" {
  void vec_add(const float *A, const float *B, float *C, uint64_t N) {
    forall(size_t i = 0; i < N; ++i)
      C[i] = A[i] + B[i];
  }
}



