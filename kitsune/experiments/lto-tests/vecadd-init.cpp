#include <cstdlib>
#include <kitsune.h>

extern "C" {
  void fill(float *data, uint64_t N) {
    float base_value = rand() / (float)RAND_MAX;
    forall(size_t i = 0; i < N; ++i)
      data[i] = base_value + i;
  }
}



