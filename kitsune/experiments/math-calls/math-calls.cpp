// See the README file for details.
#include <iostream>
#include <fstream>
#include <chrono>
#include <kitsune.h>
#include <cmath>

using namespace std;

const size_t DEFAULT_ARRAY_SIZE = 1024 * 1024 * 128;

struct testit {
  float a, b;
};

template <typename T>
void random_fill(T* data, size_t N) {
  for(size_t i = 0; i < N; ++i)
    data[i] = (2.0 * 3.142) * (rand() / (T)RAND_MAX);
}

__attribute__((noinline))
void struct_test(testit *ti) {
  ti->a = 4;
  ti->b = ti->a + 4;
}

template <typename T>
__attribute__((noinline))
T math_call(T value) {
  testit t;
  struct_test(&t);
  return fminf(value, 1234.56 - t.a + t.b);
}

template <typename T>
__attribute__((noinline))
T math_call2(T value) {
  return sqrtf(value);
}

template <typename T>
void parallel_work(T* dst, const T* src, int N) {
  forall(size_t i = 0; i < N; i++) {
    dst[i] = fminf(math_call(src[i]) + math_call2(src[i]), 100.0);
  } 
}

void print(float *data, size_t N) {
  for(size_t i = 0; i < N; i++) 
    printf("%f  ", data[i]);
  printf("\n\n");
}

int main(int argc, char** argv) {
  size_t array_size = DEFAULT_ARRAY_SIZE;
  if (argc >= 2)
    array_size = atol(argv[1]);
  fprintf(stdout, "array size: %ld\n", array_size);
  float *data0 = alloc<float>(array_size);
  float *data1 = alloc<float>(array_size);
  random_fill(data0, array_size);
  print(data0, 10);
  auto start = chrono::steady_clock::now();
  parallel_work(data1, data0, array_size);
  print(data1, 10);
  auto end = chrono::steady_clock::now();
  cout  << "Execution time: "
          << chrono::duration<double>(end-start).count()
          << endl;
  return 0;
}
