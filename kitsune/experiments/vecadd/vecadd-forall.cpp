#include <iostream>
#include <iomanip>
#include <chrono>
#include <kitsune.h>

template<typename T>
void random_fill(T *data, size_t N) {
  for(size_t i = 0; i < N; ++i)
    data[i] = rand() / (T)RAND_MAX;
}

int main (int argc, char* argv[]) {
  using namespace std;
  size_t size = 1024 * 1024 * 256;
  if (argc > 1)
    size = atol(argv[1]);

  cout << setprecision(5);
  cout << "\n";
    cout << "---- vector addition benchmark (forall) ----\n"
         << "  Vector size: " << size << " elements.\n\n";
  cout << "  Allocating arrays and filling with random values..." 
       << std::flush;
  float *A = alloc<float>(size);
  float *B = alloc<float>(size);
  float *C = alloc<float>(size);
  random_fill(A, size);
  random_fill(B, size);
  cout << "  done.\n\n";


  auto start_time = chrono::steady_clock::now();
  forall(int i = 0; i < size; i++) {
    C[i] = A[i] + B[i];
  }
  auto end_time = chrono::steady_clock::now();
  double elapsed_time = chrono::duration<double>(end_time-start_time).count();

  cout << "  Checking final result..." << std::flush;
  size_t error_count = 0;
  for(size_t i = 0; i < size; i++) {
    float sum = A[i] + B[i];
    if (C[i] != sum)
      error_count++;
  }
  if (error_count) {
    cout << "  incorrect result found! (" 
         << error_count << " errors found)\n\n";
    return 1;
  } else {
    cout << "  pass (answers match).\n\n"
         << "  Total time: " << elapsed_time
         << " seconds. (" << size / elapsed_time << " elements/sec.)\n"
         << "----\n\n";
  }

  dealloc(A);
  dealloc(B);
  dealloc(C);
  return 0;
}

