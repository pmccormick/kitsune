#include <iostream>
#include <iomanip>
#include <chrono>
#include <kitsune.h>

extern "C" void fill(float *data, uint64_t N);
extern "C" void vec_add(const float *A, const float *B, float *C, uint64_t N);

int main (int argc, char* argv[]) {
  using namespace std;
  uint64_t size = 1024 * 1024 * 256;
  unsigned int iterations = 10;
  cout << setprecision(5);
  cout << "\n";
  cout << "---- vector addition benchmark (forall) ----\n"
       << "  Vector size: " << size << " elements.\n\n";
  cout << "  Allocating arrays and filling with random values..." 
       << std::flush;
  float *A = alloc<float>(size);
  float *B = alloc<float>(size);
  float *C = alloc<float>(size);
  fill(A, size);
  fill(B, size);
  cout << "  done.\n\n";

  double elapsed_time;
  double min_time = 100000.0;
  double max_time = 0.0;
  for(unsigned t = 0; t < iterations; t++) {
    auto start_time = chrono::steady_clock::now();
    vec_add(A, B, C, size);
    auto end_time = chrono::steady_clock::now();
    elapsed_time = chrono::duration<double>(end_time-start_time).count();
    if (elapsed_time < min_time)
      min_time = elapsed_time;
    if (elapsed_time > max_time)
      max_time = elapsed_time;
    cout << "\t" << t << ". iteration time: " << elapsed_time << ".\n";
  }
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
         << "*** " << min_time << ", " << max_time << "\n"      
         << "----\n\n";
  }

  return error_count;
}

