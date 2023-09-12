// See the README file for details.
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <kitsune.h>

template <typename T>
void random_fill(T* data, size_t N) {
  for(size_t i = 0; i < N; ++i)
    data[i] = rand() / (T)RAND_MAX;
}

template <typename T>
bool check(const T* data0, const T* data1, size_t N) {
  for(size_t i = 0; i < N; ++i) {
    if (data0[i] != data1[i])
      return false;
  }
  return true;
}

template <typename T>
void array_copy(T* dst, const T* src, int N) {
  for(int i = 0; i < N; i++)
    dst[i] = src[i];
}

int main(int argc, char** argv) {
  using namespace std;

  size_t array_size = 1024 * 1024 * 256;
  unsigned int iterations = 10;
  if (argc >= 2)
    array_size = atol(argv[1]);
  if (argc == 3)
    iterations = atoi(argv[2]);

  cout << setprecision(5);
    
  cout << "\n";
  cout << "---- Simple noforall copy benchmark (serial) ----\n"
       << "  Array size: " << array_size << "\n"
       << "  Iterations: " << iterations << "\n\n";
  cout << "Allocating arrays and filling with random values..." << std::flush;
  float *data0 = alloc<float>(array_size);
  float *data1 = alloc<float>(array_size);
  random_fill(data0, array_size);
  cout << "  done.\n\n";


  cout << "Starting benchmark...\n";
  unsigned int mb_size = (sizeof(float) * array_size) / (1024 * 1024);
  
  double total_copy_time = 0.0;
  double min_time = 100000.0;
  double max_time = 0.0;
  auto start_time = chrono::steady_clock::now();
  for(int i = 0; i < iterations; i++) {
    auto copy_start_time = chrono::steady_clock::now();
    array_copy(data1, data0, array_size);
    auto copy_end_time = chrono::steady_clock::now();
    auto elapsed_time =
      chrono::duration<double>(copy_end_time-copy_start_time).count();
    if (elapsed_time < min_time)
      min_time = elapsed_time;
    if (elapsed_time > max_time)
      max_time = elapsed_time;
    cout  << "\t" << i << ". copy time: "
          << elapsed_time 
          << " sec., " << mb_size / elapsed_time << " MB/sec.\n";
    total_copy_time += elapsed_time;
  }
  auto end_time = chrono::steady_clock::now();

  if (not check(data0, data1, array_size)) {
    cerr << "error!  copy differs!\n";
    return 1;
  } else {
    cout << "pass (copy identical)\n";
  }

  cout << "Total time: "
       << chrono::duration<double>(end_time-start_time).count()
       << endl;
  cout << "Average copy time: "
       << total_copy_time / iterations
       << endl;
  cout << "*** " << min_time << ", " << max_time << "\n";
  cout << "----\n\n";
  return 0;
}

