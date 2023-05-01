#include "Kokkos_Core.hpp"
#include "Kokkos_DualView.hpp"

#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <iomanip>


template <typename T>
using View = Kokkos::DualView<T*>;

template <typename T>
void parallel_copy(View<T> &dst, View<T> &src, int N) {
  src.sync_device();
  dst.sync_device();
  Kokkos::parallel_for("copy", N, KOKKOS_LAMBDA(const int &i) {
    dst.d_view(i) = src.d_view(i);
  });
  Kokkos::fence();
  dst.modify_device();
}

template <typename T>
void random_fill(View<T> &data, size_t N) {
  for(size_t i = 0; i < N; i++)
    data.h_view(i) = rand() / (T)RAND_MAX;
  data.modify_host();
}

template <typename T>
bool check(View<T> &data0, View<T> &data1, size_t N) {
  data0.sync_host();
  data1.sync_host();
  for(size_t i = 0; i < N; ++i) {
    if (data0.h_view(i) != data1.h_view(i))
      return false;
  }
  return true;
}


int main(int argc, char** argv)
{
  using namespace std;
  size_t array_size = 1024 * 1024 * 256;
  unsigned int iterations = 10;  
  if (argc >= 2)
    array_size = atol(argv[1]);
  if (argc == 3)
    iterations = atoi(argv[2]);

  cout << setprecision(5);  

  cout << "\n";
  cout << "---- Simple copy benchmark (forall) ----\n"
       << "  Array size: " << array_size << "\n"
       << "  Iterations: " << iterations << "\n\n";
  cout << "Allocating arrays and filling with random values..." << std::flush;
  auto start = chrono::steady_clock::now();
  Kokkos::initialize(argc, argv); {
    View<float> data0("data0", array_size);
    View<float> data1("data1", array_size);
    random_fill(data0, array_size);
    cout << "  done.\n\n";

    cout << "Starting benchmark...\n";
    unsigned int mb_size = (sizeof(float) * array_size) / (1024 * 1024);

    auto start_time = chrono::steady_clock::now();
    double total_copy_time = 0.0;
    double min_time = 100000.0;
    double max_time = 0.0;
    for(int i = 0; i < iterations; i++) {
      auto copy_start_time = chrono::steady_clock::now();
      parallel_copy(data1, data0, array_size);
      auto copy_end_time = chrono::steady_clock::now();
      
      auto elapsed_time =
	chrono::duration<double>(copy_end_time-copy_start_time).count();
      cout  << "\t" << i << ". copy time: "
	    << elapsed_time 
	    << " sec., " << mb_size / elapsed_time << " MB/sec.\n";
    if (elapsed_time < min_time)
      min_time = elapsed_time;
    if (elapsed_time > max_time)
      max_time = elapsed_time;
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

  }  Kokkos::finalize();

  return 0;
}

