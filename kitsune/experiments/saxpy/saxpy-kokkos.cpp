#include "Kokkos_DualView.hpp"

#include <iostream>
#include <iomanip>
#include <chrono>

typedef Kokkos::DualView<float*, Kokkos::LayoutRight, 
                         Kokkos::DefaultExecutionSpace>
  SaxpyDualView;

// Unlike kitsune+tapir Kokkos has no way to handle these values as
// global constants with a dynamic allocation/assignment.  As a result,
// we have to explicitly use gpu-centric declarations.
// Kokkos probably has some equivalent to this... 
#ifdef __clang__ 
  __device__ float DEFAULT_X_VALUE = 23114.0;
  __device__ float DEFAULT_Y_VALUE = 81109.0;
  __device__ float DEFAULT_A_VALUE = 65231.0;
#else 
  __managed__ __device__ float DEFAULT_X_VALUE;
  __managed__ __device__ float DEFAULT_Y_VALUE;
  __managed__ __device__ float DEFAULT_A_VALUE;
#endif 

bool check_saxpy(const SaxpyDualView &v, size_t N) {
#ifdef __clang__
  float DEFAULT_X_VALUE = 23114.0;
  float DEFAULT_Y_VALUE = 81109.0;
  float DEFAULT_A_VALUE = 65231.0;
#endif 
  float err = 0.0f;
  for(size_t i = 0; i < N; i++) {
    err = err + fabs(v.h_view(i) - (DEFAULT_A_VALUE * DEFAULT_X_VALUE + DEFAULT_Y_VALUE));
  }
  return err == 0.0f;
}

int main(int argc, char *argv[]) {
  using namespace std;
  size_t size = 1 << 28;
  unsigned int iterations = 10;
  if (argc > 1) {
    size = atol(argv[1]);
    if (argc == 2) 
      iterations = atoi(argv[2]);
    else {
      cout << "usage: saxpy [size] [iterations]\n";
      return 1;
    }
  }

  cout << setprecision(5);
  cout << "\n";
  cout << "---- saxpy benchmark (kokkos) ----\n"
       << "  Problem size: " << size << " elements.\n\n";
  cout << "  Allocating arrays..." 
       << std::flush;

  Kokkos::initialize(argc, argv); {

    SaxpyDualView x = SaxpyDualView("x", size);
    SaxpyDualView y = SaxpyDualView("y", size);
    cout << "  done.\n\n";

    cout << "  Starting benchmark...\n" << std::flush;
    double iteration_total_time = 0;

#ifndef __clang__ 
    DEFAULT_X_VALUE = rand() % 1000000;
    DEFAULT_Y_VALUE = rand() % 1000000;
    DEFAULT_A_VALUE = rand() % 1000000;
#endif 
    double min_time = 100000.0;
    double max_time = 0.0;
    auto start_total_time = chrono::steady_clock::now();


    for(unsigned int t = 0; t < iterations; t++) {
      auto start_time = chrono::steady_clock::now();

      x.modify_device();
      y.modify_device();
      Kokkos::parallel_for("init", size, KOKKOS_LAMBDA(const int i) {
        x.d_view(i) = DEFAULT_X_VALUE;
        y.d_view(i) = DEFAULT_Y_VALUE;
      });
      Kokkos::fence();

      y.modify_device();
      Kokkos::parallel_for("saxpy", size, KOKKOS_LAMBDA(const int &i) {
        y.d_view(i) = DEFAULT_A_VALUE * x.d_view(i) + y.d_view(i);
      });
      Kokkos::fence();

      auto end_time = chrono::steady_clock::now();
      double elapsed_time = chrono::duration<double>(end_time-start_time).count();
      if (elapsed_time < min_time)
        min_time = elapsed_time;
      if (elapsed_time > max_time)
        max_time = elapsed_time;    
      cout << "\t" << t << ". iteration time: " << elapsed_time << " seconds.\n";
      iteration_total_time += elapsed_time;
    }
    y.sync_host();   // can't just leave the data in place to check so we add the cost.

    cout << "\n  Checking final result..." << std::flush;

    if (not check_saxpy(y, size)) {
      cout << "  incorrect result found!\n";
      return 1;
    } else { 
      auto end_total_time = chrono::steady_clock::now();
      double elapsed_total_time = chrono::duration<double>(end_total_time-start_total_time).count();
      cout << "  pass (answers match).\n\n"
           << "  Total time: " << elapsed_total_time << " seconds.\n"
           << "  Average iteration time: " << iteration_total_time / iterations << " seconds.\n"
           << "*** " << min_time << ", " << max_time << "\n"      	
           << "----\n\n";
    }
  } Kokkos::finalize();
  return 0;
}

