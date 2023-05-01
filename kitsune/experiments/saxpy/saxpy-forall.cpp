#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <kitsune.h>

const float DEFAULT_X_VALUE = rand() % 1000000;
const float DEFAULT_Y_VALUE = rand() % 1000000;
const float DEFAULT_A_VALUE = rand() % 1000000;

bool check_saxpy(const float *v, size_t N) {
  float err = 0.0f;
  for (size_t i = 0; i < N; i++) {
    err = err +
          fabs(v[i] - (DEFAULT_A_VALUE * DEFAULT_X_VALUE + DEFAULT_Y_VALUE));
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
  cout << "---- saxpy benchmark (forall) ----\n"
       << "  Problem size: " << size << " elements.\n\n";
  cout << "  Allocating arrays..." << std::flush;
  float *x = alloc<float>(size);
  float *y = alloc<float>(size);
  cout << "  done.\n\n";

  cout << "  Starting benchmark...\n" << std::flush;

  double iteration_total_time = 0;
  
  auto start_total_time = chrono::steady_clock::now();
  double min_time = 100000.0;
  double max_time = 0.0;  
  for (unsigned int t = 0; t < iterations; t++) {

    auto start_time = chrono::steady_clock::now();
    forall(size_t i = 0; i < size; i++) {
      x[i] = DEFAULT_X_VALUE;
      y[i] = DEFAULT_Y_VALUE;
    }

    forall(size_t i = 0; i < size; i++) 
      y[i] = DEFAULT_A_VALUE * x[i] + y[i];
    auto end_time = chrono::steady_clock::now();
    double elapsed_time =
        chrono::duration<double>(end_time - start_time).count();
    if (elapsed_time < min_time)
      min_time = elapsed_time;
    if (elapsed_time > max_time)
      max_time = elapsed_time;    
    cout << "\t" << t << ". iteration time: " << elapsed_time << " seconds.\n";
    iteration_total_time += elapsed_time;
  }

  cout << "\n  Checking final result..." << std::flush;
  if (not check_saxpy(y, size)) {
    cout << "  incorrect result found!\n";
    return 1;
  } else {
    auto end_total_time = chrono::steady_clock::now();
    double elapsed_total_time =
        chrono::duration<double>(end_total_time - start_total_time).count();
    cout << "  pass (answers match).\n\n"
         << "  Total time: " << elapsed_total_time << " seconds.\n"
         << "  Average iteration time: " << iteration_total_time / iterations
         << " seconds.\n"
         << "*** " << min_time << ", " << max_time << "\n"      
         << "----\n\n";
  }
  dealloc(x);
  dealloc(y);
  return 0;
}
