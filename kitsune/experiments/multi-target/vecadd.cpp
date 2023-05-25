#include <iostream>
#include <iomanip>
#include <chrono>
#include <kitsune.h>

using namespace std;

const size_t ARRAY_SIZE = 1024 * 1024 * 256;

void cpu_fill(float *data, size_t N, int inc) {
  forall(size_t i = 0; i < N; ++i)
    data[i] = float(i+inc);
}

void gpu_fill(float *data, size_t N, int inc) {
  [[tapir::target("cuda")]]
  forall(size_t i = 0; i < N; ++i)
    data[i] = float(i+inc);
}

void foo(float *A, float *B, float *C, size_t N) {
  [[tapir::target("cuda")]]
  forall(size_t i = 0; i < N/2; i++)
    C[i] = A[i] + B[i];
}


int main (int argc, char* argv[]) {
  using namespace std;
  size_t size = ARRAY_SIZE;
  unsigned int iterations = 10;
  if (argc >= 2)
    size = atol(argv[1]);
  if (argc == 3)
    iterations = atoi(argv[2]);  

  cout << setprecision(5);
  cout << "---- multi-target vector addition (forall) ----\n"
       << "  Vector size: " << size << " elements.\n\n";
  cout << "  Allocating arrays..."   
       << std::flush;
  float *A = alloc<float>(size);
  float *B = alloc<float>(size);
  float *C = alloc<float>(size);
  cout << "  done.\n\n";

  size_t error_count;
  bool found_error = false;
  for(int t = 0; t < iterations; t++) {
    spawn fill_a {
      cpu_fill(A, size, t);
    }
    spawn fill_b {
      gpu_fill(B, size, t);  
    }
    sync fill_a;
    sync fill_b;
    
    spawn add_gpu {
      forall(size_t i = 0; i < size/2; i++)
        C[i] = A[i] + B[i];
    }

    spawn add_cpu {
      forall(size_t i = size/2; i < size; i++)
        C[i] = A[i] + B[i];
    }
    sync add_cpu;
    sync add_gpu;

    cout << "  checking result..." << std::flush;
    error_count = 0;
    for(size_t i = 0; i < size; i++) {
      float sum = A[i] + B[i];
      if (C[i] != sum)
	error_count++;
    }
    if (error_count > 0) {
      cout << "  incorrect result found!\n";
      found_error = true;
    } else
      cout << "  ok\n";
  }
  
  return int(found_error);
}
