#include <iostream>
#include <iomanip>
#include <chrono>
#include <kitsune.h>


template<typename T>
void random_fill(T *data, size_t N) {
  T base_value = rand() / (T)RAND_MAX;
  forall(size_t i = 0; i < N; ++i)
    data[i] = base_value + i;
}

template<typename T>
void random_gpu_fill(T *data, size_t N) {
  T base_value = rand() / (T)RAND_MAX;
  [[tapir::target("cuda")]]  
  forall(size_t i = 0; i < N; ++i)
    data[i] = base_value + i;
}

int main (int argc, char* argv[]) {
  using namespace std;
  const size_t ARRAY_SIZE = 1024 * 1024 * 256;
  unsigned int iterations = 10;

  double elapsed_time;

  cout << setprecision(5);
  cout << "---- multi-target vector addition (forall) ----\n"
       << "  Vector size: " << ARRAY_SIZE << " elements.\n\n";
  cout << "  Allocating arrays..." << std::flush;
  float *A = alloc<float>(ARRAY_SIZE);
  float *B = alloc<float>(ARRAY_SIZE);
  float *C = alloc<float>(ARRAY_SIZE);
  cout << "  done.\n\n";

  cout << "running back-to-back (sequential) parallel fills...";
  auto start_time = chrono::steady_clock::now();
  random_fill(A, ARRAY_SIZE);
  random_fill(B, ARRAY_SIZE);
  auto end_time = chrono::steady_clock::now();
  elapsed_time = chrono::duration<double>(end_time-start_time).count();  
  cout << "  done.\n";
  cout << "elapsed time: " << elapsed_time << " seconds.\n";

  cout << "running concurrent gpu parallel fills...";
  start_time = chrono::steady_clock::now();  
  spawn fill_a {
    random_gpu_fill(A, ARRAY_SIZE);
  }
  spawn fill_b {
    random_gpu_fill(B, ARRAY_SIZE);    
  }
  sync fill_a; sync fill_b;
  end_time = chrono::steady_clock::now();

  elapsed_time = chrono::duration<double>(end_time-start_time).count();  
  cout << "  done.\n";
  cout << "elapsed time: " << elapsed_time << " seconds.\n";  

  cout << "running vector addition...";
  start_time = chrono::steady_clock::now();    
  [[tapir::target("cuda")]]
  forall(size_t i = 0; i < ARRAY_SIZE; i++)
      C[i] = A[i] + B[i];
  end_time = chrono::steady_clock::now();
  elapsed_time = chrono::duration<double>(end_time-start_time).count();  
  cout << "  done.\n";
  
  cout << "Checking final result..." << std::flush;
  size_t error_count = 0;
  for(size_t i = 0; i < ARRAY_SIZE; i++) {
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
         << " seconds. (" << ARRAY_SIZE / elapsed_time << " elements/sec.)\n"
         << "----\n\n";
  }

  return 0;
}
