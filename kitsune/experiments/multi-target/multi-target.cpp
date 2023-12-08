#include <iostream>
#include <iomanip>
#include <chrono>
#include <kitsune.h>
#include "kitrt/cuda.h"
#include "kitsune/timer.h"

#define USE_SPAWN

using namespace std;
//using namespace kitsune;

const size_t ARRAY_SIZE = 1024 * 1024 * 256;

void random_fill(float *data, size_t N) {
  for(size_t i = 0; i < N; ++i)
    data[i] = rand() / (float)RAND_MAX;
}

void fill(float *data, size_t N) {
  for(size_t i = 0; i < N; ++i)
    data[i] = float(i);
}

int main (int argc, char* argv[]) {
  using namespace std;
  size_t size = 1024 * 1024 * 256;
  unsigned int iterations = 10;
  if (argc >= 2)
    size = atol(argv[1]);
  if (argc == 3)
    iterations = atoi(argv[2]);  

  fprintf(stdout, "problem size: %ld\n", size);
  float *A = (float *)malloc(sizeof(float) * size);
  float *B = (float *)malloc(sizeof(float) * size);
  float *C = (float *)malloc(sizeof(float) * size);
  float *Ak = (float *)__kitrt_cuMemAllocManaged(sizeof(float) * size);
  float *Bk = (float *)__kitrt_cuMemAllocManaged(sizeof(float) * size);
  float *Ck = (float *)__kitrt_cuMemAllocManaged(sizeof(float) * size);

  fill(A, size);
  fill(B, size);  
  fill(Ak, size);
  fill(Bk, size);  

  /////////////////////////////////////////////////
  // seet Attr.td for acceptable target attributes
  /////////////////////////////////////////////////

//   forall(size_t iii = 0; iii < size; iii++)
//     C[iii] = A[iii] + B[iii];
//   printf("%f\n", C[5]);

kitsune::timer t;

#ifdef USE_SPAWN
spawn region 
#endif
{  
    [[tapir::target("cuda")]]
  forall(size_t iii = 0; iii < size; iii++)
    Ck[iii] = Ak[iii] + Bk[iii];
}

//   [[tapir::target("serial")]]
//   forall(size_t jjj = 0; jjj < size; jjj++)
//     C[jjj] = A[jjj] + B[jjj];
//   printf("%f\n", C[15]);

#ifdef USE_SPAWN
spawn region2 
#endif
{  
    [[tapir::target("cilk")]]
  forall(size_t kkk = 0; kkk < size; kkk++)
    C[kkk] = A[kkk] + B[kkk];
}

#ifdef USE_SPAWN
sync region; 
sync region2; 
#endif

  printf("%f\n", Ck[10]);
  printf("%f\n", C[20]);

printf("%f\n",t.seconds());
  return 0;
}

