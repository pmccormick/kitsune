// Very simple test of kokkos with two common forms of the 
// parallel_for construct.  We should be able to transform 
// all constructs from lambda into simple loops... 
#include <cstdio>
#include <cstddef>
#include <cstdlib>
#include <cmath>
#include <kitsune.h>
#include <cilk/cilk.h>
#include <vector>

using namespace std;

int main (int argc, char* argv[]) {
  setbuf(stdout, NULL); 

  printf("Initializing..."); 
  int n = argc > 1 ? atoi(argv[1]) : 1024; 

  vector<double> A(n);
  vector<vector<double>> B(n, vector<double>(n)); 
  vector<double> C(n); 
  vector<double> ANS(n); 

  for(size_t i = 0; i < n; ++i) {
      A[i] = (double)rand() / (double)RAND_MAX;
      for(size_t j = 0; j < n ; j++)
        B[i][j] = (double)rand() / (double)RAND_MAX;
      C[i] = 0; 
      ANS[i] = 0; 
  }
  printf("Done.\n"); 

  printf("Computing matrix vector product..."); 
  forall(size_t j = 0; j < n; j++)
    for(size_t i = 0; i < n; i++) 
      for(size_t k = 0; k < n; k++)
        C[j] += B[i][k] * A[k]; 
  printf("Done.\n"); 

  printf("Checking result..."); 
  for(size_t i = 0; i < n; i++) 
    for(size_t j = 0; j < n; j++)
      for(size_t k = 0; k < n; k++)
        ANS[j] += B[i][k] * A[k]; 

  for(size_t i = 0; i < n; i++)
    if(ANS[i] != C[i]){
      printf("Failure at %ld: %f != %f!\n", i, ANS[i], C[i]); 
      return i; 
    }
  printf("Success.\n"); 

  return 0;
}

