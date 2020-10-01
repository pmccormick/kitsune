#include<vector>
#include<iostream>
#include<kitsune.h>
#include<assert.h>

using namespace std; 

int main(int argc, char** argv){
  int n = argc > 1 ? atoi(argv[1]) : 1024; 
  vector<double> A(n, 3.14); 
  vector<double> B(n, 6.28); 
  vector<double> C(n); 
  forall(int i=0; i<n; i++){
    C[i] = A[i] + B[i];  
  }
  for(int i=0; i<n; i++){
    if(C[i] != A[i] + B[i]){
      printf("Failure");
      exit(i);
    }
  }
  cout << "Success" << endl; 
}
