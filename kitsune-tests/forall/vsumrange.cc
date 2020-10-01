#include<vector>
#include<boost/range/irange.hpp>
#include<iostream>
#include<kitsune.h>
#include<assert.h>

using namespace std; 

int main(int argc, char** argv){
  int n = argc > 1 ? atoi(argv[1]) : 1024; 
  vector<double> A(n, 3.14); 
  vector<double> B(n, 6.28); 
  vector<double> C(n); 
  forall(auto i : boost::irange(n)){
    C[i] = A[i] + B[i];  
  }
  for(auto i : boost::irange(n)){
    assert(C[i] == A[i] + B[i]);
  }
  cout << "Success" << endl; 
}
