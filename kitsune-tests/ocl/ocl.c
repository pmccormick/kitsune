#include<stdio.h>
#include<kitsune.h>
#include<stdlib.h>

void f(double* a, int n){
  ocl_mmap(a, n); 
  forall(int i=0; i<n; i++) {
    a[i] = 3.14159; 
  }
}

