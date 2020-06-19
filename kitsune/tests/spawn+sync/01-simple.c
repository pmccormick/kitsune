#include<kitsune.h>
#include<stdio.h> 

int main(){

  spawn p1 {
    for(int i=0; i<10; i++) 
      printf("Task 1: Hello %d\n", i);
  }
  
  spawn p2 {
    for(int i=10; i<20; i++) 
      printf("Task 2: Hello %d\n", i);
  }
  
  sync p1;
  sync p2;
  return 0;
}

