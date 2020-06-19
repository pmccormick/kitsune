#include<kitsune.h>
#include<stdio.h> 

int main(){

  for(int i=0; i<10; i++) spawn lb {
      printf("Hello %d\n", i);
  }
  printf("end loop.\n");
  sync lb;
  return 0;
}

