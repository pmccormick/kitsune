#include <stdio.h>
#include <stdlib.h>
#include <kitsune.h>

int main(int argc, char *argv[]) {
   int steps = argc > 1 ? atoi(argv[1]) : 16;
   forall(int i = 0; i < steps; ++i) {
     printf("foo\n");
   }
   return 0;
}
