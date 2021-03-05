#include <stdio.h>
#include "kitsune_realm_c.h"

int main(int argc, char *argv[]) {
  realmInitRuntime(argc, argv);
  printf("number of realm processors: %ld\n", realmGetNumProcs());
  return 0;
}
