#include <iostream>
using namespace std;
#include <kitsune.h>

const unsigned int N = 4;

int main(int argc, char *argv[])
{
  [[tapir::rt_target("openmp")]]
  forall(int i = 0; i < N; i++) {
    cout << "i : " << i << endl;
  }

  return 0;
}

