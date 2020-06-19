#include <iostream>
using namespace std;

const unsigned int N = 4;

int main(int argc, char *argv[])
{
  [[tapir::rt_target("cilk","cuda")]]
  for(int i = 0; i < N; i++) {
    cout << "i : " << i << endl;
  }

  return 0;
}

