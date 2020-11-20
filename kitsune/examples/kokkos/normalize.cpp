// 
#include <cstdio>
#include <cmath>
#include <kitsune/timer.h>
#include <Kokkos_Core.hpp>

using namespace std;
using namespace kitsune;

const size_t VEC_SIZE = 1024 * 1024 * 1024;

void random_fill(double *data, size_t N) {
  for(size_t i = 0; i < N; ++i)
    data[i] = (rand() / (double)RAND_MAX) * 10.0;
}

__attribute__((const, noinline)) double norm(const double *in, size_t N) {
  double sum = 0.0;
  for(size_t i = 0; i < N; ++i)
    sum += in[i] * in[i];
  return sqrt(sum);
}

int main (int argc, char* argv[]) {
  timer t;  
  double *in = new double[VEC_SIZE];
  double *out = new double[VEC_SIZE];

  random_fill(in, VEC_SIZE);
  
  Kokkos::initialize (argc, argv);
  double secs = t.seconds();
  fprintf(stdout, "initialization time: %lf seconds.\n", secs);
  t.reset();
  {
    Kokkos::parallel_for(VEC_SIZE, KOKKOS_LAMBDA(const size_t i) {
      out[i] = in[i] / norm(in, VEC_SIZE);
    });
  }
  double loop_secs = t.seconds();
  t.reset();

  fprintf(stdout, "%lf, %lf, %lf, %lf\n", out[0], out[22], out[3000], out[4096]); 
  fprintf(stdout, "seconds = %lf\n", loop_secs);
  delete []in;
  delete []out;
  Kokkos::finalize();
  return 0;
}
