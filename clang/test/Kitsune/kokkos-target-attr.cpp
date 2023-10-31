// RUN: %clang_cc1 -fkokkos -fkokkos-no-init -ftapir=cuda -fsyntax-only -verify %s
#include "Kokkos_Core.hpp"
#include <kitsune.h>
int main(int argc, char *argv[]) {
  float *A = alloc<float>(1024);

  Kokkos::initialize(argc, argv); {

    [[tapir::target("i860")]] // expected-error {{unknonwn tapir target}}
    Kokkos::parallel_for(1024, KOKKOS_LAMBDA(const int i) {
	A[i] = i;
    });

    [[tapir::target(cuda)]] // expected-error {{'target' attribute requires a string}}
    Kokkos::parallel_for(1024, KOKKOS_LAMBDA(const int i) {
	A[i] = i;
    });
  

    [[tapir::target()]] // expected-error {{'target' attribute takes one argument}}
    Kokkos::parallel_for(1024, KOKKOS_LAMBDA(const int i) {
	A[i] = i;
    });  
  
    [[tapir::target("cuda","-03")]] // expected-error {{'target' attribute takes one argument}}
    Kokkos::parallel_for(1024, KOKKOS_LAMBDA(const int i) {
      A[i] = i;
    });    

    [[tapir::target("cuda")]] // expected-warning {{tapir target attribute on unsupported statement}}
    if (argc == 1) {
      forall(int i = 0; i < 1024; ++i)
	Kokkos::parallel_for(1024, KOKKOS_LAMBDA(const int i) {
	  A[i] = i;
        });        
    }
    
  } Kokkos::finalize(); 
  return 0;
}

