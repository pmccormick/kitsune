// RUN: %clang_cc1 -fsyntax-only -verify %s
#include <kitsune.h>
int main(int argc, char *argv[]) {
  float a[1024];

  [[tapir::target("i860")]] // expected-error {{unknonwn tapir target}}
  forall(int i = 0; i < 1024; ++i)
    a[i] = i;

  [[tapir::target(cuda)]] // expected-error {{'target' attribute requires a string}}
  forall(int i = 0; i < 1024; ++i)
    a[i] = i;

  [[tapir::target()]] // expected-error {{'target' attribute takes one argument}}
  forall(int i = 0; i < 1024; ++i)
    a[i] = i;
  
  [[tapir::target("cuda","-03")]] // expected-error {{'target' attribute takes one argument}}
  forall(int i = 0; i < 1024; ++i)
    a[i] = i;      

  [[tapir::target("cuda")]] // expected-warning {{tapir target attribute on unsupported statement}}
  if (argc == 1) {
    forall(int i = 0; i < 1024; ++i)
      a[i] = i;
  }
  
  return 0;
}

