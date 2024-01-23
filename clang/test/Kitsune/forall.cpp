// RUN: %clang_cc1 -ftapir=serial -fsyntax-only -verify %s
#include<kitsune.h>

void f1() {
  int M;
  forall(size_t N = 0; N < 10; N++)
    ;
  forall(M = 0; M < 10; M++)
    ;


  forall(n = 0 n < 10; n++); // expected-error {{expected ';' in 'for'}}
  forall(n = 0; n < 10 n++); // expected-error {{expected ';' in 'for'}}

  forall (int n = 0 n < 10; n++); // expected-error {{expected ';' in 'for'}}
  forall (int n = 0; n < 10 n++); // expected-error {{expected ';' in 'for'}}

  forall (n = 0 bool b = n < 10; n++); // expected-error {{expected ';' in 'for'}}
  forall (n = 0; bool b = n < 10 n++); // expected-error {{expected ';' in 'for'}}

  forall (n = 0 n < 10 n++); // expected-error 2{{expected ';' in 'for'}}

  forall (;);
  // expected-error@20 {{expected ';' in 'for'}}
  // expected-error@20 {{forall statement must have an initialization expression}}
}
