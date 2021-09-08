#include"kernelHeader.h"

void f(double* x){
	x[gtid()] = 1.0;
}

