all: test

cl.o: cl.cc
	clang++ cl.cc -c  -Wall

test: kernel.spv test.c cl.o
	clang test.c cl.o -g -o test -lOpenCL -lstdc++ -lm -Wall

kernel.bc: kernel.cl 
	clang -cc1 -triple spir64 kernel.cl -O0 -emit-llvm-bc -o kernel.bc -finclude-default-header 

kernel.spv: kernel.bc
	llvm-spirv $< -o $@

