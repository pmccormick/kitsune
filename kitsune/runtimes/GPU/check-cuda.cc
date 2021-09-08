#if __has_include("cuda.h")
#include"llvm-cuda.cc"
#else
#warning "Couldn't find cuda, not building cuda support"
#include"nocuda.cc"
#endif
