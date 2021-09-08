#if __has_include("hip/hip_runtime.h")
#include"llvm-hip.cc"
#else
#warning "Couldn't find hip, not building hip support"
#include"nohip.cc"
#endif
