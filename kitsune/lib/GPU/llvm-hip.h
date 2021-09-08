#include<llvm/Target/TargetMachine.h>

void* hipManagedMalloc(size_t n);
int initHIP();
void* launchHIPKernel(llvm::Module& m, void** args, size_t n); 
void waitHIPKernel(void* wait); 
