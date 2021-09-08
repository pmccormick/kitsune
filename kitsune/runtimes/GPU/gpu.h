#include<stdint.h>
#include<stddef.h>

#ifdef __cplusplus
#include<llvm/IR/Module.h>
extern "C" {
void* launchKernel(llvm::Module& bc, void** args, size_t n); 
#endif
void *gpuManagedMalloc(size_t n); 
void initRuntime(); 
void* launchBCKernel(const char* bc, void** args, size_t n); 
void waitKernel(void* wait); 
#ifdef __cplusplus
}
#endif
