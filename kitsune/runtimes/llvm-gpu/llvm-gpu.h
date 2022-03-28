#include<stdint.h>
#include<stddef.h>

#ifdef __cplusplus
namespace llvm {
class Module;
}

extern "C" {
  void* launchKernel(llvm::Module& bc, void** args, uint64_t n);
#endif
  void *gpuManagedMalloc(uint64_t n);
  void initRuntime();
  void* launchBCKernel(const char* bc, uint64_t bcsize, void** args, uint64_t n);
  void waitKernel(void* wait);
#ifdef __cplusplus
}

#endif
