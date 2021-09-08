#include<llvm/IR/LLVMContext.h>
#include<llvm/IR/Module.h>
#include<llvm/IRReader/IRReader.h>
#include<llvm/Support/SourceMgr.h>
#include "gpu.h"
#include<iostream>

int main(){
	initRuntime(); 	

  llvm::LLVMContext C; 
  llvm::SMDiagnostic SMD; 
  std::unique_ptr<llvm::Module> ExternalModule =
      parseIRFile("kernel.bc", SMD, C);

  int n = 10; 
  double* x = (double*) gpuManagedMalloc(n*sizeof(double)); 

  void* args[] = { &x }; 
  auto w = launchKernel(*ExternalModule.get(), args, 10); 
  std::cout << "Launched kernel...";
  waitKernel(w); 
  std::cout << "done" << std::endl; 
  for(int i=0; i<n; i++){
    printf("%f\n", x[i]);
  }
}


