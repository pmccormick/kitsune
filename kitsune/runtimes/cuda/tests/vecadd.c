//
// Copyright(c) 2020 Triad National Security, LLC
// All rights reserved.
//
// This file is part of the kitsune / llvm project.  It is released under
// the LLVM license.
//
// This is yet another version of the vector sum example that has various
// other implementations within the code base.  In this case, the code 
// that implements the vector addition is actually in LLVM IR form with 
// the NVVM intrinsics used for generating a CUDA/PTX kernel.  See the 
// vecadd.ll file in this directory for the LLVM IR source.
//
// This example shows various aspects of using the cuabi runtime. 
//
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <assert.h>

#include "cuda_abi.h"

#ifndef SOURCE_DIR
#error "Define SOURCE_DIR to it points to vecadd.ll."
#endif 

int *allocHostMem(size_t nbytes) {
  int *buffer = (int *)malloc(nbytes);
  assert(buffer != NULL);
  return buffer;
}

int main(int argc, char *argv[]) {
  gpu_id_t gpuID = cuabiInit();
  printf("system has %zu gpus available, calling thread assigned to GPU #%d.\n",
         cuabiNumberOfGPUs(), gpuID);

  /* Read the LLVM source (as text) from vecadd.ll. */
  size_t llvmBufferSize = 0;
  const char *llvmIR = cuabiReadLLVMKernel(SOURCE_DIR "/vecadd.ll", &llvmBufferSize);
  assert(llvmIR != NULL && "failed to load the IR file.");

  /* Use NVVM compile the LLVM IR to PTX. */
  const char *ptxBuffer = cuabiLLVMToPTX(llvmIR, llvmBufferSize, "vecadd");
  assert(ptxBuffer && "failed to compile the llvm ir to ptx.");

  /* Setup the kernel parameters and allocate host side storage to correspond
   * to them.  
   */
  const unsigned int nThreads = 32;
  const unsigned int nBlocks = 1;
  const size_t vectorSize = nThreads * nBlocks * sizeof(int);
  int *host_a, *host_b, *host_c = NULL;
  host_a = allocHostMem(vectorSize);
  host_b = allocHostMem(vectorSize);
  host_c = allocHostMem(vectorSize);

  /* Populate the data with something simple. */
  for(unsigned int i = 0; i < nThreads; i++) {
    host_a[i] = (int)i;
    host_b[i] = (int)i;
  }

  /* Allocate the device side storage. */
  CUdeviceptr dev_a, dev_b, dev_c;
  cuMemAlloc(&dev_a, vectorSize);
  cuMemAlloc(&dev_b, vectorSize);
  cuMemAlloc(&dev_c, vectorSize);

  /* Load the PTX code (for JIT'ing) and get it configured as an executable kernel. */
  kernel_t kern = cuabiLoadPTXKernel(ptxBuffer, "vecadd");
  if ( cuabiValidKernel(kern) ) {
    /* Copy data to the device, flush out the kernel parameters 
     * and then launch the kernel. 
     */
    cuMemcpyHtoD(dev_a, host_a, vectorSize);
    cuMemcpyHtoD(dev_b, host_b, vectorSize);

    void *params[] = {&dev_c, &dev_a, &dev_b};
    cuabiLaunchKernel(kern, gpuID, 
                      nBlocks, 1, 1, 
                      nThreads, 1, 1, 
                      params);
    /* Copy data back from device and dump it out as a sanity check. */
    cuMemcpyDtoH(host_c, dev_c, vectorSize);
    fprintf(stdout, "result: ");
    for(unsigned int i = 0; i < nThreads; ++i) {
      fprintf(stdout, "%d ", host_c[i]);
    }
    fprintf(stdout, "\n");
    /* clean up... */
  }
  
  free((void*)llvmIR);
  free((void*)ptxBuffer);
  free((void*)host_a);
  free((void*)host_b);
  free((void*)host_c);
  cuMemFree(dev_a);
  cuMemFree(dev_b);
  cuMemFree(dev_c);

  return 0;
  }
