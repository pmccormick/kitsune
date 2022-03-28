//
//===- llvm-cuda.cpp - Kitsune ABI runtime target CUDA support    ---------===//
//
// TODO: Need to update LANL/Triad Copyright notice.
//
// Copyright (c) 2021, Los Alamos National Security, LLC.
// All rights reserved.
//
//  Copyright 2021. Los Alamos National Security, LLC. This software was
//  produced under U.S. Government contract DE-AC52-06NA25396 for Los
//  Alamos National Laboratory (LANL), which is operated by Los Alamos
//  National Security, LLC for the U.S. Department of Energy. The
//  U.S. Government has rights to use, reproduce, and distribute this
//  software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY,
//  LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY
//  FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
//  derivative works, such modified software should be clearly marked,
//  so as not to confuse it with the version available from LANL.
//
//  Additionally, redistribution and use in source and binary forms,
//  with or without modification, are permitted provided that the
//  following conditions are met:
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above
//      copyright notice, this list of conditions and the following
//      disclaimer in the documentation and/or other materials provided
//      with the distribution.
//
//    * Neither the name of Los Alamos National Security, LLC, Los
//      Alamos National Laboratory, LANL, the U.S. Government, nor the
//      names of its contributors may be used to endorse or promote
//      products derived from this software without specific prior
//      written permission.
//
//  THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
//  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
//  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
//  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//  DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR
//  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
//  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
//  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
//  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
//  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
//  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
//  OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
//  SUCH DAMAGE.
//
//===----------------------------------------------------------------------===//

#include "kitrt-debug.h"
#include "llvm-cuda.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <sstream>
#include <stdbool.h>

// Has the runtime been initialized (successfully)?
static bool _kitrtIsInitialized = false;

static CUdevice _kitrtCUdevice = -1;
static CUcontext _kitrtCUcontext = nullptr;

#define declare(name) extern decltype(name) *name##_p = NULL;
declare(cuInit);
declare(cuStreamCreate);
declare(cuStreamDestroy_v2);
declare(cuStreamSynchronize);
declare(cuLaunchKernel);
declare(cuDeviceGet);
declare(cuGetErrorName);
declare(cuModuleLoadDataEx);
declare(cuModuleLoadData);
declare(cuModuleGetFunction);
declare(cuModuleUnload);
declare(cuCtxCreate_v2);
declare(cuCtxDestroy_v2);
declare(cuCtxSetCurrent);
declare(cuMemAllocManaged);
declare(cuDeviceGetAttribute);

#define CU_SAFE_CALL(x)                                                  \
    do {                                                                 \
        CUresult result = x;                                             \
        if (result != CUDA_SUCCESS) {                                    \
            const char *msg;                                             \
            cuGetErrorName_p(result, &msg);                              \
            fprintf(stderr, "kitrt: %s failed with error %s\n", #x, msg);\
            exit(1);                                                     \
        }                                                                \
    } while(0)


static bool __kitrt_load_dlsyms() {

  #define DLSYM_LOAD(FN)                                       \
    if (! (FN##_p = (decltype(FN)*)dlsym(dlHandle, #FN))) {    \
      fprintf(stderr, "kitrt: failed to load dlsym 'FN##'.");  \
      return false;}

  static void *dlHandle = nullptr;
  if (dlHandle)
    return true;

  if (dlHandle = dlopen("libcuda.so", RTLD_LAZY)) {
    DLSYM_LOAD(cuInit);
    DLSYM_LOAD(cuStreamCreate);
    DLSYM_LOAD(cuStreamDestroy_v2);
    DLSYM_LOAD(cuStreamSynchronize);
    DLSYM_LOAD(cuLaunchKernel);
    DLSYM_LOAD(cuDeviceGet);
    DLSYM_LOAD(cuGetErrorName);
    DLSYM_LOAD(cuModuleLoadData);
    DLSYM_LOAD(cuModuleLoadDataEx);
    DLSYM_LOAD(cuModuleGetFunction);
    DLSYM_LOAD(cuModuleUnload);
    DLSYM_LOAD(cuCtxCreate_v2);
    DLSYM_LOAD(cuCtxDestroy_v2);
    DLSYM_LOAD(cuCtxSetCurrent);
    DLSYM_LOAD(cuMemAllocManaged);
    DLSYM_LOAD(cuDeviceGetAttribute);
    return true;
  } else {
    fprintf(stderr, "kitrt: Failed to load CUDA dynamic library.\n");
    fprintf(stderr, "kitrt: Is CUDA in your LD_LIBRARY_PATH?\n");
    return false;
  }
}

bool __kitrt_cuInit() {

  if (! __kitrt_load_dlsyms()) {
    fprintf(stderr, "kitrt: Unable to resolve dynamic symbols.\n");
    fprintf(stderr, "kitrt: aborting...\n");
    _kitrtIsInitialized = false;
    abort();  // For now we'll abort in the runtime to avoid having to
              // codegen error checks... 
  }

  // TODO: A few things need to be hardened:
  // 
  //   1. Resource clean up (malloced memory, CUDA state and 
  //      resources such as  modules, streams, etc.). 
  //   2. We could save some time by creating a cached state
  //      of kernels, modules, streams, etc.  That might help
  //      reduce some overheads but more experimentation needs
  //      to be done to see what the benefits are. 
  //   3. Multiple GPU support?
  //   4. Etc... 
  // 
  CUresult result;
  result = cuInit_p(0);
  if (result != CUDA_SUCCESS) {
    const char *msg;
    cuGetErrorName_p(result, &msg);
    fprintf(stderr, "kitrt: Failed to initialize CUDA.\n");
    fprintf(stderr, "kitrt: cuInit() error: '%s'\n", msg);
    abort();
    // TODO: Do we want a configurable failure mode here (e.g., hard error
    // vs. return failure state?  Going with a hard failure mode for now.
  } else {
    CU_SAFE_CALL(cuDeviceGet_p(&_kitrtCUdevice, 0));
    CU_SAFE_CALL(cuCtxCreate_v2_p(&_kitrtCUcontext, 0, _kitrtCUdevice));
    _kitrtIsInitialized = true;
  }
  return _kitrtIsInitialized;
}

void *__kitrt_cuMemAllocManaged(size_t size) {
  if (!_kitrtIsInitialized) {
    if (!__kitrt_cuInit())
      return nullptr;
  }

  CUdeviceptr devp;
  CU_SAFE_CALL(cuMemAllocManaged_p(&devp, size, CU_MEM_ATTACH_HOST));
  return (void*)devp;
}

static void __kitrt_cuGetLaunchParameters(size_t &threadsPerBlock,
                                          size_t &blocksPerGrid,
                                          size_t numElements) {
  int warpSize;
  CU_SAFE_CALL(cuDeviceGetAttribute_p(&warpSize,
                                    CU_DEVICE_ATTRIBUTE_WARP_SIZE,
                                    _kitrtCUdevice));
  unsigned blockSize = 4 * warpSize;
  assert(numElements % blockSize == 0);
  // TODO: There is likely quite a bit of improvement to be had here 
  // in terms of connecting the compiler and runtime components for a 
  // better set of launch parameters.   Might be best to combine code
  // analysis from codegen time with runtime calls to bind parameters 
  // to a specific launch... 
  threadsPerBlock = 256;
  blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
}

void *__kitrt_cuLaunchFBKernel(const void *fatBin,
                               const char *kernelName,
                               void **fatBinArgs,
                               uint64_t numElements) {
  assert(fatBin && "request to launch null fat binary image!");
  assert(kernelName && "request to launch kernel w/ null name!");

  // TODO: we probably want to calculate the launch compilers during code
  // generation (when we have some more information about the actual kernel
  // code -- in that case, we should pass launch parameters to this call.
  size_t threadsPerBlock, blocksPerGrid;
  __kitrt_cuGetLaunchParameters(threadsPerBlock, blocksPerGrid, numElements);

  // TODO: We need a better path here for binding and tracking
  // allcoated CUDA resources -- as it stands we will "leak"
  // modules, streams, and functions...
  CUmodule module;
  CU_SAFE_CALL(cuModuleLoadData_p(&module, fatBin));
  CUfunction kFunc;
  CU_SAFE_CALL(cuModuleGetFunction_p(&kFunc, module, kernelName));
  CUstream stream = nullptr;
  CU_SAFE_CALL(cuStreamCreate_p(&stream, 0));
  CU_SAFE_CALL(cuLaunchKernel_p(kFunc, blocksPerGrid, 1, 1, threadsPerBlock, 1,
                                1, 0, stream, fatBinArgs, NULL));
  return (void *)stream;
}


// TODO: This call currently uses a hard-coded kernel name in the
// launch.  Once that is fixed in the ABI code, we can repalce this
// call with the FB ("fat binary") launch call above...
CUstream __kitrt_cuLaunchELFKernel(void *elf, 
                                   void **args, 
                                   size_t numElements) {
  CUmodule module;
  CUfunction kernel;
  CU_SAFE_CALL(cuModuleLoadDataEx_p(&module, elf, 0, 0, 0));
  CU_SAFE_CALL(cuModuleGetFunction_p(&kernel, module, "kitsune_kernel"));
  CUstream stream = nullptr;
  CU_SAFE_CALL(cuStreamCreate_p(&stream, 0));
  size_t threadsPerBlock, blocksPerGrid;
  __kitrt_cuGetLaunchParameters(threadsPerBlock, blocksPerGrid, numElements);
  CU_SAFE_CALL(cuLaunchKernel_p(kernel, blocksPerGrid, 1, 1, // grid dim
                                threadsPerBlock, 1, 1,       // block dim
                                0, stream,    // shared mem and stream
                                args, NULL)); // arguments
  return stream;
}

void *__kitrt_cuLaunchKernel(llvm::Module &m, void **args, size_t n) {
  std::string ptx = __kitrt_cuLLVMtoPTX(m, _kitrtCUdevice);
  void *elf = __kitrt_cuPTXtoELF(ptx.c_str());
  void *stream = (void *)__kitrt_cuLaunchELFKernel(elf, args, n);
  return stream;
}

void __kitrt_cuStreamSynchronize(void *vs) {
  assert(vs && "request to synchrnoize null stream!");
  CU_SAFE_CALL(cuStreamSynchronize_p((CUstream)vs));
}
