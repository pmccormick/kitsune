/*
 * Copyright (c) 2020 Triad National Security, LLC
 *                         All rights reserved.
 *
 * This file is part of the kitsune/llvm project.  It is released under 
 * the LLVM license.
 */


// 
// TODO: Need to change launch code for performance improvements: 
//     https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH
// 
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "CudaRT/cudart.h"

#ifndef NDEBUG

  #define CHECK_KITCUDART_ERROR(err) __kitrt_cudart_err_check(err, __FILE__, __LINE__)

  extern "C"
  inline void __kitrt_cudart_err_check(CUresult ErrorID, const char *Filename, int LineNumber) {
    if (ErrorID != CUDA_SUCCESS) {
      const char *ErrorMessage = 0;
      const char *ErrorName = 0;
      cuGetErrorString(ErrorID, &ErrorMessage);
      cuGetErrorName(ErrorID, &ErrorName);
      fprintf(stderr, "%s:%d (%04d:%s) -- cuda (driver api) runtime error: '%s'\n",
         	    Filename, LineNumber, ErrorID, ErrorName, ErrorMessage);
      exit(EXIT_FAILURE);
    }
  }

#else

  #define CHECK_KITCUDART_ERROR(err) // no-op

#endif


// ---- Kernel-specific runtime data structures. 

// We flag data as blocks that need explicit copies between host and device 
// memories (or vice versa) and scalars that can be handled automatically by
// the driver API.  This affords us some simple logic to automatically invoke 
// data movement calls before and after kernel launches. 
const CudaRTArgKind CUDART_ARG_TYPE_UNKNOWN  = 0x0;     // unknown flag -- initialized state. 
const CudaRTArgKind CUDART_ARG_TYPE_SCALAR   = 0x1;     // scalar argument 
const CudaRTArgKind CUDART_ARG_TYPE_DATA_BLK = 0x2;     // data block that must explicitly be moved. 

// Flag each argument with information about how it is accessed (e.g., read-from, 
// written to, both).  This helps determine what kernel arguments must be copied 
// to/from the device and back to host memory prior and after kernel launch. 
const CudaRTArgAccessMode CUDART_UNKNOWN_ACCESS = 0x3;  // unknown access -- initialized state. 
const CudaRTArgAccessMode CUDART_ARG_READ_ONLY  = 0x4;  // LHS access within the kernel code. 
const CudaRTArgAccessMode CUDART_ARG_WRITE_ONLY = 0x5;  // RHS access within the kernel code. 
const CudaRTArgAccessMode CUDART_ARG_READ_WRITE = 0x6;  // Both LHS and RHS access within the kernel.  

// The information needed by the runtime to manage a single kernel parameter/argument. 
struct CudaRTKernelArg {
  CudaRTKernelArg() {
    kind     = CUDART_ARG_TYPE_UNKNOWN; 
    access   = CUDART_UNKNOWN_ACCESS;
    host_ptr = 0;
    dev_ptr  = 0;
    size     = 0;
  };

  CudaRTArgKind        kind;
  CudaRTArgAccessMode  access;
  size_t               size;
  CUdeviceptr          dev_ptr;
  void                 *host_ptr;
};

struct CudaRTKernelInfo {
  CudaRTKernelInfo()  { 
    args.reserve(16);
    module = 0;
    function = 0;
    dev_id = -1;

    grid_dims[0]  = grid_dims[1]  = grid_dims[2]  = -1;
    block_dims[0] = block_dims[2] = block_dims[2] = -1;
  }
    
  CUmodule    module;
  CUfunction  function;
  int         dev_id;
  int         grid_dims[3];
  int         block_dims[3];
  std::vector<CudaRTKernelArg> args;
};


// ---- Device/GPU-specific runtime data structures. 
// TODO: Flush this out with compute capabilities, etc. 
struct CudaRTDeviceInfo {
  CUcontext    Context;
  CUdevice     Device;
};

static bool __kitsune_cudart_initialized = false;
static bool __kitsune_verbose_rt = false;
static std::vector<CudaRTDeviceInfo> devices;
static std::vector<CudaRTKernelInfo> kernels;


extern "C" 
void __kitsune_check_env_vars() {
  if (getenv("KITSUNE_ENV_VERBOSE_RT"))
    __kitsune_verbose_rt = true;
}

extern "C"
void __kitsune_cudart_initialize() {

  __kitsune_check_env_vars();
    
  CHECK_KITCUDART_ERROR( cuInit(0) );
  
  int DeviceCount = 0;
  CHECK_KITCUDART_ERROR( cuDeviceGetCount(&DeviceCount) );
  
  CudaRTDeviceInfo DeviceInfo;
  for(unsigned int devID = 0; devID < DeviceCount; ++devID) {
    CHECK_KITCUDART_ERROR( cuDeviceGet(&DeviceInfo.Device, devID) );
    CHECK_KITCUDART_ERROR( cuCtxCreate(&DeviceInfo.Context, 0, DeviceInfo.Device) );
    devices.push_back(DeviceInfo);
  }

  if (DeviceCount > 0)
    __kitsune_cudart_initialized = true;
}

extern "C"
void __kitsune_cudart_finalize() {
  assert(__kitsune_cudart_initialized == true && "attempt at finalizing uninitialized runtime");

  for(int dev_id = 0; dev_id < __kitsune_cudart_ndevices(); ++dev_id)
    CHECK_KITCUDART_ERROR( cuCtxDestroy( devices[dev_id].Context) );
}


extern "C"
inline int __kitsune_cudart_ndevices() {
  assert(__kitsune_cudart_initialized == true && "runtime was not initialized prior to call");
  return devices.size();
}


extern "C"
inline int __kitsune_cudart_nkernels() {
  // TODO: We probably want something per context/device here... 
  assert(__kitsune_cudart_initialized == true && "runtime was not initialized prior to call");
  return kernels.size();
}

extern "C"
int __kitsune_cudart_create_kernel(int devID, const char *ptxSource, const char *funcName) {
  assert(__kitsune_cudart_initialized == true && "runtime was not initialized prior to call");
  assert(ptxSource != nullptr && "null ptx source string");
  assert(funcName != nullptr && "null function name string");
  assert(devID < __kitsune_cudart_ndevices() && "invalid device id provided");
  

  CudaRTKernelInfo KernInfo;
  CHECK_KITCUDART_ERROR( cuModuleLoadData(&KernInfo.module, ptxSource) );
  CHECK_KITCUDART_ERROR( cuModuleGetFunction(&KernInfo.function, KernInfo.module, funcName) );

  kernels.push_back(KernInfo);
  return kernels.size() - 1;
}


extern "C"
void __kitsune_cudart_add_arg(int kernID, void *hostArgPtr, size_t argSizeInBytes,
                              CudaRTArgKind kind, CudaRTArgAccessMode modeFlag) {

  assert(__kitsune_cudart_initialized == true && "runtime was not initialized prior to call");
  assert(kernID < __kitsune_cudart_nkernels() && "invaild kernel id provided");
  assert(hostArgPtr != 0 && "null host data pointer");
  assert(argSizeInBytes > 0 && "zero-sized argument");

  CudaRTKernelArg ArgInfo;
  
  ArgInfo.host_ptr = hostArgPtr;
  ArgInfo.kind = kind;
  ArgInfo.access = modeFlag;  
  ArgInfo.size = argSizeInBytes;

   // Data blocks (arrays) get device-side memory allocations.
  if (kind == CUDART_ARG_TYPE_DATA_BLK) {
    CHECK_KITCUDART_ERROR( cuMemAlloc(&(ArgInfo.dev_ptr), ArgInfo.size) );
    // Copy down to device iff a 'read-from' target.
    if (modeFlag == CUDART_ARG_READ_ONLY || modeFlag == CUDART_ARG_READ_WRITE) 
      CHECK_KITCUDART_ERROR( cuMemcpyHtoD(ArgInfo.dev_ptr, ArgInfo.host_ptr, argSizeInBytes) );
  } else
    ArgInfo.dev_ptr = 0;


  kernels[kernID].args.push_back(ArgInfo);
}

extern "C"
void __kitsune_cudart_set_grid_dims(int kernID, int dimX, int dimY, int dimZ) {
  assert(dimX > 0 && dimY > 0 && dimZ > 0 && "grid dims must all be > 0.");
  assert(kernID < __kitsune_cudart_nkernels() && "invalid kernel id provided");
  kernels[kernID].grid_dims[0] = dimX;
  kernels[kernID].grid_dims[1] = dimY;
  kernels[kernID].grid_dims[2] = dimZ;
}

extern "C"
void __kitsune_cudart_set_block_dims(int kernID, int dimX, int dimY, int dimZ) {
  assert(dimX > 0 && dimY > 0 && dimZ > 0 && "block dims must all be > 0.");
  assert(kernID < __kitsune_cudart_nkernels() && "invalid kernel id provided");
  kernels[kernID].block_dims[0] = dimX;
  kernels[kernID].block_dims[1] = dimY;
  kernels[kernID].block_dims[2] = dimZ;
}

extern "C" 
void __kitsune_cudart_set_kernel_params(int kernID, 
                                        int gridDims[3],
                                        int blockDims[3]) {
  assert(kernID < kernels.size() && "invalid kernel ID");
  __kitsune_cudart_set_grid_dims(kernID, gridDims[0], gridDims[1], gridDims[2]);
  __kitsune_cudart_set_block_dims(kernID, blockDims[0], blockDims[1], blockDims[2]);
}

extern "C"
void __kitsune_cudart_launch_kernel(int kernID) {
  assert(kernID < kernels.size() && "invalid kernel ID");

  CudaRTKernelInfo &Kinfo = kernels[kernID];
  assert(Kinfo.block_dims[0] > 0 && Kinfo.block_dims[1] > 0 && 
         Kinfo.block_dims[2] > 0 && "kernel must have block dimensions >= 1");
  assert(Kinfo.grid_dims[0] > 0 && Kinfo.grid_dims[1] > 0 && 
         Kinfo.grid_dims[2] > 0 && "kernel must have grid dimensions >= 1");

  int argCount = Kinfo.args.size();
  void *Kargs[16];
  assert(argCount <= 16 && "kernel has too many arguments!");

  memset(Kargs, 0, 16 * sizeof(void*));

  for(int i = 0; i < argCount; ++i) {
    if (Kinfo.args[i].kind == CUDART_ARG_TYPE_DATA_BLK)
      Kargs[i] = &(Kinfo.args[i].dev_ptr);
    else
      Kargs[i] = Kinfo.args[i].host_ptr;
  }

  CHECK_KITCUDART_ERROR( cuCtxSynchronize() );

  int *grid_dims  = &(Kinfo.grid_dims[0]);
  int *block_dims = &(Kinfo.block_dims[0]); 
  CHECK_KITCUDART_ERROR( cuLaunchKernel(Kinfo.function, 
                                        grid_dims[0], grid_dims[1], grid_dims[2],
                                        block_dims[0], block_dims[1], block_dims[2], 
                                        0, NULL, Kargs, NULL) );

  // Copy data back after kernel has run... 
  for(int i = 0; i < kernels[kernID].args.size(); ++i) {
    if (Kinfo.args[i].access == CUDART_ARG_WRITE_ONLY ||
	      Kinfo.args[i].access == CUDART_ARG_READ_WRITE) {
        assert(Kinfo.args[i].dev_ptr != 0 && "attempted copy from null device pointer");
        CHECK_KITCUDART_ERROR( cuMemcpyDtoH((void*)Kinfo.args[i].host_ptr,
					                                  Kinfo.args[i].dev_ptr, Kinfo.args[i].size) );
    }
  }
}
