/*
 * Copyright (c) 2020 Triad National Security, LLC
 *                         All rights reserved.
 *
 * This file is part of the kitsune/llvm project.  It is released under
 * the LLVM license.
 */
#include <cassert>
#include <vector>
#include <thread>
#include <mutex>
#include "cuda_abi.h"
#include "cuabi_utils.h"

#ifndef CUABI_MAX_NUM_GPUS 
#define CUABI_MAX_NUM_GPUS  8
#endif

static bool _cuda_initialized = false;
static int _cuda_num_devices = 0;
static int _next_dev_id = -1;
static std::mutex _cuabi_mutex;

struct gpu_info_t {
  CUdevice device;
  CUcontext context;
};
static std::vector<gpu_info_t> _cuabi_gpu_info(CUABI_MAX_NUM_GPUS);

struct kernel_info_t {
  CUmodule   module;
  CUfunction kernel;
  unsigned int grid_x, grid_y, grid_z;
  unsigned int block_x, block_y, block_z;
};

inline CUdevice get_cuda_device(gpu_id_t id) {
  assert(id >= 0 && id < _cuda_num_devices && "gpu id out of range!");
  return _cuabi_gpu_info[id].device;
}

inline CUcontext get_cuda_context(gpu_id_t id) {
  assert(id >= 0 && id < _cuda_num_devices && "gpu id out of range!");
  return _cuabi_gpu_info[id].context;
}

static gpu_id_t __cuabi_initialize_cuda() {

  // The calling conventions for cuInit() and multi-threading are not quite
  // obvious in the driver API documentation.  In general, it appears that
  // in order to access GPUs simultaneously each thread will need to call
  // into the cuInit()...
  CU_CHECK( cuInit(0) );
  gpu_id_t id = -1;

  { // mutex lifetime scope...

    // We should only enter this at startup so we're a bit lazy about the full
    // mutex coverage here (as well as the implementation -- could probably be
    // a bit more clever...).
    std::lock_guard<std::mutex> lck(_cuabi_mutex);
    if (! _cuda_initialized) {
      CU_CHECK( cuDeviceGetCount(&_cuda_num_devices) );
      assert(_cuda_num_devices <= CUABI_MAX_NUM_GPUS);
      _next_dev_id = 0;
      _cuda_initialized = true;
    }

    if (_next_dev_id < _cuda_num_devices) {
      id = _next_dev_id;
      _next_dev_id++;
    }
  } // end mutex lifetime

  return id;
}

extern "C"
gpu_id_t cuabiInit() {

  gpu_id_t id = __cuabi_initialize_cuda();
  if (id != -1) { 
    // The calling thread has been assigned a valid GPU device. 
    gpu_info_t &info = _cuabi_gpu_info[id];
    CU_CHECK( cuDeviceGet(&info.device, id) );
    CU_CHECK( cuCtxCreate(&info.context, 0, info.device) );
  }
  return id;
}

extern "C"
kernel_t cuabiLoadPTXKernel(const char *ptxBuffer,
                            const char *kernelName) {
  assert(ptxBuffer != NULL && "null ptx source pointer!");
  assert(kernelName != NULL && "null kernel name pointer!");

  kernel_info_t *kinfo = new kernel_info_t;

  CU_CHECK( cuModuleLoadDataEx(&(kinfo->module), ptxBuffer, 0, 0, 0) );
  CU_CHECK( cuModuleGetFunction(&(kinfo->kernel), kinfo->module, kernelName) );
  return (kernel_t)kinfo;
}

extern "C" size_t cuabiNumberOfGPUs() { return _cuda_num_devices; }

extern "C"
kernel_t cuabiLoadLLVMKernel(const char *LLVMbuffer,
                             size_t bufferSize,
                             const char *modName) {
  const char *ptx_buf = cuabiLLVMToPTX(LLVMbuffer, bufferSize, modName);
  return cuabiLoadPTXKernel(ptx_buf, modName);
}

extern "C"
bool cuabiValidKernel(const kernel_t kernel) { return kernel != 0; }

extern "C"
void cuabiSetKernelGridDims(kernel_t kernel, 
                            unsigned int xDim,
                            unsigned int yDim, 
                            unsigned int zDim) {
  kernel_info_t *kinfo = (kernel_info_t*)kernel;
  assert(kinfo != 0);
  kinfo->grid_x = xDim;
  kinfo->grid_y = yDim;
  kinfo->grid_z = zDim;
}

extern "C"
void cuabSetKernelBlockDims(kernel_t kernel, 
                            unsigned int xDim, 
                            unsigned int yDim,
                            unsigned int zDim) {
  kernel_info_t *kinfo = (kernel_info_t*)kernel;
  assert(kinfo != 0);
  kinfo->block_x = xDim;
  kinfo->block_y = yDim;
  kinfo->block_z = zDim;
}

extern "C"
void cuabiLaunchKernel(kernel_t kernel, gpu_id_t id,
                       unsigned int gridDimX, unsigned int gridDimY,
                       unsigned int gridDimZ, unsigned int blockDimX,
                       unsigned int blockDimY, unsigned int blockDimZ,
                       void **params) {
  assert(kernel != 0 && "null kernel!");
  kernel_info_t *kinfo = (kernel_info_t *)kernel;

  CU_CHECK( cuLaunchKernel(kinfo->kernel,
                           gridDimX, gridDimY, gridDimZ, 
                           blockDimX, blockDimY, blockDimZ, 
                           0, NULL, params, NULL) );
}
