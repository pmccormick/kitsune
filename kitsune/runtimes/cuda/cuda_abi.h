/*
 * Copyright (c) 2020 Triad National Security, LLC
 *                         All rights reserved.
 *
 * This file is part of the kitsune/llvm project.  It is released under
 * the LLVM license.
 */
#ifndef __CUDA_ABI_H__
#define __CUDA_ABI_H__

#include <stdint.h>
#ifndef __cplusplus 
#include <stdbool.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

  /* General notes:
   *
   * - To simplify code generation the runtime does not gracefully fail. 
   *   Instead it keeps these details opaque to the caller and uses hard 
   *   errors that terminate program execution.  
   *
   * - The current model for multiple GPUs is that a thread-of-control for
   *   each GPU in a system should call cuabiInit().  This will establish 
   *   a path for a single CUDA context per thread as well as support 
   *   asynchronous execution across multile GPUs. 
   */

  typedef int   gpu_id_t;     // A unique ID for a GPU in the system. 
  typedef void* gpu_dev_t;    // An opaque handle to the GPU device. 
  typedef void* kernel_t;     // An opaque handle to a GPU kernel. 

  /*
   * Initialize the ABI layer.  Upon success the call returns the ID of the 
   * GPU assigned to the calling thread.  Any internal errors will result in 
   * a hard error, an error message to stderr, and then a call to abort().
   * If there are more threads calling into cuabiInit() than there are GPUs
   * on the system some of those calls will receive -1 as the return value; 
   * this means that there were no available GPUs for the calling thread. 
   */
  extern gpu_id_t cuabiInit();

  /*
   * Return the number of GPUs available on the system.
   */
  extern size_t cuabiNumberOfGPUs();

  /*
   * Check to see if the given gpu device ID is valid. The GPU device ID is a
   * unique integer numbered from 0 to N-1, where N is the number of GPUs
   * successfully intialized via a call to cuabiInit().
   */
  extern bool cuabiValidID(const gpu_id_t id);

  /*
   * Read a kernel represented in LLVM IR from the given file.  A pointer to
   * the character buffer holding the IR is returned if the read is successful,
   * otherwise a null pointer is returned.  In addition to the IR buffer,
   * the size of the allocated buffer is returned in the 'bufSize' parameter.
   * 
   * NOTE: The bufSize parameter addresses potential null characters within LLVM 
   * bitcode files.
   */
  extern const char *cuabiReadLLVMKernel(const char *filename, size_t *bufSize);

  /* 
   * Read a kernel represented in PTX from the given file.  A pointer to the
   * character buffer holding the IR is returned if the read is successful, 
   * otherwise a null pointer is returned.  In addition, to the PTX buffer
   * the size of the allocated buffer is returned in the 'bufSize' parameter. 
   * 
   * NOTE: Not sure we need the bufSize parameter here.  Not sure that there 
   * is a binary form of PTX so it could always be null terminated but opted 
   * to keep interface identical to the LLVM IR call above. 
   */
  extern const char *cuabiReadPTXKernel(const char *filename, size_t *bufSize);

  /*
   * Convert the given LLVM IR buffer, of the given size, into a PTX form
   * suitable for loading and execution by the CUDA Driver API.  The resulting
   * PTX form should use the given module name as the entry point of the
   * kernel.  This call returns a text buffer containing the resulting PTX, or
   * null if the transformation failed (any specific errors will be sent to
   * stderr).
   *
   * TODO: Should we consider another path for handling errors during the
   * LLVM-to-PTX transformation?
   */
  extern const char * cuabiLLVMToPTX(const char *LLVMbuffer,
                                     size_t bufSize,
                                     const char *modName);
 
  /*
   * Load the given PTX kernel so it is available for use by the runtime
   * for kernel launches.  The call takes a null-terminated buffer holding
   * the PTX source and the name of the kernel entry point.  The returned
   * value is a unique handle associated with the kernel.  It should be
   * used in later calls to launch, update, and destroy the kernel.  If
   * an error occurs when loading the kernel the handle will be equal to
   * zero (using cuabi_is_valid_kernel() is the preferred path for kernel
   * validation).
   */
  extern kernel_t cuabiLoadPTXKernel(const char *PTXbuffer,
                                     const char *kernelName);

  /*
   * Load the given LLVM IR kernel so it is available for use by the runtime
   * for kernel launches. The call takes a null-terminated buffer holding
   * the LLVM IR and the name of the associated kernel (for identifying the
   * entry point). If an error occurs when loading the kernel the handle
   * will be equal to zero (using cuabi_is_valid_kernel() is the preferred
   * path for kernel validation).
   */
  extern kernel_t cuabiLoadLLVMKernel(const char *LLVMbuffer,
                                      size_t bufferSize, 
                                      const char *kernelName);

  /*
   * Check for a valid kernel. 
   */
  bool cuabiValidKernel(const kernel_t kernel);

  void cuabiSetKernelGridDims(kernel_t kernel, 
                              unsigned int gridDimX,
                              unsigned int gridDimY,
                              unsigned int gridDimZ);

  void cuabiSetKernelBlockDims(kernel_t kernel,
                               unsigned int blockDimX,
                               unsigned int blockDimY,
                               unsigned int blockDimZ);

  /*
   * Launch the given kernel with the given cuda-style parameters.
   */
  extern void cuabiLaunchKernel(kernel_t kernel, gpu_id_t gpu_id, 
                                unsigned int gridDimX, unsigned int gridDimY,
                                unsigned int gridDimZ, unsigned int blockDimX,
                                unsigned int blockDimY, unsigned int blockZDim,
                                void **params);
#ifdef __cplusplus 
}
#endif

#endif
