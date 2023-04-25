//
//===- hip.h - Kitsune ABI runtime debug support    --------------===//
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

#ifndef KitRT_HIP_H_
#define KitRT_HIP_H_

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

// ---- Initialization, properties, clean up, etc.


/// @brief Initialize HIP support within the kitsune runtime.
/// @return True on success, false otherwise.
bool __kitrt_hipInit();

/// @brief Handle a +xnack-specific configuration. call before __kitrt_hipInit(). 
/// @return void 
void __kitrt_hipEnableXnack(); 

/// @brief Clean up -- release resources for HIP support.
void __kitrt_hipDestroy();

// ---- Managed memory allocation, tracking, etc.

/// @brief Allocate a region of managed memory.
/// @param size: The number of bytes to allocate.
/// @return A pointer to the allcoated region.
__attribute__((malloc)) void *__kitrt_hipMemAllocManaged(size_t size);

/// @brief Free an allocated region of managed memory.
/// @param vp The pointer to the allocated region.
void __kitrt_hipMemFree(void *vp);

/// @brief Is the given pointer within a managed memory allocation?
/// @param vp The pointer in question.
/// @return True if the pointer is within a managed region, false otherwise.
bool __kitrt_hipIsMemManaged(void *vp);

/// @brief Enable "auto" prefetch mode for managed memory kernel parameters.
void __kitrt_hipEnablePrefetch();

/// @brief Disable  "auto" prefetch mode for managed memory kernel parameters.
void __kitrt_hipDisablePrefetch();

/// @brief Asynchronously prefetch data to the GPU.
/// @param vp: The starting address to prefetch.
/// @param size: The number of bytes to prefetch.
void __kitrt_hipMemPrefetchAsync(void *vp, size_t size);

/// @brief  Prefetch the specified data to the GPU.
/// @param vp: Pointer to kitsune runtime managed memory.
void __kitrt_hipMemPrefetch(void *vp);

/// @brief Prefetch the specified data on the given stream.
/// @return A new stream the prefetch was issued on.
/// @param vp: starting address to prefetch.
void* __kitrt_hipStreamMemPrefetch(void *vp);

/// @brief Prefetch the specified data on the given stream.
/// @param vp: starting address to prefetch.
/// @param stream: the stream to issue the prefetch on.
void __kitrt_hipMemPrefetchOnStream(void *vp, void *stream);

/// @brief Copy the given symbol from host to device.
/// @param hostPtr: Pointer to the host-side symbol.
/// @param devPtr: Pointer to the destination symbol on the GPU.
/// @param size: Size in bytes to copy.
void __kitrt_hipMemcpySymbolToDevice(void *hostPtr, void *devPtr, size_t size);

// ---- Kernel operations, launching, streams, etc.
void *__kitrt_hipModuleLoadData(const void *image);

void *__kitrt_hipGetGlobalSymbol(const char *symName, void *mod);

void __kitrt_hipLaunchKernel(const void *fatBin,     // fat binary w/ kernel
                             const char *kernelName, // kernel to launch
                             void *kernelArgs,       // args to kernel
                             uint64_t numElements,   // trip count
                             void *stream,           // stream to run in
                             size_t argSize);        // size in bytes of KernelArgs

void  __kitrt_hipLaunchModuleKernel(void *module,      // module handle
                              const char *kernelName,  // kernel to launch
                              void       *kernelArgs,  // args to kernel
                              uint64_t    numElements, // trip count
                              void       *stream,      // stream to run in
                              size_t     argSize);     // size of kernel args (bytes)

void __kitrt_hipStreamSynchronize(void *vStream);

void __kitrt_hipSynchronizeStreams();

// ---- Event management and handling.
void *__kitrt_hipCreateEvent();

void __kitrt_hipDestroyEvent(void *E);

void __kitrt_hipEventRecord(void *E);

void __kitrt_hipSynchronizeEvent(void *E);

float __kitrt_hipElapsedEventTime(void *start, void *stop);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
