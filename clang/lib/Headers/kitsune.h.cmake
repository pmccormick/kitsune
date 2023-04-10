
/*
 * Copyright (c) 2020 Triad National Security, LLC
 *                         All rights reserved.
 *
 * This file is part of the kitsune/llvm project.  It is released under
 * the LLVM license.
 */
#ifndef __CLANG_KITSUNE_H__
#define __CLANG_KITSUNE_H__

#include <stdint.h>

#cmakedefine01 KITSUNE_ENABLE_OPENMP_ABI_TARGET
#cmakedefine01 KITSUNE_ENABLE_QTHREADS_ABI_TARGET
#cmakedefine01 KITSUNE_ENABLE_CUDA_ABI_TARGET
#cmakedefine01 KITSUNE_ENABLE_GPU_ABI_TARGET
#cmakedefine01 KITSUNE_ENABLE_REALM_ABI_TARGET
#cmakedefine01 KITSUNE_ENABLE_OPENCL_ABI_TARGET
#cmakedefine01 KITSUNE_ENABLE_HIP_ABI_TARGET

#include "kitsune_rt.h"

#if defined(KITSUNE_ENABLE_OPENCL_ABI_TARGET)
#define ocl_mmap(a, n) __kitsune_opencl_mmap_marker((void*)a, n)
#ifdef __cplusplus
extern "C" {
#endif
  void __kitsune_opencl_mmap_marker(void* ptr, uint64_t n);
#ifdef __cplusplus
}
#endif
#endif

#if defined(spawn)
#warning encountered multiple definitions of spawn!
#else
#define spawn _kitsune_spawn
#endif

#if defined(sync)
#warning encountered multiple definitions of sync!
#else
#define sync _kitsune_sync
#endif

#if defined(forall)
#warning encountered multiple definitions of forall!
#else
#define forall _kitsune_forall
#endif


#if defined(_tapir_cuda_target)
  #ifdef __cplusplus
    extern "C" void* __kitrt_cuMemAllocManaged(size_t);
    template <typename T>
    inline __attribute__((always_inline))
      T* alloc(size_t N) {
      return (T*)__kitrt_cuMemAllocManaged(sizeof(T) * N);
    }
  
    extern "C" void __kitrt_cuMemFree(void*);
    template <typename T>
    void dealloc(T* array) {
      __kitrt_cuMemFree((void*)array);
    }
  #else 
    inline __attribute__((always_inline))
    void *alloc(size_t total_bytes) {
      return __kitrt_cuMemAllocManaged(total_bytes);
    }

    inline __attribute__((always_inline))
    void dealloc(void *array) {
      __kitrt_cuMemFree(array);
    }
  #endif
#elif defined(_tapir_hip_target)
  #ifdef __cplusplus
    extern "C" void* __kitrt_hipMemAllocManaged(size_t);
    template <typename T>
    inline __attribute__((always_inline))
      T* alloc(size_t N) {
      return (T*)__kitrt_hipMemAllocManaged(sizeof(T) * N);
    }
  
    extern "C" void __kitrt_hipMemFree(void*);
    template <typename T>
    void dealloc(T* array) {
      __kitrt_hipMemFree((void*)array);
    }
  #else 
    inline __attribute__((always_inline))
    void *alloc(size_t total_bytes) {
      return __kitrt_hipMemAllocManaged(total_bytes);
    }

    inline __attribute__((always_inline))
    void dealloc(void *array) {
       __kitrt_hipMemFree(array);
    }
  #endif 
#else 
  // For CPU-side runtimes just use malloc... 
  #ifdef __cplusplus
    #include <cstdlib>
    template <typename T>
    inline __attribute__((always_inline))
      T* alloc(size_t N) {
      return (T*)malloc(sizeof(T) * N);	
    }
  
    extern "C" void __kitrt_hipMemFree(void*);
    template <typename T>
    void dealloc(T* array) {
      free(array);
    }
  #else 
    #include <stdlib.h>
    inline __attribute__((always_inline))
    void *alloc(size_t total_bytes) {
      return (void*)malloc(total_bytes);
    }

    inline __attribute__((always_inline))
    void dealloc(T* array) {
       free(array);
    }
  #endif // __cplusplus
#endif // cpu targets

#endif
