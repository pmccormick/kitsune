/*
 * Copyright (c) 2020 Triad National Security, LLC
 *                         All rights reserved.
 *
 * This file is part of the kitsune/llvm project.  It is released under 
 * the LLVM license.
 */
#ifdef KITSUNE_CONFIG_H
#error kitsune config.h should only be included once.
#else
#define KITSUNE_CONFIG_H

/* Automatic command line arguments to add when compiling with
 * kokkos-centric support enabled (via -fkokkos).
 */ 
#cmakedefine01 KITSUNE_ENABLE_KOKKOS 
#define KOKKOS_INCLUDE_DIR       "${Kokkos_INCLUDE_DIR}"
#define KOKKOS_LIBRARY_DIR       "${Kokkos_LIBRARY_DIR}"
#define KOKKOS_LINK_LIBS         "${Kokkos_LINK_LIBS}"
#define KOKKOS_EXTRA_LINK_LIBS   "${KOKKOS_EXTRA_LINK_LIBS}"

/* Automatic command line arguments to add when compiling with
 * the Cilk runtime backend target (via -ftapir=cilk).
 */ 
#cmakedefine01 KITSUNE_ENABLE_CILKRTS_TARGET 
#define CILKRTS_LIBRARY_DIR      "${CilkRTS_LIBRARY_DIR}"
#define CILKRTS_LINK_LIBS        "${CilkRTS_LINK_LIBS}"
#define CILKRTS_EXTRA_LINK_LIBS  "${CILKRTS_EXTRA_LINK_LIBS}"

/* Automatic command line arguments to add when compiling with
 * the OpenCilk runtime backend target (via -ftapir=opencilk).
 */ 
#cmakedefine01 KITSUNE_ENABLE_OPENCILK_TARGET 
#define OPENCILK_LIBRARY_DIR      "${OpenCilk_LIBRARY_DIR}"
#define OPENCILK_LINK_LIBS        "${OpenCilk_LINK_LIBS}"
#define OPENCILK_EXTRA_LINK_LIBS  "${OPENCILK_EXTRA_LINK_LIBS}"

/* Automatic command line arguments to add when compiling with
 * the Qthreads runtime backend target (via -ftapir=qthreads).
 */ 
#cmakedefine01 KITSUNE_ENABLE_QTHREADS_TARGET   
#define QTHREADS_LIBRARY_DIR     "${Qthreads_LIBRARY_DIR}"
#define QTHREADS_LINK_LIBS       "${Qthreads_LINK_LIBS}"
#define QTHREADS_EXTRA_LINK_LIBS "${QTHREADS_EXTRA_LINK_LIBS}"

/* Automatic command line arguments to add when compiling with
 * the Realm runtime backend target (via -ftapir=realm).
 */ 
#cmakedefine01 KITSUNE_ENABLE_REALM_TARGET
#define KITSUNE_REALM_LIBRARY_DIR        "${Realm_LIBRARY_DIR}"
#define KITSUNE_REALM_WRAPPER_LIBRARY_DIR  "${Realm_WRAPPER_LIBRARY_DIR}"
#define KITSUNE_REALM_LINK_LIBS          "${Realm_LINK_LIBS} ${KITSUNE_EXTRA_REALM_LINK_LIBS}"

/* Automatic command line arguments to add when compiling with
 * the OpenMP runtime backend target (via -ftapir=omp).
 */ 
#cmakedefine01 KITSUNE_ENABLE_OPENMP_TARGET
#define OPENMP_LINK_LIBS          "-lomp"

/* Automatic command line arguments to add when compiling with
 * the CUDA/PTX runtime backend target (via -ftapir=cuda).
 */ 
#cmakedefine01 KITSUNE_ENABLE_CUDA_TARGET
#define CUDA_LIBRARY_DIR         "${CUDAToolkit_LIBRARY_DIR}"
#define CUDA_LINK_LIBS           "${CUDAToolkit_LINK_LIBS}"
#define CUDA_EXTRA_LINK_LIBS     "${CUDA_EXTRA_LINK_LIBS}"

/* Automatic command line arguments to add when compiling with 
 * the Kitsune CUDA runtime backend target (via -ftapir=kitcuda). 
 */
 #cmakedefine01 KITSUNE_ENABLE_KITCUDA_TARGET 
 #define KITCUDA_LIBRARY_DIR        "${KITSUNE_LIBRARY_DIR}"
 #define KITCUDA_LINK_LIBS          "${KITCUDA_LINK_LIBS}"
 #define KITCUDA_EXTRA_LINK_LIBS    "${KITCUDA_EXTRA_LINK_LIBS}"

/* Automatic command line arguments to add when compiling with
 * the OpenCL/SPIR-V runtime backend target (via -ftapir=opencl).
 */ 
#cmakedefine01 KITSUNE_ENABLE_OPENCL_TARGET
#define OPENCL_LIBRARY_DIR         "${OPENCL_LIBRARY_DIR}"
#define OPENCL_LINK_LIBS           "${OPENCL_LIBRARIES}"
#define OPENCL_EXTRA_LINK_LIBS     "${OPENCL_EXTRA_LINK_LIBS}"


#endif
