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
#define KITSUNE_KOKKOS_INCLUDE_DIR    "${Kokkos_INCLUDE_DIR}"
#define KITSUNE_KOKKOS_LIBRARY_DIR    "${Kokkos_LIBRARY_DIR}"
#define KITSUNE_KOKKOS_LINK_LIBS      "${Kokkos_LINK_LIBS}${KITSUNE_EXTRA_KOKKOS_LINK_LIBS}"

/* Automatic command line arguments to add when compiling with
 * the Cilk runtime backend target (via -ftapir=cilk).
 */ 
#cmakedefine01 KITSUNE_ENABLE_CILKRTS_TARGET 
#define KITSUNE_CILKRTS_LIBRARY_DIR      "${CilkRTS_LIBRARY_DIR}"
#define KITSUNE_CILKRTS_LINK_LIBS        "${CilkRTS_LINK_LIBS}${KITSUNE_EXTRA_CILKRTS_LINK_LIBS}"

/* Automatic command line arguments to add when compiling with
 * the Qthreads runtime backend target (via -ftapir=qthreads).
 */ 
#cmakedefine01 KITSUNE_ENABLE_QTHREADS_TARGET   
#define KITSUNE_QTHREADS_LIBRARY_DIR     "${Qthreads_LIBRARY_DIR}"
#define KITSUNE_QTHREADS_LINK_LIBS       "${Qthreads_LINK_LIBS}${KITSUNE_EXTRA_QTHREADS_LINK_LIBS}"

/* Automatic command line arguments to add when compiling with
 * the Realm runtime backend target (via -ftapir=realm).
 */ 
#cmakedefine01 KITSUNE_ENABLE_REALM_TARGET
#define KITSUNE_REALM_LIBRARY_DIR        "${Realm_LIBRARY_DIR}"
#define KITSUNE_REALM_LINK_LIBS          "${Realm_LINK_LIBS} ${KITSUNE_EXTRA_REALM_LINK_LIBS}"

/* Automatic command line arguments to add when compiling with
 * the OpenMP runtime backend target (via -ftapir=omp).
 */ 
#cmakedefine01 KITSUNE_ENABLE_OPENMP_TARGET
#define KITSUNE_OPENMP_LINK_LIBS          "-lomp"

/* Automatic command line arguments to add when compiling with
 * the CUDA/PTX runtime backend target (via -ftapir=cuda).
 */ 
#cmakedefine01 KITSUNE_ENABLE_CUDA_TARGET
#define KITSUNE_CUDA_LIBRARY_DIR         "${CUDAToolkit_LIBRARY_DIR}"
#define KITSUNE_CUDA_LINK_LIBS           "${KITSUNE_EXTRA_CUDART_LINK_LIBS} -lcuda"

/* Automatic command line arguments to add when compiling with
 * the OpenCL/SPIR-V runtime backend target (via -ftapir=opencl).
 */ 
#cmakedefine01 KITSUNE_ENABLE_OPENCL_TARGET
#define KITSUNE_OPENCL_LIBRARY_DIR         "${OPENCL_LIBRARY_DIR}"
#define KITSUNE_OPENCL_LINK_LIBS           "${OPENCL_LIBRARIES}"

#endif
