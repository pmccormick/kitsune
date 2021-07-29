/*
 * Copyright (c) 2020 Triad National Security, LLC
 *                         All rights reserved.
 *
 * This file is part of the kitsune/llvm project.  It is released under
 * the LLVM license.
 */
#ifdef KITSUNE_CONFIG_H
#error clang/Config/kitsune.h should only be included once.
#else
#define KITSUNE_CONFIG_H

/* This should really never happen but just in case some 
 * old code sneaks in... 
 */
#ifndef CLANG_ENABLE_KITSUNE
#error kitsune config included when CLANG_ENABLE_KITSUNE undefined. 
#endif

#cmakedefine01 KITSUNE_MANGLE_PARALEL_KEYWORDS 

#cmakedefine01 KITSUNE_ENABLE_KOKKOS_SUPPORT

#cmakedefine01 KITSUNE_ENABLE_OPENMP_TARGET 
#camkedefine01 KITSUNE_ENABLE_QTHREADS_TARGET
#cmakedefine01 KITSUNE_ENABLE_REALM_TARGET 
#cmakedefine01 KITSUNE_ENABLE_CUDATK_TARGET 
#cmakedefine01 KITSUNE_ENABLE_HIP_TARGET 
#cmakedefine01 KITSUNE_ENABLE_OPENCL_TARGET

/* kitsune-centric cfg file search location */
#cmakedefine KITSUNE_CONFIG_FILE_DIR "${KITSUNE_CONFIG_FILE_DIR}"

/* kitsune+tapir ABI/runtime codegen targets */
/* Note: The OpenCilk target is (now) always enabled */
#cmakedefine01 KITSUNE_ENABLE_OPENMP_TARGET 
#cmakedefine01 KITSUNE_ENABLE_QTHREADS_TARGET 
#cmakedefine01 KITSUNE_ENABLE_CUDATK_TARGET
#cmakedefine01 KITSUNE_ENABLE_REALM_TARGET 

/* kitsune+tapir default configuration filenames */
#cmakedefine KITSUNE_KOKKOS_CFG_FILENAME "${KITSUNE_KOKKOS_CFG_FILENAME}"
#cmakedefine TAPIR_SERIAL_TARGET_CFG_FILENAME "${TAPIR_SERIAL_TARGET_CFG_FILENAME}"
#cmakedefine TAPIR_OPENCILK_TARGET_CFG_FILENAME "${TAPIR_OPENCILK_TARGET_CFG_FILENAME}"
#cmakedefine TAPIR_CUDA_TARGET_CFG_FILENAME "${TAPIR_CUDA_TARGET_CFG_FILENAME}"
#cmakedefine TAPIR_REALM_TARGET_CFG_FILENAME "${TAPIR_REALM_TARGET_CFG_FILENAME}"
#cmakedefine TAPIR_OPENMP_TARGET_CFG_FILENAME "${TAPIR_OPENMP_TARGET_CFG_FILENAME}"
#cmakedefine TAPIR_QTHREADS_TARGET_CFG_FILENAME "${TAPIR_QTHREADS_TARGET_CFG_FILENAME}"
#cmakedefine TAPIR_OPENCL_TARGET_CFG_FILENAME "${TAPIR_OPENCL_TARGET_CFG_FILENAME}"
#cmakedefine TAPIR_HIP_TARGET_CFG_FILENAME "${TAPIR_HIP_TARGET_CFG_FILENAME}"

#endif
