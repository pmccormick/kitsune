/*
 * Copyright (c) 2020 Triad National Security, LLC
 *                         All rights reserved.
 *
 * This file is part of the kitsune/llvm project.  It is released under
 * the LLVM license.
 */
#include <assert.h>
#include <cstdio>
#include "cuabi_utils.h"

extern "C" 
const char *cuabiLLVMToPTX(const char *llvmBuffer, size_t bufSize,
                           const char *modName) {
  assert(llvmBuffer != NULL && "null LLVM IR buffer!");
  assert(modName != NULL && "null module name!");
  assert(bufSize > 0 && "zero-sized buffer specified!");

  nvvmResult result;
  nvvmProgram program;
  if ((result = nvvmCreateProgram(&program)) != NVVM_SUCCESS) {
    _cuabi_report_nvvm_error(result, "unable to create NVVM program!", 
            __FILE__, __LINE__);
    return NULL;
  }

  result = nvvmAddModuleToProgram(program, llvmBuffer, bufSize, modName);
  if (result != NVVM_SUCCESS) {
    _cuabi_report_nvvm_error(result, "error adding module to program!",
                             __FILE__, __LINE__);
    nvvmDestroyProgram(&program);
    return NULL;
  }

  result = nvvmCompileProgram(program, 0, NULL);
  if (result != NVVM_SUCCESS) {
    _cuabi_report_nvvm_compile_error(program, NULL, __FILE__, __LINE__);
    nvvmDestroyProgram(&program);
    return NULL;
  }

  size_t ptxBufSize = 0;
  result = nvvmGetCompiledResultSize(program, &ptxBufSize);
  if (result != NVVM_SUCCESS) {
    _cuabi_report_nvvm_error(result, "error getting size of compiled program!",
                             __FILE__, __LINE__);
    nvvmDestroyProgram(&program);
    return NULL;
  }
  fprintf(stderr, "ptx buffer size = %ld\n", ptxBufSize);
  char *ptxBuffer = (char *)malloc(ptxBufSize);
  if (ptxBuffer == NULL) {
    _cuabi_report_error("memory allocation request failed!", __FILE__, __LINE__);
    nvvmDestroyProgram(&program);
    return NULL;
  }
  result = nvvmGetCompiledResult(program, ptxBuffer);
  if (result != NVVM_SUCCESS) {
    _cuabi_report_nvvm_error(result, "error getting compiled result.", 
                             __FILE__, __LINE__);
    free(ptxBuffer);
    nvvmDestroyProgram(&program);
    return NULL;
  }

  // Once we have the PTX buffer we no longer need the NVVM program...
  // TODO: Is there any reason we might want to keep track of the corresponding
  // program (e.g., JIT'ing?).
  result = nvvmDestroyProgram(&program);
  if (result != NVVM_SUCCESS)
    _cuabi_report_nvvm_error(result, "error when destroying NVVM program!",
                             __FILE__, __LINE__);
  return ptxBuffer;
}
