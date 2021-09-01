/*
 * Copyright (c) 2020 Triad National Security, LLC
 *                         All rights reserved.
 *
 * This file is part of the kitsune/llvm project.  It is released under
 * the LLVM license.
 */
#ifndef __CUDA_ABI_UTILS_H__
#define __CUDA_ABI_UTILS_H__

#include <cuda.h>
#include <nvvm.h>

extern "C" const char *_cuabi_read_kernel_file(const char *filename,
                                               size_t *bufSize);

#if CUDA_VERSION < 6050
#error CUDA version 6.5 or later required.
#else
#define CU_CHECK(cuCall)                                                     \
  do {                                                                       \
    CUresult _cures = (cuCall);                                              \
    if (_cures != CUDA_SUCCESS) {                                            \
      const char *name, *msg;                                                \
      cuGetErrorName(_cures, &name);                                         \
      cuGetErrorString(_cures, &msg);                                        \
      std::fprintf(stderr, "%s:%d cuabi cuda error - %s() -> %s : %s (%d)\n",\
              __FILE__, __LINE__, #cuCall, name, msg, _cures);               \
      assert(0);                                                             \
      exit(1);                                                               \
    }                                                                        \
  } while (0)
#endif

extern "C" {

// Report a CUABI-centric error message.
extern void _cuabi_report_error(const char *mesg, 
                                const char *filename, int line);

// Report a CUDA-centric error message.  If CUABI_ABORT_ON_CUDA_ERROR
// is defined the function call will abort program execution.  The function
// takes the CUDA error result, a custom error message to display and
// the filename and line number where the error was encountered.
extern void _cuabi_report_cu_error(CUresult result, const char *mesg,
                                    const char *filename, int line);

// Report an NVVM specific error message.  If CUABI_ABORT_ON_NVVM_ERROR
// is defined the function call will abort program execution.  The
// function takes the NNVM error result, a custom error message to
// display and the filename and line number where the error was encountered.
extern void _cuabi_report_nvvm_error(nvvmResult result, const char *mesg,
                                      const char *filename, int line);

// Report an NVVM specific compilation error.  This is geared specifically
// for the LLVM-to-PTX code transformation/generation operation and will
// pull the errors related to this call within NVVM.
extern void _cuabi_report_nvvm_compile_error(nvvmProgram prog,
                                             const char *mesg,
                                             const char *filename, 
                                             int line);
}

#endif
