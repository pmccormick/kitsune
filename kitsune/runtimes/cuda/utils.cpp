
/*
 * Copyright (c) 2020 Triad National Security, LLC
 *                         All rights reserved.
 *
 * This file is part of the kitsune/llvm project.  It is released under
 * the LLVM license.
 */
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <sys/stat.h>
#include <cuda.h>
#include <nvvm.h>


extern "C"
const char *_cuabi_read_kernel_file(const char *filename, size_t *bufSize) {
  FILE *fp;
  char *buffer = 0;

  if ((fp = fopen(filename, "rb")) != NULL) {
    struct stat fileStats;
    if (stat(filename, &fileStats) == -1) {
      perror("cuabi_load_llvm() -- unable to stat input file!");
      fclose(fp);
      return NULL;
    }

    /* TODO: This code is geared to read text-based versions
     * of LLVM IR or PTX.  As such we append a null character at
     * the end of the file's contents -- not sure if this will trip
     * us up when reading llvm bitcode files.
     */
    *bufSize = fileStats.st_size;
    buffer = (char *)malloc(*bufSize + 1);
    assert(buffer != NULL && "failed to allocate buffer!");

    size_t itemsRead = fread(buffer, sizeof(char), *bufSize, fp);
    assert(itemsRead == *bufSize &&
           "cuabi_load_llvm() : incorrect read count on llvm file!");
    buffer[*bufSize - 1] = '\0';
    fclose(fp);
  }

  return buffer;
}

extern "C" 
const char *cuabiReadLLVMKernel(const char *filename, size_t *bufSize) {
  assert(filename != NULL && "null filename!");
  assert(bufSize != NULL && "null buffer size parameter!");
  return _cuabi_read_kernel_file(filename, bufSize);
}

extern "C"
const char *cuabiReadPTXKernel(const char *filename, size_t *bufSize) {
  assert(filename != NULL && "null filename!");
  assert(bufSize != NULL && "null buffer size parameter!");
  return _cuabi_read_kernel_file(filename, bufSize);
}

extern "C" 
void _cuabi_report_error(const char *mesg,
                         const char *filename, int line) {
  fprintf(stderr, "cuabi error (%s:%d): %s\n", filename, line, mesg);
  abort();
}

extern "C"
void __cuabi_report_cu_error(CUresult result, const char *mesg,
                             const char *filename, int line) {
  const char *cuErrorMessage = 0;
  const char *cuErrorName = 0;
  cuGetErrorString(result, &cuErrorMessage);
  cuGetErrorName(result, &cuErrorName);
  if (mesg)
    fprintf(stderr, "%s:%d (cuda error -- %s:'%s') : %s\n", filename, line,
            cuErrorName, cuErrorMessage, mesg);
  else
    fprintf(stderr, "%s:%d (cuda error -- %s:'%s') : <no additional info provided>\n", filename, line,
            cuErrorName, cuErrorMessage);
  #ifdef CUABI_ABORT_ON_CUDA_ERROR
  abort();
  #endif
}

extern "C"
void _cuabi_report_nvvm_error(nvvmResult result, const char *mesg,
                               const char *filename, int line) {
  if (mesg)
    fprintf(stderr, "%s:%d (nvvm error -- %04d,'%s') : %s\n", filename, line,
            result, nvvmGetErrorString(result), mesg);
  else
    fprintf(stderr, "%s:%d (nvvm error -- %04d,'%s') : <no additional info provided>\n", filename, line,
            result, nvvmGetErrorString(result));

  #ifdef CUABI_ABORT_ON_NVVM_ERROR
  abort();
  #endif
}

extern "C"
void _cuabi_report_nvvm_compile_error(nvvmProgram prog, const char *mesg,
                                       const char *filename, int line) {
  fprintf(stderr, "(%s,%d) : nvvm compilation error.\n", filename, line);
  if (mesg != 0)
    fprintf(stderr, "\ncuabi error: %s\n\n");
  char *logMesg = NULL;
  size_t logSize = 0;
  nvvmGetProgramLogSize(prog, &logSize);
  logMesg = (char *)malloc(logSize);
  nvvmGetProgramLog(prog, logMesg);
  fprintf(stderr, "%s\n\n", logMesg);
  free(logMesg);
}
