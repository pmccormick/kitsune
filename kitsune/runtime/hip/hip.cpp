//===- hip.cpp - Kitsune runtime HIP support    ------------------------===//
//
// TODO:
//     - Need to update LANL/Triad Copyright notice.
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

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <list>
#include <map>
#include <sstream>
#include <stdbool.h>
#include <unordered_map>
#include <vector>

// NOTE: HIP sprinkles some templated functions into the API and
// that can trip us up -- given we are trying to simplify with a
// C-friendly API to ease code generation woes.  Fortunately, we
// can currently disable inclusion of these calls...
#define __HIP_DISABLE_CPP_FUNCTIONS__
#define __HIP_PLATFORM_HCC__ 1
#include <hip/hip_runtime.h>

#include "../debug.h"
#include "../dlutils.h"
#include "../kitrt.h"
#include "../memory_map.h"

// === Overall runtime status, state, and details.  For
// now the runtime only supports a single GPU device and
// we keep that ID as a global state.  For HIP we also
// maintain a handle to the device properties that are
// valid upon successful initialization.
static bool _kitrt_hipIsInitialized = false;
static int _kitrt_hipDeviceID = -1;
static hipDeviceProp_t _kitrt_hipDeviceProps;

// Enable auto-prefetching of managed memory pointers.
// This is a very simple approach that likely will
// likely have limited success.  Note that it can
// significantly avoid page miss costs.
static bool _kitrt_hipEnablePrefetch = true;

// When the compiler generates a prefetch-driven series of kernel
// launches we have multiple prefetch-to-launch streams to
// synchronize -- we keep these streams in an active list that
// will be synchronized and then destroyed post the outter prefetch
// loop construct.
typedef std::list<hipStream_t> KitRTActiveStreamsList;
static KitRTActiveStreamsList _kitrtActiveStreams;

// Declare the various dynamic entry points into the the HIP
// API for use in the runtime.  This is helpful feature for
// use during JIT compilation or related similar operations.

// ---- Initialize, properties, error handling, clean up, etc.
DECLARE_DLSYM(hipInit);
DECLARE_DLSYM(hipDeviceSynchronize);
DECLARE_DLSYM(hipSetDevice);
DECLARE_DLSYM(hipGetDevice);
DECLARE_DLSYM(hipGetDeviceCount);
DECLARE_DLSYM(hipGetDeviceProperties);
DECLARE_DLSYM(hipDeviceGetAttribute);
DECLARE_DLSYM(hipDeviceReset);
DECLARE_DLSYM(hipGetErrorName);
DECLARE_DLSYM(hipGetErrorString);
// ---- Managed memory allocation, tracking, etc.
DECLARE_DLSYM(hipMallocManaged);
DECLARE_DLSYM(hipFree);
DECLARE_DLSYM(hipMemAdvise);
DECLARE_DLSYM(hipMemRangeGetAttribute);
DECLARE_DLSYM(hipPointerGetAttributes);
DECLARE_DLSYM(hipMemcpyHtoD);
DECLARE_DLSYM(hipMemPrefetchAsync);
// ---- Kernel operations, modules, launching, streams, etc.
DECLARE_DLSYM(hipModuleLoadData);
DECLARE_DLSYM(hipModuleGetGlobal);
DECLARE_DLSYM(hipStreamCreate);
DECLARE_DLSYM(hipStreamCreateWithFlags);
DECLARE_DLSYM(hipStreamDestroy);
DECLARE_DLSYM(hipStreamSynchronize);
DECLARE_DLSYM(hipModuleGetFunction);
DECLARE_DLSYM(hipLaunchKernel);
DECLARE_DLSYM(hipModuleLaunchKernel);
// ---- Event management and handling.
DECLARE_DLSYM(hipEventCreate);
DECLARE_DLSYM(hipEventDestroy);
DECLARE_DLSYM(hipEventRecord);
DECLARE_DLSYM(hipEventSynchronize);
DECLARE_DLSYM(hipEventElapsedTime);

// The runtime maintains a map from fat binary images to CUDA modules
// (CUmodule).  This avoids a redundant load of the fat binary into a
// module when looking up kernels from the generated code.
//
// TODO: Is there a faster path here for lookup?  Is a map more
// complicated than necessary?
typedef std::unordered_map<const void *, hipModule_t> KitRTModuleMap;
static KitRTModuleMap _kitrtModuleMap;

// Alongside the module map the runtime also maintains a map from
// kernel name to CUDA function (CUfunction).  Like the modules this
// avoids a call into the module to search for the kernel.
//
// TODO: Ditto from above.  Is there a faster path here for lookup?
// Is a map more complicated than necessary?
typedef std::unordered_map<const char *, hipFunction_t> KitRTKernelMap;
static KitRTKernelMap _kitrtKernelMap;

// TODO: Should we want to move this into a cmake-level
// configuration option?
static const char *HIP_DSO_LIBNAME = "libamdhip64.so";

// Load the dynamic symbols from HIP that are
// used by the runtime   Note, that you will
// need to keep the entries in this call sync'ed
// up with the dynamic symbol declarations used
// above.
static bool __kitrt_hipLoadDLSyms() {

  static void *dlHandle = nullptr;
  if (dlHandle)
    return true; // don't reload symbols...

  if ((dlHandle = dlopen(HIP_DSO_LIBNAME, RTLD_LAZY))) {
    // ---- Initialize, properties, error handling, clean up, etc.
    DLSYM_LOAD(hipInit);
    DLSYM_LOAD(hipDeviceSynchronize);
    DLSYM_LOAD(hipSetDevice);
    DLSYM_LOAD(hipGetDevice);
    DLSYM_LOAD(hipGetDeviceCount);
    DLSYM_LOAD(hipGetDeviceProperties);
    DLSYM_LOAD(hipDeviceGetAttribute);
    DLSYM_LOAD(hipDeviceReset);
    DLSYM_LOAD(hipGetErrorName);
    DLSYM_LOAD(hipGetErrorString);
    // ---- Managed memory allocation, tracking, etc.
    DLSYM_LOAD(hipMallocManaged);
    DLSYM_LOAD(hipFree);
    DLSYM_LOAD(hipMemAdvise);
    DLSYM_LOAD(hipMemRangeGetAttribute);
    DLSYM_LOAD(hipPointerGetAttributes);
    DLSYM_LOAD(hipMemcpyHtoD);
    DLSYM_LOAD(hipMemPrefetchAsync);
    // ---- Kernel operations, modules, launching, streams, etc.
    DLSYM_LOAD(hipModuleLoadData);
    DLSYM_LOAD(hipModuleGetGlobal);
    DLSYM_LOAD(hipLaunchKernel);
    DLSYM_LOAD(hipModuleGetFunction);
    DLSYM_LOAD(hipModuleLaunchKernel);
    DLSYM_LOAD(hipStreamCreate);
    DLSYM_LOAD(hipStreamCreateWithFlags);
    DLSYM_LOAD(hipStreamDestroy);
    DLSYM_LOAD(hipStreamSynchronize);
    // ---- Event management and handling.
    DLSYM_LOAD(hipEventCreate);
    DLSYM_LOAD(hipEventDestroy);
    DLSYM_LOAD(hipEventRecord);
    DLSYM_LOAD(hipEventSynchronize);
    DLSYM_LOAD(hipEventElapsedTime);

    return true;
  } else {
    fprintf(stderr, "kitrt: failed to open dynamic library '%s'",
            HIP_DSO_LIBNAME);
    return false;
  }
}

extern "C" {

#define HIP_SAFE_CALL(x)                                                       \
  {                                                                            \
    hipError_t hip_result = x;                                                 \
    if (hip_result != hipSuccess) {                                            \
      fprintf(stderr, "kitrt: %s:%d:\n", __FILE__, __LINE__);                  \
      const char *msg;                                                         \
      msg = hipGetErrorName_p(hip_result);                                     \
      fprintf(stderr, "  %s failed ('%s')\n", #x, msg);                        \
      msg = hipGetErrorString_p(hip_result);                                   \
      fprintf(stderr, "  error: '%s'\n", msg);                                 \
      abort();                                                                 \
    }                                                                          \
  }

// ---- Initialization, properties, clean up, etc.
//

static bool _kitrt_enableXnack = false;

void __kitrt_hipEnableXnack() { _kitrt_enableXnack = true; }

bool __kitrt_hipInit() {

  if (_kitrt_hipIsInitialized) {
    fprintf(stderr,
            "kitrt: warning, encountered multiple hip initialization paths!\n");
    return true;
  }

  if (_kitrt_enableXnack) {
    (void)setenv("HSA_XNACK", "1", 1);
    if (__kitrt_verboseMode())
      fprintf(stderr, "kitrt: hip -- enabling xnack.\n");
    fprintf(stderr, "       HSA_XNACK has been automatically set.\n");
  }

  if (!__kitrt_hipLoadDLSyms()) {
    fprintf(stderr, "kitrt: unable to resolve dynamic symbols for HIP.\n");
    fprintf(stderr, "       check environment settings and installation.\n");
    fprintf(stderr, "kitrt: aborting...\n");
    abort();
  }

  __kitrt_CommonInit();

  // Note: From the HIP docs, "most HIP APIs implicitly initialize the
  // HIP runtime. This [call] provides control over the timing of the
  // initialization.".  The compiler will insert this call into a global
  // ctor so we'll (play it safe?) make the explicit call.
  HIP_SAFE_CALL(hipInit_p(0));

  // Make sure we have at least one compute device available.
  int count;
  HIP_SAFE_CALL(hipGetDeviceCount_p(&count));
  if (count <= 0) {
    fprintf(stderr, "kitrt: hip -- no devices found!\n");
    abort();
    return false;
  }

  _kitrt_hipDeviceID = 0;
  HIP_SAFE_CALL(hipSetDevice_p(_kitrt_hipDeviceID));
  HIP_SAFE_CALL(
      hipGetDeviceProperties_p(&_kitrt_hipDeviceProps, _kitrt_hipDeviceID));

  // For both ease of code generation on part of the compiler and
  // humans we require managed memory support.  This can introduce
  // performance problems but significantly reduces some aspects
  // of memory and data movement on the programmer.
  //
  // HIP is somewhat challenging on this front as things seem to be
  // inconsistent with behaviors and documentation.  Code often
  // seems to fail in unfortunate ways -- i.e., GPU reading from
  // host memory but the details are not always clear with the
  // exception of poor performance (bordering on horrible performance).
  //
  // We do our best to capture an accurate state of things here...
  //
  // YMMV...
  //
  // From AMD's documentation:
  //
  //   "Managed memory [snip] is supported in the HIP combined
  //   host/device compilation. Through unified memory allocation,
  //   managed memory allows data to be shared and accessible to
  //   both the CPU and GPU using a single pointer. The allocation
  //   is managed by the AMD GPU driver using the Linux
  //   Heterogeneous Memory Management (HMM) mechanism. The user
  //   can call managed memory API hipMallocManaged to allocate a
  //   large chunk of HMM memory, execute kernels on a device, and
  //   fetch data between the host and device as needed.
  //
  //   In a HIP application, it is recommended to do a capability
  //   check before calling the managed memory APIs."
  int hasManagedMemory = 0;
  HIP_SAFE_CALL(hipDeviceGetAttribute(
      &hasManagedMemory, hipDeviceAttributeManagedMemory, _kitrt_hipDeviceID));
  if (!hasManagedMemory) {
    fprintf(stderr, "kitrt: hip -- device does not support managed memory!\n");
    abort(); // TODO: eventually want to return false so JIT runtime won't fail.
  }

  // AMD's example code suggests this is a preferred (additional?) check
  // prior to using managed memory...
  int supportsConcurrentManagedAccess = 0;
  HIP_SAFE_CALL(hipDeviceGetAttribute_p(
      &supportsConcurrentManagedAccess,
      hipDeviceAttributeConcurrentManagedAccess, _kitrt_hipDeviceID));
  if (!supportsConcurrentManagedAccess) {
    fprintf(stderr, "kitrt: hip -- device does not support concurrent "
                    "managed memory accesses!\n");
    abort(); // TODO: eventually want to return false so JIT runtime won't fail.
  } else {
    if (__kitrt_verboseMode())
      fprintf(stderr, "kitrt: hip -- successfully initialized.\n");
    _kitrt_hipIsInitialized = true;
  }

  return _kitrt_hipIsInitialized;
}

void __kitrt_hipDestroy() {
  if (_kitrt_hipIsInitialized) {
    extern void __kitrt_hipFreeManagedMem(void *);
    __kitrt_destroyMemoryMap(__kitrt_hipFreeManagedMem);
    HIP_SAFE_CALL(hipDeviceReset_p());
    _kitrt_hipIsInitialized = false;
    if (__kitrt_verboseMode())
      fprintf(stderr, "kitrt: shutting down hip runtime component.\n");
  }
}

// ---- Managed memory allocation, tracking, etc.

void *__kitrt_hipMemAllocManaged(size_t size) {
  assert(_kitrt_hipIsInitialized && "kitrt: hip has not been initialized!");
  void *memPtr;
  HIP_SAFE_CALL(hipMallocManaged_p(&memPtr, size, hipMemAttachGlobal));
  HIP_SAFE_CALL(hipMemAdvise_p(memPtr, size, hipMemAdviseSetAccessedBy,
                               _kitrt_hipDeviceID));
  HIP_SAFE_CALL(hipMemAdvise_p(memPtr, size, hipMemAdviseSetCoarseGrain,
                               _kitrt_hipDeviceID));
  __kitrt_registerMemAlloc(memPtr, size);

  if (__kitrt_verboseMode())
    fprintf(stderr,
            "kitrt: hip -- allocated managed memory "
            "(%ld bytes @ %p).\n",
            size, memPtr);

  return (void *)memPtr;
}

void __kitrt_hipMemFree(void *memPtr) {
  assert(memPtr != nullptr && "unexpected null pointer!");

  if (__kitrt_verboseMode())
    fprintf(stderr, "kitrt: hip -- freed managed memory @ %p.\n", memPtr);

  __kitrt_unregisterMemAlloc(memPtr);
  HIP_SAFE_CALL(hipFree_p(memPtr));
}

void __kitrt_hipFreeManagedMem(void *memPtr) {
  HIP_SAFE_CALL(hipFree_p(memPtr));
}

bool __kitrt_hipIsMemManaged(void *vp) {
  assert(vp && "unexpected null pointer!");
  hipPointerAttribute_t attrib;
  HIP_SAFE_CALL(hipPointerGetAttributes_p(&attrib, vp));
  if (__kitrt_verboseMode()) {
    if (attrib.isManaged)
      fprintf(stderr, "kitrt: hip -- address %p reported as managed.\n",
              vp);
    else
      fprintf(stderr, "kitrt: hip -- address %p reported as unmanaged.\n",
              vp);
  }
  return (attrib.isManaged != 0);
}

void __kitrt_hipEnablePrefetch() {
  _kitrt_hipEnablePrefetch = true;
  if (__kitrt_verboseMode())
    fprintf(stderr, "kitrt: hip -- enable prefetching.\n");
}

void __kitrt_hipDisablePrefetch() {
  _kitrt_hipEnablePrefetch = false;
  if (__kitrt_verboseMode())
    fprintf(stderr, "kitrt: hip -- disable prefetching.\n");
}

void __kitrt_hipMemPrefetchOnStream(void *vp, void *stream) {
  assert(vp && "unexpected null pointer!");
  // If prefetching is disabled or the allocation has already
  // been prefetched there is nothing further for us to do...
  //
  // TODO: The logic behind a pointer being prefetched or not
  // is pretty sketchy at this point.  Given the use of managed
  // memory, the impact is performance vs. correctness but more
  // work needs to be done on the runtime's tracking and
  // possibly stronger connections with the compiler analysis.
  if (not _kitrt_hipEnablePrefetch || __kitrt_isMemPrefetched(vp)) {
    if (__kitrt_verboseMode())
      fprintf(stderr, "kitrt: hip -- no-op prefetch for pointer %p", vp);
    return;
  }

  if (__kitrt_hipIsMemManaged(vp)) {
    size_t size = __kitrt_getMemAllocSize(vp);
    if (size > 0) {
      hipDevice_t device;
      HIP_SAFE_CALL(hipMemRangeGetAttribute(
          &device, sizeof(device), hipMemRangeAttributeLastPrefetchLocation, vp,
          size));
      if (device != _kitrt_hipDeviceID) {
        HIP_SAFE_CALL(hipMemPrefetchAsync_p(vp, size, _kitrt_hipDeviceID,
                                            (hipStream_t)stream));
        __kitrt_markMemPrefetched(vp);
        if (__kitrt_verboseMode())
          fprintf(stderr,
                  "kitrt: hip -- issued prefetch for %p (bytes = %ld), "
                  "sync'ing device",
                  vp, size);
        HIP_SAFE_CALL(hipDeviceSynchronize());
      }
    }
  }
}

void *__kitrt_hipStreamMemPrefetch(void *vp) {
  hipStream_t stream;
  HIP_SAFE_CALL(hipStreamCreateWithFlags_p(&stream, hipStreamNonBlocking));
  __kitrt_hipMemPrefetchOnStream(vp, stream);
  _kitrtActiveStreams.push_back(stream);
  return (void *)stream;
}

void __kitrt_hipMemPrefetch(void *vp) {
  assert(vp && "unexpected null pointer!");
  __kitrt_hipMemPrefetchOnStream(vp, nullptr);
}

void __kitrt_hipMemcpySymbolToDevice(void *hostPtr, void *devPtr, size_t size) {
  assert(devPtr != 0 && "unexpected null device pointer!");
  assert(hostPtr != nullptr && "unexpected null host pointer!");
  assert(size != 0 && "requested a 0 byte copy!");
  if (__kitrt_verboseMode())
    fprintf(stderr,
            "kitrt: hip -- copy symbol (%ld bytes) to device (%p --> %p).\n",
            size, hostPtr, devPtr);
  HIP_SAFE_CALL(hipMemcpyHtoD_p((hipDeviceptr_t)devPtr, hostPtr, size));
}

// ---- Kernel operations, launching, streams, etc.

void *__kitrt_hipModuleLoadData(const void *image) {
  assert(image && "unexpected null binary object pointer!");
  hipModule_t module;

  HIP_SAFE_CALL(hipModuleLoadData_p(&module, image));
  if (__kitrt_verboseMode()) {
    fprintf(stderr, "kitrt: hip - created module from fat binary...\n");
    fprintf(stderr, "\tmodule addr %p\n", (void *)module);
  }
  return (void *)module;
}

void *__kitrt_hipGetGlobalSymbol(const char *symName, void *mod) {
  assert(symName && "unexpected null symbol name!");
  assert(mod && "unexpected null module pointer!");

  // TODO: Might need to revisit the details here to make sure they
  // fit the HIP API details.
  hipDeviceptr_t devPtr;
  size_t bytes;
  HIP_SAFE_CALL(
      hipModuleGetGlobal_p(&devPtr, &bytes, (hipModule_t)mod, symName));

  if (__kitrt_verboseMode())
    fprintf(stderr, "kitrt: hip -- found symbol '%s' in module.\n", symName);
  return (void *)devPtr;
}

void __kitrt_hipLaunchModuleKernel(void *module, const char *kernelName,
                                   void *kernelArgs, size_t numElements,
                                   void *stream, uint64_t argsSize) {
  assert(module && "request to launch kernel w/ null module!");
  assert(kernelName && "request to launch kernel w/ null name!");
  assert(kernelArgs && "request to launch kernel w/ null args!");
  int threadsPerBlock, blocksPerGrid;

  __kitrt_getLaunchParameters(numElements, threadsPerBlock, blocksPerGrid);
  if (__kitrt_verboseMode()) {
    fprintf(stderr, "launching kernel '%s'.\n", kernelName);
    fprintf(stderr, "\tnumber of elements: %ld\n", numElements);
    fprintf(stderr, "\tthreads/block = %d\n", threadsPerBlock);
    fprintf(stderr, "\tblocks/grid = %d\n", blocksPerGrid);
    fprintf(stderr, "\targument size = %ld\n", argsSize);
  }

  hipFunction_t kFunc;
  HIP_SAFE_CALL(
      hipModuleGetFunction_p(&kFunc, (hipModule_t)module, kernelName));

  void *configArgs[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, kernelArgs,
                        HIP_LAUNCH_PARAM_BUFFER_SIZE, &argsSize,
                        HIP_LAUNCH_PARAM_END};

  // Note the discrepancy between hip and cuda here -- for hip the module
  // launch is the same as the "standard" cuda launch.
  HIP_SAFE_CALL(hipModuleLaunchKernel_p(
      kFunc, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0, (hipStream_t)stream,
      nullptr, (void **)&configArgs[0]));
}

// Launch a kernel on the default stream.
void __kitrt_hipLaunchKernel(const void *fatBin, const char *kernelName,
                             void *kernelArgs, size_t numElements, void *stream,
                             size_t argsSize) {
  assert(fatBin && "request to launch with null fat binary image!");
  assert(kernelName && "request to launch kernel w/ null name!");
  assert(kernelArgs && "request to launch kernel w/ null fatbin args!");
  int threadsPerBlock, blocksPerGrid;
  hipFunction_t kFunc;

  // TODO: Not sure of the actual advantages from the kernel and module
  // maps here. Needs some work to determine if it is really worthwhile...
  KitRTKernelMap::iterator kern_it = _kitrtKernelMap.find(kernelName);
  if (kern_it == _kitrtKernelMap.end()) {
    // Look for the hip module associated with the fat binary. If it
    // isn't found we need to create one and cache it away for further
    // invocations.
    hipModule_t module;
    KitRTModuleMap::iterator mod_it = _kitrtModuleMap.find(fatBin);
    if (mod_it == _kitrtModuleMap.end()) {
      HIP_SAFE_CALL(hipModuleLoadData_p(&module, fatBin));
      _kitrtModuleMap[fatBin] = module;
    } else
      module = mod_it->second;
    HIP_SAFE_CALL(hipModuleGetFunction_p(&kFunc, module, kernelName));
    _kitrtKernelMap[kernelName] = kFunc;
  } else {
    kFunc = kern_it->second;
  }

  __kitrt_getLaunchParameters(numElements, threadsPerBlock, blocksPerGrid);
  if (__kitrt_verboseMode()) {
    fprintf(stderr, "launching kernel '%s'.\n", kernelName);
    fprintf(stderr, "\tnumber of elements: %ld\n", numElements);
    fprintf(stderr, "\tthreads/block = %d\n", threadsPerBlock);
    fprintf(stderr, "\tblocks/grid = %d\n", blocksPerGrid);
    fprintf(stderr, "\targument size = %ld\n", argsSize);
  }

  void *configArgs[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, kernelArgs,
                        HIP_LAUNCH_PARAM_BUFFER_SIZE, &argsSize,
                        HIP_LAUNCH_PARAM_END};

  // Note the discrepancy between hip and cuda here -- for hip the module
  // launch is the same as the "standard" cuda launch.
  HIP_SAFE_CALL(hipModuleLaunchKernel_p(
      kFunc, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0, (hipStream_t)stream,
      nullptr, (void **)&configArgs[0]));
}

void __kitrt_hipStreamSynchronize(void *vStream) {
  HIP_SAFE_CALL(hipStreamSynchronize_p((hipStream_t)vStream));
}

void __kitrt_hipSynchronizeStreams() {
  // If the active stream is empty, our launch path went through the
  // default stream.  Otherwise, we need to sync on each of the active
  // streams.
  HIP_SAFE_CALL(hipSetDevice_p(_kitrt_hipDeviceID));
  HIP_SAFE_CALL(hipDeviceSynchronize_p());
  while (not _kitrtActiveStreams.empty()) {
    hipStream_t stream = _kitrtActiveStreams.front();
    HIP_SAFE_CALL(hipStreamDestroy(stream));
    _kitrtActiveStreams.pop_front();
  }
}

// ---- Event management and handling.

void *__kitrt_hipCreateEvent() {
  hipEvent_t e;
  HIP_SAFE_CALL(hipEventCreate_p(&e));
  return (void *)e;
}

void __kitrt_hipDestroyEvent(void *E) {
  assert(E && "unexpected null event!");
  HIP_SAFE_CALL(hipEventDestroy_p((hipEvent_t)E));
}

void __kitrt_hipEventRecord(void *E) {
  assert(E && "unexpected null event!");
  // TODO: The HIP documentation is not entirely clear about the
  // existence of a default stream... This could break...
  HIP_SAFE_CALL(hipEventRecord_p((hipEvent_t)E, nullptr));
}

void __kitrt_hipSynchronizeEvent(void *E) {
  assert(E && "unexpected null event!");
  HIP_SAFE_CALL(hipEventSynchronize_p((hipEvent_t)E));
}

float __kitrt_hipElapsedEventTime(void *start, void *stop) {
  assert(start && "unexpected null start event!");
  assert(stop && "unexpected null stop event!");
  float msecs;
  HIP_SAFE_CALL(
      hipEventElapsedTime_p(&msecs, (hipEvent_t)start, (hipEvent_t)stop));
  return msecs;
}

// ---- Event management for timing, etc.

void __kitrt_hipEnableEventTiming(unsigned report) {
  //_kitrtEnableTiming = true;
  //_kitrtReportTiming = report > 0;
}

void __kitrt_hipDisableEventTiming() {
  //_kitrtEnableTiming = false;
  //_kitrtReportTiming = false;
  //_kitrtLastEventTime = 0.0;
}

double __kitrt_hipGetLastEventTime() { 
  // return _kitrtLastEventTime; 
  return 0.0;
}

} // extern "C"
