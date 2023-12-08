//===- cuda.cpp - Kitsune runtime CUDA support    ------------------------===//
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

// TODO:
//   * Need a few options for stream usage that will likely require some 
//     compiler-side static analysis and additional entry points for 
//     runtime tuning.  (consider: blocked prefetches, prefetch streams, etc.).

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <dlfcn.h>
#include <iostream>
#include <list>
#include <map>
#include <mutex>
#include <sstream>
#include <stdbool.h>
#include <sys/syscall.h>
#include <unordered_map>
#include <unistd.h>

#define gettid() syscall(SYS_gettid)

#include "../debug.h"
#include "../dlutils.h"
#include "../kitrt.h"
#include "../memory_map.h"
#include "./cuda.h"

// #define _KITRT_VERBOSE_
//  Has the runtime been initialized?
static bool _kitrt_cuIsInitialized = false;
static int _kitrt_cuDeviceID = -1;

// Measure internal timing of launched kernels.
static bool _kitrtEnableTiming = false;
// Automatically report kernel execution times to stdout.
static bool _kitrtReportTiming = false;
// Last measured kernel execution time.
static double _kitrtLastEventTime = 0.0f;

// Use heuristic-based launch parameters.
static bool _kitrtUseHeuristicLaunchParameters = false;

// Default number of threads to use per block for kernel
// launches.
static unsigned _kitrtDefaultThreadsPerBlock = 256;

// Default number of blocks per grid (allows for custom
// settings but otherwise automatically computed).
static unsigned _kitrtDefaultBlocksPerGrid = 0;

// Enable external settings for kernel launch parameters.
static bool _kitrtUseCustomLaunchParameters = false;

// CUDA device (-1 flags an uninitialized state). At present
// runtime only supports a single device.
static CUdevice _kitrtCUdevice = -1;

// Default CUDA context (a nullptr flags an uninitialized state).
// At present the runtime only supports a single context.
static CUcontext _kitrtCUcontext = nullptr;

// When the compiler generates a prefetch-driven series of kernel
// launches we have multiple prefetch-to-launch streams to
// synchronize -- we keep these streams in an active list that
// will be synchronized and then destroyed post the outter prefetch
// loop construct.
typedef std::list<CUstream> KitRTActiveStreamsList;
static KitRTActiveStreamsList _kitrtActiveStreams;
extern unsigned _kitrt_MaxPrefetchStreams;
static unsigned _kitrt_CurPrefetchStream = 0;
std::vector<CUstream> _kitrt_PrefetchStreams;

typedef std::unordered_map<unsigned int, CUstream> KitRTCUStreamMap;
static KitRTCUStreamMap _kitrtCUStreamMap;
static CUstream __kitrt_getThreadStream(pid_t thread_id);
static void __kitrt_deleteThreadStream(pid_t thread_id);
static void __kitrt_destroyThreadStreams();

struct KitRTPrefetchRequest {
  void *addr;
  size_t size;
};

std::list<KitRTPrefetchRequest> _kitrt_PrefetchRequests;

// NOTE: Over a series of CUDA releases it is worthwhile to
// check in on the header files for replacement versioned
// entry points into the driver API.  These are typically
// denoted with a '*_vN' naming scheme and don't always
// play well with older entry points.  If you suddenly
// start to see context errors this is certainly worth
// digging into.  We are vulnerable to this issue because
// we are loading dynamic symbols by name and must therefore
// match version details explicitly in the code.

// ---- Initialize, properties, clean up, etc.
DECLARE_DLSYM(cuInit);
DECLARE_DLSYM(cuDeviceGetCount);
DECLARE_DLSYM(cuDeviceGet);
DECLARE_DLSYM(cuCtxCreate_v3);
DECLARE_DLSYM(cuDevicePrimaryCtxRetain);
DECLARE_DLSYM(cuDevicePrimaryCtxRelease_v2);
DECLARE_DLSYM(cuDevicePrimaryCtxReset_v2);
DECLARE_DLSYM(cuCtxDestroy_v2);
DECLARE_DLSYM(cuCtxSetCurrent);
DECLARE_DLSYM(cuCtxPushCurrent_v2);
DECLARE_DLSYM(cuCtxPopCurrent_v2);
DECLARE_DLSYM(cuCtxGetCurrent);
DECLARE_DLSYM(cuStreamCreate);
DECLARE_DLSYM(cuStreamDestroy_v2);
DECLARE_DLSYM(cuStreamSynchronize);
DECLARE_DLSYM(cuStreamAttachMemAsync);
DECLARE_DLSYM(cuLaunchKernel);
DECLARE_DLSYM(cuEventCreate);
DECLARE_DLSYM(cuEventRecord);
DECLARE_DLSYM(cuEventSynchronize);
DECLARE_DLSYM(cuEventElapsedTime);
DECLARE_DLSYM(cuEventDestroy_v2);
DECLARE_DLSYM(cuGetErrorName);
DECLARE_DLSYM(cuGetErrorString);
DECLARE_DLSYM(cuModuleLoadDataEx);
DECLARE_DLSYM(cuModuleLoadData);
DECLARE_DLSYM(cuModuleLoadFatBinary);
DECLARE_DLSYM(cuModuleGetFunction);
DECLARE_DLSYM(cuModuleUnload);

DECLARE_DLSYM(cuMemAllocManaged);
DECLARE_DLSYM(cuMemAllocHost);
DECLARE_DLSYM(cuMemHostAlloc);
DECLARE_DLSYM(cuMemsetD8Async);
DECLARE_DLSYM(cuMemFree_v2);
DECLARE_DLSYM(cuMemPrefetchAsync);
DECLARE_DLSYM(cuMemAdvise);
DECLARE_DLSYM(cuPointerGetAttribute);
DECLARE_DLSYM(cuPointerSetAttribute);
DECLARE_DLSYM(cuDeviceGetAttribute);
DECLARE_DLSYM(cuCtxSynchronize);
DECLARE_DLSYM(cuModuleGetGlobal_v2);
DECLARE_DLSYM(cuMemcpy);
DECLARE_DLSYM(cuMemcpyHtoD_v2);
DECLARE_DLSYM(cuOccupancyMaxPotentialBlockSize);

// The runtime maintains a map from fat binary images to CUDA modules
// (CUmodule).  This avoids a redundant load of the fat binary into a
// module when looking up kernels from the generated code.
//
// TODO: Is there a faster path here for lookup?  Is a map more
// complicated than necessary?
typedef std::unordered_map<const void *, CUmodule> KitRTModuleMap;
static KitRTModuleMap _kitrtModuleMap;

// Alongside the module map the runtime also maintains a map from
// kernel name to CUDA function (CUfunction).  Like the modules this
// avoids a call into the module to search for the kernel.
//
// TODO: Ditto from above.  Is there a faster path here for lookup?
// Is a map more complicated than necessary?
typedef std::unordered_map<const char *, CUfunction> KitRTKernelMap;
static KitRTKernelMap _kitrtKernelMap;

static bool __kitrt_cuLoadDLSyms() {

  static void *dlHandle = nullptr;
  if (dlHandle)
    return true;

  if ((dlHandle = dlopen("libcuda.so", RTLD_LAZY))) {
    DLSYM_LOAD(cuInit);
    DLSYM_LOAD(cuGetErrorName);
    DLSYM_LOAD(cuGetErrorString);
    DLSYM_LOAD(cuDeviceGetCount);
    DLSYM_LOAD(cuDeviceGet);
    DLSYM_LOAD(cuDevicePrimaryCtxRetain);
    DLSYM_LOAD(cuDevicePrimaryCtxRelease_v2);
    DLSYM_LOAD(cuDevicePrimaryCtxReset_v2);
    DLSYM_LOAD(cuCtxCreate_v3);
    DLSYM_LOAD(cuCtxDestroy_v2);
    DLSYM_LOAD(cuCtxSetCurrent);
    DLSYM_LOAD(cuCtxPushCurrent_v2);
    DLSYM_LOAD(cuCtxPopCurrent_v2);
    DLSYM_LOAD(cuCtxGetCurrent);

    DLSYM_LOAD(cuMemAllocManaged);
    DLSYM_LOAD(cuMemAllocHost);
    DLSYM_LOAD(cuMemHostAlloc);
    DLSYM_LOAD(cuMemsetD8Async);
    DLSYM_LOAD(cuMemFree_v2);
    DLSYM_LOAD(cuMemPrefetchAsync);
    DLSYM_LOAD(cuMemAdvise);

    DLSYM_LOAD(cuModuleLoadData);
    DLSYM_LOAD(cuModuleLoadDataEx);
    DLSYM_LOAD(cuModuleLoadFatBinary);
    DLSYM_LOAD(cuModuleGetFunction);
    DLSYM_LOAD(cuModuleGetGlobal_v2);
    DLSYM_LOAD(cuModuleUnload);

    DLSYM_LOAD(cuStreamCreate);
    DLSYM_LOAD(cuStreamDestroy_v2);
    DLSYM_LOAD(cuStreamSynchronize);
    DLSYM_LOAD(cuStreamAttachMemAsync);
    DLSYM_LOAD(cuLaunchKernel);

    DLSYM_LOAD(cuEventCreate);
    DLSYM_LOAD(cuEventRecord);
    DLSYM_LOAD(cuEventSynchronize);
    DLSYM_LOAD(cuEventElapsedTime);
    DLSYM_LOAD(cuEventDestroy_v2);

    DLSYM_LOAD(cuPointerGetAttribute);
    DLSYM_LOAD(cuPointerSetAttribute);
    DLSYM_LOAD(cuDeviceGetAttribute);
    DLSYM_LOAD(cuCtxSynchronize);

    DLSYM_LOAD(cuMemcpy);
    DLSYM_LOAD(cuMemcpyHtoD_v2);
    DLSYM_LOAD(cuOccupancyMaxPotentialBlockSize);
    return true;
  } else {
    fprintf(stderr, "kitrt: Failed to load CUDA dynamic library.\n");
    fprintf(stderr, "kitrt: Is CUDA in your LD_LIBRARY_PATH?\n");
    return false;
  }
}

extern "C" {

#define CU_SAFE_CALL(x)                                                        \
  {                                                                            \
    CUresult result = x;                                                       \
    if (result != CUDA_SUCCESS) {                                              \
      const char *msg;                                                         \
      cuGetErrorName_p(result, &msg);                                          \
      fprintf(stderr, "kitrt %s:%d:\n", __FILE__, __LINE__);                   \
      fprintf(stderr, "  %s failed ('%s')\n", #x, msg);                        \
      cuGetErrorString_p(result, &msg);                                        \
      fprintf(stderr, "  error: '%s'\n", msg);                                 \
      exit(1);                                                                 \
    }                                                                          \
  }

// ---- Initialization, properties, clean up, etc.

bool __kitrt_cuInit() {

  if (_kitrt_cuIsInitialized) {
    fprintf(stderr, "kitrt: warning, multiple cuda initialization paths!\n");
    return true;
  }

  if (__kitrt_verboseMode())
    fprintf(stderr, "kitrt: initializing cuda.\n");

  if (not __kitrt_cuLoadDLSyms()) {
    fprintf(stderr, "kitrt: unable to resolve dynamic symbols for CUDA.\n");
    fprintf(stderr, "       check enviornment settings and installation.\n");
    fprintf(stderr, "kitrt: aborting...\n");
    abort();
  }

  __kitrt_CommonInit();

  int deviceCount = 0;
  CU_SAFE_CALL(cuInit_p(0));
  CU_SAFE_CALL(cuDeviceGetCount_p(&deviceCount));
  if (deviceCount == 0) {
    fprintf(stderr, "kitrt: warning -- no CUDA devices found!\n");
    abort();
  }

  extern int _kitrt_DefaultDeviceID;
  if (_kitrt_DefaultDeviceID == -1)
    _kitrt_cuDeviceID = 0;
  else 
    _kitrt_cuDeviceID = _kitrt_DefaultDeviceID;

  if (__kitrt_verboseMode())
    fprintf(stderr, "\tdevice count: %d\n", deviceCount);

  CU_SAFE_CALL(cuDeviceGet_p(&_kitrtCUdevice, _kitrt_cuDeviceID));
  CU_SAFE_CALL(cuDevicePrimaryCtxRetain_p(&_kitrtCUcontext, _kitrtCUdevice));
  // NOTE: It seems we have to explicitly set the context but that seems
  // to be different than what the driver API docs suggest...
  CU_SAFE_CALL(cuCtxSetCurrent_p(_kitrtCUcontext));
  _kitrt_cuIsInitialized = true;

  char *envValue;
  if ((envValue = getenv("KITRT_USE_OCCUPANCY_HEURISTIC"))) {
    _kitrtUseHeuristicLaunchParameters = true;
  } else {
    _kitrtUseHeuristicLaunchParameters = false;
  }

  if (__kitrt_prefetchStreamsEnabled()) {
    for(unsigned si = 0; si < _kitrt_MaxPrefetchStreams; si++) {
      CUstream stream;
      CU_SAFE_CALL(cuStreamCreate_p(&stream, CU_STREAM_DEFAULT));
      _kitrt_PrefetchStreams.push_back(stream);
    }
  }
  
  return _kitrt_cuIsInitialized;
}

void __kitrt_cuDestroy() {

  if (_kitrt_cuIsInitialized) {
    void __kitrt_cuFreeManagedMem(void *vp);
    __kitrt_destroyMemoryMap(__kitrt_cuFreeManagedMem);

    if (__kitrt_prefetchStreamsEnabled()) {
      for(unsigned si = 0; si < _kitrt_MaxPrefetchStreams; si++) {
        CUstream stream = _kitrt_PrefetchStreams[si];
        CU_SAFE_CALL(cuStreamDestroy_v2_p(stream));
      }
    }

    __kitrt_destroyThreadStreams();

    // Note that all resources associated with the context will be destroyed.
    CU_SAFE_CALL(cuDevicePrimaryCtxReset_v2_p(_kitrtCUdevice));
    _kitrt_cuIsInitialized = false;
  }
}

void __kitrt_cuCheckCtxState() {
  if (_kitrt_cuIsInitialized) {
    CUcontext c;
    CU_SAFE_CALL(cuCtxGetCurrent_p(&c));
    if (c != _kitrtCUcontext) {
      fprintf(stderr, "kitrt: warning! current cuda context mismatch!\n");
    }
  } else {
    fprintf(stderr,
            "kitrt: context check encountered uninitialized CUDA state!\n");
  }
}

// ---- Managed memory allocation, tracking, etc.

static std::mutex _kitrt_mem_alloc_mutex;
__attribute__((malloc)) 
void *__kitrt_cuMemAllocManaged(size_t size) {
  if (not _kitrt_cuIsInitialized)
    __kitrt_cuInit();

  _kitrt_mem_alloc_mutex.lock();

  CUdeviceptr devp;
  CU_SAFE_CALL(cuMemAllocManaged_p(&devp, size, CU_MEM_ATTACH_GLOBAL));

  // Flag the allocation with some CUDA specific flags.  At present these
  // have little impact given the use of a single device and the default
  // stream.  Recall that the current practice is for the actual allocation
  // to occur on first touch -- thus our 'prefetch' status here is a bit
  // misleading (technically we are not prefetched to either host nor device).
  //
  CU_SAFE_CALL(
      cuMemAdvise_p(devp, size, CU_MEM_ADVISE_SET_ACCESSED_BY, _kitrtCUdevice));
  CU_SAFE_CALL(cuMemAdvise_p(devp, size, CU_MEM_ADVISE_SET_PREFERRED_LOCATION,
                             _kitrtCUdevice));

  int enable = 1;
  CU_SAFE_CALL(
      cuPointerSetAttribute_p(&enable, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, devp));
  // Register this allocation so the runtime can help track the
  // locality (and affinity) of data.
  __kitrt_registerMemAlloc((void *)devp, size);
  _kitrt_mem_alloc_mutex.unlock();

  pid_t tid = gettid();
  CU_SAFE_CALL(cuMemPrefetchAsync_p(devp, size, _kitrtCUdevice, __kitrt_getThreadStream(tid)));
  return (void *)devp;
}

__attribute__((malloc)) 
void *__kitrt_cuMemCallocManaged(size_t count, size_t element_size) {
  assert(count != 0 && "zero-valued item count!");
  assert(element_size != 0 && "zero-valued element size!");
  size_t nbytes = count * element_size;
  CUdeviceptr memp = (CUdeviceptr)__kitrt_cuMemAllocManaged(nbytes);

  // TODO: Is there a risk of a race here?  From the driver API docs:
  //
  //   The cudaMemset functions are asynchronous with respect to the host 
  //   except when the target memory is pinned host memory. The Async 
  //   versions are always asynchronous with respect to the host.
  // 
  // Given our use of UVM we might also be able to use a straight memset() 
  // call... which would, of course, place all pages on the host... 
  //
  // TODO: We're not set to run on anything but the default stream... 
  CU_SAFE_CALL(cuMemsetD8Async_p(memp, 0, nbytes, NULL));
  return (void*)memp;
}

__attribute__((malloc))
void *__kitrt_cuMemReallocManaged(void *ptr, size_t size) {
  assert(size != 0 && "zero-valued size!");
  void *memptr = nullptr;
  if (ptr == nullptr) 
    memptr = __kitrt_cuMemAllocManaged(size);
  else {
    // Check to make sure this is a pointer we're actually managing.
    bool read_only, write_only;
    size_t nbytes = __kitrt_getMemAllocSize(ptr, &read_only, &write_only);
    if (nbytes == 0) {
      fprintf(stderr, "kitrt: warning, realloc() request on untracked allocation!\n");
      return nullptr;
    }
    
    if (size > nbytes) {
      // requested size is larger than currently tracked allocation.  Replace it.
      memptr = __kitrt_cuMemAllocManaged(size);
      cuMemcpy_p(/* dest */(CUdeviceptr)memptr, /* source */(CUdeviceptr)ptr, nbytes);
      // note: realloc does not guarantee initialized memory outside of existing data... 
      __kitrt_cuMemFree(ptr);
    } else if (size < nbytes) {
      memptr = __kitrt_cuMemAllocManaged(size);
      cuMemcpy_p(/* dest */(CUdeviceptr)memptr, /* source */(CUdeviceptr)ptr, size);
      __kitrt_cuMemFree(ptr);
    } else
      // same size, just return it. 
      memptr = ptr;
  }
  return memptr;
}

void __kitrt_cuMemFree(void *vp) {
  assert(vp && "unexpected null pointer!");
  // We first remove the allocation from the runtime's
  // map, and then actually release it via CUDA...
  // Note that the versioned free calls are important
  // here -- a non-v2 version will actually result in
  // crashes...
  if (not _kitrt_cuIsInitialized)
    __kitrt_cuInit();

  _kitrt_mem_alloc_mutex.lock();
  __kitrt_unregisterMemAlloc(vp);
  CU_SAFE_CALL(cuMemFree_v2_p((CUdeviceptr)vp));
  _kitrt_mem_alloc_mutex.unlock();
}

void __kitrt_cuFreeManagedMem(void *vp) {
  if (not _kitrt_cuIsInitialized)
    __kitrt_cuInit();
  _kitrt_mem_alloc_mutex.lock();
  CU_SAFE_CALL(cuMemFree_v2_p((CUdeviceptr)vp));
  __kitrt_unregisterMemAlloc(vp);
  _kitrt_mem_alloc_mutex.unlock();
}

bool __kitrt_cuIsMemManaged(void *vp) {
  assert(vp && "unexpected null pointer!");
  if (not _kitrt_cuIsInitialized)
    __kitrt_cuInit();
  CUdeviceptr devp = (CUdeviceptr)vp;
  unsigned int is_managed;
  CUresult r = cuPointerGetAttribute_p(&is_managed,
                                       CU_POINTER_ATTRIBUTE_IS_MANAGED, devp);
  return (r == CUDA_SUCCESS) && is_managed;
}

// ---- Memory/data prefetch and data movement support.

void __kitrt_cuPrefetchRequest(void *vp) {
  size_t size = 0;
  if (not __kitrt_isMemPrefetched(vp, &size)) {
    if (size > 0) {
      struct KitRTPrefetchRequest R;
      R.addr = vp;
      R.size = size;
      _kitrt_PrefetchRequests.push_back(R);
    }
  }
}


void __kitrt_cuMemPrefetchOnStream(void *vp, void *stream) {
  assert(vp && "unexpected null pointer!");

  size_t size = 0;
  if (not __kitrt_isMemPrefetched(vp, &size)) {
    if (size > 0) {
      
      //if (is_read_only) {
      //  CU_SAFE_CALL(cuMemAdvise_p((CUdeviceptr)vp, size,
      //                             CU_MEM_ADVISE_SET_READ_MOSTLY,
      //                             _kitrtCUdevice));
      //} else {
      //  CU_SAFE_CALL(cuMemAdvise_p((CUdeviceptr)vp, size,
      //                             CU_MEM_ADVISE_UNSET_READ_MOSTLY,
      //                             _kitrtCUdevice));
      //}
      
      // Our semantics assume that a prefetch request suggests an inbound
      // kernel launch.   Setting the preferred location does not cause
      // data to migrate to that location immediately. Instead, it guides
      // the migration policy when a fault occurs on that memory region. If
      // the data is already in its preferred location and the faulting
      // processor can establish a mapping without requiring the data to be
      // migrated, then data migration will be avoided. On the other hand, if
      // the data is not in its preferred location or if a direct mapping cannot
      // be established, then it will be migrated to the processor accessing it.
      // It is important to note that setting the preferred location does not
      // prevent data prefetching done using cuMemPrefetchAsync(). Having a
      // preferred location can override the page thrash detection and
      // resolution logic in the Unified Memory driver. Normally, if a page is
      // detected to be constantly thrashing between host and device
      // memory, the page may eventually be pinned to host memory. But if the
      // preferred location is set as device memory, then the page will continue
      // to thrash indefinitely. If CU_MEM_ADVISE_SET_READ_MOSTLY is also set on
      // this memory region or any subset of it, then the policies associated
      // with that advice will override the policies of this advice, unless read
      // accesses from device will not result in a read-only copy being created
      // on that device as outlined in description for the advice
      // CU_MEM_ADVISE_SET_READ_MOSTLY.
      CU_SAFE_CALL(cuMemAdvise_p((CUdeviceptr)vp, size,
                                 CU_MEM_ADVISE_SET_PREFERRED_LOCATION,
                                _kitrtCUdevice));
      pid_t tid = gettid();
      CU_SAFE_CALL(cuMemPrefetchAsync_p((CUdeviceptr)vp, size, _kitrtCUdevice,
                                        __kitrt_getThreadStream(tid)));
      __kitrt_markMemPrefetched(vp);
    }
    
  }
}

  
void __kitrt_cuMemPrefetch(void *vp) {
  assert(vp && "unexpected null pointer!");
  __kitrt_cuMemPrefetchOnStream(vp, __kitrt_getThreadStream(gettid()));
}


void __kitrt_cuStreamSetMemPrefetch(void *vp) {
  // Prefetching with streams has some rules that make a guarenteed
  // behavior difficult...  For a busy stream, the prefetch is
  // deferred to a background thread by the driver to maintain stream
  // ordering. This background thread executes the prefetch when all
  // prior operations in the stream are completed. For idle streams,
  // the driver can either defer the operation or not, but the driver
  // often (how often?) does not defer because of the associated
  // overhead.  The exact details for when the driver may defer vary
  // across driver versions.
  assert(vp && "unexpected null pointer!");
  CUstream stream = _kitrt_PrefetchStreams[_kitrt_CurPrefetchStream];
  __kitrt_cuMemPrefetchOnStream(vp, (void*)stream);
  _kitrt_CurPrefetchStream++;
  if (_kitrt_CurPrefetchStream == _kitrt_MaxPrefetchStreams)
    _kitrt_CurPrefetchStream = 0; // wrap for round-robin... 
}

void *__kitrt_cuStreamMemPrefetch(void *vp) {
  CUstream stream;
  CU_SAFE_CALL(cuStreamCreate_p(&stream, CU_STREAM_NON_BLOCKING));
  __kitrt_cuMemPrefetchOnStream(vp, stream);
  _kitrtActiveStreams.push_back(stream);
  return (void *)stream;
}

void __kitrt_cuMemHostPrefetch(void *vp) {
  assert(vp && "unexpected null pointer!");
  // Prefetch to the host if we previously issued a device prefetch.
  if (__kitrt_isMemPrefetched(vp)) {
    bool is_read_only, is_write_only;
    size_t size = __kitrt_getMemAllocSize(vp, &is_read_only, &is_write_only);
    if (size > 0) {
      // If the data is mostly going to be read from and only occasionally
      // written to we want the unified memory driver to create a read-only copy
      // of at least the accessed pages in the target processor's memory. We
      // then call cuMemPrefetchAsync() to create that read-only copy.  Any
      // writes to this region will force all copies of the corresponding page
      // to be invalidated except for the one where the write occurred. (Note:
      // the device argument is ignored and that for a page to be
      // read-duplicated, the accessing processor must either be the CPU or a
      // GPU that has a non-zero value for the device attribute
      // CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS.)
      if (is_read_only) {
        CU_SAFE_CALL(cuMemAdvise_p((CUdeviceptr)vp, size,
                                   CU_MEM_ADVISE_SET_READ_MOSTLY,
                                   CU_DEVICE_CPU));
      } else {
        // for now we treat "write only" as disabling the read only advice...
        CU_SAFE_CALL(cuMemAdvise_p((CUdeviceptr)vp, size,
                                   CU_MEM_ADVISE_UNSET_READ_MOSTLY,
                                   CU_DEVICE_CPU));
      }
      // Our semantics assume that a prefetch request suggests an inbound
      // kernel launch.   Setting the preferred location does not cause
      // data to migrate to that location immediately. Instead, it guides
      // the migration policy when a fault occurs on that memory region. If
      // the data is already in its preferred location and the faulting
      // processor can establish a mapping without requiring the data to be
      // migrated, then data migration will be avoided. On the other hand, if
      // the data is not in its preferred location or if a direct mapping cannot
      // be established, then it will be migrated to the processor accessing it.
      // It is important to note that setting the preferred location does not
      // prevent data prefetching done using cuMemPrefetchAsync(). Having a
      // preferred location can override the page thrash detection and
      // resolution logic in the Unified Memory driver. Normally, if a page is
      // detected to be constantly thrashing between host and device
      // memory, the page may eventually be pinned to host memory. But if the
      // preferred location is set as device memory, then the page will continue
      // to thrash indefinitely. If CU_MEM_ADVISE_SET_READ_MOSTLY is also set on
      // this memory region or any subset of it, then the policies associated
      // with that advice will override the policies of this advice, unless read
      // accesses from device will not result in a read-only copy being created
      // on that device as outlined in description for the advice
      // CU_MEM_ADVISE_SET_READ_MOSTLY.
      CU_SAFE_CALL(cuMemAdvise_p((CUdeviceptr)vp, size,
                                 CU_MEM_ADVISE_SET_PREFERRED_LOCATION,
                                 CU_DEVICE_CPU));

      CU_SAFE_CALL(cuMemPrefetchAsync_p((CUdeviceptr)vp, size, CU_DEVICE_CPU,
                                        (CUstream) nullptr));
      __kitrt_setMemPrefetch(vp, false);
    }
  }
}

void __kitrt_cuMemcpySymbolToDevice(void *hostPtr, uint64_t devPtr,
                                    size_t size) {
  assert(devPtr != 0 && "unexpected null device pointer!");
  assert(hostPtr != nullptr && "unexpected null host pointer!");
  assert(size != 0 && "requested a 0 byte copy!");
  CU_SAFE_CALL(cuMemcpyHtoD_v2_p(devPtr, hostPtr, size));
}

// ---- Kernel operations, launching, streams, etc.

void __kitrt_cuSetCustomLaunchParameters(unsigned BlocksPerGrid,
                                         unsigned ThreadsPerBlock) {
  _kitrtUseCustomLaunchParameters = true;
  _kitrtDefaultBlocksPerGrid = BlocksPerGrid;
  _kitrtDefaultThreadsPerBlock = ThreadsPerBlock;
}

void __kitrt_cuSetDefaultThreadsPerBlock(unsigned ThreadsPerBlock) {
  _kitrtDefaultThreadsPerBlock = ThreadsPerBlock;
}

static void __kitrt_cuMaxPotentialBlockSize(int &blocksPerGrid,
                                            int &threadsPerBlock, CUfunction F,
                                            size_t numElements) {

  CU_SAFE_CALL(
      cuOccupancyMaxPotentialBlockSize_p(&blocksPerGrid, &threadsPerBlock, F, 0,
                                         0, // no dynamic shared memory...
                                         0));
#ifdef _KITRT_VERBOSE_
  fprintf(stderr, "occupancy returned: %d, %d\n", blocksPerGrid,
          threadsPerBlock);
#endif
  blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
}

void *__kitrt_cuCreateFBModule(const void *fatBin) {
  assert(fatBin && "unexpected null fatbinary image!");
  if (not _kitrt_cuIsInitialized)
    __kitrt_cuInit();
  CUmodule module;
  CU_SAFE_CALL(cuModuleLoadData_p(&module, fatBin));
  // CU_SAFE_CALL(cuModuleLoadFatBinary_p(&module, fatBin));
  return (void *)module;
}

uint64_t __kitrt_cuGetGlobalSymbol(const char *SN, void *CM) {
  assert(SN && "null symbol name (SN)!");
  assert(CM && "null (opaque) CUDA module");
  CUmodule Module = (CUmodule)CM;

  // NOTE: The device pointer and size ('bytes') parameters for
  // cuModuleGetGlobal are optional.  To simplify our code gen
  // work we ignore the size parameter (which is NULL below).
  CUdeviceptr devPtr;
  size_t bytes;
  CUresult result;
  if ((result = cuModuleGetGlobal_v2_p(&devPtr, &bytes, Module, SN)) !=
      CUDA_SUCCESS) {
    fprintf(stderr, "kitrt: error finding symbol '%s'.\n", SN);
    const char *msg;
    cuGetErrorName_p(result, &msg);
    fprintf(stderr, "kitrt %s:%d:\n", __FILE__, __LINE__);
    fprintf(stderr, "  cuModuleGetGlobal() failed ('%s')\n", msg);
    cuGetErrorString_p(result, &msg);
    fprintf(stderr, "  error: '%s'\n", msg);
    fprintf(stderr, "symbol name: %s\n", SN);
    exit(1);
  }
  return devPtr;
}

void *__kitrt_cuLaunchModuleKernel(void *mod, const char *kernelName,
                                   void **fatBinArgs, uint64_t numElements) {
  int threadsPerBlock, blocksPerGrid;

  CUfunction kFunc;
  CUmodule module = (CUmodule)mod;
  CU_SAFE_CALL(cuModuleGetFunction_p(&kFunc, module, kernelName));

  if (_kitrtUseHeuristicLaunchParameters)
    __kitrt_cuMaxPotentialBlockSize(blocksPerGrid, threadsPerBlock, kFunc,
                                    numElements);
  else
    __kitrt_getLaunchParameters(numElements, threadsPerBlock, blocksPerGrid);

  CUevent start, stop;
  if (_kitrtEnableTiming) {
    // Recall that we have to take a bit of care about how we time the
    // launched kernel's execution time.  The problem with using host-device
    // synchronization points is that they can potentially stall the entire
    // GPU pipeline, which we want to avoid to enable asynchronous data
    // movement and the execution of other kernels on the GPU.
    //
    // A nice overview for measuring performance in CUDA:
    //
    //   https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
    //
    // TODO: What event creation flags do we really want here?   See:
    //
    //   https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html
    //
    cuEventCreate_p(&start, CU_EVENT_DEFAULT);
    cuEventCreate_p(&stop, CU_EVENT_DEFAULT);
    cuEventRecord_p(start, 0);
  }
#ifdef _KITRT_VERBOSE_
  fprintf(stderr, "launch parameters:\n");
  fprintf(stderr, "\tnumber of overall elements: %ld\n", numElements);
  fprintf(stderr, "\tblocks/grid = %d\n", blocksPerGrid);
  fprintf(stderr, "\tthreads/block = %d\n", threadsPerBlock);
#endif
  CUstream stream = __kitrt_getThreadStream(gettid());
  CU_SAFE_CALL(cuLaunchKernel_p(kFunc, blocksPerGrid, 1, 1, threadsPerBlock, 1,
                                1, 0, stream, fatBinArgs, NULL));

  if (_kitrtEnableTiming) {
    cuEventRecord_p(stop, 0);
    cuEventSynchronize_p(stop);
    float msecs = 0;
    cuEventElapsedTime_p(&msecs, start, stop);
    if (_kitrtReportTiming)
      printf("%.8lg\n", msecs / 1000.0);
    _kitrtLastEventTime = msecs / 1000.0;
    cuEventDestroy_v2_p(start);
    cuEventDestroy_v2_p(stop);
  }

  return nullptr;
}

void __kitrt_cuLaunchFBKernelOnStream(const void *fatBin,
                                      const char *kernelName, void **fatBinArgs,
                                      uint64_t numElements, void *unused_stream) {
  assert(fatBin && "request to launch null fat binary image!");
  assert(kernelName && "request to launch kernel w/ null name!");
  int threadsPerBlock, blocksPerGrid;

  // TODO: We need a better path here for binding and tracking
  // allcoated resources -- as it stands we will "leak"
  // modules, streams, functions, etc.
  static bool module_built = false;
  static CUmodule module;
  if (!module_built) {
    CU_SAFE_CALL(cuModuleLoadData_p(&module, fatBin));
    module_built = true;
  }
  CUfunction kFunc;
  CU_SAFE_CALL(cuModuleGetFunction_p(&kFunc, module, kernelName));
  CUstream cu_stream = (CUstream)unused_stream;
  if (_kitrtUseHeuristicLaunchParameters)
    __kitrt_cuMaxPotentialBlockSize(blocksPerGrid, threadsPerBlock, kFunc,
                                    numElements);
  else {
    __kitrt_getLaunchParameters(numElements, threadsPerBlock, blocksPerGrid);
  }

  CUevent start, stop;
  if (_kitrtEnableTiming) {
    // Recall that we have to take a bit of care about how we time the
    // launched kernel's execution time.  The problem with using host-device
    // synchronization points is that they can potentially stall the entire
    // GPU pipeline, which we want to avoid to enable asynchronous data
    // movement and the execution of other kernels on the GPU.
    //
    // A nice overview for measuring performance in CUDA:
    //
    //   https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
    //
    // TODO: What event creation flags do we really want here?   See:
    //
    //   https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html
    //
    cuEventCreate_p(&start, CU_EVENT_DEFAULT);
    cuEventCreate_p(&stop, CU_EVENT_DEFAULT);
    cuEventRecord_p(start, cu_stream);
  }

#ifdef _KITRT_VERBOSE_
  fprintf(stderr, "launch parameters:\n");
  fprintf(stderr, "\tnumber of overall elements: %ld\n", numElements);
  fprintf(stderr, "\tblocks/grid = %d\n", blocksPerGrid);
  fprintf(stderr, "\tthreads/block = %d\n", threadsPerBlock);
#endif

  CUstream stream = __kitrt_getThreadStream(gettid());
  CU_SAFE_CALL(cuLaunchKernel_p(kFunc, blocksPerGrid, 1, 1, threadsPerBlock, 1,
                                1, 0, stream, fatBinArgs, NULL));
  if (_kitrtEnableTiming) {
    cuEventRecord_p(stop, cu_stream);
    cuEventSynchronize_p(stop);
    float msecs = 0;
    cuEventElapsedTime_p(&msecs, start, stop);
    if (_kitrtReportTiming)
      printf("%.8lg\n", msecs / 1000.0);
    _kitrtLastEventTime = msecs / 1000.0;
    cuEventDestroy_v2_p(start);
    cuEventDestroy_v2_p(stop);
  }
}

// Launch a kernel on the default stream.
void __kitrt_cuLaunchKernel(const void *fatBin, const char *kernelName,
                            void **fatBinArgs, uint64_t numElements,
                            void *unused_stream) {
  assert(fatBin && "request to launch with null fat binary image!");
  assert(kernelName && "request to launch kernel w/ null name!");
  assert(fatBinArgs && "request to launch kernel w/ null fatbin args!");
  int threadsPerBlock, blocksPerGrid;
  CUfunction kFunc;

  KitRTKernelMap::iterator kern_it = _kitrtKernelMap.find(kernelName);
  if (kern_it == _kitrtKernelMap.end()) {
    CUmodule module;
    KitRTModuleMap::iterator mod_it = _kitrtModuleMap.find(fatBin);
    if (mod_it == _kitrtModuleMap.end()) {
      CU_SAFE_CALL(cuModuleLoadData_p(&module, fatBin));
      _kitrtModuleMap[fatBin] = module;
    } else {
      module = mod_it->second;
    }
    CU_SAFE_CALL(cuModuleGetFunction_p(&kFunc, module, kernelName));
    _kitrtKernelMap[kernelName] = kFunc;
  } else {
    kFunc = kern_it->second;
  }

  __kitrt_getLaunchParameters(numElements, threadsPerBlock, blocksPerGrid);
#ifdef _KITRT_VERBOSE_
  fprintf(stderr, "launch parameters for %s:\n", kernelName);
  fprintf(stderr, "\tnumber of overall elements: %ld\n", numElements);
  fprintf(stderr, "\tblocks/grid = %d\n", blocksPerGrid);
  fprintf(stderr, "\tthreads/block = %d\n", threadsPerBlock);
#endif

  CUevent start, stop;
  if (_kitrtEnableTiming) {
    // An overview for measuring performance in CUDA:
    //
    // https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
    // TODO: What event creation flags do we really want here? See:
    //
    //   https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html
    //
    cuEventCreate_p(&start, CU_EVENT_BLOCKING_SYNC /*DEFAULT*/);
    cuEventCreate_p(&stop, CU_EVENT_BLOCKING_SYNC /*DEFAULT*/);
    // Kick off an event prior to kernel launch...
    cuEventRecord_p(start, 0);
  }

  CUstream stream = __kitrt_getThreadStream(gettid());
  CU_SAFE_CALL(cuLaunchKernel_p(kFunc, blocksPerGrid, 1, 1, threadsPerBlock, 1,
                                1, 0, // shared mem size
                                stream, fatBinArgs, NULL));

  if (_kitrtEnableTiming) {
    cuEventRecord_p(stop, 0);
    cuEventSynchronize_p(stop);
    float msecs = 0.0;
    cuEventElapsedTime_p(&msecs, start, stop);
    _kitrtLastEventTime = msecs / 1000.0;

    if (_kitrtReportTiming)
      printf("kitrt: kernel '%s' runtime, %.8lg seconds\n", kernelName,
             _kitrtLastEventTime);
    cuEventDestroy_v2_p(start);
    cuEventDestroy_v2_p(stop);
  }
}

void __kitrt_cuStreamSynchronize(void *vs) {
  CU_SAFE_CALL(cuStreamSynchronize_p(__kitrt_getThreadStream(gettid())));
}

void __kitrt_cuSynchronizeStreams() {
  CU_SAFE_CALL(cuStreamSynchronize_p(__kitrt_getThreadStream(gettid())));
}

// ---- Event management for timing, etc.

void __kitrt_cuEnableEventTiming(unsigned report) {
  _kitrtEnableTiming = true;
  _kitrtReportTiming = report > 0;
}

void __kitrt_cuDisableEventTiming() {
  _kitrtEnableTiming = false;
  _kitrtReportTiming = false;
  _kitrtLastEventTime = 0.0;
}

void __kitrt_cuToggleEventTiming() {
  _kitrtEnableTiming = _kitrtEnableTiming ? false : true;
  _kitrtLastEventTime = 0.0;
}

double __kitrt_cuGetLastEventTime() { return _kitrtLastEventTime; }

void *__kitrt_cuCreateEvent() {
  CUevent e;
  CU_SAFE_CALL(cuEventCreate_p(&e, CU_EVENT_DEFAULT));
  return (void *)e;
}

void __kitrt_cuDestroyEvent(void *E) {
  assert(E && "unexpected null event!");
  CU_SAFE_CALL(cuEventDestroy_v2_p((CUevent)E));
}

void __kitrt_cuRecordEvent(void *E) {
  assert(E && "unexpected null event!");
  CU_SAFE_CALL(cuEventRecord_p((CUevent)E, 0));
}

void __kitrt_cuSynchronizeEvent(void *E) {
  assert(E && "unexpected null event!");
  CU_SAFE_CALL(cuEventSynchronize_p((CUevent)E));
}

float __kitrt_cuElapsedEventTime(void *start, void *stop) {
  assert(start && "unexpected null start event!");
  float msecs;
  CU_SAFE_CALL(cuEventElapsedTime_p(&msecs, (CUevent)start, (CUevent)stop));
  return (msecs / 1000.0f);
}


static std::mutex _kitrt_stream_mutex;
CUstream __kitrt_getThreadStream(pid_t tid) {
  // TODO: Is find thread safe under a potential insertion via another thread?
  KitRTCUStreamMap::iterator sit = _kitrtCUStreamMap.find(tid);
  CUstream stream;  
  if (sit == _kitrtCUStreamMap.end()) {
    _kitrt_stream_mutex.lock();
    CU_SAFE_CALL(cuStreamCreate_p(&stream, CU_STREAM_DEFAULT));
    _kitrtCUStreamMap[tid] = stream;
    _kitrt_stream_mutex.unlock();
  } else {
    stream = sit->second;
  }
  return stream;
}

void __kitrt_deleteThreadStream(pid_t tid) {
  KitRTCUStreamMap::iterator sit = _kitrtCUStreamMap.find(tid);
  if (sit != _kitrtCUStreamMap.end()) {
    CU_SAFE_CALL(cuStreamDestroy_v2_p(sit->second));
    _kitrt_stream_mutex.lock();
    _kitrtCUStreamMap.erase(sit);            
    _kitrt_stream_mutex.unlock();
  }
}  

void __kitrt_destroyThreadStreams() {
  _kitrt_stream_mutex.lock();
  for (auto &entry : _kitrtCUStreamMap)
    CU_SAFE_CALL(cuStreamDestroy_v2_p(entry.second));
  _kitrtCUStreamMap.clear();
  _kitrt_stream_mutex.unlock();
}

  
  
} // extern "C"
