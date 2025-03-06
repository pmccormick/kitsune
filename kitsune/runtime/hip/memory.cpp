/*
 *===---- memory.cpp HIP memory management support-------------------------===
 *
 * Copyright (c) 2021, 2023 Los Alamos National Security, LLC.
 * All rights reserved.
 *
 *  Copyright 2021, 2023. Los Alamos National Security, LLC. This software
 *  was produced under U.S. Government contract DE-AC52-06NA25396 for Los
 *  Alamos National Laboratory (LANL), which is operated by Los Alamos
 *  National Security, LLC for the U.S. Department of Energy. The
 *  U.S. Government has rights to use, reproduce, and distribute this
 *  software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY,
 *  LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY
 *  FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
 *  derivative works, such modified software should be clearly marked,
 *  so as not to confuse it with the version available from LANL.
 *
 *  Additionally, redistribution and use in source and binary forms,
 *  with or without modification, are permitted provided that the
 *  following conditions are met:
 *
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *
 *    * Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following
 *      disclaimer in the documentation and/or other materials provided
 *      with the distribution.
 *
 *    * Neither the name of Los Alamos National Security, LLC, Los
 *      Alamos National Laboratory, LANL, the U.S. Government, nor the
 *      names of its contributors may be used to endorse or promote
 *      products derived from this software without specific prior
 *      written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
 *  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 *  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 *  OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 *  SUCH DAMAGE.
 *
 *===----------------------------------------------------------------------===
 */
#include "kithip.h"
#include "kithip_rtinfo.h"

#include "memory_map.h"
#include <mutex>
#include <string>

// Note that there are close bindings between the runtime and compiler
// and the details of when different settings are enabled/disabled can
// be driven by the compiler via command line arguments and/or language
// level attributes/directives.  In general, using direct runtime calls
// in source can be tricky without understanding the compiler-runtime
// details.
//
// To help overcome some hassle when debugging (e.g., having to
// continually edit-rebuilt-test), the runtime accepts various settings
// via environment variables.  In most cases these environment variables
// completely override the runtime/compiler settings.  Throughout the
// runtime code various settings include their associated environment
// variable in the comments.

static std::mutex _kithip_mem_alloc_mutex;

extern "C" {

// XNACK NOTES
//
// XNACK enables retries of memory access when a page fault occurs and is
// specifically desired when using managed memory (aka UVM). Without it
// on-demand migration of pages does not occur and data access will occur
// over host memory (pages remain in place).  This (obviously) kills
// performance...
//
// There are a few steps required to enable it when using AMD GPUs.  The
// first is the using the HSA_XNACK=1 environment setting.  The AMD docs
// also suggest,
//
//   "While this environment variable is required at kernel runtime to enable
//   page migration, it is also helpful to enable this environment variable at
//   compile time, which can change the performance of any compiled kernels"
//
// However, there are no settings within all of the open-source LLVM project
// that references this variable -- AMD's own releases potentially use this
// but it is doubtful to impact compile time w/ Kitsune.
//
// Next, XNACK requires kernel support. Specifically, the kernel must have
// HMM support (likely, the more recent the better).
//
// It is also important to configure the runtime details to get XNACK support
// to work well -- these details are commented in the code below that goes
// along with management memory allocations.
//
// TODO: We have not done a significant amount of engineering work to get XNACK
// and all possible compilation targets handled by either the compiler nor
// the runtime.
//
// *** Checking for XNACK support ***
//
//  Using rocminfo and setting HSA_XNACK you should be able to determine if
//  your system supports XNACK:
//
//    $ HSA_XNACK=1 rocminfo
//
//  Output should include the string 'xnack+' if supported.
//
// IMPORTANT CAVEAT: As mentioned above, compiling with xnack enabled and then
// attempting to disable it at runtime is risky.  It can either result in
// kernel failure or even corruption/silent errors.   The runtime will spit
// out warnings if it encounters a mismatch but will not terminate execution.

void __kithip_initialze_memory_limits(const hipDeviceProp_t &) {
  using namespace kithip_rt;

  // no-op for now...
}

// Check for the xnack setting in the enviornment. In favor of
// avoiding more enviornment variable clutter, we reuse the AMD
// HSA flag.  Note that the runtime will also set this variable
// if xnack is enabled via compiler flags.
bool __kithip_xnack_env_check() {
  using namespace kithip_rt;

  int use_xnack;
  if (not __kitrt_get_env_value("HSA_XNACK", use_xnack))
    use_xnack = 0;
  return (use_xnack == 1);
}

// This is the compiler's entry point for xnack support.
// At this level all that is done is internal tracking
// and the setting of HSA_XNACK in the enviornment.
void __kithip_enable_xnack() {
  using namespace kithip_rt;

  int set_xnack;
  if (not __kitrt_get_env_value("HSA_XNACK", set_xnack)) {
    // We are looking to enable XNACK but there is not a supporting
    // environment variable...  For correct low-level behavior from
    // rocm (and perhaps even the kernel) we need to set "HAS_XNACK".
    __kitrt_set_env("HSA_XNACK", std::to_string(1).c_str());
    if (__kitrt_verbose_mode())
      fprintf(stderr, "kitrt[hip]: auto-set HAS_XNACK=1\n");
  }

  if (__kitrt_verbose_mode())
    fprintf(stderr, "kitrt[hip]: xnack enabled.\n");
}

// Enable/Disable the xnack operation.  This setting is largely
// in the hands of the hsa/rocm runtime and it can be a touchy
// setting as compiler-side enablement and disablement in the
// runtime can cause conflict and potenitally introduce errors.
// Care is recommended when mixing conflicting settings...
void __kithip_set_xnack(bool flag) {
  using namespace kithip_rt;

  bool xnack_env = __kithip_xnack_env_check();
  if (xnack_env != flag) {
    fprintf(stderr, "kitrt[hip]: note HSA_XNACK setting overriding/"
                    "conflicting with runtime settings.\n"
                    "  stability and/or correctness issues may occur.\n");
    setXnack(xnack_env);
  } else {
    setXnack(flag);
  }

  if (__kitrt_verbose_mode())
    fprintf(stderr, "kitrt[hip]: xnack mode is %s.\n",
            xnackEnabled() ? "enabled\n" : "disabled\n");
}

// Allocate a block of managed memory (UVM) of 'size' bytes.
__attribute__((malloc)) void *__kithip_mem_alloc_managed(size_t size) {
  using namespace kithip_rt;

  if (not isInitialized()) {
    // Note: compiler handles this in a global ctor but we
    // do this here to make writing test programs a bit
    // easier...
    __kithip_initialize();
  }

  void *alloced_ptr = nullptr;
  HIP_SAFE_CALL(hipSetDevice(deviceID()));
  HIP_SAFE_CALL(hipMallocManaged(&alloced_ptr, size, hipMemAttachGlobal));

  // LOCK
  _kithip_mem_alloc_mutex.lock();
  __kitrt_register_mem_alloc(alloced_ptr, size);
  _kithip_mem_alloc_mutex.unlock();
  // UNLOCK

  assert(alloced_ptr != nullptr && "kitrt[hip]: unexpected null allocation!");

  if (__kitrt_verbose_mode())
    fprintf(stderr,
            "kitrt[hip]: allocated and registered %ld bytes"
            " of management memory (address = %p).\n",
            size, alloced_ptr);

  // Attempt to provide the hip/rocm runtime with extra information
  // about the block of memory that might help improve performance.
  HIP_SAFE_CALL(hipMemAdvise(alloced_ptr, size,
                             hipMemAdviseSetPreferredLocation, deviceID()));
  HIP_SAFE_CALL(
      hipMemAdvise(alloced_ptr, size, hipMemAdviseSetAccessedBy, deviceID()));
  HIP_SAFE_CALL(
      hipMemAdvise(alloced_ptr, size, hipMemAdviseSetReadMostly, deviceID()));
  // This call currently seems to be the most signifcant in terms of improving
  // performance -- others appear to be mostly ignored...
  HIP_SAFE_CALL(
      hipMemAdvise(alloced_ptr, size, hipMemAdviseSetCoarseGrain, deviceID()));
  return alloced_ptr;
}

__attribute__((malloc)) void *__kithip_mem_calloc_managed(size_t count,
                                                          size_t element_size) {
  assert(count != 0 && "kitrt[hip]: zero-valued item count!");
  assert(element_size != 0 && "kitrt[hip]: zero-valued element size!");

  using namespace kithip_rt;

  size_t nbytes = count * element_size;
  void *memp = __kithip_mem_alloc_managed(nbytes);

  // TODO: Is there a risk of a race here?
  HIP_SAFE_CALL(hipMemsetD8Async(memp, 0, nbytes,
                                 (hipStream_t)__kithip_get_thread_stream()));
  return (void *)memp;
}

__attribute__((malloc)) void *__kithip_mem_realloc_managed(void *ptr,
                                                           size_t size) {
  assert(size != 0 && "kitrt[hip]: zero-valued size!");

  using namespace kithip_rt;

  void *memptr = nullptr;
  size_t alloced_nbytes = 0;

  if (ptr == nullptr) {
    // just a malloc() equivalent call...
    return __kithip_mem_alloc_managed(size);
  } else {
    // Check to make sure this is a pointer we're actually managing.
    bool read_only, write_only;
    alloced_nbytes = __kitrt_get_mem_alloc_size(ptr, &read_only, &write_only);
    assert(alloced_nbytes != 0 &&
           "kitrt[hip]: realloc() on untracked allocation!");
  }

  if (size > alloced_nbytes) {
    memptr = __kithip_mem_alloc_managed(size);
    HIP_SAFE_CALL(hipMemcpy(memptr /* dest */, ptr /* source */, alloced_nbytes,
                            hipMemcpyDefault));
    // TODO: Race?  Do we need to lock the free here?
    __kithip_mem_free(ptr);
  } else if (size < alloced_nbytes) {
    memptr = __kithip_mem_alloc_managed(size);
    HIP_SAFE_CALL(hipMemcpy(memptr /* dest */, ptr /* source */, alloced_nbytes,
                            hipMemcpyDefault));
    // TODO: Race?  Do we need to lock the free here?
    __kithip_mem_free(ptr);
  } else
    memptr = ptr; // TODO: does this match realloc() behavior?

  return memptr;
}

void __kithip_mem_free(void *vp) {
  assert(vp && "kitrt[hip]: unexpected null pointer!");

  using namespace kithip_rt;

  // LOCK
  _kithip_mem_alloc_mutex.lock();
  __kitrt_unregister_mem_alloc(vp);
  _kithip_mem_alloc_mutex.unlock();
  // UNLOCK

  HIP_SAFE_CALL(hipFree(vp));
}

void __kithip_mem_destroy(void *vp) {
  using namespace kithip_rt;

  assert(vp && "kitrt[hip]: unexpected null pointer!");

  // This entry point is used to clean up only the
  // HIP portions of an allocation -- it is used
  // by the runtime at program exit.
  HIP_SAFE_CALL(hipFree(vp));
}

bool __kithip_is_mem_managed(void *vp) {
  assert(vp && "kitrt[hip]: unexpected null pointer!");
  assert(__kithip_is_initialized() && "kitrt[hip]: runtime not initialized!");

  using namespace kithip_rt;

  unsigned int is_managed;
  // NOTE: We don't wrap in a HIP-safe call here as we could be
  // passing in a bogus pointer -- if we get a HIP error we will
  // assume the pointer is unmanaged and return false accordingly.
  hipError_t r =
      hipPointerGetAttribute(&is_managed, HIP_POINTER_ATTRIBUTE_IS_MANAGED, vp);
  return (r == hipSuccess) && is_managed;
}

// NOTE: See within the code below for notes about the prefetching
// semantics.
void *__kithip_mem_gpu_prefetch(void *vp, void *opaque_stream) {
  assert(vp && "kitrt[hip]: unexpected null pointer!");

  using namespace kithip_rt;

  size_t size = 0;
  // TODO: Prefetching details and approaches need to be further
  // explored.  In particular, in concert with compiler analysis
  // and code generation.
  //
  // The semantics here are simplistic as we don't have enough
  // information to really guarantee any really "smart" behaviors.
  // If we have previously issued a prefetch on a given pointer
  // it will show here as prefetched (we presently cannot
  // track host-side 'touches' where a subsequent prefetch might
  // be beneficial).  Some testing suggests blindly reissuing
  // prefetches introduces overhead for already resident allocations.
  //
  // More advanced features will require compiler-side analysis.
  // We've aimed to keep things on the simple side now vs. more
  // complicated code base to maintain and debug.
  if (not __kitrt_is_mem_prefetched(vp, &size)) {
    hipStream_t hip_stream;
    if (opaque_stream)
      hip_stream = (hipStream_t)opaque_stream;
    else
      hip_stream = (hipStream_t)__kithip_get_thread_stream();

    if (__kitrt_verbose_mode())
      fprintf(stderr,
              "kitrt[hip]: issue prefetch(address=%p, size=%ld, stream=%p)\n",
              vp, size, (void *)hip_stream);

    HIP_SAFE_CALL(hipSetDevice(deviceID()));
    HIP_SAFE_CALL(hipMemPrefetchAsync(vp, size, deviceID(), hip_stream));

    // LOCK
    _kithip_mem_alloc_mutex.lock();
    __kitrt_mark_mem_prefetched(vp);
    _kithip_mem_alloc_mutex.unlock();
    // UNLOCK
    return (void *)hip_stream;
  }

  return opaque_stream;
}

void __kithip_mem_host_prefetch(void *vp) {
  assert(vp && "kitrt[hip]: unexpected null pointer!");
  // TODO: Prefetching details and approaches need to be further
  // explored.  In particular, in concert with compiler analysis and
  // code generation.
  //
  // The semantics here are tricky and we don't have enough
  // information to guarantee "smart" behavior.  If we have ever
  // issued a prefetch to the device (gpu) it will show here as
  // prefetched.  In this case we assume a prefetch back to the host
  // is preferred and will let it proceed.  There are obviously cases
  // where this is helpful and others where it will lead to page
  // faults and evictions.  Little work has been done with host-side
  // prefetch requests.

  using namespace kithip_rt;

  size_t size;
  if (__kitrt_is_mem_prefetched(vp, &size)) {
    if (size > 0) {
      // The logic here resets the memory advice from being
      // GPU-centric to host-side preferred.  The logic is
      // to assume that host-side access suggests pending
      // operations after completion of a kernel (the inverse
      // model of what happens prior to a kernel launch).
      //
      // TODO: A lot of work needs to go into seeing if we can be
      // smarter about device- and host-side prefetching.
      //HIP_SAFE_CALL(
      //    hipMemAdvise(vp, size, hipMemAdviseSetPreferredLocation, deviceID()));
      // Issue a prefetch request on the stream associated with the
      // calling thread. Once issued go ahead and mark the memory as
      // no long being prefetched to the device/GPU.  This "mark" does
      // not guarantee prefetching is complete it simply flags that
      // the "instruction" has been issued by the runtime.
      HIP_SAFE_CALL(hipMemPrefetchAsync(
          vp, size, deviceID(), (hipStream_t)__kithip_get_thread_stream()));
      __kitrt_set_mem_prefetch(vp, false);
    }
  }
}

void __kithip_memcpy_sym_to_device(void *hostPtr, void *devPtr, size_t size) {
  assert(devPtr != 0 && "kitrt[hip]: unexpected null device pointer!");
  assert(hostPtr != nullptr && "kitrt[hip]: unexpected null host pointer!");
  assert(size != 0 && "kitrt[hip]: requested a 0 byte copy!");
  HIP_SAFE_CALL(hipMemcpyHtoD(devPtr, hostPtr, size));
}

} // extern "C"
