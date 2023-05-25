/*
 * ===- kithip.cpp - Kitsune runtime HIP support    ---------------------===//
 *
 * Copyright (c) 2021, 2023 Los Alamos National Security, LLC.
 * All rights reserved.
 *
 * Copyright 2021, 2023. Los Alamos National Security, LLC. This
 *  software was produced under U.S. Government contract
 *  DE-AC52-06NA25396 for Los Alamos National Laboratory (LANL), which
 *  is operated by Los Alamos National Security, LLC for the
 *  U.S. Department of Energy. The U.S. Government has rights to use,
 *  reproduce, and distribute this software.  NEITHER THE GOVERNMENT
 *  NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS
 *  OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.
 *  If software is modified to produce derivative works, such modified
 *  software should be clearly marked, so as not to confuse it with
 *  the version available from LANL.
 *
 *  Additionally, redistribution and use in source and binary forms,
 *  with or without modification, are permitted provided that the
 *  following conditions are met:
 *
 * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
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
 *  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 *  AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *===----------------------------------------------------------------------===
 */

// TODO: Add support for roctracer (see https://github.com/ROCm/roctracer)

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <iostream>
#include <list>
#include <map>
#include <mutex>
#include <sstream>
#include <stdbool.h>
#include <sys/syscall.h>

#include "kithip.h"
#include "kithip_dylib.h"
#include "kitrt.h"
#include "memory_map.h"

#ifdef KITHIP_ENABLE_ROCTX
#error "rocTX is not yet supported."
#endif

// Some global state -- code outside this file should use the accesses
// via the helper functions.
bool _kithip_initialized = false;
int _kithip_device_id = -1;
static hipDeviceProp_t _kithip_device_props;
static int _kithip_max_threads_per_blk;
static bool _kithip_use_xnack = false;

// TODO: These don't need to be global -- we're not using them elsewhere beyond
// the initialization code for verbose feedback durring runtime...

static int _kithip_ecc_enabled;
static int _kithip_num_async_engines;
static int _kithip_can_map_host_mem;
static int _kithip_concurrent_kernels;
static int _kithip_uses_host_page_table;

extern "C" {

bool __kithip_initialize() {

  if (_kithip_initialized) {
    fprintf(stderr, "kithip: warning, multiple initialization calls!\n");
    return true;
  }

  __kitrt_initialize();

  if (_kithip_use_xnack) {
    (void)setenv("HSA_XNACK", "1", 1);
    if (__kitrt_verbose_mode()) {
      fprintf(stderr, "kithip: xnack enabled.\n");
      fprintf(stderr, "        HSA_XNACK automatically set in running environment.\n");
    }
  }

  if (not __kithip_load_symbols()) {
    // TODO: This error block is repetitive in the runtime...  Probably best
    // to collapse them down to a call so that we can get consistent messages
    // and mechanisms across the runtime...
    fprintf(stderr, "kithip: FATAL ERROR - "
                    "unable to resolve dynamic symbols for HIP.\n");
    fprintf(stderr, "kithip: aborting...\n");
    __kitrt_print_stack_trace();
    abort();
  }

  // NOTE: Even though the HIP docs suggest that, "most HIP [calls]
  // implicitly initialize the runtime. This [call] provides control
  // over the timing of the initialization."  Instead of relying on
  // HIP to deal with initialization we prefer to be explicit about
  // it -- given we're a code gen target, do some funny things via a
  // global ctor, and "most" is undefined/unclarified in the HIP docs.
  HIP_SAFE_CALL(hipInit_p(0));

  int device_count;
  HIP_SAFE_CALL(hipGetDeviceCount_p(&device_count));
  if (device_count <= 0) {
    fprintf(stderr, "kithip: FATAL ERROR -- "
                    "no suitable HIP devices found!\n");
    fprintf(stderr, "kithip: aborting...\n");
    __kitrt_print_stack_trace();
    abort();
  }

  // Note that instead of sharing a common device id across runtime
  // components (e.g., with CUDA and HIP) we instead isolate them
  // within each sub-component.  This isolation provides some
  // flexibility, future-proofing of details, and a possibility of
  // keeping support isolated for bizarre cases where multiple
  // devices from different vendors are available on a single
  // system.

  if (!__kitrt_get_env_value("KITHIP_DEVICE_ID", _kithip_device_id))
    _kithip_device_id = 0;

  assert(_kithip_device_id < device_count &&
         "kithip: KITHIP_DEVICE_ID value exceeds available number"
         " of devices.");

  HIP_SAFE_CALL(hipSetDevice_p(_kithip_device_id));
  HIP_SAFE_CALL(
      hipGetDeviceProperties_p(&_kithip_device_props, _kithip_device_id));

  // For ease of code generation on part of the compiler and humans
  // (mostly writing the runtime and compiler support) we currently
  // require managed memory support.  While this can introduce
  // performance problems, it significantly reduces explicit
  // requirements to manage data movement at the program level as
  // well as trying to track these details within the runtime.
  //
  // HIP is somewhat challenging on this front as we have found
  // some details to be inconsistent in terms of what is documented
  // and how it actually behaves in practice.  Given this has touch
  // points all the way into the kernel's page management it is not
  // entirely straightforward to debug and analyze -- work needs to
  // be on the HIP side here to better understand what's going on...
  //
  // From AMD's documentation:
  //
  //   "Managed memory [snip] is supported in the HIP combined
  //   host/device compilation. Through unified memory allocation,
  //   managed memory allows data to be shared and accessible to
  //   both the CPU and GPU using a single pointer. The allocation
  //   is managed by the AMD GPU driver using the Linux
  //   Heterogeneous Memory Management (HMM) mechanism. The user
  //   can call managed memory API hipMallocManaged() to allocate a
  //   large chunk of HMM memory, execute kernels on a device, and
  //   fetch data between the host and device as needed.
  //
  //   In a HIP application, it is recommended to do a capability
  //   check before calling the managed memory APIs."
  int has_managed_memory = 0;
  HIP_SAFE_CALL(hipDeviceGetAttribute(
      &has_managed_memory, hipDeviceAttributeManagedMemory, _kithip_device_id));
  if (!has_managed_memory) {
    fprintf(stderr, "kithip: device does not support managed memory!\n");
    fprintf(stderr, "kithip: aborting...\n");
    __kitrt_print_stack_trace();
    abort();
  }

  // AMD's example code suggests this is a preferred (additional?
  // required?) check that should be done prior to using managed
  // memory.  TODO: It is unclear why there is a disagreement
  // between docs and code samples.  We could possibly clarify or
  // simplify code here by looking at this in more detail.
  int has_concurrent_access = 0;
  HIP_SAFE_CALL(hipDeviceGetAttribute_p(
      &has_concurrent_access, hipDeviceAttributeConcurrentManagedAccess,
      _kithip_device_id));

  if (!has_concurrent_access) {
    fprintf(stderr, "kithip: device does not have support for concurrent"
                    " managed memory accesses!\n");
    fprintf(stderr, "kithip: aborting...\n");
    __kitrt_print_stack_trace();
    abort();
  }

  // At this point we're ready to go as far as the basic HIP
  // initialization requirements go.  Let's consider this a success.
  _kithip_initialized = true;

  // Grab some device-specific details that we will need either at
  // runtime or to provide some verbose feedback during execution to
  // help provide platform-/target-specific details for performance,
  // debugging, etc.
  HIP_SAFE_CALL(hipDeviceGetAttribute_p(&_kithip_max_threads_per_blk,
                                        hipDeviceAttributeMaxThreadsPerBlock,
                                        _kithip_device_id));
  HIP_SAFE_CALL(hipDeviceGetAttribute_p(
      &_kithip_ecc_enabled, hipDeviceAttributeEccEnabled, _kithip_device_id));
  //HIP_SAFE_CALL(hipDeviceGetAttribute_p(&_kithip_num_async_engines,
  //                                      hipDeviceAttributeAsyncEngineCount,
  //                                      _kithip_device_id));
  HIP_SAFE_CALL(hipDeviceGetAttribute_p(&_kithip_can_map_host_mem,
                                        hipDeviceAttributeCanMapHostMemory,
                                        _kithip_device_id));
  HIP_SAFE_CALL(hipDeviceGetAttribute_p(&_kithip_concurrent_kernels,
                                        hipDeviceAttributeConcurrentKernels,
                                        _kithip_device_id));
  HIP_SAFE_CALL(hipDeviceGetAttribute_p(
      &_kithip_uses_host_page_table,
      hipDeviceAttributePageableMemoryAccessUsesHostPageTables,
      _kithip_device_id));

  if (__kitrt_verbose_mode()) {
    fprintf(stderr, "kithip: found %d devices.\n", device_count);
    fprintf(stderr, "        using device:         %d\n", _kithip_device_id);
    fprintf(stderr, "        max threads/blk       %d\n",
            _kithip_max_threads_per_blk);
    fprintf(stderr, "        ecc enabled:          %s\n",
            _kithip_ecc_enabled ? "yes" : "no");
    fprintf(stderr, "        no. async engines:    %d\n",
            _kithip_num_async_engines);
    fprintf(stderr, "        can map host mem:     %s\n",
            _kithip_can_map_host_mem ? "yes" : "no");
    fprintf(stderr, "        uses host page table: %s\n",
            _kithip_uses_host_page_table ? "yes" : "no");
    fprintf(stderr, "        concurrent kernels:   %s\n",
            _kithip_concurrent_kernels ? "yes" : "no");
  }

  int threads_per_block = 1024;
  if (__kitrt_get_env_value("KITHIP_THREADS_PER_BLOCK", threads_per_block)) {
    if (threads_per_block > _kithip_max_threads_per_blk)
      threads_per_block = _kithip_max_threads_per_blk;
    __kithip_set_default_threads_per_blk(threads_per_block);
    if (__kitrt_verbose_mode())
      fprintf(stderr, "kithip: threads/block: %d\n", threads_per_block);
  }

    bool enable_occupancy_calc = true;
    if (__kitrt_get_env_value("KITHIP_USE_OCCUPANCY_LAUNCH",
			      enable_occupancy_calc))
      __kithip_use_occupancy_launch(true);
    if (__kitrt_verbose_mode())
      fprintf(stderr, "  kithip: occupancy-based launches enabled.\n");  

  return _kithip_initialized;
}

void __kithip_enable_xnack() {
  _kithip_use_xnack = true;
}

void __kithip_destroy() {
  if (not _kithip_initialized)
    return;

  __kithip_destroy_thread_streams();
  __kitrt_destroy_memory_map(__kithip_mem_destroy);
  HIP_SAFE_CALL(hipDeviceReset_p());
  _kithip_initialized = false;
}

} // extern "C"
