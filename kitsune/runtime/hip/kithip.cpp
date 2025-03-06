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
#include "kithip_rtinfo.h"
#include "kitrt.h"
#include "memory_map.h"

namespace kithip_rt {
kithip_rt_info_t rt_info;
}

#ifdef KITHIP_ENABLE_ROCTX
#error "rocTX is not yet supported."
#endif

// All "compiler-facing" calls are C calling convention to avoid having to
// codegen C++ managled names...

extern "C" {

void __kithip_dump_dev_properties(hipDeviceProp_t &props) {
  using namespace std;
  using namespace kithip_rt;
  cerr << "kitrt[hip]: ###### DEVICE PROPERTIES ######\n";
  cerr << "  GPU name/model: " << props.name << endl;
  cerr << "    - PCI bus info ***\n";
  cerr << "        bus id: " << props.pciBusID << endl;
  cerr << "        device id: " << props.pciDeviceID << endl;
  cerr << "        domain id: " << props.pciDomainID << endl;
  cerr << "   - Compute Unit (CU) info ***\n";
  cerr << "        multi-processor (mp) count: " << props.multiProcessorCount
       << endl;
  cerr << "      max threads / mp: " << props.maxThreadsPerMultiProcessor
       << endl;
  cerr << "      max threads / block: " << props.maxThreadsPerBlock << endl;
  cerr << "      max registers / block: " << props.regsPerBlock << endl;
  cerr << "      warp size: " << props.warpSize << endl;
  cerr << "      max threads x dim: " << props.maxThreadsDim[0] << endl;
  cerr << "      max threads y dim: " << props.maxThreadsDim[1] << endl;
  cerr << "      max threads z dim: " << props.maxThreadsDim[2] << endl;
  cerr << "      max grid x size: " << props.maxGridSize[0] << endl;
  cerr << "      max grid y size: " << props.maxGridSize[1] << endl;
  cerr << "      max grid z size: " << props.maxGridSize[2] << endl;
  cerr << "   *** General memory specs ***\n";
  cerr << "      total constant memory: " << props.totalConstMem << endl;
  cerr << "      shared memory / block: " << props.sharedMemPerBlock << endl;
  cerr << "      L2 cache size: " << props.l2CacheSize << endl;
  cerr << "      total global memory: " << bytesToGBytes(props.totalGlobalMem)
       << " GB" << endl;
}

bool __kithip_is_initialized() {
  using namespace kithip_rt;
  return rt_info.initialized;
}

bool __kithip_initialize() {

  using namespace kithip_rt;

  if (isInitialized()) {
    fprintf(stderr,
            "kitrt[hip]: warning, mutliple initialization attempts...\n");
    return true;
  }

  // initialize common runtime data structures and settings.
  __kitrt_initialize();

  // AMD's documentation suggests that there is no need to explicilty
  // call hipInit() as all API entry points will initialize when
  // necessary.  For now, we'll just follow the more direct path.
  //HIP_SAFE_CALL(hipInit(0));

  HIP_SAFE_CALL(hipGetDeviceCount(&rt_info.device_count));
  if (rt_info.device_count <= 0) {
    fprintf(stderr, "kitrt[hip]: no suitable hip-enabled devices found!\n");
    __kitrt_print_stack_trace();
    abort();
  }

  // TODO: We need to consider multi-gpu device support and use cases
  // where multiple threads (parallel constructs) can drive multiple
  // gpus.

  // There have been several cases where debugging and performance
  // regressions have tripped us up on multi-gpu systems. To help
  // out in these cases we provide a path to pick the device id via
  // the environment.
  if (not __kitrt_get_env_value("KITHIP_DEVICE_ID", rt_info.device_id)) {
    setDeviceID(0); // Default is always the first device.
  } else {
    assert(deviceID() < deviceCount() && deviceID() >= 0 &&
           "kitrt[hip]: KITHIP_DEVICE_ID is out of range/invalid!");
    fprintf(stderr, "kitrt[hip]: env override, using device: %d.\n",
            deviceID());
  }

  HIP_SAFE_CALL(hipSetDevice(deviceID()));
  HIP_SAFE_CALL(hipGetDeviceProperties(&rt_info.props, deviceID()));
  if (__kitrt_verbose_mode())
    __kithip_dump_dev_properties(rt_info.props);

  if (not supportsManagedMemory()) {
    fprintf(stderr,
            "kitrt[hip]: device/system does not support managed memory!\n");
    fprintf(stderr, "  kitsune does not support this platform.\n");
    abort();
  }

  // We should be good to go...
  setInitialized(true);
  return isInitialized();
}

void __kithip_destroy() {
  using namespace kithip_rt;
  if (not isInitialized())
    return;
  __kithip_destroy_thread_streams();
  __kitrt_destroy_memory_map(__kithip_mem_destroy);
  HIP_SAFE_CALL(hipDeviceReset());
  setInitialized(false);
}

} // extern "C"
