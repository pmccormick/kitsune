/*
 *===- kithip.h - HIP runtime interface   ---------------------------------===
 *
 * Copyright (c) 2021, 2023, 2025 Los Alamos National Security, LLC.
 * All rights reserved.
 *
 * Copyright 2021, 2023, 2025. Los Alamos National Security, LLC. This
 * software was produced under U.S. Government contract DE-AC52-06NA25396
 * for Los Alamos National Laboratory (LANL), which is operated by Los Alamos
 *  National Security, LLC for the U.S. Department of Energy. The
 *  U.S. Government has rights to use, reproduce, and distribute this
 *  software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY,
 *  LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY
 *  FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
 *  derivative works, such modified software should be clearly marked,
 *  so as not to confuse it with the version available from LANL.
 *
 *  Additionally, redistribution and use in source and binary forms,
 *   with or without modification, are permitted provided that the
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

#ifndef __KITHIP_RTINFO_H_
#define __KITHIP_RTINFO_H_

#define __HIP_DISABLE_CPP_FUNCTIONS__ // skip extra c++ cruft

// we're only interested in AMD GPUs and HIP (no CUDA).
#define __HIP_PLATFORM_AMD__ 1
#define __HIP_PLATFORM_HCC__ 1
#include <hip/hip_runtime.h>

namespace kithip_rt {

struct kithip_rt_info_t {
  bool initialized = false;
  int device_id = 0; // KITHIP_DEVICE_ID
  int device_count = -1;
  hipDeviceProp_t props{0};

  // compiler/enviornment driven settings.
  bool xnack = true;
  bool ylaunch = false;
  int max_threads_per_block = -1; // KITHIP_MAX_THREADS_PER_BLOCK
  int threads_per_block = -1;     // KITHIP_THREADS_PER_BLOCK
};

// Yep, it is true... A C++ faux pas...  Our current design
// borrows a bit of C++'ish practices but in general we try
// (in true runtime fashion) to stay a bit closer to C building
// blocks -- primarily to maintain an easier compiler target
// path.  This is meant to be lower-level "ugly" code...
extern kithip_rt_info_t rt_info;

inline bool isInitialized() { return rt_info.initialized; }
inline void setInitialized(bool is_initialized) {
  rt_info.initialized = is_initialized;
}

inline int deviceID() { return rt_info.device_id; }
inline void setDeviceID(int ID) {
  assert(ID >= 0 && ID < rt_info.device_count && "device id out of bounds!");
  rt_info.device_id = ID;
}

inline int deviceCount() { return rt_info.device_count; }
inline void setDeviceCount(int count) { rt_info.device_count = count; }

inline bool supportsManagedMemory() {
  return static_cast<bool>(rt_info.props.managedMemory &&
                           rt_info.props.concurrentManagedAccess);
}

inline int multiProcessorCount() { return rt_info.props.multiProcessorCount; }
inline int warpSize() { return rt_info.props.warpSize; }
inline int maxThreadsPerMultiProcessor() {
  return rt_info.props.maxThreadsPerMultiProcessor;
}
inline int maxThreadsPerBlock() { return rt_info.props.maxThreadsPerBlock; }
inline int numRegistersPerBlock() { return rt_info.props.regsPerBlock; }
inline int maxThreadsDim(unsigned axis) {
  assert(axis > 0 && axis < 2 &&
         "maxThreadsDim axis must be 0 (X), 1 (Y), or 2 (Z)!");
  return rt_info.props.maxThreadsDim[axis];
}

inline bool xnackEnabled() { return rt_info.xnack; }
inline void setXnack(bool enabled) { rt_info.xnack = enabled; }

inline bool useYLaunch() { return rt_info.ylaunch; }
inline void setYLaunch(bool enable) { rt_info.ylaunch = enable; }

inline int kitMaxThreadsPerBlock() { return rt_info.max_threads_per_block; }
inline void kitSetMaxThreadsPerBlock(int tpb) {
  rt_info.max_threads_per_block = tpb;
}

inline int kitThreadsPerBlock() { return rt_info.threads_per_block; }
inline void kitSetThreadsPerBlock(int tpb) { rt_info.threads_per_block = tpb; }

inline double bytesToGBytes(size_t size) {
  return static_cast<double>(size) / (1024.0 * 1024.0 * 1024.0);
}

inline double bytesToMBytes(size_t size) {
  return static_cast<double>(size) / (1024.0 * 1024.0);
}

inline double bytesToKBytes(size_t size) {
  return static_cast<double>(size) / 1024.0;
}
} // namespace kithip_rt

#endif
