//===---- streams.cpp HIP streams support----------------------------------===
//
// Copyright (c) 2021, 2023 Los Alamos National Security, LLC.
//
// All rights reserved.
//
//  Copyright 2021, 2023. Los Alamos National Security, LLC. This software 
//  was produced under U.S. Government contract DE-AC52-06NA25396 for Los
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
//   * Redistributions of source code must retain the above copyright
//     notice, this list of conditions and the following disclaimer.
//
//   * Redistributions in binary form must reproduce the above
//     copyright notice, this list of conditions and the following
//     disclaimer in the documentation and/or other materials provided
//     with the distribution.
//
//   * Neither the name of Los Alamos National Security, LLC, Los
//     Alamos National Laboratory, LANL, the U.S. Government, nor the
//     names of its contributors may be used to endorse or promote
//     products derived from this software without specific prior
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

#include "kithip.h"
#include "kithip_dylib.h"
#include <mutex>
#include <deque>
#include <algorithm>
#include <sys/syscall.h>
#include <unistd.h>


// On older systems gettid() is not exposed and the syscall()
// interface must be used.  It won't hurt us to just use that
// everywhere as there are still systems in use with older
// system libraries that are missing the call...
#define gettid() syscall(SYS_gettid)


// NOTE: HIP and streams...  HIP supports a per-thread default stream
// -- it is implicit that is local to *both* the calling thread and
// the currently selected device.  Thus, commands issued to the
// default stream does not sync with other streams.  In addition, the
// per-thread stream is blocking and will synchronize with the global
// "null" stream if both are used...  Use of the per-thread stream is
// enabled via compilation with "-fgpu-default-stream=per-thread".
// There are ways to use this within the runtime calls (use
// hipStreamPerThread as the stream handle).
//
// TODO: It is unclear if HIP's per-thread stream might provide an
// advantage vs. what we are currently using to create a stream
// per calling thread...  Details below... 
//
// TODO: Rethink stream and thread mapping with an eye towards
// reducing resource usage.
//
// The runtime currently tracks a unique thread per calling thread.
// This allows us to separate entry points and support concurrent
// kernels and other similar operations on a per-stream basis.  For
// large thread counts this can be problematic as we potentially run
// the risk of exhausting resources (e.g., multiple threads can
// potentially enter similar runtime call sequences over a
// long-running code and thus every thread will end up with its own
// stream.
// Stream creation can be expensive.  We "recycle" them when possible. 
typedef std::deque<hipStream_t> KitHipStreamList;
static KitHipStreamList _kithip_streams;
static std::mutex _kithip_stream_mutex;

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

void *__kithip_get_thread_stream() {
  #ifdef KITHIP_USE_DEFAULT_STREAM
  return (void*)0;
  #else
  hipStream_t hip_stream;
  _kithip_stream_mutex.lock();
  if (not _kithip_streams.empty()) {
    hip_stream = _kithip_streams.front();
    _kithip_streams.pop_front();
    if (__kitrt_verbose_mode())
      fprintf(stderr, "kithip: use recycled stream [stream=%p, poolsize=%zu].\n",
            (void*)hip_stream, _kithip_streams.size());
  } else {
    HIP_SAFE_CALL(hipSetDevice_p(__kithip_get_device_id()));          
    HIP_SAFE_CALL(hipStreamCreate(&hip_stream));
  }
  _kithip_stream_mutex.unlock();
  return (void*)hip_stream;
  #endif
}

void __kithip_sync_thread_stream(void *opaque_stream) {
  HIP_SAFE_CALL(hipSetDevice_p(__kithip_get_device_id()));
  if (opaque_stream) {
    hipStream_t hip_stream = (hipStream_t)opaque_stream;
    HIP_SAFE_CALL(hipStreamSynchronize_p(hip_stream));
    _kithip_stream_mutex.lock();
    _kithip_streams.push_back(hip_stream);
    _kithip_stream_mutex.unlock();
  } else // null streams sync to full device
    HIP_SAFE_CALL(hipDeviceSynchronize_p()); 
}

void __kithip_sync_context() {
  HIP_SAFE_CALL(hipSetDevice_p(__kithip_get_device_id()));            
  HIP_SAFE_CALL(hipDeviceSynchronize_p());
}

void __kithip_delete_thread_stream(void *opaque_stream) {
   assert(opaque_stream != nullptr && "unexpected null stream pointer!");
   hipStream_t hip_stream = (hipStream_t)opaque_stream;
   _kithip_stream_mutex.lock();
   auto sit = std::find(_kithip_streams.begin(), _kithip_streams.end(), hip_stream);
   if (sit != _kithip_streams.end()) {
     _kithip_streams.erase(sit);
   }
   HIP_SAFE_CALL(hipStreamDestroy_p(hip_stream));
   _kithip_stream_mutex.unlock();
}

void __kithip_destroy_thread_streams() {
  _kithip_stream_mutex.lock();
  for(auto &entry : _kithip_streams)
    HIP_SAFE_CALL(hipStreamDestroy_p(entry));
  _kithip_streams.clear();
  _kithip_stream_mutex.unlock();
}
  
} // extern "C"
