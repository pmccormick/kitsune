//===- streams.cpp - Kitsune runtime CUDA streams support    --------------===//
//
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

#include "kitcuda.h"
#include "kitcuda_dylib.h"
#include <mutex>
#include <stdio.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <unordered_map>

// On older systems gettid() is not exposed and the syscall()
// interface must be used.  It won't hurt us to just use that
// everywhere as there are still systems in use with older
// system libraries that are missing the call...
#define gettid() syscall(SYS_gettid)

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
typedef std::unordered_map<unsigned int, CUstream> KitCudaStreamMap;
static KitCudaStreamMap _kitcuda_stream_map;

static std::mutex _kitcuda_stream_mutex;

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

CUstream __kitcuda_get_thread_stream() {
  KIT_NVTX_PUSH("kitcuda:get_thread_stream", KIT_NVTX_STREAM);

  pid_t tid = gettid();

  _kitcuda_stream_mutex.lock();
  KitCudaStreamMap::iterator sit = _kitcuda_stream_map.find(tid);
  CUstream stream;
  if (sit == _kitcuda_stream_map.end()) {
    CU_SAFE_CALL(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    _kitcuda_stream_map[tid] = stream;
  } else
    stream = sit->second;
  _kitcuda_stream_mutex.unlock();
  KIT_NVTX_POP();
  return stream;
}

void __kitcuda_sync_thread_stream() {
  // We could call get_thread_stream here but we don't want
  // to create a new stream simply to sync on it...
  KIT_NVTX_PUSH("kitcuda:sync_thread_stream", KIT_NVTX_STREAM);
  pid_t tid = gettid();
  _kitcuda_stream_mutex.lock();
  KitCudaStreamMap::iterator sit = _kitcuda_stream_map.find(tid);
  _kitcuda_stream_mutex.unlock();
  if (sit != _kitcuda_stream_map.end()) {
    CU_SAFE_CALL(cuStreamSynchronize_p(sit->second));
  } else {
    fprintf(stderr,
            "kitcuda: warning -- unexpected failure finding thread stream!\n");
  }
  KIT_NVTX_POP();
}

void __kitcuda_sync_context() {
  KIT_NVTX_PUSH("kitcuda:sync_context", KIT_NVTX_STREAM);
  CUcontext ctx;
  // TODO: We have multiple calls to set the context for the calling
  // thread -- should probably wrap it in a function.
  CU_SAFE_CALL(cuCtxGetCurrent_p(&ctx));
  if (ctx == NULL)
    CU_SAFE_CALL(cuCtxSetCurrent_p(__kitcuda_get_context()));
  CU_SAFE_CALL(cuCtxSynchronize_p());
  KIT_NVTX_POP();
}

void __kitcuda_delete_thread_stream() {
  KIT_NVTX_PUSH("kitrt:delete_thread_stream", KIT_NVTX_STREAM);
  pid_t tid = gettid();
  _kitcuda_stream_mutex.lock();
  KitCudaStreamMap::iterator sit = _kitcuda_stream_map.find(tid);
  if (sit != _kitcuda_stream_map.end()) {
    CU_SAFE_CALL(cuStreamDestroy_v2_p(sit->second));
    _kitcuda_stream_map.erase(sit);
  }
  _kitcuda_stream_mutex.unlock();
  KIT_NVTX_POP();
}

void __kitcuda_destroy_thread_streams() {
  KIT_NVTX_PUSH("kitrt:delete_thread_streams", KIT_NVTX_STREAM);
  _kitcuda_stream_mutex.lock();
  // Walk all streams, sync, and destroy them.
  for (auto &entry : _kitcuda_stream_map)
    CU_SAFE_CALL(cuStreamDestroy_v2_p(entry.second));
  _kitcuda_stream_map.clear();
  _kitcuda_stream_mutex.unlock();
  KIT_NVTX_POP();
}

} // extern "C"
