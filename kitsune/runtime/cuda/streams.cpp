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
#include <deque>
#include <algorithm>


// On older systems gettid() is not exposed and the syscall()
// interface must be used.  It won't hurt us to just use that
// everywhere as there are still systems in use with older
// system libraries that are missing the call...
#define gettid() syscall(SYS_gettid)

// Stream creation can be expensive.  We "recycle" them when possible. 
typedef std::deque<CUstream> KitCudaStreamList;
static KitCudaStreamList _kitcuda_streams;
static std::mutex _kitcuda_stream_mutex;

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

void *__kitcuda_get_thread_stream() {
  KIT_NVTX_PUSH("kitcuda:get_thread_stream", KIT_NVTX_STREAM);

  CUstream cu_stream;
  if (not _kitcuda_streams.empty()) {
    _kitcuda_stream_mutex.lock();
    cu_stream = _kitcuda_streams.front();
    _kitcuda_streams.pop_front();
    _kitcuda_stream_mutex.unlock();
    if (__kitrt_verbose_mode())
       fprintf(stderr, "reusing thread stream.\n");
  } else {
    if (__kitrt_verbose_mode())
       fprintf(stderr, "creating new thread stream.\n");
    CU_SAFE_CALL(cuStreamCreate(&cu_stream, CU_STREAM_NON_BLOCKING));
  }
  KIT_NVTX_POP();
  if (__kitrt_verbose_mode())
    fprintf(stderr, "returning thread stream: %p\n", cu_stream);
  return (void *)cu_stream;
}

void __kitcuda_sync_thread_stream(void *opaque_stream) {
  assert(opaque_stream != nullptr && "unexpected null pointer!");
  KIT_NVTX_PUSH("kitcuda:sync_thread_stream", KIT_NVTX_STREAM);
  CUstream stream = (CUstream)opaque_stream;
  CU_SAFE_CALL(cuStreamSynchronize_p(stream));
  // In our current use case a synchronized stream is done doing
  // any useful work.  Recycle it for later use... 
  _kitcuda_stream_mutex.lock();
  _kitcuda_streams.push_back(stream);
  _kitcuda_stream_mutex.unlock();
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

void __kitcuda_delete_thread_stream(void *opaque_stream) {
  KIT_NVTX_PUSH("kitrt:delete_thread_stream", KIT_NVTX_STREAM);
  CUstream stream = (CUstream)opaque_stream;
  // Do a quick check to make sure we don't need to clean up the deque.
  // We are a bit lazy with the scope of the lock here but we don't expect
  // this to happen often (if ever).  Streams will be aggressively reused 
  // vs. explicitly destroyed in the current implementation... 
  _kitcuda_stream_mutex.lock();
  auto sit = std::find(_kitcuda_streams.begin(), _kitcuda_streams.end(), stream);
  if (sit != _kitcuda_streams.end()) {
    _kitcuda_streams.erase(sit);
  }
  CU_SAFE_CALL(cuStreamDestroy_v2_p(stream));
  _kitcuda_stream_mutex.unlock();
  KIT_NVTX_POP();
}

void __kitcuda_destroy_thread_streams() {
  KIT_NVTX_PUSH("kitrt:delete_thread_streams", KIT_NVTX_STREAM);
  _kitcuda_stream_mutex.lock();
 
  for (auto &entry : _kitcuda_streams)
    CU_SAFE_CALL(cuStreamDestroy_v2_p(entry));
  _kitcuda_streams.clear();
  _kitcuda_stream_mutex.unlock();
  KIT_NVTX_POP();
}

} // extern "C"
