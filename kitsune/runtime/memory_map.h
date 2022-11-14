//===- memory_map.h - Kitsune runtime support    -------------------------===//
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

#ifndef __KITRT_MEMORY_MAP_H__
#define __KITRT_MEMORY_MAP_H__

#include <stddef.h>

/// Both the CUDA and HIP versions of the runtime track managed memory
/// allocations.  This is done by providing a map from the allocated
/// pointer to a set of details about the allocation.  The current
/// information tracked per entry include:
///
///    * The size in bytes of the allocation.
///    * If the allocated region has been 'prefetched'.
///
/// Prefetching suggests a hint to the runtime/driver/OS to migrate
/// all pages to the corresponding device's memory.
///
/// TODO: Prefetch is probably better tracked as a location vs. a
/// one-sided boolean relationship...  Better tracking will require
/// this change.
struct KitRTAllocMapEntry {
  size_t     size;       // size of the allocated buffer in bytes.
  bool       prefetched; // has the data been prefetched?
};

/// Register a memory allocation with the runtime.  The allocation
/// is assumed be successful at this point and pointed to by the
/// supplied pointer (addr) and be 'numBytes' in size.
extern void __kitrt_registerMemAlloc(void *addr,
                                     size_t numBytes,
                                     bool prefetched = false);

/// Set the prefetch status of the given memory allocation entry.
extern void __kitrt_setMemPrefetch(void *addr, bool prefetched);

/// Mark the given memory allocation entry as prefetched.
inline void __kitrt_markMemPrefetched(void *addr) {
  __kitrt_setMemPrefetch(addr, true);
}

/// Mark the given memory allocation entry as not prefetched (i.e.,
/// entirely resident on CPU-/host-side memory).
inline void __kitrt_markMemNeedsPrefetch(void *addr) {
  __kitrt_setMemPrefetch(addr, false);
}

/// Return the prefetch status of the given allocation entry.
/// (true == prefetched, false == not prefetched)
bool __kitrt_isMemPrefetched(void *addr);

/// Return the size of the given memory allocation.
size_t __kitrt_getMemAllocSize(void *addr);

/// Unregister a memory allocation.  If the supplied pointer is not
/// found in the allocation map the runtime will throw an assertion
/// and terminate.  This call does not free the memory allocation;
/// that management is assumed to be managed elsewhere.
extern void __kitrt_unregisterMemAlloc(void *addr);

/// @brief  Mark the given managed memory allocation to need prefetching.
/// @param addr: The pointer to the managed memory allocation.
extern void __kitrt_memNeedsPrefetch(void *addr);

/// Destroy the memory map and call the function pointed to by
/// 'freeFP' to free the actual memory allocation (runtime target
/// dependent).  Note we keep this as a C function to simplify
/// things when dealing with existing APIs (e.g., CUDA).
extern "C" void __kitrt_destroyMemoryMap(void (*freeFP)(void *));

#endif
