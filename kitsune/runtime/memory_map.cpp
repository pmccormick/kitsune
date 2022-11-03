//===- memory_map.cpp - Kitsune runtime  CUDA support    -----------------===//
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

#include <cstdio>
#include <cassert>
#include <unordered_map>
#include <map>
#include "memory_map.h"

//#define _KITRT_VERBOSE_


typedef std::unordered_map<void *, KitRTAllocMapEntry> KitRTAllocMap;
static KitRTAllocMap _kitrtAllocMap;

void __kitrt_registerMemAlloc(void *addr, size_t size, bool prefetched) {
  assert(addr != nullptr && "unexpected null pointer!");
  KitRTAllocMapEntry entry;
  entry.size = size;
  entry.prefetched = false;
  _kitrtAllocMap[addr] = entry;
  #ifdef _KITRT_VERBOSE_
  fprintf(stderr, "kitrt: registered memory allocation (%p) "
		  "of %ld bytes.\n", addr, size);
  #endif
}

void __kitrt_setMemPrefetch(void *addr, bool prefetched) {
  assert(addr != nullptr && "unexpected null pointer!");
  KitRTAllocMap::iterator ait = _kitrtAllocMap.find(addr);
  if (ait != _kitrtAllocMap.end()) {
    ait->second.prefetched = prefetched;
    #ifdef _KITRT_VERBOSE
    fprintf(stderr, "kitrt: __kitrt_setMemPrefetch() -- "
            "marked memory at %p, size %ld, as '%s'.\n", addr, ait->second.size,
            prefetched ? "prefetched" : "not prefetched");
    #endif
  } else {
    #ifdef _KITRT_VERBOSE_
    fprintf(stderr,
            "kitrt: __kitrt_setMemPrefetch() -- "
            "warning, address %p not found in memory map.\n",
            addr);
    #endif
  }
}

bool __kitrt_isMemPrefetched(void *addr) {
  assert(addr != nullptr && "unexpected null pointer!");
  KitRTAllocMap::const_iterator cit = _kitrtAllocMap.find(addr);
  if (cit != _kitrtAllocMap.end()) {
    return cit->second.prefetched;
  } else {
    #ifdef _KITRT_VERBOSE_
    fprintf(stderr,
            "kitrt: __kitrt_isMemPrefetched() -- "
            "warning, address %p not found in address map. "
            "returning false state!\n",
            addr);
    #endif
    return true;
  }
}

size_t __kitrt_getMemAllocSize(void *addr) {
  assert(addr != nullptr && "unexpected null pointer!");
  KitRTAllocMap::const_iterator cit = _kitrtAllocMap.find(addr);
  if (cit != _kitrtAllocMap.end()) {
    return cit->second.size;
  } else {
    #ifdef _KITRT_VERBOSE_
    fprintf(stderr,
            "kitrt: __kitrt_getMemAllocSize() -- "
            "warning, address %p not found in address map. "
            "returning zero size.\n",
            addr);
    #endif
    return 0;
  }
}

void __kitrt_unregisterMemAlloc(void *addr) {
  assert(addr != nullptr && "unexpected null pointer!");
  KitRTAllocMap::iterator ait = _kitrtAllocMap.find(addr);
  if (ait != _kitrtAllocMap.end()) {
    _kitrtAllocMap.erase(ait);
  } else {
    #ifdef _KITRT_VERBOSE_
    fprintf(stderr,
            "kitrt: __kitrt_unregisterMemAlloc() -- "
            "warning, address %p not found in address map.\n",
            addr);
    #endif
  }
}

void __kitrt_memNeedsPrefetch(void *addr) {
  assert(addr != nullptr && "unexpected null pointer!");
  KitRTAllocMap::iterator it = _kitrtAllocMap.find(addr);
  if (it != _kitrtAllocMap.end()) {
    #ifdef _KITRT_VERBOSE_
    fprintf(stderr,
            "kitrt: allocation (%p) needs prefetching (updated on host).\n",
            addr);
    #endif
    it->second.prefetched = false;
  }
}

extern "C" void __kitrt_destroyMemoryMap(void (*freeMem)(void *)) {
  for (auto &mapEntry : _kitrtAllocMap)
    freeMem(mapEntry.first);
}

