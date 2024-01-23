//===- memory_map.cpp - Kitsune runtime high-level memory map ---------------===//
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
#include "kitrt.h"
#include "memory_map.h"

typedef std::unordered_map<void *, KitRTAllocMapEntry> KitRTAllocMap;
static KitRTAllocMap _kitrt_alloc_map;

void __kitrt_register_mem_alloc(void *addr, size_t size) {
  assert(addr != nullptr && "unexpected null pointer!");
  KitRTAllocMapEntry entry;
  entry.size = size;
  entry.prefetched = false;
  entry.read_only = false;
  entry.write_only = false;
  _kitrt_alloc_map[addr] = entry;
  if (__kitrt_verbose_mode())
    fprintf(stderr, "kitrt: registered memory allocation (%p) "
	    "of %ld bytes.\n", addr, size);
}

void __kitrt_set_mem_prefetch(void *addr, bool prefetched) {
  assert(addr != nullptr && "unexpected null pointer!");
  KitRTAllocMap::iterator ait = _kitrt_alloc_map.find(addr);
  if (ait != _kitrt_alloc_map.end()) {
    ait->second.prefetched = prefetched;
    if (__kitrt_verbose_mode())
      fprintf(stderr, "kitrt: marked memory at %p, size %ld, as '%s'.\n",
	      addr, ait->second.size,
	      prefetched ? "prefetched" : "not prefetched");
  }
  // We could consider a diagnostic here reporting use of an unregistered
  // pointer.  However, this is tricky with the compiler generating calls
  // as it currently has no way to distinguish managed pointer types.  At
  // present we quietly ignore bogus requests...
  //
  // TODO: Should we introduce something more specific at the langauge
  // level to denote managed memory?
}

void __kitrt_mark_mem_read_only(void *addr) {
  assert(addr != nullptr && "unexpected null pointer!");
  KitRTAllocMap::iterator ait = _kitrt_alloc_map.find(addr);
  if (ait != _kitrt_alloc_map.end()) {
    ait->second.read_only = true;
  }
}

bool __kitrt_is_mem_read_only(void *addr) {
  assert(addr != nullptr && "unexpected null pointer!");
  KitRTAllocMap::iterator ait = _kitrt_alloc_map.find(addr);
  if (ait != _kitrt_alloc_map.end())
    return ait->second.read_only;
  else 
    return false;
}

/// @brief Flag the given memory allocation as write only.
/// @param addr: the pointer to the managed memory allocation. 
extern void __kitrt_mark_mem_write_only(void *addr) {
  assert(addr != nullptr && "unexpected null pointer!");
  KitRTAllocMap::iterator ait = _kitrt_alloc_map.find(addr);
  if (ait != _kitrt_alloc_map.end()) {
    ait->second.write_only = true;
  }
}


bool __kitrt_is_mem_write_only(void *addr) {
  assert(addr != nullptr && "unexpected null pointer!");
  KitRTAllocMap::iterator ait = _kitrt_alloc_map.find(addr);
  if (ait != _kitrt_alloc_map.end())
    return ait->second.write_only;
  else 
    return false;
}

void __kitrt_clear_mem_advice(void *addr) {
  assert(addr != nullptr && "unexpected null pointer!");
  KitRTAllocMap::iterator ait = _kitrt_alloc_map.find(addr);
  if (ait != _kitrt_alloc_map.end()) {
    ait->second.read_only = false;
    ait->second.write_only = false;
  }
}

bool __kitrt_is_mem_prefetched(void *addr, size_t *size) {
  assert(addr != nullptr && "unexpected null pointer!");
  KitRTAllocMap::const_iterator cit = _kitrt_alloc_map.find(addr);
  if (cit != _kitrt_alloc_map.end()) {
    if (size != nullptr)
      *size = cit->second.size;
    return cit->second.prefetched;
  } else {
    // NOTE: This is a bit strange but we have to deal with the
    // compiler's code generation mechanisms here.  Specifically it is
    // only able to identify pointers but not pointers allocated in
    // managed memory.  For this reason we may generate requests for
    // un-managed pointers.  To "behave" we currently treat such
    // requests as prefetched (and thus avoid additional calls that
    // might presume a valid managed memory region is associated with
    // the pointer).
    return true;
  }
}

size_t __kitrt_get_mem_alloc_size(void *addr,
				  bool *read_only,
				  bool *write_only) {
  assert(addr != nullptr && "unexpected null addr pointer!");
  assert(read_only != nullptr && "unexpected null read_only pointer!");
  assert(write_only != nullptr && "unexpected null write_only pointer!");
  size_t size = 0;
  KitRTAllocMap::const_iterator cit = _kitrt_alloc_map.find(addr);
  if (cit != _kitrt_alloc_map.end()) {
    *read_only = cit->second.read_only;
    *write_only = cit->second.write_only;
    size = cit->second.size;
  } else {
    // NOTE: This is a bit strange but we have to deal with the
    // compiler's code generation mechanisms here.  Specifically it is
    // only able to identify pointers but not pointers allocated in
    // managed memory.  For this reason we may generate requests for
    // un-managed pointers.  To "behave" we currently treat such
    // requests as prefetched (and thus avoid additional calls that
    // might presume a valid managed memory region is associated with
    // the pointer).
    *read_only = false;
    *write_only = false;
  }
  
  return size;
}

void __kitrt_unregister_mem_alloc(void *addr) {
  assert(addr != nullptr && "unexpected null pointer!");
  KitRTAllocMap::iterator ait = _kitrt_alloc_map.find(addr);
  if (ait != _kitrt_alloc_map.end())
    _kitrt_alloc_map.erase(ait);

  // NOTE: We currently silently ignore requests to unregister
  // an pointer that was not found in the map.  This mostly has
  // to do with the nuanaces of the compiler's code generation
  // and its inability to distinguish between various pointer
  // types.
}

void __kitrt_mem_needs_prefetch(void *addr) {
  assert(addr != nullptr && "unexpected null pointer!");
  KitRTAllocMap::iterator it = _kitrt_alloc_map.find(addr);
  if (it != _kitrt_alloc_map.end()) {
    it->second.prefetched = false;
  }
}

extern "C" void __kitrt_print_memory_map() {
  fprintf(stdout, "kitsune runtime memory allocation map:\n");
  if (_kitrt_alloc_map.empty()) 
    fprintf(stdout, "\t[... empty ...]\n");
  else {
    const size_t MBYTE = 1024 * 1024;
    size_t total_allocated = 0;
    unsigned int num_allocations = 0;
    for(auto &entry : _kitrt_alloc_map) {
      void *addr = entry.first;
      const KitRTAllocMapEntry *alloc_entry = &entry.second;
      total_allocated += alloc_entry->size;
      num_allocations++;
      fprintf(stderr, "\tAddress: %p --> [size: %6.2f Mbytes, prefetched: %8s, "
              "read-only: %8s, write-only: %8s]\n",
              addr,
	      alloc_entry->size / (double)MBYTE,
	      alloc_entry->prefetched ? "true" : "false", 
              alloc_entry->read_only ? "true" : "false", 
              alloc_entry->write_only ? "true": "false");
    }
    fprintf(stderr, "\n");
    fprintf(stdout, "\ttotal memory allocation: %6.2f Mbytes\n",
	    total_allocated / (double)MBYTE);
    fprintf(stderr, "\taverage size per allocation: %6.2f Mbytes\n", 
            (total_allocated / (double)MBYTE)/ num_allocations);
  }
}

extern "C" void __kitrt_destroy_memory_map(void (*free_mem_call)(void *)) {
  assert(free_mem_call != nullptr && "unexpected null function pointer!");
  for (auto &entry : _kitrt_alloc_map)
    free_mem_call(entry.first);
  _kitrt_alloc_map.clear();
}

