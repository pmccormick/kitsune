//
//===- kitrt.h - Kitsune ABI runtime debug support     ------------------===//
//
// TODO: Need to update LANL/Triad Copyright notice.
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

#ifndef __KITRT_H__
#define __KITRT_H__

#include <cstdio>
#include <cassert>
#include <stdint.h>
#include <stdlib.h>
#include <cstring>
#include <execinfo.h>
#include <type_traits>
#include <ctype.h>

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

 /**
   * Initialize the core kitsune runtime components that are shared
   * across all the target runtimes.  Note that this call is typically
   * invoked by each specific runtime target (e.g., CUDA) vs. having
   * this call initialization each target runtime.  It can be called
   * multiple times as it is guarded to avoid repeated initialization.
   */
  extern void __kitrt_initialize();
  
  /**
   * Set the runtime system to operate in verbose mode.
   */
  inline void __kitrt_enable_verbose_mode() {
    extern bool _kitrt_verbose_mode;
    _kitrt_verbose_mode = true;
  }

  /**
   * Disable the runtime system's verbose reporting mode.
   */
  inline void __kitrt_disable_verbose_mode() {
    extern bool _kitrt_verbose_mode;
    _kitrt_verbose_mode = false;
  }

  /**
   * Enable/Disable the runtime's verbose mode.  Note that this can
   * also be enabled at runtime by setting the KITRT_VERBOSE
   * environment variable.
   *
   * @param enable - if `true` enable verbose mode, disable if `false`.
   */
  inline void __kitrt_set_verbose_mode(bool enable) {
    extern bool _kitrt_verbose_mode;
    _kitrt_verbose_mode = enable;
  }

  /**
   * Return the runtime's verbose operating mode.  If `true` the
   * runtime should provide status details on stderr during execution,
   * otherwise it is quiet.
   */
  inline bool __kitrt_verbose_mode() {
    extern bool _kitrt_verbose_mode;
    return _kitrt_verbose_mode;
  }

  /**
   * Provide a backtrace to stderr to help track down runtime crashes.
   */
  extern void __kitrt_print_stack_trace();

  extern unsigned __kitrt_getNumPrefetchStreams();
  extern bool __kitrt_prefetchEnabled();
  extern void __kitrt_enablePrefetching();
  extern bool __kitrt_prefetchStreamsEnabled();
  extern void __kitrt_enablePrefetchStreams();

#ifdef __cplusplus
} // extern "C"
#endif
  
/**
 * Return the value of the given environment variable. If the
 * variable does not exist in the environment return `false`.
 * Otherwise, `true` is returned and the value is returned in
 * the caller provided parameter.
 */
template <typename ValueType>
bool __kitrt_get_env_value(const char *var_name,
			   ValueType &value) {
  assert(var_name && "unexpected null variable name!");
  bool found = false;
  char *value_string;
  if ((value_string = getenv(var_name))) {

    if constexpr (std::is_same_v<ValueType, int>) {
      value = atoi(value_string);
      found = true;
    } else if constexpr (std::is_same_v<ValueType, unsigned>) {
      value = atoi(value_string);
      found = true;
    } else if constexpr (std::is_same_v<ValueType, bool>) {
      found = true;
      for(int i = 0; value_string[i]; i++) 
        value_string[i] = tolower(value_string[i]);
      if (!strcmp(value_string, "true") || !strcmp(value_string, "1"))
        value = true;
      else if (!strcmp(value_string, "false") || !strcmp(value_string, "0"))
        value = false;
      else {
        fprintf(stderr, "kitsune_rt: warning, boolean environment variable "
                        "'%s' not set to true or false.\nTreating presence "
                        "as an implied true setting.\n", 
                        var_name);
        value = true;
      } 
    } else if constexpr (std::is_same_v<ValueType, long>) {
      value = atol(value_string);
      found = true;
    } else if constexpr (std::is_same_v<ValueType, unsigned long>) {
      value = atol(value_string);
      found = true;	  	
    } else if constexpr (std::is_same_v<ValueType, float>) {
      value = (float)atof(value_string);
      found = true;
    } else if constexpr (std::is_same_v<ValueType, double>) {
      value = atof(value_string);
      found = true;
    } else {
      fprintf(stderr, "kitrt: warning unhandled type encountered.\n");
      fprintf(stderr, "       location %s:%d\n", __FILE__, __LINE__);
      __kitrt_print_stack_trace();
    }
  }
    
  return found;
}

#endif // __KITRT_H__

