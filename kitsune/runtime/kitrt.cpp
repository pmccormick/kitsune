//
//===- kitrt.h - Kitsune ABI runtime debug support    -----------------------===//
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

#include "kitrt.h"
#include "debug.h"
#include <stdlib.h>


// Overall runtime configuration settings.
static bool _kitrtVerboseMode          = false;
static bool _kitrtReportKernelRuntimes = false;
static bool _kitrtEnableStackTraces    = false;

// Shared GPU parameter settings/interface.
static unsigned _kitrtDefaultThreadsPerBlock = 256;
static bool _kitrtUseCustomLaunchParameters = false;
static unsigned _kitrtThreadsPerBlock = 0;
static unsigned _kitrtBlocksPerGrid = 0;


#ifdef __cplusplus
extern "C" {
#endif

  void __kitrt_setVerboseMode(bool Enable) {
    KITRT_DEBUG( kitrt::kitdbgs() << "kitrt: verbose mode "
                                  << Enable : "on.\n" ? "off.\n");
    _kitrtVerboseMode = Enable;
  }

  bool __kitrt_verboseMode() {
    return _kitrtVerboseMode;
  }

  void __kitrt_setReportRuntimes(bool Enable) {
    KITRT_DEBUG( kitrt::kitdbgs() << "kitrt: execution time reporting "
                                  << Enable : "on.\n" ? "off.\n");
    _kitrtReportKernelRuntimes = Enable;
  }

  bool __kitrt_reportRuntimes() {
    return _kitrtReportKernelRuntimes;
  }

  void __kitrt_setReportStackTraces(bool Enable) {
    KITRT_DEBUG( kitrt::kitdbgs() << "kitrt: stack trace reporting "
                                  << Enable : "on.\n" ? "off.\n");
    _kitrtEnableStackTraces = Enable;
  }

  bool __kitrt_ReportStackTraces() {
    return _kitrtEnableStackTraces;
  }

  bool __kitrt_init() {
    // Check for environment variables that impact the runtime
    // behavior...
    char *envValue;
    if ((envValue = getenv("KITRT_VERBOSE")))
      __kitrt_enableVerboseMode();
    else
      __kitrt_disableVerboseMode();

    if ((envValue = getenv("KITRT_TIMING_REPORTS")))
      __kitrt_enableRuntimeReports();
    else
      __kitrt_disableRuntimeReports();

    if ((envValue = getenv("KITRT_STACK_TRACES")))
      __kitrt_enableStackTraces();
    else
      __kitrt_disableStackTraces();

    // Initialize the supported runtime layer(s).
    bool __kitrt_runtimesInit();
    return __kitrt_runtimesInit();
  }

  void __kitrt_getLaunchParameters(size_t numElements,
                                   int &threadsPerBlock,
                                   int &blocksPerGrid) {
    if (not _kitrtUseCustomLaunchParameters) {
      threadsPerBlock = _kitrtDefaultThreadsPerBlock;
      blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    } else {
      threadsPerBlock = _kitrtThreadsPerBlock;
      blocksPerGrid = _kitrtBlocksPerGrid;
    }
  }

  void __kitrt_setDefaultGPUThreadsPerBlock(unsigned threadsPerBlock) {
    _kitrtDefaultThreadsPerBlock = threadsPerBlock;
  }

  void __kitrt_overrideLaunchParameters(unsigned threadsPerBlock,
                                        unsigned blocksPerGrid) {
    _kitrtUseCustomLaunchParameters = true;
    _kitrtThreadsPerBlock = threadsPerBlock;
    _kitrtBlocksPerGrid = blocksPerGrid;
  }

  void __kitrt_resetLaunchParameters() {
    _kitrtUseCustomLaunchParameters = false;
  }

#ifdef __cplusplus
} // extern "C"
#endif


