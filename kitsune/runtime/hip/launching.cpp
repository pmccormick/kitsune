/*
 *===- launching.cpp - HIP kernel launching support   ---------------------===
 *
 * Copyright (c) 2021, 2023 Los Alamos National Security, LLC.
 * All rights reserved.
 *
 * Copyright 2021, 2023. Los Alamos National Security, LLC. This
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
#include "kithip.h"
#include "kithip_dylib.h"
#include <mutex>
#include <unordered_map>

// TODO: There is some inclination to want to share some code here
// with the CUDA runtime implementation.  However, the benefits are
// really not significant and given some difference in behaviors
// and maturity between CUDA and HIP we've currently decided to
// keep them separated. Right now the most apparent difference is
// primarily the types used in map entries (see below) -- until we
// get HIP a bit more stable we've punted thinking any harder about
// a more common code base.

// *** EXPERIMENTAL: The runtime maintains a map from fatbinary images
// to a supporting HIP module.  The primary reason for this is
// exploring reducing runtime overheads.
//
// TODO: Finish exploration of map vs. HIP call overheads.
typedef std::unordered_map<const void *, hipModule_t> KitHipModuleMap;
static KitHipModuleMap _kithip_module_map;
static std::mutex _kithip_module_map_mutex;

// *** EXPERIMENTAL: The runtime maintains a map from kernel names to
// kernel functions.  This avoids searching a module repeatedly at
// kernel launch time and is primarily focused on reducing runtime
// overheads.
//
// TODO: Finish exploration of map vs. CUDA call overheads.
typedef std::unordered_map<const char *, hipFunction_t> KitHipKernelMap;
static KitHipKernelMap _kithip_kernel_map;

extern "C" {

// *** EXPERIMENTAL: The details of picking launch parameters can be a
// challenge and occupancy is often one of the driving factors.  Occupancy
// is defined, in "CUDA-ese", as the ratio of the number of active warps
// per multiprocessor to the maximum number of active warps. Importantly,
// having a higher occupancy does not guarantee better performance. It is
// simply a reasonable metric for the latency hiding ability of a particular
// kernel.
//
// TODO: Would it behoove us to keep a record of launch parameters
// for each kernel based on `trip_count`?  This might be the case
// if the details of computing the parameters grows costly -- it is
// unlikely to hurt us on the HIP side at present.

// Without any tweaks from the environment or other runtime calls,
// this is the default number of threads we'll launch per block (in
// CUDA speak).
static bool _kithip_use_occupancy_calc = true;
static int _kithip_default_max_threads_per_blk = 1024;
static int _kithip_default_threads_per_blk = _kithip_default_max_threads_per_blk;

void __kithip_use_occupancy_launch(bool enable) {
  _kithip_use_occupancy_calc = enable;
}

void __kithip_set_default_max_threads_per_blk(int num_threads) {
  _kithip_default_max_threads_per_blk = num_threads;
}

void __kithip_set_default_threads_per_blk(int threads_per_blk) {
  if (threads_per_blk > _kithip_default_max_threads_per_blk) 
    threads_per_blk = _kithip_default_max_threads_per_blk;
  _kithip_default_threads_per_blk = threads_per_blk;
}

void __kithip_get_launch_params(size_t trip_count, hipFunction_t kfunc,
                                int &threads_per_blk, int &blks_per_grid) {

  if (_kithip_use_occupancy_calc) {
    // Frustratingly there are a bunch of inlined type templated calls lurking
    // behind HIP's occupancy calls.  This makes it difficult here with the
    // dynamic loading and other details where we are a bit more accustomed to a
    // C-style approach in the runtime...  It turns out all calls eventually
    // make it to the "ModuleOccupancy" call used below and we can find a valid
    // dylib entry point for it.
    int min_grid_size; // currently ignored...
    HIP_SAFE_CALL(hipModuleOccupancyMaxPotentialBlockSize_p(
        &min_grid_size, &threads_per_blk, kfunc, 0, 0));
  } else {
    threads_per_blk = _kithip_default_threads_per_blk;
  }

  // Need to round-up based on array size/trip count.
  blks_per_grid = (trip_count + threads_per_blk - 1) / threads_per_blk;
}

void __kithip_launch_kernel(const void *fat_bin, const char *kernel_name,
                            void **kern_args, uint64_t trip_count) {

  assert(fat_bin && "kithip: launch with null fat binary!");
  assert(kernel_name && "kithip: launch with null name!");
  assert(kern_args && "kithip: launch with null args!");
  assert(trip_count != 0 && "kithip: launch with zero trips!");

  HIP_SAFE_CALL(hipSetDevice_p(__kithip_get_device_id()));

  // Multiple threads can launch kernels in our current design.  If a
  // thread enters without having previously set the device the runtime
  // becomes unhappy with us.  Make sure we're following the rules.
  hipFunction_t kern_func;
  _kithip_module_map_mutex.lock();
  KitHipKernelMap::iterator kernit = _kithip_kernel_map.find(kernel_name);
  if (kernit == _kithip_kernel_map.end()) {
    // We have not yet encountered this kernel function...  Check to see
    // if we already have a supporting module for the fat binary.
    hipModule_t hip_module;
    KitHipModuleMap::iterator modit = _kithip_module_map.find(fat_bin);
    if (modit == _kithip_module_map.end()) {
      // Create a supporting module and "register" the fat binary
      // image in the map...
      HIP_SAFE_CALL(hipModuleLoadData_p(&hip_module, fat_bin));
      _kithip_module_map[fat_bin] = hip_module;
    } else
      hip_module = modit->second;

    // Look up the kernel function in the module.
    HIP_SAFE_CALL(hipModuleGetFunction_p(&kern_func, hip_module, kernel_name));
    _kithip_kernel_map[kernel_name] = kern_func;
  } else
    kern_func = kernit->second;

  _kithip_module_map_mutex.unlock();

  int threads_per_blk, blks_per_grid;
  __kithip_get_launch_params(trip_count, kern_func, threads_per_blk,
                             blks_per_grid);

  if (__kitrt_verbose_mode()) {
    fprintf(stderr, "kithip: kernel '%s' launch parameters:\n", kernel_name);
    fprintf(stderr, "  blocks: %d, 1, 1\n", blks_per_grid);
    fprintf(stderr, "  threads: %d, 1, 1\n", threads_per_blk);
    fprintf(stderr, "  trip count: %ld\n", trip_count);
    fprintf(stderr, "  args address: %p\n", kern_args);
  }

  hipStream_t hip_stream = __kithip_get_thread_stream();
  HIP_SAFE_CALL(hipModuleLaunchKernel_p(kern_func, blks_per_grid, 1, 1,
                                        threads_per_blk, 1, 1,
                                        0, // shared mem size
                                        hip_stream, kern_args, NULL));
}

void *__kithip_get_global_symbol(void *fat_bin, const char *sym_name) {
  assert(fat_bin && "null fat binary!");
  assert(sym_name && "null symbol name!");

  hipModule_t hip_module;
  _kithip_module_map_mutex.lock();
  KitHipModuleMap::iterator modit = _kithip_module_map.find(fat_bin);
  if (modit == _kithip_module_map.end()) {
    HIP_SAFE_CALL(hipModuleLoadData_p(&hip_module, fat_bin));
    _kithip_module_map[fat_bin] = hip_module;
  } else
    hip_module = modit->second;

  // NOTE: The device pointer and size ('bytes') parameters for the
  // call to cuModuleGetGlobal are optional.  To simplify the compiler's
  // code generation details we ignore the size parameter...
  hipDeviceptr_t sym_ptr;
  size_t bytes;
  HIP_SAFE_CALL(hipModuleGetGlobal_p(&sym_ptr, &bytes, hip_module, sym_name));
  return sym_ptr;
}
}
