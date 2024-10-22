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
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <unordered_map> // IWYU pragma: keep (clang-tidy+tempaltes == bad???)

// TODO: The hip runtime shares common implementation details
// with cuda.  At present we have decided to keep them separated
// given the potential differences in the underlying runtimes
// and stability issues w/ hip; further down the road, it might be
// possible to share common code between the two.

//   TODO: The experimental features below have not yet been
//   validated to actually reduce overheads.

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
// simply a metric of the latency hiding ability of a particular kernel.
//
// TODO: Would it behoove us to keep a record of launch parameters
// for each kernel based on `trip_count`?

// Without any tweaks from the environment or other runtime calls,
// this is the default number of threads we'll launch per block (in
// CUDA speak).
static bool _kithip_use_occupancy_calc = true;
bool _kithip_refine_occupancy_calc = true;
static int _kithip_default_max_threads_per_blk = 1024;
static int _kithip_default_threads_per_blk = 256;

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

typedef std::unordered_map<std::string, int> KitHipLaunchParamMap;
static KitHipLaunchParamMap _kithip_launch_param_map;

namespace {

// we "borrow" this from cuda...
extern int next_lowest_factor(int n, int m);

/**
 * Get the launch parameters for a given kernel and trip count based
 * an occupancy-based heuristic.  The behavior of this call will depend
 * on various runtime configuration details.
 *
 * This call is used when the `use_occupancy_launch` flag is set.  The
 * behavior of the call can be further refined if `tune_occupancy` is
 * also set.  Details of how this tuning is accomplished is described
 * within the implementation (and is far from an exact science...).
 *
 * @param trip_count - how many elements to process
 * @param kfunc - the actual CUDA function / kernel.
 * @param threads_per_blk - computed threads per block for launch
 * @param blks_per_grid - computed blocks per grid for launch
 */
void __kithip_get_occ_launch_params(size_t trip_count, hipFunction_t kfunc,
                                    int &threads_per_blk, int &blks_per_grid,
                                    const KitRTInstMix *inst_mix) {
  assert(_kithip_use_occupancy_calc && "called when occupancy mode is false!");

  // As a default starting point, the hip occupancy heuristic to get
  // an initial occupancy-driven threads-per-block figure.
  int min_grid_size;
  HIP_SAFE_CALL(hipModuleOccupancyMaxPotentialBlockSize_p(
      &min_grid_size, &threads_per_blk, kfunc, 0, 0));
  int active_blks;
  HIP_SAFE_CALL(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(&active_blks,
								   kfunc,
								   threads_per_blk,
								   0));

  if (__kitrt_verbose_mode()) {
    fprintf(stderr, "kithip: occupancy launch figures -----------------------\n");
    fprintf(stderr, "   - minimum grid size:       %d\n", min_grid_size);
    fprintf(stderr, "   - recommended block size:  %d (threads)\n", threads_per_blk);
    fprintf(stderr, "   - active blocks per mp:    %d\n\n", active_blks);
  }

  if (_kithip_refine_occupancy_calc) {
    // Assume that the occupancy heuristic is flawed and look to refine
    // its threads-per-block result such that it utilizes the full GPU
    // (i.e., it is not uncommon for the heuristic to return values
    // that only use a limited number of available resources -- most
    // often under-utilizing the number of available multi-processors).
    extern int _kithip_device_id;

    int num_multiprocs = 0;
    HIP_SAFE_CALL(hipDeviceGetAttribute_p(&num_multiprocs,
                                          hipDeviceAttributeMultiprocessorCount,
                                          _kithip_device_id));
    int max_threads_per_blk = 0;
    HIP_SAFE_CALL(hipDeviceGetAttribute_p(&max_threads_per_blk,
                                          hipDeviceAttributeMaxThreadsPerBlock,
                                          _kithip_device_id));
    
    // Estimate how many multi-processors we are using with the provided
    // threads-per-block value..
    int block_count = (trip_count + threads_per_blk - 1) / threads_per_blk;
    float sm_load = ((float)block_count / num_multiprocs) * 100.0;

    if (__kitrt_verbose_mode()) {
      fprintf(stderr, "kithip: initial occupancy kernel workload --------------\n");
      fprintf(stderr, "  Number of multi-procs:  %d\n", num_multiprocs);
      fprintf(stderr, "  Max threads per block:  %d\n", max_threads_per_blk);
      fprintf(stderr, "  Trip count:             %ld\n", trip_count);
      fprintf(stderr, "  Occupancy-driven TPB:   %d\n", threads_per_blk);
      fprintf(stderr, "  Multi-proc utilization: %3.2f%%\n", sm_load);
    }


    if (threads_per_blk == max_threads_per_blk) {
      // Maxing out the threads per blk is a frequent occurrence when calling 
      // HIP's occupancy heuristic.  Let's shuffle things downward a bit to try
      // and avoid a bottleneck.
      // 
      // TODO: There is a ton of work to do here!  Need to inject some more sanity
      // into this process... 
      do { 
        threads_per_blk = threads_per_blk - (max_threads_per_blk / 32);
        block_count = (trip_count + threads_per_blk - 1) / threads_per_blk;
        sm_load = ((float)block_count / num_multiprocs) * 100.0;
	HIP_SAFE_CALL(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(&active_blks,
									 kfunc,
									 threads_per_blk,
									 0));
	if (__kitrt_verbose_mode()) 
	  fprintf(stderr, "\t+ active blocks/mp=%d, threads/block=%d\n", active_blks, threads_per_blk);
	
      } while (active_blks < 8 && threads_per_blk > 768);
    }

    if (__kitrt_verbose_mode()) {
      fprintf(stderr, "+kithip: REFINED launch parameters ---------------------\n");
      fprintf(stderr, "\tthreads-per-block: %d\n", threads_per_blk);
      fprintf(stderr, "\tnumer of blocks:   %d\n", block_count);
      fprintf(stderr, "\tmulti-proc load:   %3.2f%%\n", sm_load);
      fprintf(stderr, "--------------------------------------------------------\n\n");
    }
  }

  blks_per_grid = (trip_count + (threads_per_blk - 1)) / threads_per_blk;
}

} // namespace

void __kithip_get_launch_params(size_t trip_count, hipFunction_t kfunc,
                                const char *kfunc_name, int &threads_per_blk,
                                int &blks_per_grid,
                                const KitRTInstMix *inst_mix) {
  std::string map_entry_name(kfunc_name);
  map_entry_name += std::to_string(trip_count);

  KitHipLaunchParamMap::iterator lpit =
      _kithip_launch_param_map.find(map_entry_name);
  if (lpit != _kithip_launch_param_map.end())
    // use previously determined parameters.
    threads_per_blk = lpit->second;
  else {
    if (_kithip_use_occupancy_calc)
      __kithip_get_occ_launch_params(trip_count, kfunc, threads_per_blk,
                                     blks_per_grid, inst_mix);
    else
      threads_per_blk = _kithip_default_threads_per_blk;
    _kithip_launch_param_map[map_entry_name] = threads_per_blk;
  }
  
  blks_per_grid = (trip_count + threads_per_blk - 1) / threads_per_blk;
}

void *__kithip_launch_kernel(const void *fat_bin, const char *kernel_name,
                             void **kern_args, uint64_t trip_count,
                             int threads_per_blk, const KitRTInstMix *inst_mix,
                             void *opaque_stream) {

  assert(fat_bin && "kithip: launch with null fat binary!");
  assert(kernel_name && "kithip: launch with null name!");
  assert(kern_args && "kithip: launch with null args!");
  assert(trip_count != 0 && "kithip: launch with zero trips!");

  // HIP_SAFE_CALL(hipSetDevice_p(__kithip_get_device_id()));

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

  int blks_per_grid;
  if (threads_per_blk == 0)
    __kithip_get_launch_params(trip_count, kern_func, kernel_name,
                               threads_per_blk, blks_per_grid, inst_mix);
  else
    blks_per_grid = (trip_count + threads_per_blk - 1) / threads_per_blk;

  hipStream_t hip_stream = nullptr;
  if (opaque_stream == nullptr)
    hip_stream = (hipStream_t)__kithip_get_thread_stream();
  else
    hip_stream = (hipStream_t)opaque_stream;

  if (__kitrt_verbose_mode()) {
    fprintf(stderr, "kithip: '%s' launch parameters:\n", kernel_name);
    fprintf(stderr, "  blocks:     %d, 1, 1\n", blks_per_grid);
    fprintf(stderr, "  threads:    %d, 1, 1\n", threads_per_blk);
    fprintf(stderr, "  trip count: %ld\n", trip_count);
    fprintf(stderr, "  stream ptr: %p\n", (void*)hip_stream);
  }

  HIP_SAFE_CALL(hipModuleLaunchKernel_p(kern_func,
					blks_per_grid, 1, 1,
                                        threads_per_blk, 1, 1,
                                        0, // shared mem size
                                        hip_stream, kern_args, NULL));
  return (void *)hip_stream;
}

void *__kithip_get_global_symbol(void *fat_bin, const char *sym_name) {
  assert(fat_bin && "null fat binary!");
  assert(sym_name && "null symbol name!");

  HIP_SAFE_CALL(hipSetDevice_p(__kithip_get_device_id()));

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

} // extern C
