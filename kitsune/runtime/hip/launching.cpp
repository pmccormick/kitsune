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
#include <string>
#include <unordered_map>

//
// TODO: Needs to be brought into line with HIP/AMDGPU architecture specifics... 
// 
// *** EXPERIMENTAL: In kitsune the details of picking launch parameters 
// are not entirely aligned with the general high-level model of parallelism.  
// In general, the challenge it that many approaches focus on occupancy as 
// the primary factor for selecting launch parameters but that can lead to 
// under-utilization of the GPU (e.g., idle SMs) so occupancy isn not the sole
// parameter to optimize for launching any given kernel.
// 
// The HIP approach mirrors that of the current CUDA version, occupancy is 
// defined as the ratio of the number of active warps per multiprocessor to 
// the maximum number of active warps supported by the target architecture. 
// Importantly, having a higher occupancy does not guarantee better performance. 
// It is simply a reasonable metric for the latency hiding ability of a particular
// kernel.  
//
// The runtime has the following modes when it comes to launch parameters: 
//
//   1. Occupancy-based launches: This uses HIP's heuristics for 
//      determination of the parameters.  Given it tries to maximize 
//      occupancy it can also significantly under-utilize the GPU (e.g,
//      a small number of active SMs).
// 
//   2. Refined occupancy launches: This uses the same occupancy 
//      details in #1 but then attempts to adjust the parameters 
//      to provide better overall utilization of the GPU's SMs. 
//
//   3. Custom launches: This uses runtime specific parameters to 
//      set the launch parameters.  These can be specified by the 
//      compiler, via direct runtime calls, or via the environment 
//      (variables). 
// 
//   4. Default launches: This uses a set of hard-coded parameters 
//      that are built into the runtime. These can be set at build
//      time. 
// 


namespace {
  // TODO: Need to figure out if the experimental pieces below actually
  // provide any significant wins...

  // *** EXPERIMENTAL: The runtime maintains a map from fat-binary images
  // to a supporting HIP module.  The primary reason for this is
  // exploring reducing runtime overheads.
  typedef std::unordered_map<const void *, hipModule_t> KitHipModuleMap;
  KitHipModuleMap module_map;
  std::mutex module_map_mutex;

  // *** EXPERIMENTAL: The runtime maintains a map from kernel names to
  // kernel functions.  This avoids searching a module repeatedly at
  // kernel launch time and is primarily focused on reducing runtime
  // overheads.
  typedef std::unordered_map<const char *, hipFunction_t> KitHipKernelMap;
  KitHipKernelMap kernel_map;

  // Runtime state for HIP launch configuration and modes of operation.
  bool use_occupancy_launch = KITRT_USE_OCCUPANCY_LAUNCH;
  bool refine_occupancy_launch = KITRT_USE_REFINED_OCCUPANCY_LAUNCH;
  int  default_threads_per_blk = KITRT_DEFAULT_THREADS_PER_BLOCK;
  int  max_threads_per_blk = KITRT_DEFAULT_THREADS_PER_BLOCK;
}

extern "C" {

  // *** EXPERIMENTAL: The details of picking launch parameters can be a
  // challenge and many approaches push occupancy as the primary factor.
  // From a CUDA-terminology viewpoint, occupancy is defined as the ratio
  // of the number of active warps per multiprocessor to the maximum number
  // of active warps supported by the target architecture. Importantly,
  // having a higher occupancy does not guarantee better performance. It is
  // simply a reasonable metric for the latency hiding ability of a particular
  // kernel.  We are actively looking to address this with a different approach
  // for supporting Kitsune.
  //
  // The runtime has the following modes when it comes to launch parameters:
  //
  //   1. Occupancy-based launches: This uses CUDA's heuristics for
  //      determination of the parameters.  Given it tries to maximize
  //      occupancy it can also significantly under-utilize the GPU (e.g,
  //      a small number of active SMs).
  //
  //   2. Refined occupancy launches: This uses the same occupancy
  //      details in #1 but then attempts to adjust the parameters
  //      to provide better overall utilization of the GPU's SMs.
  //
  //   3. Custom launches: This uses runtime specific parameters to
  //      set the launch parameters.  These can be specified by the
  //      compiler, via direct runtime calls, or via the environment
  //      (variables).
  //
  //   4. Default launches: This uses a set of hard-coded parameters
  //      that are built into the runtime. These can be set at build
  //      time.
  //

void __kithip_use_occupancy_launch(bool enable) {
  int threads_per_block = 0;
  use_occupancy_launch = enable;
  if (__kitrt_get_env_value("KITHIP_THREADS_PER_BLOCK", threads_per_block)) {
    if (enable) {
      fprintf(stderr, "kithip: note - environment setting "
                      "of 'KITHIP_THREADS_PER_BLOCK' (%d) "
                      "overrides occupancy launch.\n", threads_per_block);
      use_occupancy_launch = false;
    }
  }
  
  if (__kitrt_verbose_mode()) 
    fprintf(stderr, "kithip: %s occupancy-computed launch parameters.\n", 
            use_occupancy_launch ? "enabled" : "disabled");
}

void __kithip_refine_occupancy_launches(bool enable) {
  if (enable) {
    // refined occupancy launches requires occupancy 
    // launches to be enabled.  Yep, logic is ugly but
    // was a bit more minimalistic here... 
    use_occupancy_launch = enable;
    // occupancy launch can be overridden by the environment 
    // so follow suit with our refinement setting... 
    refine_occupancy_launch = use_occupancy_launch;
  } else
    refine_occupancy_launch = false;

  if (__kitrt_verbose_mode())
    fprintf(stderr, "kithip: runtime '%s' refine occupancy launches.\n", 
            refine_occupancy_launch ? "will" : "will not");
}

void __kithip_set_default_max_threads_per_blk(int num_threads) {
  // TODO: This value should be clamped to hardware-specific limits. 
  max_threads_per_blk = num_threads;
}

void __kithip_set_default_threads_per_blk(int threads_per_blk) {
  if (threads_per_blk > max_threads_per_blk)
    threads_per_blk = max_threads_per_blk;
  default_threads_per_blk = threads_per_blk;
}

typedef std::unordered_map<std::string, int> KitHipLaunchParamMap;
static KitHipLaunchParamMap _kithip_launch_param_map;

void __kithip_get_occ_launch_params(size_t trip_count, hipFunction_t kfunc,
                                    int &threads_per_blk, int &blks_per_grid,
                                    const KitHipInstMix *inst_mix) {
  assert(use_occupancy_launch &&
         "called when occupancy mode is false!");
  // HIP frustratingly uses a bunch of inlined template-based calls and that
  // makes it difficult to weed out actual (dylib) friendly entry points.
  // Eventually chasing the details down the entry point we need (to match the
  // overall runtime design) is the 'ModuleOccupancy' call used below.  Of
  // course this is orthogonal to the HIP documentation details...
  int min_grid_size; // currently ignored...
  HIP_SAFE_CALL(hipModuleOccupancyMaxPotentialBlockSize_p(
      &min_grid_size, &threads_per_blk, kfunc, 0, 0));

  if (refine_occupancy_launch) {
    extern int _kithip_device_id;

    int num_multiprocs = 0;
    HIP_SAFE_CALL(hipDeviceGetAttribute_p(&num_multiprocs,
                                          hipDeviceAttributeMultiprocessorCount,
                                          _kithip_device_id));

    // The occupancy measure isn't the only aspect of launch performance
    // to consider...  Specifically, the heuristic ignores trip counts
    // and therefore can lead to an under-subscription of GPU resources.
    // In these cases performance can suffer.  To address this we start
    // by looking at an estimate based on the number of multi-processors
    // that can be kept busy by the calculated threads per block value.
    int block_count = (trip_count + threads_per_blk - 1) / threads_per_blk;
    float mp_load = ((float)block_count / num_multiprocs) * 100;

    if (__kitrt_verbose_mode()) {
      fprintf(stderr,
              "kithip: Kernel launch multi-processor load details -------\n");
      fprintf(stderr, "  number of multi-processors:  %d\n", num_multiprocs);
      fprintf(stderr, "  trip count:                  %ld\n", trip_count);
      fprintf(stderr, "  occupancy-driven TPB:        %d\n", threads_per_blk);
      fprintf(stderr, "  multi-processor utilization: %f3.2f%%\n", mp_load);
    }
    // EXPERIMENTAL:
    // If we are under-utilizing the available multi-processors on the GPU we
    // reduce the threads-per-block count until we hit a "decent" utilization
    // (i.e., we increase the block count to fill up the multi-processors).
    // The determination of when to make this adjustment is based on the
    // percentage of multi-processors that would be used and it must be
    // adjusted such that the resulting block count does not exceed the
    // number of multi-processors available.
    if (mp_load < 75) { // TODO: stop hard-coding 75...
      if (__kitrt_verbose_mode())
        fprintf(stderr,
                "    --*-GPU is underutilized -- tuning block count...\n");

      int warp_size = 0;
      HIP_SAFE_CALL(hipDeviceGetAttribute_p(
          &warp_size, hipDeviceAttributeWarpSize, _kithip_device_id));
      while (block_count < num_multiprocs && threads_per_blk > warp_size) {
        threads_per_blk =
            __kitrt_next_lowest_factor(threads_per_blk, warp_size);
        block_count = (trip_count + threads_per_blk - 1) / threads_per_blk;
        mp_load = ((float)block_count / num_multiprocs) * 100.0;
      }

      if (__kitrt_verbose_mode()) {
        fprintf(stderr,
                "    --*-GPU is underutilized -- tuning block count...\n");
        fprintf(stderr, "      updated launch parameters:");
        fprintf(stderr, "        threads-per-block:      %d\n",
                threads_per_blk);
        fprintf(stderr, "        numer of blocks:        %d\n", block_count);
        fprintf(stderr, "        multi-proc utilization: %3.2f%%\n", mp_load);
        fprintf(stderr,
                "----------------------------------------------------------\n");
      }
    }
  }

  blks_per_grid = (trip_count + threads_per_blk - 1) / threads_per_blk;
}

void __kithip_get_launch_params(size_t trip_count, hipFunction_t kfunc,
                                int &threads_per_blk, int &blks_per_grid) {

  if (use_occupancy_launch) {
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
    threads_per_blk = default_threads_per_blk;
  }

  // Need to round-up based on array size/trip count.
  blks_per_grid = (trip_count + threads_per_blk - 1) / threads_per_blk;
}

void __kithip_launch_kernel(const void *fat_bin, const char *kernel_name,
                            void **kern_args, uint64_t trip_count,
                            int threads_per_blk, const KitHipInstMix *inst_mix,
                            void *opaque_stream) {

  assert(fat_bin && "kithip: launch with null fat binary!");
  assert(kernel_name && "kithip: launch with null name!");
  assert(kern_args && "kithip: launch with null args!");
  assert(trip_count != 0 && "kithip: launch with zero trips!");

  HIP_SAFE_CALL(hipSetDevice_p(__kithip_get_device_id()));

  // Multiple threads can launch kernels in our current design.  If a
  // thread enters without having previously set the device the runtime
  // becomes unhappy with us.  Make sure we're following the rules.
  hipFunction_t kern_func;
  module_map_mutex.lock();
  KitHipKernelMap::iterator kernit = kernel_map.find(kernel_name);
  if (kernit == kernel_map.end()) {
    // We have not yet encountered this kernel function...  Check to see
    // if we already have a supporting module for the fat binary.
    hipModule_t hip_module;
    KitHipModuleMap::iterator modit = module_map.find(fat_bin);
    if (modit == module_map.end()) {
      // Create a supporting module and "register" the fat binary
      // image in the map...
      HIP_SAFE_CALL(hipModuleLoadData_p(&hip_module, fat_bin));
      module_map[fat_bin] = hip_module;
    } else
      hip_module = modit->second;

    // Look up the kernel function in the module.
    HIP_SAFE_CALL(hipModuleGetFunction_p(&kern_func, hip_module, kernel_name));
    kernel_map[kernel_name] = kern_func;
  } else
    kern_func = kernit->second;

  module_map_mutex.unlock();

  int blks_per_grid;
  __kithip_get_launch_params(trip_count, kern_func, threads_per_blk,
                             blks_per_grid);

  if (__kitrt_verbose_mode()) {
    fprintf(stderr, "kithip: kernel '%s' launch parameters:\n", kernel_name);
    fprintf(stderr, "  blocks: %d, 1, 1\n", blks_per_grid);
    fprintf(stderr, "  threads: %d, 1, 1\n", threads_per_blk);
    fprintf(stderr, "  trip count: %ld\n", trip_count);
    fprintf(stderr, "  args address: %p\n", kern_args);
  }

  hipStream_t hip_stream = (hipStream_t)opaque_stream;
  HIP_SAFE_CALL(hipModuleLaunchKernel_p(kern_func, blks_per_grid, 1, 1,
                                        threads_per_blk, 1, 1,
                                        0, // shared mem size
                                        hip_stream, kern_args, NULL));
}

void *__kithip_get_global_symbol(void *fat_bin, const char *sym_name) {
  assert(fat_bin && "null fat binary!");
  assert(sym_name && "null symbol name!");

  hipModule_t hip_module;
  module_map_mutex.lock();
  KitHipModuleMap::iterator modit = module_map.find(fat_bin);
  if (modit == module_map.end()) {
    HIP_SAFE_CALL(hipModuleLoadData_p(&hip_module, fat_bin));
    module_map[fat_bin] = hip_module;
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
