//===- kitcuda-launch.cpp - Kitsune runtime CUDA launch support -----------===//
//
// Copyright (c) 2021, 2023 Los Alamos National Security, LLC.
// All rights reserved.
//
//  Copyright 2021, 2023. Los Alamos National Security, LLC. This
//  software was produced under U.S. Government contract
//  DE-AC52-06NA25396 for Los Alamos National Laboratory (LANL), which
//  is operated by Los Alamos National Security, LLC for the
//  U.S. Department of Energy. The U.S. Government has rights to use,
//  reproduce, and distribute this software.  NEITHER THE GOVERNMENT
//  NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS
//  OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.
//  If software is modified to produce derivative works, such modified
//  software should be clearly marked, so as not to confuse it with
//  the version available from LANL.
//
//  Additionally, redistribution and use in source and binary forms,
//  with or without modification, are permitted provided that the

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
#include <string>
#include <unordered_map>

//
// *** EXPERIMENTAL: In kitsune the details of picking launch parameters 
// are not entirely aligned with the general high-level model of parallelism.  
// In general, the challenge it that many approaches focus on occupancy as 
// the primary factor for selecting launch parameters but that can lead to 
// under-utilization of the GPU (e.g., idle SMs) so occupancy isn not the sole
// parameter to optimize for launching any given kernel.
// 
// From a CUDA-terminology viewpoint, occupancy is defined as the ratio
// of the number of active warps per multiprocessor to the maximum number 
// of active warps supported by the target architecture. Importantly,
// having a higher occupancy does not guarantee better performance. It is
// simply a reasonable metric for the latency hiding ability of a particular
// kernel.  
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



namespace {
  // TODO: Need to evaluate the overall advantage (or disadvantage) for
  // the runtime in terms of the benefits provided by the given experimental
  // code below... 

  // *** EXPERIMENTAL: The runtime maintains a map from fat-binary images
  // to a supporting CUDA module.  The primary goal for this is to avoid 
  // rebuilding modules (or creating too many modules).  Early benefits 
  // showed a minor reduction in overheads but the details need to be 
  // quantified to see if the long-term use makes a lot of sense... 
  typedef std::unordered_map<const void *, CUmodule> KitCudaModuleMap;
  KitCudaModuleMap module_map;
  std::mutex module_map_mutex;

  // *** EXPERIMENTAL: The runtime maintains a map from kernel names to
  // kernel functions.  This avoids searching a module repeatedly at
  // kernel launch time and is primarily focused on reducing runtime
  // overheads.  The savings here still need to explored in more detail;
  // it is not clear how map overheads compare to the runtime lookup...
  typedef std::unordered_map<const char *, CUfunction> KitCudaKernelMap;
  KitCudaKernelMap kernel_map;

  // Runtime state for CUDA launch configuration and modes of operation.
  bool use_occupancy_launch = KITRT_USE_OCCUPANCY_LAUNCH;
  bool refine_occupancy_launch = KITRT_USE_REFINED_OCCUPANCY_LAUNCH;
  int  default_threads_per_blk = KITRT_DEFAULT_THREADS_PER_BLOCK;
  int  max_threads_per_blk = KITRT_DEFAULT_THREADS_PER_BLOCK;
}

extern "C" {

/* 
 * Enable the use of CUDA's heuristic-based launch parameter selection.
 */
void __kitcuda_use_occupancy_launch(bool enable) {

  int threads_per_block = 0;
  use_occupancy_launch = enable;
  
  if (__kitrt_get_env_value("KITCUDA_THREADS_PER_BLOCK", threads_per_block)) {
    if (enable) {
      fprintf(stderr, "kitcuda: note - environment setting "
                      "of 'KITCUDA_THREADS_PER_BLOCK' (%d) "
                      "overrides occupancy launch.\n", threads_per_block);
      use_occupancy_launch = false;
    }
  }

  if (__kitrt_verbose_mode()) 
    fprintf(stderr, "kitcuda: %s occupancy-computed launch parameters.\n", 
            enable ? "enabled" : "disabled");
}

void __kitcuda_refine_occupancy_launches(bool enable) {
  if (enable) {
    // refined occupancy launches requires occupancy 
    // launches to be enabled.  Yep, logic is ugly but
    // was a bit more minimalistic here... 
    __kitcuda_use_occupancy_launch(enable);
    // occupancy launch can be overridden by the environment 
    // so follow suit with our refinement setting... 
    refine_occupancy_launch = use_occupancy_launch;
  } else
    refine_occupancy_launch = false;

  if (__kitrt_verbose_mode())
    fprintf(stderr, "kitcuda: runtime '%s' refine occupancy launches.\n", 
            refine_occupancy_launch ? "will" : "will not");
}

void __kitcuda_set_max_threads_per_blk(int num_threads) {
  // TODO: This value should be clamped to hardware-specific limits. 
  max_threads_per_blk = num_threads;
  if (__kitrt_verbose_mode()) 
    fprintf(stderr, "kitcuda: setting maximum threads-per-block -- %d\n", 
            max_threads_per_blk);
}

void __kitcuda_set_default_threads_per_blk(int threads_per_blk) {
  if (threads_per_blk > max_threads_per_blk)
    threads_per_blk = max_threads_per_blk;
  default_threads_per_blk = threads_per_blk;
  if (__kitrt_verbose_mode())
    fprintf(stderr, "kitcuda: setting default threads-per-block -- %d\n", 
            threads_per_blk);
}

typedef std::unordered_map<std::string, int> KitCudaLaunchParamMap;
static KitCudaLaunchParamMap _kitcuda_launch_param_map;

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
 * @param cu_func - the actual CUDA function / kernel.
 * @param threads_per_blk - computed threads per block for launch
 * @param blks_per_grid - computed blocks per grid for launch
 */
void __kitcuda_get_occ_launch_params(size_t trip_count, CUfunction cu_func,
                                     int &threads_per_blk, int &blks_per_grid,
                                     const KitCudaInstMix *inst_mix) {
  assert(use_occupancy_launch && "called when occupancy mode is false!");
  KIT_NVTX_PUSH("kitcuda:get_occupancy_launch_params", KIT_NVTX_LAUNCH);

  // As a default starting point, use CUDA's occupancy heuristic to get
  // an initial occupancy.
  int min_grid_size;
  CU_SAFE_CALL(cuOccupancyMaxPotentialBlockSize_p(
      &min_grid_size, &threads_per_blk, cu_func, 0, 0, 0));

  if (refine_occupancy_launch) {
    extern int _kitcuda_device_id;

    int num_multiprocs = 0;
    CU_SAFE_CALL(cuDeviceGetAttribute_p(
        &num_multiprocs, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        _kitcuda_device_id));

    // The occupancy measure isn't the only aspect of launch performance
    // to consider.  Specifically, the heuristic ignores trip counts that
    // that can lead to an under-subscription of GPU resources.  In these
    // cases performance can significantly suffer.
    //
    // To address trip count impacts we start by looking at an estimate of
    // the number of SM's that can be kept busy by the provided
    // threads-per-block value.  We do this by getting a block count and looking
    // at that number in comparison to the number of SMs.
    int block_count = (trip_count + threads_per_blk - 1) / threads_per_blk;
    float sm_load = ((float)block_count / num_multiprocs) * 100.0;

    if (__kitrt_verbose_mode()) {
      fprintf(stderr,
              "kitcuda: Kernel Launch SM Load Details --------------\n");
      fprintf(stderr, "  number of SMs:        %d\n", num_multiprocs);
      fprintf(stderr, "  kernel trip count:    %ld\n", trip_count);
      fprintf(stderr, "  occupancy-driven TPB: %d\n", threads_per_blk);
      fprintf(stderr, "  SM utilization:  %3.2f%%\n", sm_load);
    }
    // If we are under-utilizing the available SMs on the GPU we reduce the
    // threads-per-block count until we hit a decent utilization (i.e., we
    // increase the block count). The determination of when to make this
    // adjustment is based on the percentage of SMs used (`sm_usage`) and
    // must be adjusted such that the resulting block count does not exceed
    // the number of SMs available.
    //
    // As a starting point we will adjust launch parameters if we are utilizing
    // less than 75% of the GPU's SMs.  TODO: Make this a tweak-able parameter?
    if (sm_load < 75) {
      if (__kitrt_verbose_mode())
        fprintf(stderr,
                "  ***-GPU is underutilized -- adjusting block size...\n");

      int warp_size = 0;
      CU_SAFE_CALL(cuDeviceGetAttribute_p(
          &warp_size, CU_DEVICE_ATTRIBUTE_WARP_SIZE, _kitcuda_device_id));
      while (block_count < num_multiprocs && threads_per_blk > warp_size) {
        threads_per_blk = __kitrt_next_lowest_factor(threads_per_blk, warp_size);
        block_count = (trip_count + threads_per_blk - 1) / threads_per_blk;
        sm_load = ((float)block_count / num_multiprocs) * 100.0;
      }
      if (__kitrt_verbose_mode()) {
        fprintf(stderr, "  ***-new launch parameters:");
        fprintf(stderr, "\tthreads-per-block: %d\n", threads_per_blk);
        fprintf(stderr, "\tnumer of blocks:   %d\n", block_count);
        fprintf(stderr, "\tSM utilization:    %3.2f%%\n", sm_load);
        fprintf(stderr,
                "-----------------------------------------------------\n\n");
      }
    }
  }

  blks_per_grid = (trip_count + threads_per_blk - 1) / threads_per_blk;
  KIT_NVTX_POP();
}

/**
 * Get the launch parameters for a given kernel and trip count.  The
 * behavior of this call will depend on various runtime
 * configuration details.  If `use_occupancy_launch` is set the
 * kernel we be analyzed to determine an approximate measure of
 * occupancy (via CUDA), if custom parameters have been set they
 * will be used, or a simple default determination based on a
 * default number of threads-per-block will be the default.  See
 * the code in kitcuda-launch.cpp for more details.
 *
 * @param trip_count - how many elements to process
 * @param cu_func - the actual CUDA function / kernel.
 * @param threads_per_blk - computed threads per block for launch
 * @param blks_per_grid - computed blocks per grid for launch
 */
void __kitcuda_get_launch_params(size_t trip_count, CUfunction cu_func,
                                 int &threads_per_blk, int &blks_per_grid,
                                 const KitCudaInstMix *inst_mix) {
  KIT_NVTX_PUSH("kitcuda:get_launch_params", KIT_NVTX_LAUNCH);

  // EXPERIMENTAL: Our 'forall' kernels have zero shared memory usage so
  // tweak the kernel's cache configuration to prefer L1 usage vs. shared
  // or 'split' usage of the local memory.
  CU_SAFE_CALL(cuFuncSetCacheConfig_p(cu_func, CU_FUNC_CACHE_PREFER_L1));

  // EXPERIMENTAL: To reduce some overheads the runtime caches launch
  // parameters for each kernel.  Check to see if we have already set
  // the launch parameters for this kernel and trip count.
  const char *cu_func_name;
  CU_SAFE_CALL(cuFuncGetName_p(&cu_func_name, cu_func));
  std::string map_entry_name(cu_func_name);
  map_entry_name += std::to_string(trip_count);

  KitCudaLaunchParamMap::iterator lpit =
      _kitcuda_launch_param_map.find(map_entry_name);

  if (lpit != _kitcuda_launch_param_map.end())
    // use previously determined parameters.
    threads_per_blk = lpit->second;
  else {
    if (use_occupancy_launch)
      // EXPERIMENTAL: use an occupancy-based path to setting the launch
      // parameters.
      __kitcuda_get_occ_launch_params(trip_count, cu_func, threads_per_blk,
                                      blks_per_grid, inst_mix);
    else
      threads_per_blk = default_threads_per_blk;
    _kitcuda_launch_param_map[map_entry_name] = threads_per_blk;
  }

  blks_per_grid = (trip_count + threads_per_blk - 1) / threads_per_blk;
  KIT_NVTX_POP();
}

void *__kitcuda_launch_kernel(const void *fat_bin, const char *kernel_name,
                              void **kern_args, uint64_t trip_count,
                              int threads_per_blk,
                              const KitCudaInstMix *inst_mix,
                              void *opaque_stream) {
  assert(fat_bin && "kitrt: CUDA launch with null fat binary!");
  assert(kernel_name && "kitrt: CUDA launch with null name!");
  assert(kern_args && "kitrt: CUDA launch with null args!");

  KIT_NVTX_PUSH("kitcuda:launch_kernel", KIT_NVTX_LAUNCH);

  // Multiple threads can launch kernels in our current design.  If a
  // thread enters without having previously set the context the CUDA
  // runtime becomes unhappy with us.  Make sure we're following the
  // rules.
  CUcontext ctx;
  CU_SAFE_CALL(cuCtxGetCurrent_p(&ctx));
  if (ctx == NULL)
    CU_SAFE_CALL(cuCtxSetCurrent_p(_kitcuda_context));

  CUfunction cu_func;
  module_map_mutex.lock();
  KitCudaKernelMap::iterator kernit = kernel_map.find(kernel_name);
  if (kernit == kernel_map.end()) {
    // We have not yet encountered this kernel function...  Check to see
    // if we already have a supporting module for the fat binary.
    CUmodule cu_module;
    KitCudaModuleMap::iterator modit = module_map.find(fat_bin);
    if (modit == module_map.end()) {
      // Create a supporting CUDA module and "register" the fat binary
      // image in the map...
      CU_SAFE_CALL(cuModuleLoadData_p(&cu_module, fat_bin));
      module_map[fat_bin] = cu_module;
    } else
      cu_module = modit->second;

    // Look up the kernel function.
    CU_SAFE_CALL(cuModuleGetFunction_p(&cu_func, cu_module, kernel_name));
    kernel_map[kernel_name] = cu_func;
  } else
    cu_func = kernit->second;

  module_map_mutex.unlock();

  int blks_per_grid;
  if (threads_per_blk == 0)
    __kitcuda_get_launch_params(trip_count, cu_func, threads_per_blk,
                                blks_per_grid, inst_mix);
  else {
    if (__kitrt_verbose_mode())
      fprintf(stderr, "kitcuda: threads per block provided by compiler -- %d\n", 
             threads_per_blk);
    blks_per_grid = (trip_count + threads_per_blk - 1) / threads_per_blk;
  }

  if (__kitrt_verbose_mode()) {
    fprintf(stderr, "kitcuda: kernel '%s' launch parameters:\n", kernel_name);
    fprintf(stderr, "  blocks: %d, 1, 1\n", blks_per_grid);
    fprintf(stderr, "  threads: %d, 1, 1\n", threads_per_blk);
    fprintf(stderr, "  trip count: %ld\n\n", trip_count);
  }

  CUstream cu_stream = nullptr;
  if (opaque_stream == nullptr) {
    // create a stream for this launch...
    cu_stream = (CUstream)__kitcuda_get_thread_stream();
    if (__kitrt_verbose_mode())
      fprintf(stderr,
              "kitcuda: launch stream is null, requested a new stream.\n");
  } else {
    // use the provided stream for this launch...
    cu_stream = (CUstream)opaque_stream;
    if (__kitrt_verbose_mode())
      fprintf(stderr, "kitcuda: launch stream is non-null.\n");
  }

  CU_SAFE_CALL(cuLaunchKernel_p(cu_func, blks_per_grid, 1, 1, threads_per_blk,
                                1, 1,
                                0, // shared mem size
                                cu_stream, kern_args, NULL));
  KIT_NVTX_POP();
  return (void *)cu_stream;
}

uint64_t __kitcuda_get_global_symbol(void *fat_bin, const char *sym_name) {
  assert(fat_bin && "null fat binary!");
  assert(sym_name && "null symbol name!");

  KIT_NVTX_PUSH("kitcuda:get_global_symbol", KIT_NVTX_LAUNCH);

  // Multiple threads can launch kernels in the current design.  If a
  // thread enters without having previously set the context the CUDA
  // runtime becomes unhappy with us.  Make sure we're following the
  // rules.
  //
  // TODO: This code is shared verbatim w/ the kernel launch.  We should
  // move it to a shared call...
  CUcontext ctx;
  CU_SAFE_CALL(cuCtxGetCurrent_p(&ctx));
  if (ctx == NULL)
    CU_SAFE_CALL(cuCtxSetCurrent_p(_kitcuda_context));
  CUmodule cu_module;
  module_map_mutex.lock();
  KitCudaModuleMap::iterator modit = module_map.find(fat_bin);
  if (modit == module_map.end()) {
    // Create a supporting CUDA module and "register" the fat binary
    // image in the map...
    CU_SAFE_CALL(cuModuleLoadData_p(&cu_module, fat_bin));
    module_map[fat_bin] = cu_module;
  } else
    cu_module = modit->second;

  // NOTE: The device pointer and size ('bytes') parameters for the
  // call to cuModuleGetGlobal are optional.  To simplify the compiler's
  // code generation details we ignore the size parameter...
  CUdeviceptr sym_ptr;
  size_t bytes;

  // To provide some assistance in debugging our code generation we
  // avoid wrapping the following in a CUDA_SAFE_CALL...
  CUresult result;
  if ((result = cuModuleGetGlobal_v2_p(&sym_ptr, &bytes, cu_module,
                                       sym_name)) != CUDA_SUCCESS) {
    const char *msg;
    fprintf(stderr, "kitcuda: error finding global symbol '%s'.\n", sym_name);
    cuGetErrorName_p(result, &msg);
    fprintf(stderr, "kitcuda %s:%d:\n", __FILE__, __LINE__);
    fprintf(stderr, "  * cuModuleGetGlobal('%s'...) failed\n", msg);
    cuGetErrorString_p(result, &msg);
    fprintf(stderr, "  * error: '%s'\n", msg);
    __kitrt_print_stack_trace();
    abort();
  }
  KIT_NVTX_POP();
  return sym_ptr;
}

} // extern "C"
