//===- kitcuda-launch.cpp - Kitsune runtime CUDA launch support -----------===//
//
// Copyright (c) 2021, 2023, 2025 Los Alamos National Security, LLC.
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

// We maintain a map of modules (think fat binary) to avoid having to
// reprocess them over and over again.  This mapping actually goes
// directly from the compiler generated fat binary to the hip runtime
// module structure required to look up kernel calls.  The hope here
// is that the map acccess is actually faster than repeatedly loading
// a module, searching it, and returning a kernel.
#include <unordered_map>
typedef std::unordered_map<const void *, CUmodule> KitCudaModuleMap;
static KitCudaModuleMap _kitcuda_module_map;
static std::mutex _kitcuda_module_map_mutex;

// Like the module map above, the runtime maintains a map from kernel
// names to kernel functions (again avoiding a hip-driven lookup
// process).
typedef std::unordered_map<const char *, CUfunction> KitCudaKernelMap;
static KitCudaKernelMap _kitcuda_kernel_map;

extern "C" {

// NOTE: Our strategy for choosing kernel launch parameters is
// based on seeking to optimize parallelism at the level of
// SM subscription first.  We do this by picking a
// threads-per-block value that fully, or even over
// subscribes, the GPU.  This is our primary driver and we
// will then consider additional details (e.g., register
// usage, occupancy, etc) to tweak the parameters.
//
// TODO: We need to introduce features beyond the SM
// utilization for tweaking launch parameters.


// Codegen target maximum threads per block -- this is not the hardware
// limit but a configuration option that can enable more flexiblity
// for register allocation/usage in compiled kernels.  The compiler
// generates code to set this value at runtime initialization.
static int _KITCUDA_MAX_THREADS_PER_BLK = 1024;

// By default the runtime will pick this number of threads-per-block
// in a launch if no compmiler/parameter/environment settings are
// provided.
static int _KITCUDA_DEFAULT_THREADS_PER_BLK = 256;

// Enable the runtime's feature of adjusting the launch parameters
// until the full number of SMs available on the target GPU are
// utilized.  If this is false, we fall back to using the default
// number of threads per block (set by default here or via the
// user's environment).
static bool _kitcuda_refine_launches = true;

void __kitcuda_enable_launch_refinement(bool enable) {
  int threads_per_block = 0;
  if (__kitrt_get_env_value("KITCUDA_THREADS_PER_BLOCK", threads_per_block)) {
    if (enable) {
      fprintf(stderr, "kitcuda: note, setting 'KITCUDA_THREADS_PER_BLOCK' "
                      "will override refinement of launch parameters.\n");
      _kitcuda_refine_launches = false;
    } else {
      _kitcuda_refine_launches = enable;
    }
  }
}

void __kitcuda_set_max_threads_per_blk(int num_threads) {
  _KITCUDA_MAX_THREADS_PER_BLK = num_threads;
  // TODO: We could assert here on out-of-range values but
  // for now we are assuming if you are calling this, you are
  // well aware of what's going on.
}

void __kitcuda_set_default_threads_per_blk(int threads_per_blk) {
  if (threads_per_blk > _KITCUDA_MAX_THREADS_PER_BLK)
    threads_per_blk = _KITCUDA_MAX_THREADS_PER_BLK;
  _KITCUDA_DEFAULT_THREADS_PER_BLK = threads_per_blk;
}

typedef std::unordered_map<std::string, int> KitCudaLaunchParamMap;
static KitCudaLaunchParamMap _kitcuda_launch_param_map;

namespace {

int next_lowest_factor(int n, int m) {
  if (n > m && n) {
    for (int i = n - 1; i != 0; i--) {
      int r = i % m;
      if (r == 0)
        return i;
    }
  }
  return m;
}

} // namespace

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
void __kitcuda_refine_launch_params(size_t trip_count, CUfunction cu_func,
                                    int &threads_per_blk, int &blks_per_grid,
                                    const KitRTInstMix *inst_mix) {
  KIT_NVTX_PUSH("kitcuda:get_launch_params", KIT_NVTX_LAUNCH);

  // As a default starting point, use CUDA's occupancy heuristic to get
  // an initial occupancy.  At present we have seen this call do nothing
  // other than return the maximum number of allowable threads per block;
  // but we'll use it as a starting point anyways...  YMMV based on what
  // kernel code you are using/generating.
  int min_grid_size;
  CU_SAFE_CALL(cuOccupancyMaxPotentialBlockSize_p(
      &min_grid_size, &threads_per_blk, cu_func, 0, 0, 0));

  if (_kitcuda_refine_launches) {
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
      fprintf(stderr, "  Number of SMs:        %d\n", num_multiprocs);
      fprintf(stderr, "  Kernel trip count:    %ld\n", trip_count);
      fprintf(stderr, "  Occupancy-driven TPB: %d\n", threads_per_blk);
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
    if (sm_load < 85) {
      int warp_size = 0;
      CU_SAFE_CALL(cuDeviceGetAttribute_p(
          &warp_size, CU_DEVICE_ATTRIBUTE_WARP_SIZE, _kitcuda_device_id));
      int warps_per_sm = 4;
      int min_tpb = warp_size * warps_per_sm;
      while (block_count < num_multiprocs && threads_per_blk > min_tpb) {
        threads_per_blk = next_lowest_factor(threads_per_blk, min_tpb);
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

static int __kitcuda_reg_analysis(int threads_per_blk, int regs_per_thread,
                                  int max_regs_per_block) {
  int total_rcount = threads_per_blk * regs_per_thread;
  float perused = total_rcount / float(max_regs_per_block);
  extern int _kitcuda_device_id;
  int warp_size = 0;
  CU_SAFE_CALL(cuDeviceGetAttribute_p(&warp_size, CU_DEVICE_ATTRIBUTE_WARP_SIZE,
                                      _kitcuda_device_id));
  int warps_per_sm = 4;
  int min_tpb = warp_size * warps_per_sm;
  // fprintf(stderr, "kernel uses %d total registers for %d threads per
  // block.\n",
  //         total_rcount, threads_per_blk);
  // fprintf(stderr, "that's %f%% of available registers.\n",
  //         (float)total_rcount / (float)max_regs_per_block * 100.0);
  if (perused > 0.80) {
    threads_per_blk -= min_tpb;
    total_rcount = threads_per_blk * regs_per_thread;
    perused = total_rcount / float(max_regs_per_block);
    // fprintf(stderr, "ADJUSTED: %f%% registers and %d threads-per-block.\n",
    //         (float)total_rcount / (float)max_regs_per_block * 100.0,
    //         threads_per_blk);
  }

  if (threads_per_blk > _KITCUDA_MAX_THREADS_PER_BLK)
    threads_per_blk = _KITCUDA_MAX_THREADS_PER_BLK;

  return threads_per_blk;
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
                                 const KitRTInstMix *inst_mix) {
  KIT_NVTX_PUSH("kitcuda:get_launch_params", KIT_NVTX_LAUNCH);

  // EXPERIMENTAL: Our 'forall' kernels have zero shared memory usage so
  // tweak the kernel's cache configuration to prefer L1 usage vs. shared
  // or 'split' usage of the local memory.
  CU_SAFE_CALL(cuFuncSetCacheConfig_p(cu_func, CU_FUNC_CACHE_PREFER_L1));

  int regs_per_thread;
  CU_SAFE_CALL(cuFuncGetAttribute_p(&regs_per_thread,
                                    CU_FUNC_ATTRIBUTE_NUM_REGS, cu_func));

  int max_regs_per_blk;
  extern int _kitcuda_device_id;
  CU_SAFE_CALL(cuDeviceGetAttribute_p(
      &max_regs_per_blk, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
      _kitcuda_device_id));

  // EXPERIMENTAL: To reduce some overheads the runtime caches launch
  // parameters for each kernel.  Check to see if we have already set
  // the launch parameters for this kernel and trip count.
  const char *cu_func_name;
  CU_SAFE_CALL(cuFuncGetName_p(&cu_func_name, cu_func));
  std::string map_entry_name(cu_func_name);
  map_entry_name += std::to_string(trip_count);

  KitCudaLaunchParamMap::iterator lpit =
      _kitcuda_launch_param_map.find(map_entry_name);

  if (lpit != _kitcuda_launch_param_map.end()) {
    // use previously determined parameters.
    threads_per_blk = lpit->second;
  } else {
    if (_kitcuda_refine_launches) {
      __kitcuda_refine_launch_params(trip_count, cu_func, threads_per_blk,
                                     blks_per_grid, inst_mix);
      threads_per_blk = __kitcuda_reg_analysis(threads_per_blk, regs_per_thread,
                                               max_regs_per_blk);
    } else {
      threads_per_blk = _KITCUDA_DEFAULT_THREADS_PER_BLK;
    }

    // Final check to make sure we have not exceeded compile-time limits.
    int func_max_tpb;
    CU_SAFE_CALL(cuFuncGetAttribute_p(
        &func_max_tpb, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, cu_func));
    if (threads_per_blk > func_max_tpb) {
      fprintf(stderr,
              "kitcudart: warning, requested threads-per-block exceeds "
              "compile-time limits, adjusting to max possible "
              "threads-per-block (%d --> %d).\n",
              threads_per_blk, func_max_tpb);
      threads_per_blk = func_max_tpb;
    }

    _kitcuda_launch_param_map[map_entry_name] = threads_per_blk;
  }

  // TODO: This looks redundant with code in launch kernel...
  blks_per_grid = (trip_count + threads_per_blk - 1) / threads_per_blk;
  KIT_NVTX_POP();
}

void *__kitcuda_launch_kernel(const void *fat_bin, const char *kernel_name,
                              void **kern_args, uint64_t trip_count,
                              int threads_per_blk, const KitRTInstMix *inst_mix,
                              void *opaque_stream) {
  assert(fat_bin && "kitcuda: launch with null fat binary!");
  assert(kernel_name && "kitcuda: launch with null name!");
  assert(kern_args && "kitcuda: launch with null args!");
  assert(trip_count != 0 && "kitcuda: launch with zero trips!");

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
  _kitcuda_module_map_mutex.lock();
  KitCudaKernelMap::iterator kernit = _kitcuda_kernel_map.find(kernel_name);
  if (kernit == _kitcuda_kernel_map.end()) {
    // We have not yet encountered this kernel function...  Check to see
    // if we already have a supporting module for the fat binary.
    CUmodule cu_module;
    KitCudaModuleMap::iterator modit = _kitcuda_module_map.find(fat_bin);
    if (modit == _kitcuda_module_map.end()) {
      // Create a supporting CUDA module and "register" the fat binary
      // image in the map...
      CU_SAFE_CALL(cuModuleLoadData_p(&cu_module, fat_bin));
      _kitcuda_module_map[fat_bin] = cu_module;
    } else
      cu_module = modit->second;

    // Look up the kernel function.
    CU_SAFE_CALL(cuModuleGetFunction_p(&cu_func, cu_module, kernel_name));
    _kitcuda_kernel_map[kernel_name] = cu_func;
  } else
    cu_func = kernit->second;

  _kitcuda_module_map_mutex.unlock();

  int blks_per_grid;
  if (threads_per_blk == 0) {
    __kitcuda_get_launch_params(trip_count, cu_func, threads_per_blk,
                                blks_per_grid, inst_mix);
  } else {
    if (threads_per_blk > _KITCUDA_MAX_THREADS_PER_BLK) {
      fprintf(stderr,
              "kitcuda: warning, threads-per-block request exceeds bounds.\n");
      threads_per_blk = _KITCUDA_MAX_THREADS_PER_BLK;
    }

    int func_max_tpb;
    CU_SAFE_CALL(cuFuncGetAttribute_p(
        &func_max_tpb, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, cu_func));
    if (threads_per_blk > func_max_tpb) {
      fprintf(stderr,
              "kitcuda: warning, requested threads-per-block exceeds "
              "kernel's compile-time limits, adjusting to max possible "
              "threads-per-block (%d --> %d).\n",
              threads_per_blk, func_max_tpb);
      threads_per_blk = func_max_tpb;
    }
  }

  blks_per_grid = (trip_count + threads_per_blk - 1) / threads_per_blk;

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

  // LOCK
  _kitcuda_module_map_mutex.lock();
  KitCudaModuleMap::iterator modit = _kitcuda_module_map.find(fat_bin);
  if (modit == _kitcuda_module_map.end()) {
    // Create a supporting CUDA module and "register" the fat binary
    // image in the map...
    CU_SAFE_CALL(cuModuleLoadData_p(&cu_module, fat_bin));
    _kitcuda_module_map[fat_bin] = cu_module;
  } else {
    cu_module = modit->second;
  }

  _kitcuda_module_map_mutex.unlock();
  // UNLOCK

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
