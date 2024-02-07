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

#include "kitcuda.h"
#include "kitcuda_dylib.h"
#include <mutex>
#include <string>
#include <unordered_map>

// *** EXPERIMENTAL: The runtime maintains a map from fatbinary images
// to a supporting CUDA module.  The primary reason for this is
// exploring reducing runtime overheads.
//
// TODO: Finish exploration of map vs. CUDA call overheads.
typedef std::unordered_map<const void *, CUmodule> KitCudaModuleMap;
static KitCudaModuleMap _kitcuda_module_map;
static std::mutex _kitcuda_module_map_mutex;

// *** EXPERIMENTAL: The runtime maintains a map from kernel names to
// kernel functions.  This avoids searching a module repeatedly at
// kernel launch time and is primiarly focused on reducing runtime
// overheads.  The savings here still need to explored in more detail;
// it is not clear how map overheads compare to the runtime lookup...
//
// TODO: Finish exploration of map vs. CUDA call overheads.
typedef std::unordered_map<const char *, CUfunction> KitCudaKernelMap;
static KitCudaKernelMap _kitcuda_kernel_map;

extern "C" {

// *** EXPERIMENTAL: First some background. In general, the details of
// picking launch parameters can be a challenge and occupancy is often
// one of the driving factors.  Occupancy is defined as the ratio of
// the number of active warps per multiprocessor to the maximum number
// of active warps. Importantly, having a higher occupancy does not
// guarantee better performance. It is simply a reasonable metric for
// the latency hiding ability of a particular kernel.
//
// This section of calls all deal with different approaches to
// trying to determine launch parameters.  If the
// `_kitcuda_use_occupancy_calc` flag is set to `true` the runtime
// will use CUDA's support for estimating occupancy for a kernel
// function and setting associated launch parameters.  If the flag
// is `false` the runtime will fall back to using either custom
// parameters (set externally) or use a very simple default
// computation that will be hit-or-miss based on the kernel.
//
static bool _kitcuda_use_occupancy_calc = true;
static int _kitcuda_default_max_threads_per_blk = 1024;
static int _kitcuda_default_threads_per_blk =
    _kitcuda_default_max_threads_per_blk;

void __kitcuda_use_occupancy_launch(bool enable) {

  int threads_per_block = 0;
  if (__kitrt_get_env_value("KITCUDA_THREADS_PER_BLOCK", threads_per_block)) {
    if (enable)
      fprintf(
          stderr,
          "kitcuda: KITCUDA_THREADS_PER_BLOCK overriding occupancy lanuch.\n");
    _kitcuda_use_occupancy_calc = false;
  } else
    _kitcuda_use_occupancy_calc = enable;

  if (__kitrt_verbose_mode()) {
    if (enable)
      fprintf(stderr,
              "kitcuda: enabling occupancy-computed launch parameters.\n");
    else
      fprintf(stderr,
              "kitcuda: disabling occupancy-computed launch parameters.\n");
  }
}

void __kitcuda_set_default_max_threads_per_blk(int num_threads) {
  _kitcuda_default_max_threads_per_blk = num_threads;
}

void __kitcuda_set_default_threads_per_blk(int threads_per_blk) {
  if (threads_per_blk > _kitcuda_default_max_threads_per_blk)
    threads_per_blk = _kitcuda_default_max_threads_per_blk;
  _kitcuda_default_threads_per_blk = threads_per_blk;
}

typedef std::unordered_map<std::string, int> KitCudaLaunchParamMap;
static KitCudaLaunchParamMap _kitcuda_launch_param_map;

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


  // TODO: For now we are only launching forall-based kernels.  In this
  // case we tweak the kernel such that it always run in "prefer cache"
  // mode.  We will probably need to revisit this for other kernel
  // kinds (e.g., reductions).
  CU_SAFE_CALL(cuFuncSetCacheConfig_p(cu_func, CU_FUNC_CACHE_PREFER_L1));

  // TODO: Need to handle custom launch parameters on a per-launch
  // use case.  This is now a bit tricky as we can have multiple
  // threads launching and will need to pair custom parameters with
  // a thread id.
  if (_kitcuda_use_occupancy_calc) {
    extern int _kitcuda_device_id;
    int num_multiprocs = 0;
    CU_SAFE_CALL(cuDeviceGetAttribute_p(
        &num_multiprocs, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        _kitcuda_device_id));
    int max_blks_per_multiproc = 0;
    CU_SAFE_CALL(cuDeviceGetAttribute_p(
        &max_blks_per_multiproc,
        CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR, _kitcuda_device_id));
    int max_threads_per_blk = 0;
    CU_SAFE_CALL(cuDeviceGetAttribute_p(
        &max_threads_per_blk, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        _kitcuda_device_id));
    int warp_size = 0;
    CU_SAFE_CALL(cuDeviceGetAttribute_p(
        &warp_size, CU_DEVICE_ATTRIBUTE_WARP_SIZE, _kitcuda_device_id));
    int max_regs_per_blk = 0;
    CU_SAFE_CALL(cuDeviceGetAttribute_p(
        &max_regs_per_blk, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
        _kitcuda_device_id));

    if (__kitrt_verbose_mode()) {
      fprintf(stderr, "device properties:\n");
      fprintf(stderr, "\t- number of SMs: %d\n", num_multiprocs);
      fprintf(stderr, "\t- max blocks per SM: %d\n", max_blks_per_multiproc);
      fprintf(stderr, "\t- max threads per block: %d\n", max_threads_per_blk);
      fprintf(stderr, "\t- max registers per block: %d\n", max_regs_per_blk);
      fprintf(stderr, "\t- warp size: %d\n", warp_size);
    }

    // When the runtime operates in occupancy-launch mode we
    // use CUDA's built-in heuristics to get a SM-based occupancy
    // calculation -- however, it is possible that this path favors
    // parameters that under-subscribe the GPU (by focusing on a
    // single occupancy measure). We attempt to adjust for this case
    // afterwards.
    //
    // Our first step is to see if we have already set launch parameters
    // for the given kernel -- if so, we simply reuse them...
    const char *cu_func_name;
    CU_SAFE_CALL(cuFuncGetName_p(&cu_func_name, cu_func));
    std::string map_entry_name(cu_func_name);
    map_entry_name += std::to_string(trip_count);
    KitCudaLaunchParamMap::iterator lpit =
        _kitcuda_launch_param_map.find(map_entry_name);
    if (lpit != _kitcuda_launch_param_map.end()) {
      threads_per_blk = lpit->second;
    } else {
      int min_grid_size; // currently ignored...
      CU_SAFE_CALL(cuOccupancyMaxPotentialBlockSizeWithFlags_p(
          &min_grid_size, &threads_per_blk, cu_func, 0, 0, 0,
          CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE));

      int block_count = (trip_count + threads_per_blk - 1) / threads_per_blk;
      float sm_load = ((float)block_count / num_multiprocs) / num_multiprocs;

      if (__kitrt_verbose_mode()) {
        fprintf(stderr, "\toccupancy-driven threads-per-block: %d\n", threads_per_blk);
        fprintf(stderr, "\tminimum grid size (# of blocks): %d\n", min_grid_size);
        fprintf(stderr, "\tkernel trip count: %ld\n", trip_count);
        fprintf(stderr, "\tcomputed grid size: %d blocks\n", block_count);
        fprintf(stderr, "\toverall SM utilization %f of %d\n", 
                sm_load, num_multiprocs);
      }

      while (sm_load < 0.7) {
	if (threads_per_blk < 16)
	  break;
        threads_per_blk = threads_per_blk / 2;	
        if (__kitrt_verbose_mode()) {
          fprintf(stderr, "\t**** SMs are under-utilized.  Creating more blocks...\n");
	  fprintf(stderr, "\t\tthreads-per-block = %d\n", threads_per_blk);
	}
        block_count = (trip_count + threads_per_blk - 1) / threads_per_blk;
        sm_load = ((float)block_count / num_multiprocs) / num_multiprocs;
        if (sm_load > 1)
          threads_per_blk = threads_per_blk * 2 * 0.25;
	
        if (__kitrt_verbose_mode()) {
          fprintf(stderr, "\t\tnew sm compute load: %f\n", sm_load);
          fprintf(stderr, "\t\tadjusted grid size: %d blocks\n", block_count);
        }
      }
      threads_per_blk += threads_per_blk & 1;

      uint64_t total_insts = inst_mix->num_memory_ops + 
                             inst_mix->num_flops + 
                             inst_mix->num_iops;
      float mem_ratio = float(inst_mix->num_memory_ops) / total_insts;
      if (__kitrt_verbose_mode())
        fprintf(stderr, "\t\t***memory ops are %d%% of all instructions.\n", 
                int(mem_ratio*100.0));

      int reg_count;
      CU_SAFE_CALL(cuFuncGetAttribute_p(&reg_count, CU_FUNC_ATTRIBUTE_NUM_REGS,
                                        cu_func));
      const int max_regs_per_thread = 255;
      int reg_use = (float(reg_count) / max_regs_per_thread) * 100;
      if (__kitrt_verbose_mode())
        fprintf(stderr, "\t\tkernel uses %d%% of available registers (per thread).\n", 
                reg_use);

      if (reg_use > 45) {
        if (__kitrt_verbose_mode()) 
          fprintf(stderr, "moderate/high register usage -- increasing block count...\n");
        threads_per_blk = threads_per_blk / 2;
      }

      if (threads_per_blk > 32) {
        threads_per_blk =
            ((threads_per_blk - warp_size - 1) / warp_size) * warp_size;
      }

      blks_per_grid = (trip_count + threads_per_blk - 1) / threads_per_blk;

      if (__kitrt_verbose_mode()) {
        fprintf(stderr, "\tadjusted threads-per-block: %d\n", threads_per_blk);
        fprintf(stderr, "\tadjusted grid size: %d blocks\n", blks_per_grid);
      }
      _kitcuda_launch_param_map[map_entry_name] = threads_per_blk;
    }
  } else
    threads_per_blk = _kitcuda_default_threads_per_blk;

  // Need to round-up based on array size/trip count.
  blks_per_grid = (trip_count + threads_per_blk - 1) / threads_per_blk;
  KIT_NVTX_POP();
}

void __kitcuda_launch_kernel(const void *fat_bin, const char *kernel_name,
                             void **kern_args, uint64_t trip_count,
                             int threads_per_blk,
                             const KitCudaInstMix *inst_mix) {
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
  if (threads_per_blk == 0)
    __kitcuda_get_launch_params(trip_count, cu_func, threads_per_blk,
                                blks_per_grid, inst_mix);
  else
    blks_per_grid = (trip_count + threads_per_blk - 1) / threads_per_blk;

  if (__kitrt_verbose_mode()) {
    fprintf(stderr, "kitcuda: kernel '%s' launch parameters:\n", kernel_name);
    fprintf(stderr, "  blocks: %d, 1, 1\n", blks_per_grid);
    fprintf(stderr, "  threads: %d, 1, 1\n", threads_per_blk);
    fprintf(stderr, "  trip count: %ld\n\n", trip_count);
  }

  CUstream cu_stream = __kitcuda_get_thread_stream();
  CU_SAFE_CALL(cuLaunchKernel_p(cu_func, blks_per_grid, 1, 1, threads_per_blk,
                                1, 1,
                                0, // shared mem size
                                cu_stream, kern_args, NULL));
  KIT_NVTX_POP();
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
  _kitcuda_module_map_mutex.lock();
  KitCudaModuleMap::iterator modit = _kitcuda_module_map.find(fat_bin);
  if (modit == _kitcuda_module_map.end()) {
    // Create a supporting CUDA module and "register" the fat binary
    // image in the map...
    CU_SAFE_CALL(cuModuleLoadData_p(&cu_module, fat_bin));
    _kitcuda_module_map[fat_bin] = cu_module;
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
