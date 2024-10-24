/*
 *===- launching.cpp - HIP kernel launching support   ---------------------===
 *
 * Copyright (c) 2021, 2023, 2025 Los Alamos National Security, LLC.
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
#include "kithip_rtinfo.h"

#include <mutex>
#include <stdint.h>
#include <stdio.h>
#include <string>

// We maintain a map of modules (think fat binary) to avoid having to
// reprocess them over and over again.  This mapping actually goes
// directly from the compiler generated fat binary to the hip runtime
// module structure required to look up kernel calls.  The hope here
// is that the map acccess is actually faster than repeatedly loading
// a module, searching it, and returning a kernel.
#include <unordered_map>
typedef std::unordered_map<const void *, hipModule_t> KitHipModuleMap;
static KitHipModuleMap _kithip_module_map;
static std::mutex _kithip_module_map_mutex;

// Like the module map above, the runtime maintains a map from kernel
// names to kernel functions (again avoiding a hip-driven lookup
// process).
typedef std::unordered_map<const char *, hipFunction_t> KitHipKernelMap;
static KitHipKernelMap _kithip_kernel_map;

extern "C" {

// This max threads per block setting helps to drive register usage
// from the compiler side.  In this case the runtime will clamp the
// max threads per block values to ensure that we do not exceed the
// limits used when compiling...
void __kithip_set_max_threads_per_blk(int nthreads) {
  using namespace kithip_rt;
  // Check hardware bounds...
  if (nthreads > maxThreadsPerBlock()) {
    fprintf(stderr, "kitrt[hip]: requested max threads per block exceeds "
                    "hardware limits!\n");
    fprintf(stderr,
            "  %d > %d -- clamping requested value to hardware limit.\n",
            nthreads, maxThreadsPerBlock());
    // to assert or not assert???
    kitSetMaxThreadsPerBlock(maxThreadsPerBlock());
  } else {
    kitSetMaxThreadsPerBlock(nthreads);
  }
}

// This is a specific setting that will impact the value used by kernel
// launches from this call forward in the call chain.  The runtime will
// use this value directly for launches that lack any specific settings
// from compiled code.
void __kithip_set_threads_per_blk(int nthreads) {
  using namespace kithip_rt;
  if (nthreads > maxThreadsPerBlock()) {
    fprintf(
        stderr,
        "kitrt[hip]: requested threads per block exceeds hardware limits!\n");
    fprintf(stderr,
            "  %d > %d -- clamping requested value to hardware limit.\n",
            nthreads, maxThreadsPerBlock());
    // to assert or not assert???
    kitSetThreadsPerBlock(maxThreadsPerBlock());
  } else {
    kitSetThreadsPerBlock(nthreads);
  }
}

// This call is an entry point for the compiler as it must go along with
// code generation details.  It basically moves the launch threads to the
// y-axis from the default x-axis.
void __kithip_enable_ylaunch() { kithip_rt::setYLaunch(true); }

// The runtime maintains a map of the kernel launch paramters so they are
// not recalculated after the first determination.  As with most other
// maps maintained by the runtime the overarching goal is to reduce overhead
// calls into the hip runtime api and/or additional steps by the runtime that
// might also incur more significant costs.
//
// TODO: For small programs the costs here might be on par with runtime calls
// but for large code bases it could be more significant.  We have not yet
// fully evaluated the trade-offs here....
typedef std::unordered_map<std::string, int> KitHipLaunchParamMap;
static KitHipLaunchParamMap _kithip_launch_param_map;

namespace {

struct kithip_kern_attrs_t {
  int numRegisters;
  int constSize;
  int sharedSize;
  int maxThreadsPerBlock;
};

void __kithip_get_kern_attrs(kithip_kern_attrs_t &attrs, hipFunction_t kfunc) {
  HIP_SAFE_CALL(hipFuncGetAttribute(&attrs.numRegisters,
                                    HIP_FUNC_ATTRIBUTE_NUM_REGS, kfunc));
  HIP_SAFE_CALL(hipFuncGetAttribute(
      &attrs.constSize, HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, kfunc));
  HIP_SAFE_CALL(hipFuncGetAttribute(
      &attrs.sharedSize, HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, kfunc));
  HIP_SAFE_CALL(hipFuncGetAttribute(&attrs.maxThreadsPerBlock,
                                    HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                    kfunc));
}

// We currently borrow this from the kitcuda runtime... It should probably
// move up one level to kitrt...
extern int next_lowest_factor(int n, int m);

int __kithip_reg_analysis(int threads_per_blk, int regs_per_thread,
                          int max_regs_per_block) {
  using namespace kithip_rt;

  int total_rcount = threads_per_blk * regs_per_thread;
  float perused = total_rcount / float(max_regs_per_block);

  int warp_size = 0;
  HIP_SAFE_CALL(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize,
                                      deviceID()));

  int warps_per_sm = 40;
  int min_tpb = warp_size * warps_per_sm;
  if (perused > 0.90) {
    threads_per_blk -= min_tpb;
    total_rcount = threads_per_blk * regs_per_thread;
    perused = total_rcount / float(max_regs_per_block);
    // fprintf(
    //     stderr,
    //     "ADJUSTED to use %f of available registers w/ %d
    //     threads-per-block.\n", (float)total_rcount /
    //     (float)max_regs_per_block * 100.0, threads_per_blk);
  }

  if (threads_per_blk > maxThreadsPerBlock())
    threads_per_blk = maxThreadsPerBlock();

  return threads_per_blk;
}

} // namespace

void __kithip_get_launch_params(size_t trip_count, hipFunction_t kfunc,
                                const char *kfunc_name, int &threads_per_blk,
                                int &blks_per_grid,
                                const KitRTInstMix *inst_mix) {
  assert(kfunc != nullptr && "__kithip_get_launch_params(): null kernel!");
  using namespace kithip_rt;

  // We treat a kernel function and its trip count as a unique identifier
  // for a particular launch.  Instead of repeating these calculations at
  // every launch we keep a map from this pair to the launch parameters.
  std::string map_entry_name(kfunc_name);
  map_entry_name += std::to_string(trip_count);
  KitHipLaunchParamMap::iterator lpit =
      _kithip_launch_param_map.find(map_entry_name);
  if (lpit != _kithip_launch_param_map.end()) {
    // use previously determined parameters.
    threads_per_blk = lpit->second;
  } else {
    kithip_kern_attrs_t attrs;
    __kithip_get_kern_attrs(attrs, kfunc);
    if (__kitrt_verbose_mode()) {
      fprintf(stderr, "kitrt[hip]: kernel attributes:\n");
      fprintf(stderr, "  - name: %s\n", kfunc_name);
      fprintf(stderr, "  - num registers: %d\n", attrs.numRegisters);
      fprintf(stderr, "  - const size: %d\n", attrs.constSize);
      fprintf(stderr, "  - shared size: %d\n", attrs.sharedSize);
      fprintf(stderr, "  - max threads per block: %d\n",
              attrs.maxThreadsPerBlock);
      if (inst_mix != nullptr) {
        fprintf(stderr, "  - instruction mix:\n");
        fprintf(stderr, "     - # memory operations: %ld\n",
                inst_mix->numMemoryOps);
        fprintf(stderr, "     - # floating point ops: %ld\n",
                inst_mix->numFlops);
        fprintf(stderr, "     - # integer ops: %ld\n", inst_mix->numIntOps);
        fprintf(stderr, "     - # other ops: %ld\n", inst_mix->numOtherOps);
        size_t total_ops =
            inst_mix->numIntOps + inst_mix->numFlops + inst_mix->numOtherOps;
        float ops_per_memop = float(total_ops) / inst_mix->numMemoryOps;
        fprintf(stderr, "     - ops / memory op: %3.2f\n", ops_per_memop);
      }
    }

    int nblocks = 0;
    // Work with what we know about the kernel to help determine
    // an appropriate set of launch parameters. As a default starting
    // point we use the hip occupancy heuristic to get an initial
    // occupancy-driven threads-per-block figure.  Note that in
    // many (most? all?) cases this can be very dependent upon the
    // compilation and limits placed on launches there.
    int min_grid_size;
    HIP_SAFE_CALL(hipModuleOccupancyMaxPotentialBlockSize(
        &min_grid_size, &threads_per_blk, kfunc, 0, 0));
    blks_per_grid = (trip_count + threads_per_blk - 1) / threads_per_blk;
    if (__kitrt_verbose_mode()) {
      fprintf(stderr, "*** BENGIN LAUNCH\n");
      fprintf(stderr, "kitrt[hip]: occpancy kernel launch parameters:\n");
      fprintf(stderr, "  threads per block: %d\n", threads_per_blk);
      fprintf(stderr, "  tmin_grid_size: %d\n", min_grid_size);
      fprintf(stderr, "  blocks per grid: %d\n", blks_per_grid);
      // Estimate how many compute units we can use with the provided
      // threads-per-block value.
      int block_count = (trip_count + threads_per_blk - 1) / threads_per_blk;
      float blks_per_cu = (float(block_count) / multiProcessorCount());
      fprintf(stderr, "blocks per cu = %d\n", int(blks_per_cu));
      fprintf(stderr, "cu load = %f\n", blks_per_cu * 100.0f);
      fprintf(stderr, "percent of max threads/cu = %f\n",
              (float(blks_per_cu) / maxThreadsPerMultiProcessor()) * 100.0f);
    }

    /*
    // Compare the calculated blocks-per-compute-unit with the hardware
    // specs.  If we have failed to maximize the work across all compute
    // units we reduce the threads-per-block (increase the block count)
    // until we do.
    while (blks_per_cu < _kithip_dev_max_blks_per_cu &&
    threads_per_blk >= _kithip_dev_warp_size) {
    threads_per_blk = threads_per_blk - _kithip_dev_warp_size;
    block_count = (trip_count + threads_per_blk - 1) / threads_per_blk;
    blks_per_cu = float(block_count) / _kithip_dev_num_cus;
    }
    //fprintf(stderr, "kitrt[hip]: refined kernel launch parameters:\n");
    //fprintf(stderr, "  threads per block: %d\n", threads_per_blk);
    //fprintf(stderr, "  blocks per grid: %d\n", blks_per_grid);
    //fprintf(stderr, "*** END LAUNCH\n");
    */

    _kithip_launch_param_map[map_entry_name] = threads_per_blk;
  }

  blks_per_grid = (trip_count + threads_per_blk - 1) / threads_per_blk;
}

// Compiler interface notes: the 'threads_per_blk' parameter passed
// into the launch is our indication if the compiler or runtime
// should be in charge of determining the details of the launch
// parameters.  If the value is <= 0 then the runtime is responsible
// for determining the launch parameters, otherwise the launch has
// been explicitly set by the compiler (via command line options,
// user provided attributes, etc.).
//
// Environment interface notes: if the KITHIP_THREADS_PER_BLOCK
// environment variable is set it will override the threads_per_blk
// parameter when the parameter is <= 0.  Otherwise, the
// compiler-provided value for the threads-per-block value will be
// used.
void *__kithip_launch_kernel(const void *fat_bin, const char *kernel_name,
                             void **kern_args, uint64_t trip_count,
                             int threads_per_blk, const KitRTInstMix *inst_mix,
                             void *opaque_stream) {
  assert(fat_bin && "kitrt[hip]: launch with null fat binary!");
  assert(kernel_name && "kitrt[hip]: launch with null name!");
  assert(kern_args && "kitrt[hip]: launch with null args!");
  assert(trip_count != 0 && "kitrt[hip]: launch with zero trips!");

  using namespace kithip_rt;

  HIP_SAFE_CALL(hipSetDevice(deviceID()));

  // There are certain paths from the current kitsune feature set
  // where it is possible for multiple threads to launch GPU kernels.
  // For this reason we have to take some care with the maps used by
  // the runtime to avoid races... This is currently the sole reason
  // for the mutexes within the runtime code...

  // LOCK
  hipFunction_t kern_func;
  _kithip_module_map_mutex.lock();
  KitHipKernelMap::iterator kernit = _kithip_kernel_map.find(kernel_name);
  if (kernit == _kithip_kernel_map.end()) {
    // We have not encountered this kernel before. The next step is to
    // check to see if we have already created a module that corresponds
    // to the fat binary...
    hipModule_t hip_module;
    KitHipModuleMap::iterator modit = _kithip_module_map.find(fat_bin);
    if (modit == _kithip_module_map.end()) {
      // Nope, we need to create the module.
      HIP_SAFE_CALL(hipModuleLoadData(&hip_module, fat_bin));
      _kithip_module_map[fat_bin] = hip_module;
    } else {
      hip_module = modit->second;
    }

    // Now we can look up the kernel function in the module and save it so
    // we can skip hip api calls for the module and kernel searches for the
    // next go-around...
    HIP_SAFE_CALL(hipModuleGetFunction(&kern_func, hip_module, kernel_name));
    _kithip_kernel_map[kernel_name] = kern_func;
  } else {
    kern_func = kernit->second;
  }
  _kithip_module_map_mutex.unlock();
  // UNLOCK

  // Next we need to sort out how we should determine the launch parameters.
  // As mentioned above, any specific guidance from the compiler will set an
  // explicit value for the threads-per-block.  If that value is <= 0 there
  // is no guiance from the compiler/source code so the runtime will
  // determine a (hopefully suitable) value.

  int blks_per_grid;
  if (threads_per_blk <= 0) {
    __kithip_get_launch_params(trip_count, kern_func, kernel_name,
                               threads_per_blk, blks_per_grid, inst_mix);
  } else {
    // Sanity check the compiler's / programmer's guidance...
    if (threads_per_blk > maxThreadsPerBlock()) {
      fprintf(stderr,
              "kitrt[hip]: WARNING! Requested threads-per-block value execeeds "
              "hardware limits.  Adjusting to match limit...\n");
      threads_per_blk = maxThreadsPerBlock();
    }
  }

  // With the threads-per-block value nailed down, we can sort out the
  // number of blocks per grid.
  blks_per_grid = (trip_count + threads_per_blk - 1) / threads_per_blk;

  if (__kitrt_verbose_mode()) {
    fprintf(stderr, "kitrt[hip]: kernel '%s' launch parameters:\n",
            kernel_name);
    fprintf(stderr, "  kernel: %s\n", kernel_name);
    fprintf(stderr, "  blocks: [%d, 1, 1]\n", blks_per_grid);
    if (useYLaunch())
      fprintf(stderr, "  threads: [1, %d, 1]\n", threads_per_blk);
    else
      fprintf(stderr, "  threads: [%d, 1, 1]\n", threads_per_blk);
    fprintf(stderr, "  stream: %p\n", opaque_stream);
  }

  // There is a handshake between the compiler's codegen steps and the
  // streams associatd with kernel launches.  Much of the logic about
  // how streams are managed are part of the codegen but in a nutshell
  // the runtime either receives a stream or will create a new for this
  // launch (the compiler will handle the generation of sync calls and
  // stream bindings).
  hipStream_t hip_stream = nullptr;
  if (opaque_stream) {
    hip_stream = (hipStream_t)opaque_stream;
  } else {
    hip_stream = (hipStream_t)__kithip_get_thread_stream();
  }

  if (!useYLaunch()) {
    HIP_SAFE_CALL(hipModuleLaunchKernel(kern_func, blks_per_grid, 1, 1,
                                        threads_per_blk, 1, 1,
                                        0, // shared mem size
                                        hip_stream, kern_args, NULL));
  } else {
    HIP_SAFE_CALL(hipModuleLaunchKernel(kern_func, blks_per_grid, 1, 1, 1,
                                        threads_per_blk, 1,
                                        0, // shared mem size
                                        hip_stream, kern_args, NULL));
  }
  return (void *)hip_stream;
}

void *__kithip_get_global_symbol(void *fat_bin, const char *sym_name) {
  assert(fat_bin && "kitrt[hip]: null fat binary!");
  assert(sym_name && "kitrt[hip]: null symbol name!");

  hipModule_t hip_module;

  // LOCK
  _kithip_module_map_mutex.lock();
  KitHipModuleMap::iterator modit = _kithip_module_map.find(fat_bin);
  if (modit == _kithip_module_map.end()) {
    HIP_SAFE_CALL(hipModuleLoadData(&hip_module, fat_bin));
    _kithip_module_map[fat_bin] = hip_module;
  } else {
    hip_module = modit->second;
  }

  _kithip_module_map_mutex.lock();
  // UNLOCK

  // NOTE: The device pointer and size ('bytes') parameters for the
  // call to cuModuleGetGlobal are optional.  To simplify the compiler's
  // code generation details we ignore the size parameter...
  hipDeviceptr_t sym_ptr;
  size_t bytes;
  HIP_SAFE_CALL(hipModuleGetGlobal(&sym_ptr, &bytes, hip_module, sym_name));
  return sym_ptr;
}

} // extern C
