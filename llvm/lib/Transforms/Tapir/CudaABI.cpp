//===- CudaABI.cpp - Lower Tapir to the Kitsune CUDA target ----*- C++ -*-===//
//
//
//                     The LLVM Compiler Infrastructure
//
// Copyright (c) 2021, 2023, 2025 Los Alamos National Security, LLC.
//  All rights reserved.
//
// Copyright 2021, 2023, 2025. Los Alamos National Security, LLC. This
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
// Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
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
//  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
//  AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
//  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
//  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
//
//===----------------------------------------------------------------------===
//

#include "llvm/Transforms/Tapir/CudaABI.h"
#include "kitsune/Config/config.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/FMF.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SmallVectorMemoryBuffer.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/Inliner.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Tapir/Outline.h"
#include "llvm/Transforms/Tapir/TapirGPUUtils.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

using namespace llvm;

#define DEBUG_TYPE "cuabi"

// For some background material see the NVPTX target documentation
// at https://llvm.org/docs/NVPTXUsage.html.
//
// This transformation is carrying out the prep to convert Tapir to a
// kernel module suitable for codegen using the NVPTX target.
//
// NOTE: We currently do not support the full range of GPU
// architectures supported by the NVPTX backend -- this is primarily
// due to a lack of systems for testing of these GPUs.
namespace {

const std::string CUABI_PREFIX = "kitcu";
const std::string CUABI_KERNEL_LOOP_NAME_PREFIX = "kitcu_loop_";

// TO BE DEPRECATED
#ifndef _CUDAABI_DEFAULT_TARGET_ARCH
#define _CUDAABI_DEFAULT_TARGET_ARCH "sm_80"
#endif
cl::opt<std::string>
    GPUArch("cuabi-arch", cl::init(_CUDAABI_DEFAULT_TARGET_ARCH), cl::NotHidden,
            cl::desc("Target GPU architecture for CUDA ABI transformation."
                     "(default: " _CUDAABI_DEFAULT_TARGET_ARCH));

// Set a specific optimization level for the transformation's pass over the
// created "kernel-module".  By default this level will mirror that of the
// frontend but can be set specifically here -- this is primarily useful
// for exploring various details of levels between those operating on the
// Tapir IR and those after the transformation to GPU-friendly LLVM IR.
cl::opt<int> OptLevel("cuabi-opt-level", cl::init(-1), cl::Hidden,
                      cl::desc("Specify the GPU kernel optimization level."));

// Similar to the optimization level above it is possible to separately
// control the optimization level used by ptxas for creating the GPU
// binary code.  By default this flag will mirror that of the frontend's
// optimization level.  This is primarily intended to help explor the
// various aspects of code generation details in the kitsune+tapir
// pipeline(s).
cl::opt<int>
    PTXasOptLevel("cuabi-ptxas-opt-level", cl::init(-1), cl::Hidden,
                  cl::desc("Specify the optimization level for ptxas."));

// TO BE DEPRECATED
cl::opt<int> HostOptLevel(
    "cuabi-host-opt-level", cl::init(-1), cl::NotHidden,
    cl::desc(
        "The optimization level for an experimental pass over the transformed "
        "host-side code."));

// Enabled/disable compiler generated code for issuing data prefetch calls
// prior to the launch of each kernel.  The associated prefetch calls are
// directly to the kitsune runtime that will determine if the prefetch is
// "valid" (primarily this means a pointer matches the known memory
// allocations made by the runtime.
cl::opt<bool> CodeGenPrefetch("cuabi-prefetch", cl::init(true), cl::NotHidden,
                              cl::desc("Enable generation of calls to do data "
                                       "prefetching for managed memory."));

// Request that the runtime carry out an extra set of steps to attempt to
// refine the launch parameters of kernels.  In this mode of operation the
// compiler will provide some compile-time information onto the runtime for
// assisting in the analysis an refinement of launches.
cl::opt<bool> RefineLaunches(
    "cuabi-refine-launches", cl::init(true), cl::Hidden,
    cl::desc("Enable runtime's refinement of launch parameters."));

// This is meant to be a factor used for additional kernel optimizations
// but it currently unimplemented.  It should be left in its default
// state...
cl::opt<unsigned> DefaultGrainSize(
    "cuabi-default-grainsize", cl::init(1), cl::Hidden,
    cl::desc("The default grain size used by the transform "
             "when analysis fails to determine one. (default=1)"));

// Request that the transformation pass leave a set of files in place
// during operation.  Obviously most helpful for those trying to debug
// the transformation...
cl::opt<bool>
    KeepFiles("cuabi-keep-files", cl::init(false), cl::Hidden,
              cl::desc("Keep a set of intermediate files on disk during the "
                       "execution of the transformation. (default=false)"));

// Enable a level of verbose operation during both compilation and
// runtime.  This is primarily used to gain some insight into the
// transformations most important stages -- detailed debugging
// output should use -mllvm -debug-only-cuabi.
cl::opt<bool>
    VerboseMode("cuabi-verbose", cl::init(false), cl::NotHidden,
                cl::desc("Enable verbose mode for cuda toolchain components. "
                         "(default=false)"));

// The default mode of the transformation is to embed a single fat binary
// image for the selected target architecture.  With this flag set the
// PTX version of the code will also be embedded into the fat binary.
cl::opt<bool>
    EmbedPTXInFatbinaries("cuabi-embed-ptx", cl::init(false), cl::Hidden,
                          cl::desc("Embed intermediate PTX files in the "
                                   "fatbinaries used by the CUDA ABI "
                                   "transformation."));

// Set the default threads per block value used to launch generated kernels
// that do not have attributed launch specifications.  Note that this will
// also currently override any further refinement of kernel launch parameters
// by the runtime.
cl::opt<unsigned>
    DefaultThreadsPerBlock("cuabi-threads-per-block", cl::init(0), cl::Hidden,
                           cl::desc("Set the runtime system's value for "
                                    "the default number of threads per block. "
                                    "(default: 0 - do not override)"));
// TODO: Currently unused...
cl::opt<unsigned>
    DefaultBlocksPerGrid("cuabi-blocks-per-grid", cl::init(0), cl::Hidden,
                         cl::desc("Hard-code the runtime system's value for "
                                  "the number of blocks per grid in kernel "
                                  "launches. (default: 0 = disabled)"));

// Set the maximum number of threads that allowed in any generated kernel
// launches.  This number can be used to fine-tune kernel code generation
// and can impact register allocation and other details.  This will impact
// both code generation and runtime behaviors.
cl::opt<unsigned> MaxThreadsPerBlock(
    "cuabi-max-threads-per-block", cl::init(1024), cl::Hidden,
    cl::desc("Set the maximum value for the threads per "
             "block.  For register allocation tweaks. (default: 1024)"));

// Enable/Disable flush denorms-to-zero code generation.
cl::opt<bool>
    FTZCodeGen("cuabi-ftz", cl::init(false), cl::NotHidden,
               cl::desc("Use flush-denorms-to-zero code generation paths"));

// Return the matching 'compute_*' target for the given 'sm_*' architecture.
std::string virtualArchForCudaArch(StringRef Arch) {
  // TODO: We've scaled back from the full suite of Nvidia targets
  // based on systems we have available for testing. We need to
  // also cross-check with CUDA versions and what architectures
  // are continuing to be supported.
  std::string VirtArch = llvm::StringSwitch<std::string>(Arch)
                             .Case("sm_60", "compute_60") // Pascal
                             .Case("sm_61", "compute_61") //
                             .Case("sm_62", "compute_62") //
                             .Case("sm_70", "compute_70") // Volta
                             .Case("sm_72", "compute_72") //
                             .Case("sm_75", "compute_75") // Turing
                             .Case("sm_80", "compute_80") // Ampere
                             .Case("sm_86", "compute_86") //
                             .Case("sm_87", "compute_87") //
                             .Case("sm_90", "compute_90") // Hopper
                             .Default("unknown");

  if (VirtArch == "unknown") {
    errs() << "kitsune[cuabi]: unsupported architecture target '" << Arch
           << "'.\n"
           << "  Support is only available for sm_60 through sm_90.\n";
    report_fatal_error("kitsune[cuabi]: fatal error!");
  }

  if (VerboseMode)
    errs() << "kitsune[cuabi]: architecture target '" << Arch << "' ("
           << VirtArch << ").\n";
  return VirtArch;
}

// Given a cuda version, return a matching PTX version.  Note that this does
// not exactly follow the same versioning details in CUDA releases. Instead
// it is dependent upon the PTX version support available in the NVPTX
// backend.
//
// We are currently dependent upon the cmake configuration process to provide
// the CUDA version info.
std::string PTXVersionFromCudaVersion() {
#ifdef CUDATOOLKIT_VERSION
  std::string CudaVersion;
  raw_string_ostream ss(CudaVersion);
  ss << CUDATOOLKIT_VERSION_MAJOR << "." << CUDATOOLKIT_VERSION_MINOR;
#else
#pragma error("cuabi: CUDA Toolkit version info required!")
#endif
  std::string PTXVersionStr =
      llvm::StringSwitch<std::string>(CudaVersion)
          // TODO: These CUDA to PTX version translations have
          // to be watched between both CUDA and LLVM releases.
          // It is common for LLVM to lag behind CUDA in these
          // details.  The following is current for LLVM 19.x.
          //
          // For sync'ing up with CUDA check the PTX documentation
          // and release notes that provide details for the version
          // mapping:
          //   https://docs.nvidia.com/cuda/parallel-thread-execution/#release-notes
          //
          // These details will then have to be cross-checked with the
          // version detail in the NVPTX backend source.
          .Case("10.0", "+ptx63")
          .Case("10.1", "+ptx64")
          .Case("10.2", "+ptx65")
          .Case("10.0", "+ptx63")
          .Case("11.0", "+ptx70")
          .Case("11.1", "+ptx71")
          .Case("11.2", "+ptx72")
          .Case("11.3", "+ptx72")
          .Case("11.4", "+ptx72")
          .Case("11.5", "+ptx72")
          .Case("11.6", "+ptx76")
          .Case("11.7", "+ptx77")
          .Case("11.8", "+ptx78")
          .Case("12.0", "+ptx80")
          .Case("12.1", "+ptx81")
          .Case("12.2", "+ptx82")
          .Case("12.3", "+ptx83")
          .Case("12.4", "+ptx84")
          .Case("12.5", "+ptx85")
          .Case("12.6", "+ptx85")
          .Case("12.7", "+ptx85")
          .Case("12.8", "+ptx85")
          .Case("12.9", "+ptx85")
          .Default("");

  if (PTXVersionStr == "unknown") {
    errs() << "kitsune[cuabi]: no matching PTX version found for cuda toolkit: "
           << CudaVersion << "\n.  Check that PTX mapping is up-to-date.\n";
    report_fatal_error("kitsune[cuabi]: fatal error!");
  }

  if (VerboseMode)
    errs() << "kitsune[cuabi]: CUDA version " << CudaVersion
           << ", PTX version: " << PTXVersionStr << ".\n";
  return PTXVersionStr;
}

bool isDeclaration(const Function *F) { return F->size() == 0; }

} // namespace

/// Static ID for kernel naming -- each encountered kernel (loop)
/// during compilation will receive a unique ID.  TODO: This is
/// a not so great naming mechanism and certainly not thread safe...
unsigned CudaLoop::NextKernelID = 0;

CudaLoop::CudaLoop(Module &M, Module &KernelModule, const std::string &KN,
                   CudaABI *T, bool MakeUniqueName)
    : LoopOutlineProcessor(M, KernelModule), TTarget(T), KernelName(KN),
      KernelModule(KernelModule) {

  nonMicrosoftDemangle(KN, KernelName);
  KernelName += "_" + Twine(NextKernelID).str();
  NextKernelID++;

  LLVM_DEBUG(dbgs() << "debug[cuabi]: creating a cuda loop outliner.\n"
                    << "  - target kernel name: " << KernelName << "\n");

  LLVMContext &Ctx = KernelModule.getContext();
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);
  Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *VoidPtrTy = PointerType::getUnqual(Ctx);
  PointerType *VoidPtrPtrTy = VoidPtrTy->getPointerTo();
  PointerType *CharPtrTy = PointerType::getUnqual(Ctx);

  // Thread index values -- equivalent to Cuda's builtins:  threadIdx.[x,y,z].
  CUThreadIdxX = Intrinsic::getDeclaration(&KernelModule,
                                           Intrinsic::nvvm_read_ptx_sreg_tid_x);
  CUThreadIdxY = Intrinsic::getDeclaration(&KernelModule,
                                           Intrinsic::nvvm_read_ptx_sreg_tid_y);
  CUThreadIdxZ = Intrinsic::getDeclaration(&KernelModule,
                                           Intrinsic::nvvm_read_ptx_sreg_tid_z);

  // Block index values -- equivalent to Cuda's builtins: blockIndx.[x,y,z].
  CUBlockIdxX = Intrinsic::getDeclaration(
      &KernelModule, Intrinsic::nvvm_read_ptx_sreg_ctaid_x);
  CUBlockIdxY = Intrinsic::getDeclaration(
      &KernelModule, Intrinsic::nvvm_read_ptx_sreg_ctaid_y);
  CUBlockIdxZ = Intrinsic::getDeclaration(
      &KernelModule, Intrinsic::nvvm_read_ptx_sreg_ctaid_z);

  // Block dimensions -- equivalent to Cuda's builtins: blockDim.[x,y,z].
  CUBlockDimX = Intrinsic::getDeclaration(&KernelModule,
                                          Intrinsic::nvvm_read_ptx_sreg_ntid_x);
  CUBlockDimY = Intrinsic::getDeclaration(&KernelModule,
                                          Intrinsic::nvvm_read_ptx_sreg_ntid_y);
  CUBlockDimZ = Intrinsic::getDeclaration(&KernelModule,
                                          Intrinsic::nvvm_read_ptx_sreg_ntid_x);

  // Grid dimensions -- equivalent to Cuda's builtins: gridDim.[x,y,z].
  CUGridDimX = Intrinsic::getDeclaration(
      &KernelModule, Intrinsic::nvvm_read_ptx_sreg_nctaid_x);
  CUGridDimY = Intrinsic::getDeclaration(
      &KernelModule, Intrinsic::nvvm_read_ptx_sreg_nctaid_y);
  CUGridDimZ = Intrinsic::getDeclaration(
      &KernelModule, Intrinsic::nvvm_read_ptx_sreg_nctaid_z);

  // NVVM-centric barrier -- equivalent to Cuda's __sync_threads().
  CUSyncThreads =
      Intrinsic::getDeclaration(&KernelModule, Intrinsic::nvvm_barrier0);

  // Get entry points into the Cuda-centric portion of the Kitsune GPU runtime.
  KernelInstMixTy = StructType::get(Int64Ty,  // number of memory ops.
                                    Int64Ty,  // number of floating point ops.
                                    Int64Ty,  // number of integer ops.
                                    Int64Ty); // number of other ops.
  KitCudaLaunchFn = M.getOrInsertFunction(
      "__kitcuda_launch_kernel",
      VoidPtrTy,                       // return an opaque stream
      VoidPtrTy,                       // fat-binary
      VoidPtrTy,                       // kernel name
      VoidPtrPtrTy,                    // arguments
      Int64Ty,                         // trip count
      Int32Ty,                         // threads-per-block
      KernelInstMixTy->getPointerTo(), // instruction mix info
      VoidPtrTy);                      // opaque cuda stream

  KitCudaMemPrefetchFn =
      M.getOrInsertFunction("__kitcuda_mem_gpu_prefetch",
                            VoidPtrTy,  // return an opaque stream
                            VoidPtrTy,  // pointer to prefetch
                            VoidPtrTy); // opaque stream
  KitCudaGetGlobalSymbolFn =
      M.getOrInsertFunction("__kitcuda_get_global_symbol",
                            Int64Ty,    // return the device pointer for symbol.
                            VoidPtrTy,  // fat binary
                            CharPtrTy); // symbol name
  KitCudaMemcpySymbolToDeviceFn =
      M.getOrInsertFunction("__kitcuda_memcpy_symbol_to_device",
                            VoidTy,   // no return
                            Int32Ty,  // host pointer
                            Int64Ty,  // device pointer
                            Int64Ty); // number of bytes to copy
  LLVM_DEBUG(dbgs() << "  - done.\n");
}

CudaLoop::~CudaLoop() {
  LLVM_DEBUG(dbgs() << "debug[cuabi]: destroying loop outliner for kernel '"
                    << KernelName << "'.\n");
}

void CudaLoop::setupLoopOutlineArgs(Function &F, ValueSet &HelperArgs,
                                    SmallVectorImpl<Value *> &HelperInputs,
                                    ValueSet &InputSet,
                                    const SmallVectorImpl<Value *> &LCArgs,
                                    const SmallVectorImpl<Value *> &LCInputs,
                                    const ValueSet &TLInputsFixed) {

  LLVM_DEBUG(dbgs() << "debug[cuabi]: setting up loop outline arguments...\n");

  // Add the loop control inputs -- the first parameter defines
  // the extent of the index space (the number of threads to launch).
  {
    Argument *EndArg = cast<Argument>(LCArgs[1]);
    EndArg->setName("runSize");
    HelperArgs.insert(EndArg);

    Value *InputVal = LCInputs[1];
    HelperInputs.push_back(InputVal);
    // Add loop-control input to the input set.
    InputSet.insert(InputVal);
  }

  // The second parameter defines the start of the
  // index space.
  {
    Argument *StartArg = cast<Argument>(LCArgs[0]);
    StartArg->setName("runStart");
    HelperArgs.insert(StartArg);

    Value *InputVal = LCInputs[0];
    HelperInputs.push_back(InputVal);
    // Add loop-control input to the input set.
    InputSet.insert(InputVal);
  }

  // The third parameter defines the grain size, if it is
  // not constant.
  if (!isa<ConstantInt>(LCInputs[2])) {
    Argument *GrainsizeArg = cast<Argument>(LCArgs[2]);
    GrainsizeArg->setName("grainSize");
    HelperArgs.insert(GrainsizeArg);

    Value *InputVal = LCInputs[2];
    HelperInputs.push_back(InputVal);
    // Add loop-control input to the input set.
    InputSet.insert(InputVal);
  }

  // Add the loop-centric kernel parameters (i.e., variables/arrays
  // used in the loop body).
  LLVM_DEBUG(dbgs() << "  - adding loop-centric kernel arguments...\n");
  for (Value *V : TLInputsFixed) {
    HelperArgs.insert(V);
    HelperInputs.push_back(V);
    LLVM_DEBUG(dbgs() << "    - arg: " << V->getName() << "\n");
  }

  LLVM_DEBUG(dbgs() << "  - adding helper kernel arguments...\n");
  for (Value *V : HelperInputs) {
    OrderedInputs.push_back(V);
    LLVM_DEBUG(dbgs() << "    - helper arg: " << V->getName() << "\n");
  }

  LLVM_DEBUG(dbgs() << "  - done.\n");
}

unsigned CudaLoop::getIVArgIndex(const Function &F,
                                 const ValueSet &Args) const {
  // The argument for the primary induction variable is the second input.
  return 1;
}

unsigned CudaLoop::getLimitArgIndex(const Function &F,
                                    const ValueSet &Args) const {
  // The argument for the loop limit is the first input.
  return 0;
}

static std::set<GlobalValue *> &collect(Constant &c,
                                        std::set<GlobalValue *> &seen);

static std::set<GlobalValue *> &collect(BasicBlock &bb,
                                        std::set<GlobalValue *> &seen) {
  for (auto &inst : bb)
    for (auto &op : inst.operands())
      if (auto *c = dyn_cast<Constant>(&op))
        collect(*c, seen);
  return seen;
}

static std::set<GlobalValue *> &collect(Function &f,
                                        std::set<GlobalValue *> &seen) {
  seen.insert(&f);

  for (auto &bb : f)
    collect(bb, seen);
  return seen;
}

static std::set<GlobalValue *> &collect(GlobalVariable &g,
                                        std::set<GlobalValue *> &seen) {
  seen.insert(&g);
  if (g.hasInitializer())
    collect(*g.getInitializer(), seen);
  return seen;
}

static std::set<GlobalValue *> &collect(GlobalIFunc &g,
                                        std::set<GlobalValue *> &seen) {
  seen.insert(&g);
  llvm_unreachable("kitsune: GNU IFUNC not yet supported");
  return seen;
}

static std::set<GlobalValue *> &collect(GlobalAlias &g,
                                        std::set<GlobalValue *> &seen) {
  seen.insert(&g);
  llvm_unreachable("kitsune: GlobalAlias not yet supported");
  return seen;
}

static std::set<GlobalValue *> &collect(BlockAddress &blkaddr,
                                        std::set<GlobalValue *> &seen) {
  if (Function *f = blkaddr.getFunction())
    collect(*f, seen);
  if (BasicBlock *bb = blkaddr.getBasicBlock())
    collect(*bb, seen);
  return seen;
}

std::set<GlobalValue *> &collect(Constant &c, std::set<GlobalValue *> &seen) {
  if (GlobalValue *g = dyn_cast<GlobalValue>(&c))
    if (seen.find(g) != seen.end())
      return seen;

  if (auto *f = dyn_cast<Function>(&c))
    return collect(*f, seen);
  else if (auto *g = dyn_cast<GlobalVariable>(&c))
    return collect(*g, seen);
  else if (auto *g = dyn_cast<GlobalAlias>(&c))
    return collect(*g, seen);
  else if (auto *g = dyn_cast<GlobalIFunc>(&c))
    return collect(*g, seen);
  else if (auto *blkaddr = dyn_cast<BlockAddress>(&c))
    return collect(*blkaddr, seen);
  else
    for (auto &op : c.operands())
      if (auto *cop = dyn_cast<Constant>(op))
        collect(*cop, seen);
  return seen;
}

void CudaLoop::preProcessTapirLoop(TapirLoopInfo &TL, ValueToValueMapTy &VMap) {
  // TODO: process loop prior to outlining to do GPU/CUDA-specific things
  // like capturing global variables, etc.

  LLVM_DEBUG(dbgs() << "debug[cuabi]: -preprocessing loop for kernel '"
                    << KernelName << "'.\n");

  // Collect the top-level entities (Function, GlobalVariable, GlobalAlias
  // and GlobalIFunc) that are used in the outlined loop. Since the outlined
  // loop will live in the KernelModule, any GlobalValue's used in it will
  // need to be cloned into the KernelModule and then register with CUDA
  // in the CUDA-centric ctor.
  LLVM_DEBUG(dbgs() << "  - gathering and analyzing global values...\n");
  std::set<GlobalValue *> UsedGlobalValues;
  Loop &L = *TL.getLoop();

  for (Loop *SL : L)
    for (BasicBlock *BB : SL->blocks())
      collect(*BB, UsedGlobalValues);

  for (BasicBlock *BB : L.blocks())
    collect(*BB, UsedGlobalValues);

  // Clone global variables (TODO: and aliases).
  LLVM_DEBUG(dbgs() << "  - cloning global variables into kernel module...\n");
  for (GlobalValue *V : UsedGlobalValues) {

    if (GlobalVariable *GV = dyn_cast<GlobalVariable>(V)) {
      // TODO: Make sure this logic makes sense...
      // We don't necessarily need a GPU-side clone of a
      // global variable -- instead we need a location where
      // we can copy symbol information from the host.

      GlobalVariable *NewGV = nullptr;
      // If GV is a constant we can clone the entire
      // variable over, including the initializer
      // details, and deal with it as an internal
      // variable (i.e., no need to coordinate with
      // host).  TODO: make sure this is sound!
      if (GV->isConstant()) {
        NewGV = new GlobalVariable(
            KernelModule, GV->getValueType(), /* isConstant*/ true,
            GlobalValue::InternalLinkage, GV->getInitializer(),
            GV->getName() + "_devvar", (GlobalVariable *)nullptr,
            GlobalValue::NotThreadLocal);

        LLVM_DEBUG(dbgs() << "    - new constant global variable: '"
                          << NewGV->getName() << "', from '" << GV->getName()
                          << "'.\n");
      } else {
        // If GV is non-constant we will need to
        // create a device-side version that will
        // have the host-side value copied over
        // prior to launching the corresponding
        // kernel...
        NewGV = new GlobalVariable(
            KernelModule, GV->getValueType(), /* isConstant*/ false,
            GlobalValue::ExternalWeakLinkage,
            (Constant *)Constant::getNullValue(GV->getValueType()),
            GV->getName() + "_devvar", (GlobalVariable *)nullptr,
            GlobalValue::NotThreadLocal);
        TTarget->pushGlobalVariable(GV);
      }
      NewGV->setAlignment(GV->getAlign());
      VMap[GV] = NewGV;
      LLVM_DEBUG(dbgs() << "\t\t\tcreated kernel-side global variable '"
                        << NewGV->getName() << "'.\n");
    } else if (dyn_cast<GlobalAlias>(V))
      llvm_unreachable("cuabi: fatal error, GlobalAlias not implemented!");
  }

  // As part of the creation of the kernel module we need to deal with functions
  // and make sure a declaration exists within the module.  This is a simple
  // step at this point and is literally just a declaration creation pass; later
  // steps will transform these to match cuda-specific entry points, etc.
  for (GlobalValue *G : UsedGlobalValues) {
    Function *F = dyn_cast<Function>(G);
    if (F) {
      Function *DeviceF = KernelModule.getFunction(F->getName());
      if (not DeviceF) {
        DeviceF = Function::Create(F->getFunctionType(), F->getLinkage(),
                                   F->getName(), KernelModule);
        LLVM_DEBUG(dbgs() << "\tcreated device-side function declaration for '"
                          << F->getName() << "()'.\n");
      }

      for (size_t i = 0; i < F->arg_size(); i++) {
        Argument *Arg = F->getArg(i);
        Argument *NewA = DeviceF->getArg(i);
        NewA->setName(Arg->getName());
        VMap[Arg] = NewA;
      }
      VMap[F] = DeviceF;
    }
  }

  // FIXME: Support GlobalIFunc at some point. This is a GNU extension, so we
  // may not want to support it at all, but just in case, this is here.
  for (GlobalValue *V : UsedGlobalValues) {
    if (dyn_cast<GlobalIFunc>(V))
      llvm_unreachable("kitsune: GlobalIFunc not yet supported.");
  }

  // Now clone any function bodies that need to be cloned. This should be
  // done as late as possible so that the VMap is populated with any other
  // global values that need to be remapped.
  for (GlobalValue *v : UsedGlobalValues) {

    if (Function *F = dyn_cast<Function>(v)) {
      if (F->size() && not F->isIntrinsic()) {
        SmallVector<ReturnInst *, 8> Returns;
        Function *DeviceF = cast<Function>(VMap[F]);
        CloneFunctionInto(DeviceF, F, VMap,
                          CloneFunctionChangeType::DifferentModule, Returns);

        LLVM_DEBUG(dbgs() << "cuabi: cloning device function '"
                          << DeviceF->getName() << "' into kernel module.\n");

        // GPU calls are slow, try to force inlining...
        if (OptLevel > 1 && not DeviceF->hasFnAttribute(Attribute::NoInline))
          DeviceF->addFnAttr(Attribute::AlwaysInline);
      }
    }
  }
}

void CudaLoop::postProcessOutline(TapirLoopInfo &TLI, TaskOutlineInfo &Out,
                                  ValueToValueMapTy &VMap) {
  // addSyncToOutlineReturns(TLI, Out, VMap);

  LLVMContext &Ctx = M.getContext();
  Task *T = TLI.getTask();
  Loop *TL = TLI.getLoop();

  TapirLoopHints Hints(TL);

  BasicBlock *Entry = cast<BasicBlock>(VMap[TL->getLoopPreheader()]);
  BasicBlock *Header = cast<BasicBlock>(VMap[TL->getHeader()]);
  BasicBlock *Exit = cast<BasicBlock>(VMap[TLI.getExitBlock()]);
  PHINode *PrimaryIV = cast<PHINode>(VMap[TLI.getPrimaryInduction().first]);
  Value *PrimaryIVInput = PrimaryIV->getIncomingValueForBlock(Entry);

  TTarget->pushSR(T->getDetach()->getSyncRegion());

  // We no longer need the cloned sync region.
  Instruction *ClonedSyncReg =
      cast<Instruction>(VMap[T->getDetach()->getSyncRegion()]);
  ClonedSyncReg->eraseFromParent();

  // Get the kernel function for this loop and clean up
  // any stray (target related) attributes that were
  // attached as part of the host-side target that
  // occurred before outlining.
  Function *KernelF = Out.Outline;
  KernelF->setName(KernelName);

  KernelF->setLinkage(GlobalValue::LinkageTypes::ExternalLinkage);
  KernelF->removeFnAttr("target-cpu");
  KernelF->removeFnAttr("target-features");
  KernelF->removeFnAttr("personality");
  KernelF->removeFnAttr("tune-cpu");

  KernelF->addFnAttr("target-cpu", GPUArch);
  KernelF->addFnAttr("target-features",
                     PTXVersionFromCudaVersion() + "," + GPUArch);
  KernelF->addFnAttr("uniform-work-group-size", "true");
  NamedMDNode *Annotations =
      KernelModule.getOrInsertNamedMetadata("nvvm.annotations");
  SmallVector<Metadata *, 6> AV;
  AV.push_back(ValueAsMetadata::get(KernelF));
  AV.push_back(MDString::get(Ctx, "kernel"));
  AV.push_back(
      ValueAsMetadata::get(ConstantInt::get(Type::getInt32Ty(Ctx), 1)));
  // AV.push_back(MDString::get(Ctx, "maxntidx"));
  // AV.push_back(ValueAsMetadata::get(
  //     ConstantInt::get(Type::getInt32Ty(Ctx), MaxThreadsPerBlock)));
  Annotations->addOperand(MDNode::get(Ctx, AV));

  // Verify that the Thread ID corresponds to a valid iteration.  Because
  // Tapir loops use canonical induction variables, valid iterations range
  // from 0 to the loop limit with stride 1.  The End argument encodes the
  // loop limit. Get end and grainsize arguments
  Argument *End;
  Value *Grainsize;
  {
    // TODO: We only support a grain size of 1 right now.  Not clear if this
    // could be a future optimization but strip mining on our current tests
    // only results in degraded performance...
    auto OutlineArgsIter = KernelF->arg_begin();
    // End argument is the first LC arg.
    End = &*OutlineArgsIter++;

    // Get the grainsize value, which is either constant or the third LC
    // arg.
    // if (unsigned ConstGrainsize = TLI.getGrainsize())
    //  Grainsize = ConstantInt::get(PrimaryIV->getType(), ConstGrainsize);
    // else
    Grainsize =
        ConstantInt::get(PrimaryIV->getType(), DefaultGrainSize.getValue());
  }

  IRBuilder<> B(Entry->getTerminator());

  // Get the thread ID for this invocation of Helper.
  //
  // This is the classic CUDA thread ID calculation:
  //      i = blockDim.x * blockIdx.x + threadIdx.x;
  // For now we only generate 1-D thread IDs.
  Value *ThreadIdx = B.CreateCall(CUThreadIdxX);
  Value *BlockIdx = B.CreateCall(CUBlockIdxX);
  Value *BlockDim = B.CreateCall(CUBlockDimX);
  Value *ThreadIV = B.CreateIntCast(
      B.CreateAdd(ThreadIdx, B.CreateMul(BlockIdx, BlockDim, "blk_offset"),
                  "cuthread_id"),
      PrimaryIV->getType(), false, "thread_iv");

  // NOTE/TODO: Assuming that the grainsize is fixed at 1 for the
  // current codegen...
  // ThreadID = B.CreateMul(ThreadID, Grainsize);
  Value *ThreadEnd = B.CreateAdd(ThreadIV, Grainsize, "thread_end");
  Value *Cond = B.CreateICmpUGE(ThreadIV, End, "cond_thread_end");
  ReplaceInstWithInst(Entry->getTerminator(),
                      BranchInst::Create(Exit, Header, Cond));

  // Use the thread ID as the start iteration number for the primary IV.
  PrimaryIVInput->replaceAllUsesWith(ThreadIV);
  // TODO: ???? PrimaryIVInput->eraseFromParent();

  // Update cloned loop condition to use the thread-end value.
  unsigned TripCountIdx = 0;
  ICmpInst *ClonedCond = cast<ICmpInst>(VMap[TLI.getCondition()]);
  if (ClonedCond->getOperand(0) != End)
    ++TripCountIdx;
  assert(ClonedCond->getOperand(TripCountIdx) == End &&
         "End argument not used in condition!");
  ClonedCond->setOperand(TripCountIdx, ThreadEnd);
}

Function *CudaLoop::resolveLibDeviceFunction(Function *Fn, bool enableFast) {

  if (Fn->isTargetIntrinsic()) {
    LLVM_DEBUG(dbgs() << "cuabi: function '" << Fn->getName()
                      << "()' resolved as a target-specific intrinsic.\n");
    return Fn; // just return the intrinsic again (it is "built-in")
  }

  std::string NVPrefix = "__nv_";
  if (NVPrefix == Fn->getName().str().substr(0, NVPrefix.size() - 1)) {
    LLVM_DEBUG(dbgs() << "cuabi: skipping already prefixed function '"
                      << Fn->getName() << "()'.\n");
    return Fn;
  }

  // TODO #2: Add printf() support (correct codegen)...
  if (Fn->getName() == "printf" || Fn->getName() == "fprintf")
    report_fatal_error("cuabi: printf is currently unsupported "
                       "in device-side code... :-(\n");

  std::string FnName;

  LLVM_DEBUG(dbgs() << "\t\tresolving device library function '"
                    << Fn->getName() << "' for cuda device module.\n");

  if (enableFast)
    NVPrefix += "fast_";

  FnName = NVPrefix + StringSwitch<std::string>(Fn->getName().str())
                          .Case("acos", "acos")
                          .Case("acosf", "acosf")
                          .Case("acosh", "acosh")
                          .Case("acoshf", "acoshf")
                          .Case("asin", "asin")
                          .Case("asinf", "asinf")
                          .Case("asinh", "asinh")
                          .Case("asinhf", "asinhf")
                          .Case("atan2", "atan2")
                          .Case("atan2f", "atan2f")
                          .Case("atan", "atan")
                          .Case("atanf", "atahnf")
                          .Case("atanh", "atanh")
                          .Case("atanhf", "atanhf")
                          .Case("cbrt", "cbrt")
                          .Case("cbrtf", "cbrtf")
                          .Case("cos", "cos")
                          .Case("cosf", "cosf")
                          .Case("cosh", "cosh")
                          .Case("coshf", "coshf")
                          .Case("erfc", "erfc")
                          .Case("erfcf", "erfcf")
                          .Case("erf", "erf")
                          .Case("erff", "erff")
                          .Case("exp2", "exp2")
                          .Case("exp2f", "exp2f")
                          .Case("exp", "exp")
                          .Case("expf", "expf")
                          .Case("expm1", "expm1")
                          .Case("expm1f", "expm1f")
                          .Case("fmodf", "fmodf")
                          .Case("fmod", "fmod")
                          .Case("hypotf", "hypotf")
                          .Case("hypot", "hypot")
                          .Case("lgammaf", "lgammaf")
                          .Case("lgamma", "lgamma")
                          .Case("llvm.cos.f32", "cosf")
                          .Case("llvm.cos.f64", "cos")
                          .Case("llvm.exp.f32", "expf")
                          .Case("llvm.exp.f64", "exp")
                          .Case("llvm.fabs.f32", "fabsf")
                          .Case("llvm.fabs.f64", "fabs")
                          .Case("llvm.fmod.f32", "fmodf")
                          .Case("llvm.fmod.f64", "fmod")
                          .Case("llvm.maxnum.f32", "fmaxf") // correct?
                          .Case("llvm.maxnum.f64", "fmax")  // correct?
                          .Case("llvm.minnum.f32", "fminf") // correct?
                          .Case("llvm.minnum.f64", "fmin")  // correct?
                          .Case("llvm.pow.f32", "powf")
                          .Case("llvm.pow.f64", "pow")
                          .Case("llvm.sincos.f32", "sincosf")
                          .Case("llvm.sincos.f64", "sincos")
                          .Case("llvm.sin.f32", "sinf")
                          .Case("llvm.sin.f64", "sin")
                          .Case("llvm.sqrt.f32", "sqrtf")
                          .Case("llvm.sqrt.f64", "sqrt")
                          .Case("llvm.tan.f32", "tanf")
                          .Case("llvm.tan.f64", "tan")
                          .Case("llvm.tanh.f32", "tanhf ")
                          .Case("llvm.tanh.f64", "tanh")
                          .Case("log10f", "log10f")
                          .Case("log10", "log10")
                          .Case("log1pf", "log1pf")
                          .Case("log1p", "log1p")
                          .Case("log2f", "log2f")
                          .Case("log2", "log2")
                          .Case("logf", "logf")
                          .Case("log", "log")
                          .Case("powf", "powf")
                          .Case("pow", "pow")
                          .Case("sincosf", "sincosf")
                          .Case("sincos", "sincos")
                          .Case("sinf", "sinf")
                          .Case("sinhf", "sinhf")
                          .Case("sinh", "sinh")
                          .Case("sin", "sin")
                          .Case("sqrtf", "sqrtf")
                          .Case("sqrt", "sqrt")
                          .Case("tanf", "tanf")
                          .Case("tanhf", "tanhf")
                          .Case("tanh", "tanh")
                          .Case("tan", "tan")
                          .Case("tgammaf", "tgammaf")
                          .Case("tgamma", "tgamma")
                          .Default("");

  if (FnName == NVPrefix) {
    if (Fn->isIntrinsic())
      return Fn;
    else
      return nullptr;
  }

  std::unique_ptr<Module> &DevMod = TTarget->getLibDeviceModule();
  if (Function *DevFn = DevMod->getFunction(FnName)) {
    LLVM_DEBUG(dbgs() << "\t\t\tresolved mapped function '" << FnName
                      << "' in device library module.\n");
    if (Function *KF = KernelModule.getFunction(FnName))
      return KF;

    Function *DeviceF =
        Function::Create(DevFn->getFunctionType(), DevFn->getLinkage(),
                         DevFn->getName(), KernelModule);
    DeviceF->setAttributes(DevFn->getAttributes());
    return DeviceF;
  } else {
    LLVM_DEBUG(dbgs() << "\t\t\t *unresolved* function '" << FnName
                      << "()'.  Not in libdevice....\n");
    return nullptr;
  }
}

void CudaLoop::transformForPTX(Function &F) {
  LLVM_DEBUG(dbgs() << "cuabi: transforming function " << F.getName()
                    << "() for PTX code gen.\n");

  // The outlined loop is now in the form of a function in the
  // kernel module.  We need to first take a pass that looks for
  // unresolved function calls.

  std::map<CallInst *, CallInst *> Replaced;
  std::set<Function *> CalledFns;
  for (auto I = inst_begin(&F); I != inst_end(&F); I++) {
    if (auto CI = dyn_cast<CallInst>(&*I)) {
      // Look for a marked fast-math calls. If it is a
      // fast-math call we need to tell the 'resolver'
      // to specifically look for an appropriate
      // transformation for a device-side call.
      bool enableFast;
      enableFast = false;
      if (FPMathOperator *FPO = dyn_cast<FPMathOperator>(CI))
        enableFast = FPO->isFast();

      Function *CF = CI->getCalledFunction();
      if (isDeclaration(CF)) {
        // At this point we have already linked in the device
        // library.  Check to see if we can resolve this function
        // relative to other device side entries.
        if (Function *DF = resolveLibDeviceFunction(CF, enableFast)) {
          if (DF != CF) {
            // We found a device-side function (DF) to replace
            // the called function.
            CallInst *NCI = dyn_cast<CallInst>(CI->clone());
            NCI->setCalledFunction(DF);
            Replaced[CI] = NCI;
          }
        }
      } else {
        if (CF != &F)
          CalledFns.insert(CF);
      }
    }
  }

  for (auto I : Replaced) {
    CallInst *CI = I.first;
    CallInst *NCI = I.second;
    NCI->insertBefore(CI);
    CI->replaceAllUsesWith(NCI);
    CI->eraseFromParent();
  }

  for (auto *Fn : CalledFns) {
    transformForPTX(*Fn);
  }

  LLVM_DEBUG(saveFunctionToFile(&F, F.getName().str(), ".cuabi-for-ptx.ll"));
}

void CudaLoop::remapData(ValueToValueMapTy &VMap) {
  for (auto &V : OrderedInputs) {
    if (auto MappedV = VMap[V]) {
      V = MappedV;
    }
  }
}

void CudaLoop::processOutlinedLoopCall(TapirLoopInfo &TL, TaskOutlineInfo &TOI,
                                       DominatorTree &DT) {

  LLVM_DEBUG(dbgs() << "cudaloop: processing outlined loop call...\n"
                    << "\tkernel name: " << KernelName << "\n");

  LLVMContext &Ctx = M.getContext();

  // NOTE: If we are dealing with loop nests with multiple targets
  // (in this case only a CPU-target w/ a nested GPU target is
  // supported) we can end up with multiple calls to the outlined
  // loop (which has been setup for dead code elimination) but can
  // cause invalid IR that trips us up when handling the GPU module
  // code generation. This is a challenge in the Tapir design that
  // was not geared to handle some of the nuances of GPU target
  // transformations (and code gen).  To address this, we need to
  // do some clean up to keep the IR correct (or the verifier will
  // fail on us...).  Specifically, we can no longer depend upon
  // DCE as it runs too late in the GPU transformation process...
  //
  // TODO: This code can be shared between the cuda and hip targets...
  //
  Function *TargetKF = KernelModule.getFunction(KernelName);
  std::list<Instruction *> RemoveList;
  if (TargetKF) {
    LLVM_DEBUG(dbgs() << "\t*- searching for 'dangling' outline calls...\n");
    for (Use &U : TargetKF->uses()) {
      if (auto *Inst = dyn_cast<Instruction>(U.getUser())) {
        LLVM_DEBUG(dbgs() << "\t\t- marking use for removal.\n");
        if (Inst != TOI.ReplCall)
          RemoveList.push_back(Inst);
      }
    }
  }

  for (auto *I : RemoveList)
    I->eraseFromParent();

  // Make a pass to prep for PTX code generation...
  LLVM_DEBUG(dbgs() << "\t*- transform kernel for PTX code gen.\n");
  Function &F = *KernelModule.getFunction(KernelName.c_str());
  transformForPTX(*TargetKF);
  LLVM_DEBUG(dbgs() << "\t*- transform kernel for PTX code gen.\n");

  // Create two builders -- one inserts code into the entry block
  // (e.g. new "up-front" allocas) and the other is for generating
  // new code into a split BB.

  // *** NOTE: If you are going to code gen an alloca in the code
  // below it is most likely (100%?) you should use the EntryBuilder
  // vs. the NewBuilder.  If you find yourself with stack issues for
  // longer running code this is a likely bug to check...
  Function *Parent = TOI.ReplCall->getFunction();
  BasicBlock &EntryBB = Parent->getEntryBlock();
  IRBuilder<> EntryBuilder(&EntryBB.front());

  BasicBlock *RCBB = TOI.ReplCall->getParent();
  BasicBlock *NewBB = RCBB->splitBasicBlock(TOI.ReplCall);
  IRBuilder<> NewBuilder(&NewBB->front());

  // TODO: There is some potential here to share this code across both
  // the hip and cuda transforms...
  LLVM_DEBUG(dbgs() << "\t*- code gen packing of " << OrderedInputs.size()
                    << " kernel args.\n");
  PointerType *VoidPtrTy = PointerType::getUnqual(Ctx);
  ArrayType *ArrayTy = ArrayType::get(VoidPtrTy, OrderedInputs.size());
  Value *ArgArray = EntryBuilder.CreateAlloca(ArrayTy);

  Value *NullPtr = ConstantPointerNull::get(PointerType::getUnqual(Ctx));
  Value *CudaStream = ConstantPointerNull::get(PointerType::getUnqual(Ctx));

  unsigned int i = 0;
  for (Value *V : OrderedInputs) {
    Value *VP = EntryBuilder.CreateAlloca(V->getType());
    NewBuilder.CreateStore(V, VP);
    Value *VoidVPtr = NewBuilder.CreateBitCast(VP, VoidPtrTy);
    Value *ArgPtr =
        NewBuilder.CreateConstInBoundsGEP2_32(ArrayTy, ArgArray, 0, i);
    NewBuilder.CreateStore(VoidVPtr, ArgPtr);
    i++;

    if (CodeGenPrefetch && V->getType()->isPointerTy()) {
      LLVM_DEBUG(dbgs() << "\t\t- code gen prefetch for kernel arg #" << i - 1
                        << "\n");
      CudaStream = NewBuilder.CreateCall(KitCudaMemPrefetchFn, {V, CudaStream});
    }
  }

  // The next step is prep for the actual kernel launch call via
  // the kitsune runtime.  We have to add some extra levels of
  // pointers to match API details, deal with some potential
  // type mismatches, build a dummy pointer for the yet-to-be-created
  // fat binary, etc...
  const DataLayout &DL = M.getDataLayout();
  Value *argsPtr =
      NewBuilder.CreateConstInBoundsGEP2_32(ArrayTy, ArgArray, 0, 0);

  // Generate a call to launch the kernel.
  Constant *KNameCS = ConstantDataArray::getString(Ctx, KernelName);
  GlobalVariable *KNameGV =
      new GlobalVariable(M, KNameCS->getType(), true,
                         GlobalValue::PrivateLinkage, KNameCS, "kern.name");
  KNameGV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  Type *StrTy = KNameGV->getType();
  Constant *Zeros[] = {ConstantInt::get(DL.getIndexType(StrTy), 0),
                       ConstantInt::get(DL.getIndexType(StrTy), 0)};
  Constant *KNameParam =
      ConstantExpr::getGetElementPtr(KNameGV->getValueType(), KNameGV, Zeros);

  // We place *all* transformed tapir loops from the input module into a
  // single GPU target module.  At this point we can not create a complete
  // fat binary image.  However, we have all the important info for the
  // current loop so we use a 'dummy' (null) fat binary for code gen at
  // this point -- we'll post-process the module to clean this up after
  // we've processed all tapir loops.
  (void)tapir::getOrInsertFBGlobal(M, "_cuabi.dummy_fatbin", VoidPtrTy);

  // Deal with type mismatches for the trip count. A difference
  // introduced via the input source details and the runtime's
  // API type signature for the launch.
  Type *Int64Ty = Type::getInt64Ty(Ctx);
  Value *TripCount = OrderedInputs[0];
  Value *CastTripCount = nullptr;
  if (TripCount->getType() != Int64Ty) {
    CastTripCount = CastInst::CreateIntegerCast(TripCount, Int64Ty, false);
    NewBuilder.Insert(CastTripCount, "cast.tc");
  } else
    CastTripCount = TripCount;

  // At this point we need a threads-per-block value for the launch
  // call.  The runtime will determine this value if ThreadsPerBlock
  // is zero but it can also be overridden via kitsune's forall launch
  // attribute.  The catch here is the launch attribute's value for
  // this is flexible and be a computed expression vs. a compile-time
  // constant.  For this first step of creating the kernel launch, we
  // take the path of a runtime configuration vs. an attributed
  // launch.  This will get patched up as needed when we post-process
  // the module and replace the DummyFBPtr (as we will also need to
  // replace the kernel launch call parameter for threads-per-block if
  // an attributed expression is present).  See postProcessModule()'s
  // stage of finalizing the launch calls for details.
  TapirLoopHints Hints(TL.getLoop());
  unsigned TPB = Hints.getThreadsPerBlock();
  Value *TPBValue = ConstantInt::get(Type::getInt32Ty(Ctx), TPB);

  LLVM_DEBUG(dbgs() << "\tgathering kernel instruction mix....\n");
  tapir::KernelInstMixData InstMix;
  tapir::getKernelInstructionMix(&F, InstMix);
  LLVM_DEBUG(
      dbgs() << "\tinstruction mix:\n"
             << "      memory ops      : " << InstMix.numMemoryOps << "\n"
             << "      flop count      : " << InstMix.numFlops << "\n"
             << "      integer op count: " << InstMix.numIntOps << "\n"
             << "      number other ops: " << InstMix.numOtherOps << "\n");

  Constant *InstructionMix = ConstantStruct::get(
      KernelInstMixTy, ConstantInt::get(Int64Ty, InstMix.numMemoryOps),
      ConstantInt::get(Int64Ty, InstMix.numFlops),
      ConstantInt::get(Int64Ty, InstMix.numIntOps),
      ConstantInt::get(Int64Ty, InstMix.numOtherOps));

  AllocaInst *AI = EntryBuilder.CreateAlloca(KernelInstMixTy);
  NewBuilder.CreateStore(InstructionMix, AI);

  LLVM_DEBUG(dbgs() << "\t*- code gen kernel launch....\n");
  CudaStream = NewBuilder.CreateCall(
      KitCudaLaunchFn,
      {NullPtr, KNameParam, argsPtr, CastTripCount, TPBValue, AI, CudaStream});
  Type *VoidTy = Type::getVoidTy(Ctx);
  FunctionCallee KitCudaSyncFn =
      M.getOrInsertFunction("__kitcuda_sync_thread_stream", VoidTy, VoidPtrTy);
  (void)NewBuilder.CreateCall(KitCudaSyncFn, {CudaStream});

  TOI.ReplCall->eraseFromParent();
  LLVM_DEBUG(dbgs() << "*** finished processing outlined call.\n");
}

CudaABI::CudaABI(Module &M)
    : TapirTarget(M),
      KernelModule(Twine(CUABI_PREFIX + sys::path::filename(M.getName())).str(),
                   M.getContext()) {

  // A helping hand for invocation of the transform from a JIT-driven
  // environment where it can be a bit painful to get to
  std::optional<std::string> envTarget = sys::Process::GetEnv("CUDAABI_TARGET");
  if (envTarget) {
    LLVM_DEBUG(dbgs() << "cuabi: target set via environment '"
                      << envTarget.value() << "'.\n");
    GPUArch.setInitialValue(envTarget.value());
  }

  if (MaxThreadsPerBlock > 1024) {
    errs() << "warning: provided max threads per block exceeds hw limits.\n"
           << "   setting to hw limit (1024)\n";
    MaxThreadsPerBlock = 1024;
  }
  LLVM_DEBUG(dbgs() << "cuabi: setting max threads per block: "
                    << MaxThreadsPerBlock << "\n");

  // Create a module (KernelModule) to hold all device side functions for all
  // parallel constructs in the module we are processing (M). At present a loop
  // processor will be created for each construct and is then responsible for
  // the steps required to prepare the "kernel" module (KernelModule) for code
  // generation to PTX.

  // Build the details required to have a PTX code generation path ready to go
  // at completion of the module processing; see postProcessModule() for when
  // that stage is kicked off via the Tapir layer.
  std::string ArchString = "nvptx64";
  Triple TT(ArchString, "nvidia", "cuda");

  std::string PTXVersionStr = PTXVersionFromCudaVersion();
  std::string Error;
  const Target *PTXTarget = TargetRegistry::lookupTarget("", TT, Error);
  if (!PTXTarget) {
    errs() << "Target lookup failed: " << Error << "\n";
    report_fatal_error("Unable to find registered PTX target. "
                       "Was LLVM built with the NVPTX target enabled?");
  }

  CodeGenOptLevel TargetOptLevel;
  CodeModel::Model TargetCodeModel;
  switch (Level.getSpeedupLevel()) {
  case 0:
    TargetOptLevel = CodeGenOptLevel::None;
    TargetCodeModel = CodeModel::Large;
    break;
  case 1:
    TargetOptLevel = CodeGenOptLevel::Less;
    TargetCodeModel = CodeModel::Large;
    break;
  case 2:
    TargetOptLevel = CodeGenOptLevel::Default;
    TargetCodeModel = CodeModel::Large;
    break;
  case 3:
    TargetOptLevel = CodeGenOptLevel::Aggressive;
    TargetCodeModel = CodeModel::Large;
    break;
  default:
    llvm_unreachable("cuabi: unknown speed up level!");
    break;
  }

  PTXTargetMachine = PTXTarget->createTargetMachine(
      TT.getTriple(), GPUArch, PTXVersionStr.c_str(), TargetOptions(),
      Reloc::PIC_, TargetCodeModel, TargetOptLevel);

  KernelModule.setTargetTriple(TT.str());
  KernelModule.setDataLayout(PTXTargetMachine->createDataLayout());

  LLVM_DEBUG(dbgs() << "\ttarget triple: " << TT.getTriple() << "\n");
}

CudaABI::~CudaABI() { LLVM_DEBUG(dbgs() << "cuabi: destroy tapir target.\n"); }

void CudaABI::pushPTXFilename(const std::string &FN) {
  ModulePTXFileList.push_back(FN);
}

void CudaABI::pushGlobalVariable(GlobalVariable *GV) {
  GlobalVars.push_back(GV);
}

std::unique_ptr<Module> &CudaABI::getLibDeviceModule() {
  if (not LibDeviceModule) {
    LLVMContext &Ctx = KernelModule.getContext();
    llvm::SMDiagnostic SMD;
    // KITSUNE FIXME: It might be useful during development to override the
    // libdevice.10.bc function. We could do this with a command-line argument
    // that gets passed to this transform.
    // std::optional<std::string> CudaPath = sys::Process::FindInEnvPath(
    //     "CUDA_HOME", "nvvm/libdevice/libdevice.10.bc");
    // if (!CudaPath) {
    //   CudaPath = sys::Process::FindInEnvPath("CUDA_PATH",
    //                                          "nvvm/libdevice/libdevice.10.bc");
    //   if (!CudaPath)
    //     report_fatal_error("Unable to load cuda libdevice.10.bc!");
    // }

    llvm::StringRef LibDeviceBCFile = KITSUNE_CUDA_LIBDEVICE_BC;
    LibDeviceModule = parseIRFile(LibDeviceBCFile, SMD, Ctx);
    if (not LibDeviceModule)
      report_fatal_error(llvm::StringRef("Failed to parse: ") +
                         LibDeviceBCFile);
  }

  return LibDeviceModule;
}

Value *CudaABI::lowerGrainsizeCall(CallInst *GrainsizeCall) {
  // TODO: The grainsize on the GPU is a completely different beast
  // than the CPU cases Tapir was originally designed for.  At present
  // keeping the grainsize at 1 has almost always shown to yield the
  // best results.  It is obviously not the best choice for all
  // cases...
  Value *Grainsize =
      ConstantInt::get(GrainsizeCall->getType(), DefaultGrainSize.getValue());
  // Replace uses of grainsize intrinsic call with a computed grainsize value.
  GrainsizeCall->replaceAllUsesWith(Grainsize);
  GrainsizeCall->eraseFromParent();
  return Grainsize;
}

void CudaABI::lowerSync(SyncInst &SI) {
  // no-op...
  // The CUDA transformations split the code into device and host
  // side modules.
}

void CudaABI::addHelperAttributes(Function &F) { /* no-op */ }

bool CudaABI::preProcessFunction(Function &F, TaskInfo &TI,
                                 bool OutliningTapirLoops) {
  return false;
}

void CudaABI::postProcessFunction(Function &F, bool OutliningTapirLoops) {
  // no-op
}

void CudaABI::postProcessHelper(Function &F) { /* no-op */ }

void CudaABI::preProcessOutlinedTask(llvm::Function &, llvm::Instruction *,
                                     llvm::Instruction *, bool, BasicBlock *) {
  /* no-op */
}

void CudaABI::postProcessOutlinedTask(Function &F, Instruction *DetachPt,
                                      Instruction *TaskFrameCreate,
                                      bool IsSpawner, BasicBlock *TFEntry) {
  /* no-op */
}

void CudaABI::postProcessRootSpawner(Function &F, BasicBlock *TFEntry) {
  /* no-op */
}

void CudaABI::processSubTaskCall(TaskOutlineInfo &TOI, DominatorTree &DT) {
  /* no-op */
}

void CudaABI::preProcessRootSpawner(llvm::Function &, BasicBlock *TFEntry) {
  /* no-op */
}

CudaABIOutputFile CudaABI::assemblePTXFile(CudaABIOutputFile &PTXFile) {

  LLVM_DEBUG(dbgs() << "\t- assembling PTX file '" << PTXFile->getFilename()
                    << "'.\n");

  std::error_code EC;
  llvm::StringRef PTXASExe = KITSUNE_CUDA_PTXAS;
  SmallString<255> AsmFileName(PTXFile->getFilename());
  sys::path::replace_extension(AsmFileName, ".s");
  std::unique_ptr<ToolOutputFile> AsmFile;
  AsmFile = std::make_unique<ToolOutputFile>(AsmFileName, EC,
                                             sys::fs::OpenFlags::OF_None);

  // Build the command line for ptxas...  There are some target specific options
  // that we support to configure some specifics here.  See the 'opt' entries
  // near the top of this file.
  // These can be passed to the transform via '-mllvm <cuabi-option>'.
  opt::ArgStringList PTXASArgList;
  PTXASArgList.push_back(PTXASExe.data());

  // TODO: Do we need/want to add support for generating relocatable code?

  PTXASArgList.push_back("--gpu-name"); // target gpu architecture (e.g., sm_86)
  PTXASArgList.push_back(GPUArch.c_str());
  // Warn if we spill registers and provide feedback on kernel stats.
  PTXASArgList.push_back("--warn-on-spills");
  PTXASArgList.push_back("--verbose");

  if (PTXasOptLevel == -1)
    PTXasOptLevel = Level.getSpeedupLevel();

  PTXASArgList.push_back("--opt-level");
  std::string optLevelStr = std::to_string(PTXasOptLevel);
  PTXASArgList.push_back(optLevelStr.c_str());

  PTXASArgList.push_back("--output-file");
  std::string SCodeFilename = AsmFile->getFilename().str();
  PTXASArgList.push_back(SCodeFilename.c_str());
  std::string ptxfile(PTXFile->getFilename().str());
  PTXASArgList.push_back(ptxfile.c_str());

  // Build argv for exec'ing ptxas...
  SmallVector<const char *, 128> PTXASArgv;
  PTXASArgv.append(PTXASArgList.begin(), PTXASArgList.end());
  PTXASArgv.push_back(nullptr);

  auto PTXASArgs = toStringRefArray(PTXASArgv.data());
  LLVM_DEBUG(dbgs() << "\t- ptxas command line:\n";
             unsigned c = 0; for (auto dbg_arg : PTXASArgs) {
               dbgs() << "\t\t" << c << ": " << dbg_arg << "\n";
               c++;
             } dbgs() << "\n\n";);

  // Finally we are ready to run ptxas...
  std::string ErrMsg;
  bool ExecFailed;
  int ExecStat = sys::ExecuteAndWait(PTXASExe, PTXASArgs, std::nullopt, {},
                                     0, /* secs to wait -- 0 --> unlimited */
                                     0, /* memory limit -- 0 --> unlimited */
                                     &ErrMsg, &ExecFailed);
  if (ExecFailed)
    report_fatal_error("fatal error: 'ptxas' execution failed!");

  if (ExecStat != 0)
    // 'ptxas' ran but returned an error state.
    report_fatal_error("fatal error: 'ptxas' failure: " + StringRef(ErrMsg));
  AsmFile->keep();
  return AsmFile;
}

// We can't create a correct launch sequence until all the kernels within a
// (LLVM) module are generated.  When post-processing the module we create the
// fat binary and then to revisit the kernel launch calls we created at the loop
// level and replace the fat binary pointer/handle with the completed version.
//
// In addition, we must copy data for global variables from the host to the
// device prior to kernel launches.  This requires digging some additonal
// details out of the fat binary (CUDA module).
void CudaABI::finalizeLaunchCalls(Module &M, GlobalVariable *Fatbin) {

  LLVM_DEBUG(dbgs() << "\t- finalizing kernel launch calls...\n");
  LLVM_DEBUG(
      saveModuleToFile(&M, M.getName().str() + "pre-finalize-launch.host"));

  LLVMContext &Ctx = M.getContext();
  const DataLayout &DL = M.getDataLayout();
  Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *VoidPtrTy = PointerType::getUnqual(Ctx);
  PointerType *CharPtrTy = PointerType::getUnqual(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);

  // Look up a global (device-side) symbol via a module
  // created from the fat binary.
  // TODO: Move these callees to the constructor -- no need
  // to build and destory each time... Perhaps speed us up a
  // tidbit...
  FunctionCallee KitCudaGetGlobalSymbolFn =
      M.getOrInsertFunction("__kitcuda_get_global_symbol",
                            Int64Ty,    // device pointer
                            VoidPtrTy,  // fat binary
                            CharPtrTy); // symbol name

  FunctionCallee KitCudaMemcpyToDeviceFn =
      M.getOrInsertFunction("__kitcuda_memcpy_sym_to_device",
                            VoidTy,    // returns
                            VoidPtrTy, // host ptr
                            Int64Ty,   // device ptr
                            Int64Ty);  // num bytes

  // Search for kernel launch calls that we built prior to the creation
  // of the fat binary -- which we now have.  Replace the first parameter
  // in each call (which is currently null) with the fat binary pointer.
  LLVM_DEBUG(dbgs() << "\t\tsearching...\n");
  auto &FnList = M.getFunctionList();

  // If we have encountered any attributed launches (forall loops) we
  // will have a call to a dummy runtime entry point
  // (_kitrt_dummy_threads_per_blk) that we use to keep the
  // threads-per-block expression from getting DCE'ed.  When lowered
  // from clang, this call was inserted and we need to use its sole
  // parameter in place of the threads-per-block parameter in the
  // kernel launch call.  These dummy calls should be paired with
  // launch calls, so as we look for a launch we *should* first
  // encounter the threads-per-block call to pair with it.  We keep
  // track of the dummy calls so they can removed at the end of the
  // launch finalization.

  for (auto &Fn : FnList) {
    for (auto &BB : Fn) {
      for (auto &I : BB) {
        if (CallInst *CI = dyn_cast<CallInst>(&I)) {
          if (Function *CFn = CI->getCalledFunction()) {
            if (CFn->getName().starts_with("__kitcuda_launch_kernel")) {
              Value *CFatbin;
              CFatbin = CastInst::CreateBitOrPointerCast(Fatbin, VoidPtrTy,
                                                         "_cubin.fatbin", CI);
              CI->setArgOperand(0, CFatbin);

              Instruction *NI = CI->getNextNonDebugInstruction();
              // Unless someting else has monkeyed with our generated code
              // NI should be the launch call.  We need the following code
              // to go between the call instruction and the launch.
              assert(NI && "unexpected null instruction!");

              // We need to explicitly add code to sync up host- and
              // device-side global values prior to launching kernels.
              // We only have a complete awareness of this now so insert
              // the supporting runtime calls.
              //
              // TODO: This is overdone -- we copy *all* globals and not
              // just those that the kernel we're launching is using.
              //
              for (auto &HostGV : GlobalVars) {
                std::string DevVarName = HostGV->getName().str() + "_devvar";
                LLVM_DEBUG(dbgs() << "\t\t\t  processing global: '"
                                  << HostGV->getName() << "'\n");
                Value *SymName =
                    tapir::createConstantStr(DevVarName, M, DevVarName);
                Value *DevPtr =
                    CallInst::Create(KitCudaGetGlobalSymbolFn,
                                     {CFatbin, SymName}, ".cuabi_devptr", NI);
                Value *VGVPtr =
                    CastInst::CreatePointerCast(HostGV, VoidPtrTy, "", NI);
                uint64_t NumBytes = DL.getTypeAllocSize(HostGV->getValueType());
                CallInst::Create(
                    KitCudaMemcpyToDeviceFn,
                    {VGVPtr, DevPtr, ConstantInt::get(Int64Ty, NumBytes)}, "",
                    CI);
              }
            }
          }
        }
      }
    }
  }

  GlobalVariable *ProxyFB = M.getGlobalVariable("_cuabi.dummy_fatbin", true);
  if (ProxyFB) {
    Constant *CFB =
        ConstantExpr::getPointerCast(Fatbin, VoidPtrTy->getPointerTo());
    LLVM_DEBUG(dbgs() << "\tcleaning up dummy fatbin global.\n");
    ProxyFB->replaceAllUsesWith(CFB);
    ProxyFB->eraseFromParent();
  } else {
    // FIXME: If we haven't found the proxy for a fat binary the odds are we
    // have not found any parallel loops in the code...  Technically, this
    // should not be seen as a compiler error...
    report_fatal_error("internal error! unable to find proxy fatbin pointer!");
  }
}

CudaABIOutputFile CudaABI::createFatbinaryFile(CudaABIOutputFile &AsmFile) {
  std::error_code EC;
  SmallString<255> FatbinFilename(AsmFile->getFilename());
  sys::path::replace_extension(FatbinFilename, ".cufatbin");
  CudaABIOutputFile FatbinFile;
  FatbinFile = std::make_unique<ToolOutputFile>(FatbinFilename, EC,
                                                sys::fs::OpenFlags::OF_None);

  LLVM_DEBUG(dbgs() << "\t- generatng fatbinary image file '"
                    << FatbinFile->getFilename() << "'.\n");

  // // TODO: LLVM docs suggest we shouldn't be using findProgramByName()...
  // auto FatbinaryExe = sys::findProgramByName("fatbinary");
  // if ((EC = FatbinaryExe.getError()))
  //   report_fatal_error("'fatbinary' not found. "
  //                      "Is a CUDA installation in your path?");

  llvm::StringRef FatbinaryExe = KITSUNE_CUDA_FATBINARY;
  opt::ArgStringList FatbinaryArgList;
  FatbinaryArgList.push_back(FatbinaryExe.data());
  FatbinaryArgList.push_back("--64");
  FatbinaryArgList.push_back("--create");
  FatbinaryArgList.push_back(FatbinFilename.c_str());

  std::string FatbinaryImgArgs =
      "--image=profile=" + GPUArch + ",file=" + AsmFile->getFilename().str();
  FatbinaryArgList.push_back(FatbinaryImgArgs.c_str());

  std::list<std::string> PTXFilesArgList;
  if (EmbedPTXInFatbinaries) {
    std::string VArchStr = virtualArchForCudaArch(GPUArch);
    if (VArchStr == "unknown")
      report_fatal_error("cuabi: no virtual target for given gpuarch '" +
                         StringRef(GPUArch) + "'!");

    std::string PTXFixedArgStr = "--image=profile=" + VArchStr + ",file=";
    for (auto &PTXFile : ModulePTXFileList) {
      std::string arg = PTXFixedArgStr + PTXFile;
      std::list<std::string>::const_iterator it;
      it = PTXFilesArgList.emplace(PTXFilesArgList.end(), std::move(arg));
      FatbinaryArgList.push_back(it->c_str());
    }
  }

  FatbinaryArgList.push_back(nullptr);

  SmallVector<const char *, 128> FatbinaryArgv;
  FatbinaryArgv.append(FatbinaryArgList.begin(), FatbinaryArgList.end());
  auto FatbinaryArgs = toStringRefArray(FatbinaryArgv.data());

  LLVM_DEBUG(dbgs() << "\tfatbinary command line:\n";
             unsigned c = 0; for (auto dbg_arg : FatbinaryArgs) {
               dbgs() << "\t\t" << c << ": " << dbg_arg << "\n";
               c++;
             } dbgs() << "\n\n";);

  std::string ErrMsg;
  bool ExecFailed;
  int ExecStat =
      sys::ExecuteAndWait(FatbinaryExe, FatbinaryArgs, std::nullopt, {},
                          0, /* secs to wait -- 0 --> unlimited */
                          0, /* memory limit -- 0 --> unlimited */
                          &ErrMsg, &ExecFailed);
  if (ExecFailed)
    report_fatal_error("unable to execute 'fatbinary'.");

  if (ExecStat != 0)
    // 'fatbinary' ran but returned an error state.
    // TODO: Need to check what sort of actual state 'fatbinary' returns to the
    // environment -- currently assuming it matches standard practices...
    report_fatal_error("'fatbinary' error:" + StringRef(ErrMsg));

  if (EmbedPTXInFatbinaries) {
    std::list<std::string>::iterator it = PTXFilesArgList.begin();
    while (it != PTXFilesArgList.end()) {
      PTXFilesArgList.erase(it++);
    }
  }

  // TODO: Not sure we need to force 'keep' here as we return the output file
  // but will keep it here for now just to play it safe.
  FatbinFile->keep();
  return FatbinFile;
}

GlobalVariable *CudaABI::embedFatbinary(CudaABIOutputFile &FatbinaryFile) {

  LLVM_DEBUG(dbgs() << "\t- code gen for embedded fat binary image...\n");

  // Allocate a buffer to store the fat binary image in.  We
  // will then codegen it into the host-side module.
  std::unique_ptr<llvm::MemoryBuffer> Fatbinary = nullptr;
  ErrorOr<std::unique_ptr<MemoryBuffer>> FBBufferOrErr =
      MemoryBuffer::getFile(FatbinaryFile->getFilename());
  if (std::error_code EC = FBBufferOrErr.getError()) {
    report_fatal_error("cuabi: failed to load fat binary image: " +
                       StringRef(EC.message()));
  }
  Fatbinary = std::move(FBBufferOrErr.get());
  LLVM_DEBUG(dbgs() << "\t\treading fat binary.  size = "
                    << Fatbinary->getBufferSize() << " bytes.\n");

  LLVMContext &Ctx = M.getContext();
  Type *Int8Ty = Type::getInt8Ty(Ctx);
  Constant *FatbinArray = ConstantDataArray::getRaw(
      StringRef(Fatbinary->getBufferStart(), Fatbinary->getBufferSize()),
      Fatbinary->getBufferSize(), Int8Ty);

  LLVM_DEBUG(dbgs() << "\t\tcreating associated global 'fatbin' variable.\n");

  // Create a global variable to hold the fatbinary image.
  GlobalVariable *FatbinaryGV;
  FatbinaryGV = new GlobalVariable(M, FatbinArray->getType(), true,
                                   GlobalValue::PrivateLinkage, FatbinArray,
                                   "_cuabi_fatbin_ptr");
  return FatbinaryGV;
}

void CudaABI::bindGlobalVariables(Value *Handle, IRBuilder<> &B) {
  LLVMContext &Ctx = M.getContext();
  const DataLayout &DL = M.getDataLayout();
  Type *IntTy = Type::getInt32Ty(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);
  Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *VoidPtrTy = PointerType::getUnqual(Ctx);
  PointerType *VoidPtrPtrTy = VoidPtrTy->getPointerTo();
  Type *VarSizeTy = Int64Ty;
  PointerType *CharPtrTy = PointerType::getUnqual(Ctx);

  FunctionCallee RegisterVarFn = M.getOrInsertFunction(
      "__cudaRegisterVar", VoidTy, VoidPtrPtrTy, CharPtrTy, CharPtrTy,
      CharPtrTy, IntTy, VarSizeTy, IntTy, IntTy);
  for (auto &HostGV : GlobalVars) {
    uint64_t VarSize = DL.getTypeAllocSize(HostGV->getType());
    Value *VarName = tapir::createConstantStr(HostGV->getName().str(), M);
    std::string DevVarName = HostGV->getName().str() + "_devvar";
    Value *DevName = tapir::createConstantStr(DevVarName, M, DevVarName);
    llvm::Value *Args[] = {
        Handle,
        B.CreateBitCast(HostGV, VoidPtrTy),
        VarName,
        DevName,
        ConstantInt::get(IntTy, 0), // HostGV->isExternalLinkage()),
        ConstantInt::get(VarSizeTy, VarSize),
        ConstantInt::get(IntTy, HostGV->isConstant()),
        ConstantInt::get(IntTy, 0)};

    LLVM_DEBUG(dbgs() << "\t\t\thost global '" << HostGV->getName().str()
                      << "' to device '" << DevVarName << "'.\n");
    B.CreateCall(RegisterVarFn, Args);
  }
}

Function *CudaABI::createCtor(GlobalVariable *Fatbinary,
                              GlobalVariable *Wrapper) {
  LLVMContext &Ctx = M.getContext();
  Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *VoidPtrTy = PointerType::getUnqual(Ctx);
  PointerType *VoidPtrPtrTy = VoidPtrTy->getPointerTo();
  Type *IntTy = Type::getInt32Ty(Ctx);
  Type *BoolTy = Type::getInt8Ty(Ctx);

  Function *CtorFn = Function::Create(
      FunctionType::get(VoidTy, VoidPtrTy, false), GlobalValue::InternalLinkage,
      CUABI_PREFIX + ".ctor." + KernelModule.getName(), &M);

  BasicBlock *CtorEntryBB = BasicBlock::Create(Ctx, "entry", CtorFn);
  IRBuilder<> CtorBuilder(CtorEntryBB);
  const DataLayout &DL = M.getDataLayout();

  if (DefaultThreadsPerBlock != 0) {
    FunctionCallee KitRTSetDefaultThreadsPerBlockFn = M.getOrInsertFunction(
        "__kitcuda_set_default_threads_per_blk", VoidTy, IntTy);
    CtorBuilder.CreateCall(KitRTSetDefaultThreadsPerBlockFn,
                           {ConstantInt::get(IntTy, DefaultThreadsPerBlock)});
  }

  FunctionCallee KitRTSetMaxThreadsPerBlockFn =
      M.getOrInsertFunction("__kitcuda_set_max_threads_per_blk", VoidTy, IntTy);
  CtorBuilder.CreateCall(KitRTSetMaxThreadsPerBlockFn,
                         {ConstantInt::get(IntTy, MaxThreadsPerBlock)});

  FunctionCallee KitCudaInitFn =
      M.getOrInsertFunction("__kitcuda_initialize", VoidTy);
  CtorBuilder.CreateCall(KitCudaInitFn, {});

  FunctionCallee KitCudaLaunchRefinementFn = M.getOrInsertFunction(
      "__kitcuda_enable_launch_refinement", VoidTy, BoolTy);
  Value *EnableRefinedLaunches;
  if (RefineLaunches)
    EnableRefinedLaunches = ConstantInt::get(BoolTy, 1);
  else
    EnableRefinedLaunches = ConstantInt::get(BoolTy, 0);
  CtorBuilder.CreateCall(KitCudaLaunchRefinementFn, {EnableRefinedLaunches});

  // TODO: The parameters to the CUDA registration calls can be opaque about
  // specifics (e.g., types).  Once we sort out some details we should clean
  // this up.

  // The general layout of the calls for fat binary registration
  // look something like this:
  //
  // void** __cudaRegisterFatBinary(void *fatCubin);
  //
  // void __cudaRegisterVar(void **fatCubinHandle,
  //                        char  *hostVar,
  //                        char  *deviceAddress,
  //                        const char  *deviceName,
  //                        int    ext,
  //                        size_t size,
  //                        int    constant,
  //                        int    global);
  //
  // void __cudaRegisterFatBinaryEnd(void **fatCubinHandle);
  //
  FunctionCallee RegisterFatbinaryFn =
      M.getOrInsertFunction("__cudaRegisterFatBinary",
                            FunctionType::get(VoidPtrPtrTy, // cubin handle.
                                              VoidPtrTy, // fat bin device txt.
                                              false));
  CallInst *RegFatbin = CtorBuilder.CreateCall(
      RegisterFatbinaryFn, CtorBuilder.CreateBitCast(Wrapper, VoidPtrTy));

  GlobalVariable *Handle = new GlobalVariable(
      M, VoidPtrPtrTy, false, GlobalValue::InternalLinkage,
      ConstantPointerNull::get(VoidPtrPtrTy), CUABI_PREFIX + ".fbhand");
  Handle->setAlignment(Align(DL.getPointerABIAlignment(0)));
  CtorBuilder.CreateAlignedStore(RegFatbin, Handle,
                                 DL.getPointerABIAlignment(0));
  Handle->setUnnamedAddr(GlobalValue::UnnamedAddr::None);

  Value *HandlePtr = CtorBuilder.CreateLoad(VoidPtrPtrTy, Handle,
                                            CUABI_PREFIX + ".fbhand.ptr");

  // TODO: It is not 100% clear what calls we actually need to make here for
  // kernel, variable, etc. registration with CUDA.  Clang makes these calls but
  // we are targeting CUDA driver API entry points via the Kitsune runtime
  // library so these calls are potentially unneeded...
  if (!GlobalVars.empty()) {
    LLVM_DEBUG(dbgs() << "\t\tbinding host and device global variables...\n");
    bindGlobalVariables(HandlePtr, CtorBuilder);
  }

  // Wrap up fatbinary registration steps...
  FunctionCallee EndFBRegistrationFn =
      M.getOrInsertFunction("__cudaRegisterFatBinaryEnd",
                            FunctionType::get(VoidTy,
                                              VoidPtrPtrTy, // cubin handle.
                                              false));
  CtorBuilder.CreateCall(EndFBRegistrationFn, RegFatbin);

  // Now add a Dtor to help us clean up at program exit...
  if (Function *CleanupFn = createDtor(Handle)) {
    // Hook into 'atexit()'...
    FunctionType *AtExitFnTy =
        FunctionType::get(IntTy, CleanupFn->getType(), false);
    FunctionCallee AtExitFn =
        M.getOrInsertFunction("atexit", AtExitFnTy, AttributeList());
    CtorBuilder.CreateCall(AtExitFn, CleanupFn);
  }

  CtorBuilder.CreateRetVoid();
  return CtorFn;
}

Function *CudaABI::createDtor(GlobalVariable *FBHandle) {
  LLVMContext &Ctx = M.getContext();
  const DataLayout &DL = M.getDataLayout();
  Type *VoidTy = Type::getVoidTy(Ctx);
  Type *VoidPtrTy = PointerType::getUnqual(Ctx);
  Type *VoidPtrPtrTy = VoidPtrTy->getPointerTo();

  FunctionCallee UnregisterFatbinFn =
      M.getOrInsertFunction("__cudaUnregisterFatBinary",
                            FunctionType::get(VoidTy, VoidPtrPtrTy, false));

  Function *DtorFn = Function::Create(
      FunctionType::get(VoidTy, VoidPtrTy, false), GlobalValue::InternalLinkage,
      CUABI_PREFIX + ".dtor", &M);

  // TODO: Do we call into this too many times???
  BasicBlock *DtorEntryBB = BasicBlock::Create(Ctx, "entry", DtorFn);
  IRBuilder<> DtorBuilder(DtorEntryBB);
  Value *HandleValue = DtorBuilder.CreateAlignedLoad(
      VoidPtrPtrTy, FBHandle, DL.getPointerABIAlignment(0));
  DtorBuilder.CreateCall(UnregisterFatbinFn, HandleValue);

  FunctionCallee KitRTDestroyFn =
      M.getOrInsertFunction("__kitcuda_destroy", VoidTy);
  DtorBuilder.CreateCall(KitRTDestroyFn, {});

  DtorBuilder.CreateRetVoid();
  return DtorFn;
}

void CudaABI::registerFatbinary(GlobalVariable *Fatbinary) {

  LLVM_DEBUG(dbgs() << "\t- registering fat binary...\n");

  // Registering the fat binary image (and all the associated components) is an
  // undocumented portion of the CUDA API. One place to peek for some details
  // hides in the cuda header files; specifially fatbinary_section.h. This
  // shows the following struct that we need to have in the host side code.
  //
  //    struct fatbinC_Wrapper_t {
  //      int magic;
  //      int version;
  //      const unsigned long long *data;
  //      void *filename_or_fatbins;
  //    };
  //
  // * Per the header, the magic number is 0x466243B1.
  // * FATBINC_VERSION is 1 and FATBINC_LINK_VERSION is 2.
  //   (more below)
  // * Then section and segments are needed that contains
  //   the "fatbin control structure".  This loosely looks
  //   like:
  //
  //        Control section name: ".nvFatBinSegment"
  //        Fatbinary section name: ".nv_fatbin"
  //        Pre-linked relocatable section: "__nv_relfatbin"
  //
  // * The last struct member varies between versions.  In the
  //   case of version 1 it can be a offline filename and for
  //   version 2 it is an array of pre-linked fatbins.
  //
  const int FATBINARY_MAGIC_ID = 0x466243B1;
  const int FATBINARY_VERSION = 1;
  const char *FATBIN_CONTROL_SECTION_NAME = ".nvFatBinSegment";
  const char *FATBIN_DATA_SECTION_NAME = ".nv_fatbin";

  LLVMContext &Ctx = M.getContext();
  Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *VoidPtrTy = PointerType::getUnqual(Ctx);
  Type *IntTy = Type::getInt32Ty(Ctx);

  const DataLayout &DL = M.getDataLayout();

  Type *FatbinStrTy = Fatbinary->getType();
  Constant *Zeros[] = {ConstantInt::get(DL.getIndexType(FatbinStrTy), 0),
                       ConstantInt::get(DL.getIndexType(FatbinStrTy), 0)};

  Fatbinary->setSection(FATBIN_DATA_SECTION_NAME);
  Constant *FatbinaryPtr = ConstantExpr::getGetElementPtr(
      Fatbinary->getValueType(), Fatbinary, Zeros);

  // Wrap the fatbinary in struct that the CUDA runtime and tools expect
  // to exist in final objects/executables.
  StructType *WrapperTy = StructType::get(IntTy,      // magic #
                                          IntTy,      // version
                                          VoidPtrTy,  // data
                                          VoidPtrTy); // unused for now.
  Constant *WrapperS = ConstantStruct::get(
      WrapperTy, ConstantInt::get(IntTy, FATBINARY_MAGIC_ID),
      ConstantInt::get(IntTy, FATBINARY_VERSION), FatbinaryPtr,
      ConstantPointerNull::get(VoidPtrTy));

  GlobalVariable *Wrapper =
      new GlobalVariable(M, WrapperTy, true, GlobalValue::InternalLinkage,
                         WrapperS, "_cuabi_wrapper");
  Wrapper->setSection(FATBIN_CONTROL_SECTION_NAME);
  Wrapper->setAlignment(Align(DL.getPrefTypeAlign(Wrapper->getType())));

  // The rest of the registration details are tucked into a constructor
  // entry...
  Function *CtorFn = createCtor(Fatbinary, Wrapper);
  if (CtorFn) {
    FunctionType *CtorFnTy = FunctionType::get(VoidTy, false);
    Type *CtorFnPtrTy =
        PointerType::get(CtorFnTy, M.getDataLayout().getProgramAddressSpace());
    tapir::appendToGlobalCtors(M, ConstantExpr::getBitCast(CtorFn, CtorFnPtrTy),
                               65536, nullptr);
  }
}

CudaABIOutputFile CudaABI::generatePTX() {

  LLVM_DEBUG(dbgs() << "\t- generating PTX...\n");
  // Take the intermediate form code in the kernel module and
  // generate a PTX file.  The PTX file will be named the same as
  // the original input source module (M) with the extension changed
  // to PTX.
  std::string ModelPTXFileName =
      std::string(CUABI_PREFIX) + "%%-%%-%%_" + KernelModule.getName().str();
  SmallString<1024> PTXFileName;
  sys::fs::createUniquePath(ModelPTXFileName.c_str(), PTXFileName, true);
  sys::path::replace_extension(PTXFileName, ".ptx");

  std::error_code EC;
  std::unique_ptr<ToolOutputFile> PTXFile;
  PTXFile = std::make_unique<ToolOutputFile>(PTXFileName, EC,
                                             sys::fs::OpenFlags::OF_None);
  PTXFile->keep();

  KernelModule.setModuleFlag(llvm::Module::Override, "nvvm-reflect-ftz",
                             FTZCodeGen);

  OptimizationLevel KModOptLevel(Level);
  if (OptLevel != -1) {
    switch (OptLevel) {
    case 0:
      KModOptLevel = OptimizationLevel::O0;
      break;
    case 1:
      KModOptLevel = OptimizationLevel::O1;
      break;
    case 2:
      KModOptLevel = OptimizationLevel::O2;
      break;
    case 3:
      KModOptLevel = OptimizationLevel::O3;
      break;
    default:
      llvm_unreachable("unexpected optimization level!");
    }
  }

  int SpeedupLevel = KModOptLevel.getSpeedupLevel();
  if (SpeedupLevel > 0) {
    PipelineTuningOptions pto;
    pto.LoopUnrolling = SpeedupLevel > 2;
    pto.LoopInterleaving = SpeedupLevel > 2;
    pto.LoopStripmine = SpeedupLevel > 2;
    pto.LoopVectorization = false;
    pto.SLPVectorization = false;
    // !!!! NOTE !!!!  From the LLVM docs: Create the analysis
    // managers.  These must be declared in this order so that they
    // are destroyed in the correct order due to
    // inter-analysis-manager references.
    LoopAnalysisManager lam;
    FunctionAnalysisManager fam;
    CGSCCAnalysisManager cgam;
    ModuleAnalysisManager mam;

    PassBuilder pb(PTXTargetMachine, pto);
    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    PTXTargetMachine->registerPassBuilderCallbacks(pb);
    pb.crossRegisterProxies(lam, fam, cgam, mam);

    ModulePassManager mpm = pb.buildPerModuleDefaultPipeline(KModOptLevel);
    mpm.addPass(VerifierPass());
    LLVM_DEBUG(dbgs() << "\t\t* module: " << KernelModule.getName() << "\n");
    mpm.run(KernelModule, mam);
    LLVM_DEBUG(dbgs() << "\t\tpasses complete.\n");
  }

  LLVM_DEBUG(saveModuleToFile(&KernelModule,
                              KernelModule.getName().str() + ".kmod.final"));

  // Setup the passes and request that the output goes to the
  // specified PTX file.
  LLVM_DEBUG(dbgs() << "\t- PTX file: '" << PTXFileName << "'.\n");
  legacy::PassManager PassMgr;
  if (PTXTargetMachine->addPassesToEmitFile(PassMgr, PTXFile->os(), nullptr,
                                            CodeGenFileType::AssemblyFile,
                                            false))
    report_fatal_error("Cuda ABI transform -- PTX generation failed!");

  PassMgr.run(KernelModule);

  LLVM_DEBUG(dbgs() << "\t\t - ptx file: '" << PTXFile->getFilename()
                    << "'.\n");
  LLVM_DEBUG(dbgs() << "\tkernel optimizations and code gen complete.\n\n");
  return PTXFile;
}

void CudaABI::postProcessModule() {
  LLVM_DEBUG(dbgs() << "cuabi: post processing kernel and host modules...\n");
  LLVM_DEBUG(saveModuleToFile(&KernelModule,
                              M.getName().str() + ".kmod.pre-postproc"));

  // The kernel module now contains the outlined loop kernels for the
  // compilation units.  This call wraps up all required module-wide
  // transformations and clean-up to create a fat binary image that can be
  // embedded into the final executable.

  // TODO #1: Need to do some more work on debugging and debug info...
  // Make sure any outlined (cloned) debugged info is removed from the kernel
  // module (if we don't it will show up duplicated w/ the host-side module).
  StripDebugInfo(KernelModule);

  // Once the debug info is removed we need to clean up the naming used
  // in the module so it doesn't trip up PTX code generation.  This
  // follows previous renaming steps where we replace '.' with '_'...
  for (GlobalVariable &G : KernelModule.globals()) {
    auto Name = G.getName().str();
    std::replace(Name.begin(), Name.end(), '.', '_');
    G.setName(Name);
  }

  // Next, we need to link the cuda device library to help resolve nvvm-specific
  // intrinsics, math calls, etc.
  auto L = Linker(KernelModule);
  if (LibDeviceModule) {
    LLVM_DEBUG(dbgs() << "\t- linking cuda libdevice --> kernel module.\n");
    L.linkInModule(std::move(LibDeviceModule), Linker::LinkOnlyNeeded);
  }

  // Our final step on the kernel module side is to generate the PTX code,
  // assemble it, and then take the resulting binary image and embed it
  // into the host-side module.
  CudaABIOutputFile PTXFile = generatePTX();
  CudaABIOutputFile AsmFile = assemblePTXFile(PTXFile);
  CudaABIOutputFile FatbinFile = createFatbinaryFile(AsmFile);
  GlobalVariable *Fatbinary = embedFatbinary(FatbinFile);

  // On the host-side we need to now finalize the launch code; which is
  // incomplete at this point (e.g., we didn't have the completed fat binary
  // handle until we compelted the steps above).  After that, we add some
  // host-side code to register the fat binary.
  finalizeLaunchCalls(M, Fatbinary);
  registerFatbinary(Fatbinary);

  if (not KeepFiles) {
    sys::fs::remove(PTXFile->getFilename());
    sys::fs::remove(AsmFile->getFilename());
    sys::fs::remove(FatbinFile->getFilename());
  } else
    LLVM_DEBUG(dbgs() << "*** module post-processing phase complete.\n");

  LLVM_DEBUG(
      saveModuleToFile(&KernelModule, M.getName().str() + ".kmod.postproc"));
}

LoopOutlineProcessor *
CudaABI::getLoopOutlineProcessor(const TapirLoopInfo *TL,
                                 OptimizationLevel OptLevel) {
  // The outline processor handles the steps required for the
  // loop --> kernel transformation.  This is driven from
  // the upstream compilation pipeline in a callback-driven
  // fashion...

  // PTX has some issues with naming that can trip things up.
  // We use the current compilation unit (file) name as a
  // building block so clean that up...
  std::string ModuleName = sys::path::filename(M.getName()).str();
  std::replace(ModuleName.begin(), ModuleName.end(), '.', '_');
  std::replace(ModuleName.begin(), ModuleName.end(), '-', '_');

  LLVM_DEBUG(dbgs() << "cuabi: create loop outlining processor.\n");
  LLVM_DEBUG(saveModuleToFile(&M, M.getName().str() + ".input"));

  // Each outlined loop maps into a kernel function.  We name
  // the kernel based on the function that 'owns' the loop.
  // NOTE: Ordering of how loops within a function are processed
  // does not appear to always matchg program order...
  //
  Loop *TheLoop = TL->getLoop();
  Function *Fn = TheLoop->getHeader()->getParent();
  std::string KernelName = Fn->getName().str();

  // TODO #1: Need to do some more work on debugging and debug info...
  if (M.getNamedMetadata("llvm.dbg.cu") || M.getNamedMetadata("llvm.dbg")) {
    // If we have debug info in the module use a line number
    // based naming scheme for kernels.
    unsigned LineNumber = TL->getLoop()->getStartLoc()->getLine();
    KernelName = CUABI_KERNEL_LOOP_NAME_PREFIX + ModuleName + "_" +
                 Twine(LineNumber).str();
  } else {
    std::string DemangledName;
    if (llvm::nonMicrosoftDemangle(KernelName, DemangledName, false, false))
      KernelName = CUABI_KERNEL_LOOP_NAME_PREFIX + DemangledName;
    else
      KernelName = CUABI_KERNEL_LOOP_NAME_PREFIX + KernelName;
    LLVM_DEBUG(dbgs() << "\t- kernel function '" << KernelName << "()'.\n");
  }

  Level = OptLevel;

  CudaLoop *Outliner = new CudaLoop(M, KernelModule, KernelName, this);
  return Outliner;
}
