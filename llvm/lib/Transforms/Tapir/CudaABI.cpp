//===- CudaABI.cpp - Lower Tapir to the Kitsune GPU back end -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Kitsune+Tapir GPU ABI to convert Tapir
// instructions to call into the Cuda-centric portions of the Kitsune
// runtime system for GPUs (bypassing the top-level platform independent API
// that is a JIT-only interface).
//
//===----------------------------------------------------------------------===//
#include "llvm/Transforms/Tapir/CudaABI.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Option/ArgList.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SmallVectorMemoryBuffer.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Tapir/Outline.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Vectorize.h"

#include <fstream>
#include <sstream>

using namespace llvm;

#define DEBUG_TYPE "cuda-abi"


// Some default naming convensions. 
const std::string CUABI_KERNEL_NAME_PREFIX = "__cuabi_kern_";
const std::string CUABI_MODULE_NAME_PREFIX = "__cuabi_module_";


/// Selected target GPU architecture --  passed directly to ptxas.
static cl::opt<std::string>
GPUArch("cuabi-arch", cl::init("sm_75"), cl::NotHidden, 
        cl::desc("Target GPU architecture for CUDA ABI transformation."
                "(default: sm_75)"));
/// Select 32- vs. 64-bit host architecture. Passed directly to ptxas.
static cl::opt<std::string>            
HostMArch("cuabi-march", cl::init("64"), cl::NotHidden,
          cl::desc("Specify 32- or 64-bit host architecture."
                   "(default=64-bit)."));
/// Enable verbose mode.  Handled internally as well as passed on to 
/// ptxas.
static cl::opt<bool>
Verbose("cuabi-verbose", cl::init(false), cl::NotHidden, 
         cl::desc("Enable verbose mode and also print out code "
                  "generation statistics. (default=off)"));
/// Enable debug mode. Passed directly to ptxas. 
static cl::opt<bool>
Debug("cuabi-debug", cl::init(false), cl::NotHidden, 
        cl::desc("Enable debug information for GPU device code. "
                 "(default=false)")); 
/// Surpress the generation of debug sections in the final object 
/// file.  This option is ignored if not used with the debug or 
/// generate-line-info options. 
static cl::opt<bool>
SurpressDBInfo("cuabi-surpress-debug-info", cl::init(false), 
               cl::Hidden, 
               cl::desc(" Do not generate debug information sections "
                        "in final output object file."));
/// Generate line information for all generated GPU kernels.  Passed 
/// directly to 'ptxas' as '--generate-line-info'. 
static cl::opt<bool>
GenLineInfo("cuabi-generate-line-info", cl::init(false), cl::NotHidden,
              cl::desc("Generate line information for generated GPU "
                       "kernels. (default=false)"));
/// Enable stack bounds checking. Passed directly to 'ptxas', which
/// will automatically turn this on if 'device-debug' or 'opt-level=0'
/// is used.
static cl::opt<bool>
BoundsCheck("cuabi-bounds-check", cl::init(false), cl::NotHidden,
            cl::desc("Enable GPU static pointer bounds checking. "
                       "(default=false)"));
/// Set the optimization level for the compiler.  Values can be 0, 1, 
/// 2, or 3; following standard compiler practice. 
static cl::opt<unsigned>
OptLevel("cuabi-opt-level", cl::init(3), cl::NotHidden, 
         cl::desc("Specify the GPU kernel optimization level."));
/// Enable expensive optimizations to allow the compiler to use the 
/// maximum amount of resources (memory and time).  If the 
/// optimization level if >= 2 this flag is enabled. 
static cl::opt<bool>
AllowExpensiveOpts("cuabi-enable-expensive-optimizations", 
                   cl::init(false), cl::Hidden, 
                   cl::desc("Enable expensive optimizations that use "
                            "maximum available resources."));
/// Disable the generatino of floating point multiply add instructions. 
/// This is passed on to 'ptxas' to disable the contraction of floating
/// point multiply-add operations (FMAD, FFMA, or DFMA).  This is 
/// equivalent to passing '-fmad false' to 'ptxas'.
static cl::opt<bool>
DisableFMA("cuabi-disable-fma", cl::init(false), cl::Hidden, 
           cl::desc("Disable the generation of FMA instructions."
                    "(default=false)")); 
/// Disable the optimizer's constant bank optimizations.  Passed directly
/// to 'ptxas' as '--disable-optimizer-constants'.  
static cl::opt<bool>
DisableConstantBank("cuabi-disable-constant-bank", cl::init(false), 
                    cl::Hidden, 
                    cl::desc("Disable the use of the constants bank in "
                    "GPU code generation. (default=false)"));






/// Set the CUDA ABI's default grainsize value.  This is used internally 
/// by the transform.  
static cl::opt<unsigned>
DefaultGrainSize("cuabi-default-grainsize", cl::init(8), 
                 cl::Hidden, 
                 cl::desc("The default grainsize used by the "
                          "transform when analysis fails to determine one."));

/// Keep the complete set of intermediate files around after compilation.  This 
/// includes LLVM IR, PTX, and the fatbinary file. 
static cl::opt<bool>
KeepIntermediateFiles("cuabi-keep-files", cl::init(false), 
                      cl::Hidden, 
                      cl::desc("Keep all the intermediate files on disk "
                               "after successsful completion of the transforms "
                               "various steps.")); 

/// Generate code to prefetch data prior to kernel launches.  This is literally 
/// in the few lines right before a launch so obviously less than ideal. 
static cl::opt<bool>
CodeGenDisablePrefetch("cuabi-disable-prefetch", cl::init(false), 
                         cl::Hidden,
                         cl::desc("Disable insertion of calls to do data prefetching "
                                  "for UVM-based kernel parameters."));

/// Provide a hard-coded default value for the number of threads per block to 
/// use in kernel launches.  This provides a compile-time mechanisms for setting 
/// this value and it will persist throughout the execution of the associated 
/// compilation unit(s).  The runtime internally currently uses the equations, 
/// 
///   ``unsigned blockSize = 4 * warpSize;``
///   ``blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;`` 
/// 
/// to determine the overall set of launch parameters.  This is mostly meant for 
/// experimentation and testing. 
static cl::opt<unsigned>
DefaultThreadsPerBlock("cuabi-threads-per-block", cl::init(256), 
                       cl::Hidden, 
                       cl::desc("Set the runtime system's value for "
                                "the default number of threads per block. "
                                "(default=256)"));

static cl::opt<unsigned> 
DefaultBlocksPerGrid("cuabi-blocks-per-grid", cl::init(0), 
                     cl::Hidden, 
                     cl::desc("Hard-code the runtime system's value for "
                              "the number of blocks per grid in kernel "
                              "launches. (default=0=disabled)"));  


/*
 * TODO: work here needs to be done to support these additional arguments 
 * to expose more of the ptxas feature set within the ABI transform. 
 * 
 /// Specify the maximum number of registers that GPU functions can use.
 /// Until a function-specific limit, a higher value will generally 
 /// increase the performance of individual GPU threads that execute this
 /// function. However, because thread registers are allocated from a 
 /// global register pool on each GPU, a higher value of this option
 /// will also reduce the maximum thread block size, thereby reducing the 
 /// amount of thread parallelism.
 static cl::opt<int>
 MaxRegCount("cuabi-maxregcount", cl::init(-1), cl::Hidden,
             cl::desc("Specify the max number t of registers that GPU functions "
                      "can use."));
 /// Control the aggressiveness of optimizations that affect register usage.
 /// ([0..10], default = 5) Higher values aggressively optimize the source
 /// program, trading off additional register usage for potential
 /// improvements in the generated code. Lower values inhibit optimizations
 /// that aggressively increase register usage. This option can work in
 /// conjunction with -maxrregcount and kernel launch bounds. (NVIDIA states:
 /// This is a BETA feature for advanced users and there is no guarantee that
 /// the implementation stays consistent between ptxas releases.)
 static cl::opt<unsigned>
 RegUsageLevel("cuabi-register-usage-level", cl::init(5), cl::Hidden,
               cl::desc("Experimental feature -- see 'ptxas' documentaiton. "
                        "(default=5)"));
*/

/// Take the NVIDIA CUDA 'sm_' architecture format and convert it into 
/// the 'compute_' form.  Note that we require CUDA 11 or greater and 
/// we have removed support for sm_2x and sm_3x architectures. 

static std::string VirtualArchForCudaArch(StringRef Arch) {
  // TODO: We've scaled back some from the full suite of Nvidia targets
  // as we are going in assuming we will support only CUDA 11 or greater.
  // We should probably raise an error for sm_2x and sm_3x -- this is 
  // different than the current CUDA support in Clang and could be 
  // confusing to users... That said, not sure it makes a lot of sense 
  // to work on support for deprecated architectures (and thus older 
  // versions of CUDA).
  return llvm::StringSwitch<std::string>(Arch)
      // sm_20 (Fermi) is deprecated as of CUDA 9.
      // sm_3X (Kepler) is deprecated as of CUDA 11.
      .Case("sm_50", "compute_50") // Maxwell (to be deprecated w/ CUDA 12?)
      .Case("sm_52", "compute_52") //
      .Case("sm_53", "compute_53") //
      .Case("sm_60", "compute_60") // Pascal
      .Case("sm_61", "compute_61") //
      .Case("sm_62", "compute_62") //
      .Case("sm_70", "compute_70") // Volta
      .Case("sm_72", "compute_72") //
      .Case("sm_75", "compute_75") // Turing
      .Case("sm_80", "compute_80") // Ampere
      // TODO: LLVM's PTX target doesn't appear to support
      // anything past sm_80???  For LLVM 12.x.  Need to
      // update for new architectures -- 13.x will support 
      // through sm_86. 
      .Case("sm_86", "compute_86") //
      .Case("sm_87", "compute_87") //
      .Default("unknown");
}

static std::string PTXVersionFromCudaVersion() {
  #ifdef CUDATOOLKIT_VERSION
  Twine CudaVersionStr = Twine(CUDATOOLKIT_VERSION_MAJOR) + 
                   Twine(".") + 
                   Twine(CUDATOOLKIT_VERSION_MINOR);
  #else 
  #pragma message("warning, no CUDA Toolkit version info available.")
  Twine CudaVersionStr = "unknown";
  #endif

  // TODO: There could a tighter connection here with the GPU 
  // architecture choice that we are not cross referencing. 
  return llvm::StringSwitch<std::string>(CudaVersionStr.str())
      // TODO: These CUDA to PTX version translations will have 
      // to be watched between CUDA and LLVM resources.  It is 
      // not uncommon for LLVM to lag behind CUDA PTX versions.
      // The details below are based on Cuda 11.6 and LLVM 13.x.
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
      .Case("11.6", "+ptx72")
      .Default("+ptx72"); // TODO: fall back or go with latest?
}

// Helper function to configure the details of our post-Tapir transformation 
// passes. 
//
// TODO: Need to spend some time exploring the selected set of passes here. 
//
const unsigned OptLevel0  = 0;
const unsigned OptLevel1  = 1;
const unsigned OptLevel2  = 2;
const unsigned OptLevel3  = 3;
const unsigned SizeLevel0 = 0;
const unsigned SizeLevel1 = 1;
const unsigned SizeLevel2 = 2;

// NOTES: From the NVPTX target documentation.
//  (See: https://llvm.org/docs/NVPTXUsage.html)
//
//  The NVVMReflect pass should be executed early in the optimization
//  pipeline, immediately after the link stage. The internalize pass is also
//  recommended to remove unused math functions from the resulting PTX. For an
//  input IR module module.bc, the following compilation flow is recommended:
//
//  1. Save list of external functions in module.bc.
//  2. Link module.bc with libdevice.compute_XX.YY.bc.
//  3. Internalize all functions not in list from step #1.
//  4. Eliminate all unused internal functions.
//  5. Run NVVMReflect pass.
//  6. Run standard optimization pipeline.
//
//  The NVVMReflect pass replaces conditionals with constants and it will
//  often leave behind dead code. Therefore, it is recommended that
//  NVVMReflect be executed early in the optimization pipeline before
//  dead-code elimination.  The NVPTX TargetMachine knows how to schedule
//  NVVMReflect at the beginning of your pass manager; just use the following
//  code when setting up your pass manager:
//
//    std::unique_ptr<TargetMachine> TM = ...;
//    PassManagerBuilder PMBuilder(...);
//    if (TM)
//      TM->adjustPassManager(PMBuilder);
//

/// Add a set of customized optimization passes over the LLVM code 
/// post Tapir transformation but prior to conversion to PTX.  The 
/// 'OptLevel' parameter controls the compiler optimization level 
/// for performance, while 'SizeLevel' controls the size optimization
/// target. 
static 
void runOptimizationPasses(Module &KM,
                           unsigned OptLevel = OptLevel3,
                           unsigned SizeLevel = OptLevel2) {
  legacy::PassManager PM;
  PM.add(createReassociatePass());
  PM.add(createGVNPass());
  PM.add(createCFGSimplificationPass());
  PM.add(createSLPVectorizerPass());
  PM.add(createDeadCodeEliminationPass());
  PM.add(createDeadStoreEliminationPass());
  PM.add(createCFGSimplificationPass());
  PM.add(createDeadCodeEliminationPass());
  PM.add(createVerifierPass());
  PM.run(KM);
}

// NOTE: The NextKernelID variable below is not thread safe.  
// This currently isn't an issue but if LLVM at some point 
// in the future starts supporting multiple compilation threads
// (e.g. over Modules for example) the support code for kernel 
// IDs will have a race... 

/// Static ID for kernel naming -- each encountered kernel (loop)
/// during compilation will receive a unique ID. 
unsigned CudaLoop::NextKernelID = 0;


CudaLoop::CudaLoop(Module &M, const std::string &KN, bool MakeUniqueName)
    : LoopOutlineProcessor(M, KernelModule),
      KernelModule(CUABI_MODULE_NAME_PREFIX+M.getName().str(), 
                   M.getContext()), 
      KernelName(KN) {

  if (MakeUniqueName) {
    std::string UN = KN + "_" + Twine(NextKernelID).str();
    NextKernelID++;  // rand();  
    KernelName = UN;
  }        

  LLVM_DEBUG(dbgs() << "tapir cuda ABI loop outliner creation:\n"
                    << "\tbase kernel name: " << KernelName << "\n"
                    << "\tmodule name     : " << KernelModule.getName()  << "\n");

  PTXTargetMachine = nullptr;

  std::string ArchString;
  if (HostMArch == "64")
    ArchString = "nvptx64";
  else 
    ArchString = "nvptx";

  // Note the "cuda" choice as part of the OS portion of the triple selects 
  // compatability with the CUDA Driver API.
  Triple TT(ArchString, "nvidia", "cuda");

  std::string error;
  const Target *PTXTarget = TargetRegistry::lookupTarget("", TT, error);
  if (!PTXTarget) {
    errs() << "Target lookup failed: " << error << "\n";
    report_fatal_error("Unable to find registered PTX target. "
                       "Was LLVM built with the NVPTX target enabled?");
  }

  // The feature string for the target machine can be confusing (this is the 
  // 3rd parameter to createTargetMachine())).  The most common usage seems 
  // to suggest a 32- or 64-bit mode (e.g., "+ptx64").  However, digging into
  // clang's implementation it turns out the feature string is actually a 
  // PTX version specifier that goes along with CUDA version. 
  std::string PTXVersionStr = PTXVersionFromCudaVersion();

  LLVM_DEBUG(dbgs() << "Cuda abi: ptx target feature version " 
                     << PTXVersionStr << ".\n");

  PTXTargetMachine = PTXTarget->createTargetMachine(TT.getTriple(), GPUArch,
                                                    PTXVersionStr.c_str(), 
                                                    TargetOptions(),
                                                    Reloc::PIC_,
                                                    CodeModel::Small,
                                                    CodeGenOpt::Aggressive);
  KernelModule.setTargetTriple(TT.str());
  // TODO: The data layout that this generates does not match that listed in
  // the NVPTX documentation: https://llvm.org/docs/NVPTXUsage.html#data-layout
  KernelModule.setDataLayout(PTXTargetMachine->createDataLayout());

  // Insert PTX intrinsic declarations into the host module.
  LLVMContext &Ctx = KernelModule.getContext();
  Type *Int8Ty = Type::getInt8Ty(Ctx);
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);
  Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *VoidPtrTy = Type::getInt8PtrTy(Ctx);
  PointerType *VoidPtrPtrTy = VoidPtrTy->getPointerTo();
  PointerType *Int8PtrTy = Type::getInt8PtrTy(Ctx);

  GetThreadIdx = KernelModule.getOrInsertFunction("gtid", Int32Ty);

   // Thread index values -- equivalent to Cuda's builtins:  threadIdx.[x,y,z].
  CUThreadIdxX = Intrinsic::getDeclaration(&KernelModule,
                                           Intrinsic::nvvm_read_ptx_sreg_tid_x);
  CUThreadIdxY = Intrinsic::getDeclaration(&KernelModule,
                                           Intrinsic::nvvm_read_ptx_sreg_tid_y);
  CUThreadIdxZ = Intrinsic::getDeclaration(&KernelModule,
                                           Intrinsic::nvvm_read_ptx_sreg_tid_z);

  // Block index values -- equivalent to Cuda's builtins: blockIndx.[x,y,z].
  CUBlockIdxX = Intrinsic::getDeclaration(&KernelModule,
                                          Intrinsic::nvvm_read_ptx_sreg_ctaid_x);
  CUBlockIdxY = Intrinsic::getDeclaration(&KernelModule,
                                          Intrinsic::nvvm_read_ptx_sreg_ctaid_y);
  CUBlockIdxZ = Intrinsic::getDeclaration(&KernelModule,
                                          Intrinsic::nvvm_read_ptx_sreg_ctaid_z);

  // Block dimensions -- equivalent to Cuda's builtins: blockDim.[x,y,z].
  CUBlockDimX = Intrinsic::getDeclaration(&KernelModule,
                                          Intrinsic::nvvm_read_ptx_sreg_ntid_x);
  CUBlockDimY = Intrinsic::getDeclaration(&KernelModule,
                                          Intrinsic::nvvm_read_ptx_sreg_ntid_y);
  CUBlockDimZ = Intrinsic::getDeclaration(&KernelModule,
                                          Intrinsic::nvvm_read_ptx_sreg_ntid_x);

  // Grid dimensions -- equivalent to Cuda's builtins: gridDim.[x,y,z].
  CUGridDimX = Intrinsic::getDeclaration(&KernelModule,
                                         Intrinsic::nvvm_read_ptx_sreg_nctaid_x);
  CUGridDimY = Intrinsic::getDeclaration(&KernelModule,
                                         Intrinsic::nvvm_read_ptx_sreg_nctaid_y);
  CUGridDimZ = Intrinsic::getDeclaration(&KernelModule,
                                         Intrinsic::nvvm_read_ptx_sreg_nctaid_z);

  // NVVM-centric barrier -- equivalent to Cuda's __sync_threads().
  CUSyncThreads = Intrinsic::getDeclaration(&KernelModule, Intrinsic::nvvm_barrier0);
 
  // Get entry points into the Cuda-centric portion of the Kitsune GPU runtime.
  // These are a layer deeper than the interface used by the GPUABI.  While we 
  // could codegen straight to the CudaRT API, the higher level calls still help
  // to simplify the code at this level.


  KitCudaInitFn   = M.getOrInsertFunction("__kitrt_cuInit", VoidTy);
  KitCudaLaunchFn = M.getOrInsertFunction("__kitrt_cuLaunchFBKernel",
                                          VoidPtrTy,    // returns an opaque stream
                                          VoidPtrTy,    // fatbinary
                                          VoidPtrTy,    // kernel name
                                          VoidPtrPtrTy, // arguments
                                          Int64Ty);     // trip count
  KitCudaWaitFn   = M.getOrInsertFunction("__kitrt_cuStreamSynchronize",
                                          VoidTy,
                                          VoidPtrTy);
  KitCudaMemPrefetchFn = M.getOrInsertFunction("__kitrt_cuMemPrefetch",
                                          VoidTy, 
                                          VoidPtrTy);
  KitCudaSetDefaultTBPFn = 
          M.getOrInsertFunction("__kitrt_cuSetDefaultThreadsPerBlock",
                                VoidTy,
                                Int32Ty);  // threads per block.
  KitCudaSetDefaultLaunchParamsFn = 
          M.getOrInsertFunction("__kitrt_cuSetDefaultLaunchParameters",
                                VoidTy, 
                                Int32Ty,   // blocks per grid 
                                Int32Ty);  // threads per block
}


CudaLoop::~CudaLoop() 
{ }

void CudaLoop::setupLoopOutlineArgs(Function &F, ValueSet &HelperArgs,
                                    SmallVectorImpl<Value *> &HelperInputs,
                                    ValueSet &InputSet,
                                    const SmallVectorImpl<Value *> &LCArgs,
                                    const SmallVectorImpl<Value *> &LCInputs,
                                    const ValueSet &TLInputsFixed) {

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

  // The third parameter defines the grainsize, if it is
  // not constant.
  if (!isa<ConstantInt>(LCInputs[2])) {
    Argument *GrainsizeArg = cast<Argument>(LCArgs[2]);
    GrainsizeArg->setName("runStride");
    HelperArgs.insert(GrainsizeArg);

    Value *InputVal = LCInputs[2];
    HelperInputs.push_back(InputVal);
    // Add loop-control input to the input set.
    InputSet.insert(InputVal);
  }

  // Add the loop-centric kernel parameters (i.e., variables/arrays 
  // used in the loop body).
  for (Value *V : TLInputsFixed) {
    HelperArgs.insert(V);
    HelperInputs.push_back(V);
  }

  for (Value *V : HelperInputs) {
    OrderedInputs.push_back(V);
  }
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

void CudaLoop::updateKernelName(const std::string &KN, bool addID) {
  // Use the kernel ID to create a unique name.
  if (addID) {
    std::string UN = KN + "_" + Twine(NextKernelID).str();
    NextKernelID = rand();
    KernelName = UN;
  } else
    KernelName = KN;
}

void CudaLoop::postProcessOutline(TapirLoopInfo &TLI, TaskOutlineInfo &Out,
                                    ValueToValueMapTy &VMap) {
  LLVMContext &Ctx = M.getContext();
  Type *Int8Ty = Type::getInt8Ty(Ctx);
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  Task *T = TLI.getTask();
  Loop *TL = TLI.getLoop();


  BasicBlock *Entry = cast<BasicBlock>(VMap[TL->getLoopPreheader()]);
  BasicBlock *Header = cast<BasicBlock>(VMap[TL->getHeader()]);
  BasicBlock *Exit = cast<BasicBlock>(VMap[TLI.getExitBlock()]);
  PHINode *PrimaryIV = cast<PHINode>(VMap[TLI.getPrimaryInduction().first]);
  Value *PrimaryIVInput = PrimaryIV->getIncomingValueForBlock(Entry);
  Instruction *ClonedSyncReg =
      cast<Instruction>(VMap[T->getDetach()->getSyncRegion()]);

  // We no longer need the cloned sync region.
  ClonedSyncReg->eraseFromParent();

  // Set the helper function to have external linkage.
  Function *Helper = Out.Outline;
  Helper->setName(KernelName);
  Helper->setLinkage(Function::ExternalLinkage);
  // Set the target features for the helper. 
  AttrBuilder Attrs;
  Attrs.addAttribute("target-cpu", GPUArch);
  Attrs.addAttribute("target-features", PTXVersionFromCudaVersion() + "," + GPUArch);
  Helper->removeFnAttr("target-cpu");
  Helper->removeFnAttr("target-features");
  Helper->removeFnAttr("personality");
  Helper->addAttributes(AttributeList::FunctionIndex, Attrs);

  NamedMDNode *Annotations =
      KernelModule.getOrInsertNamedMetadata("nvvm.annotations");
  SmallVector<Metadata *, 3> AV;
  AV.push_back(ValueAsMetadata::get(Helper));
  AV.push_back(MDString::get(Ctx, "kernel"));
  AV.push_back(
      ValueAsMetadata::get(ConstantInt::get(Type::getInt32Ty(Ctx), 1)));
  Annotations->addOperand(MDNode::get(Ctx, AV));

  // Verify that the Thread ID corresponds to a valid iteration.  Because Tapir
  // loops use canonical induction variables, valid iterations range from 0 to
  // the loop limit with stride 1.  The End argument encodes the loop limit.
  // Get end and grainsize arguments
  Argument *End;
  Value *Grainsize;
  {
    auto OutlineArgsIter = Helper->arg_begin();
    // End argument is the first LC arg.
    End = &*OutlineArgsIter++;

    // Get the grainsize value, which is either constant or the third LC arg.
    if (unsigned ConstGrainsize = TLI.getGrainsize())
      Grainsize = ConstantInt::get(PrimaryIV->getType(), ConstGrainsize);
    else
      // Grainsize argument is the third LC arg.
      Grainsize = &*++(++OutlineArgsIter);
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
  Value *ThreadID = B.CreateIntCast(B.CreateAdd(ThreadIdx, 
                                    B.CreateMul(BlockIdx, BlockDim), 
                                    "thread_id"), PrimaryIV->getType(), false);
  ThreadID = B.CreateMul(ThreadID, Grainsize);
  Value *ThreadEnd = B.CreateAdd(ThreadID, Grainsize);
  Value *Cond = B.CreateICmpUGE(ThreadID, End);
  ReplaceInstWithInst(Entry->getTerminator(),
                      BranchInst::Create(Exit, Header, Cond));
  // Use the thread ID as the start iteration number for the primary IV.
  PrimaryIVInput->replaceAllUsesWith(ThreadID);

  // Update cloned loop condition to use the thread-end value.
  unsigned TripCountIdx = 0;
  ICmpInst *ClonedCond = cast<ICmpInst>(VMap[TLI.getCondition()]);
  if (ClonedCond->getOperand(0) != ThreadEnd)
    ++TripCountIdx;
  ClonedCond->setOperand(TripCountIdx, ThreadEnd);
  assert(ClonedCond->getOperand(TripCountIdx) == ThreadEnd &&
         "End argument not used in condition");
}

void CudaLoop::transformForPTX() {
  LLVMContext &Ctx = KernelModule.getContext();
  Function &F = *KernelModule.getFunction(KernelName.c_str());

  // ThreadID.x 
  auto tid   = Intrinsic::getDeclaration(&KernelModule,
                          Intrinsic::nvvm_read_ptx_sreg_tid_x);

  // BlockID.x 
  auto ctaid = Intrinsic::getDeclaration(&KernelModule,
                          Intrinsic::nvvm_read_ptx_sreg_ctaid_x);
  // BlockDim.x 
  auto ntid  = Intrinsic::getDeclaration(&KernelModule,
                          Intrinsic::nvvm_read_ptx_sreg_ntid_x);

  IRBuilder<> B(F.getEntryBlock().getFirstNonPHI());

  // Compute blockDim.x * blockIdx.x + threadIdx.x;
  Value *tidv = B.CreateCall(tid, {}, "thread_idx"); 
  Value *ntidv = B.CreateCall(ntid, {}, "block_idx"); 
  Value *ctaidv = B.CreateCall(ctaid, {}, "block_dimx");
  Value *tidoff = B.CreateMul(ctaidv, ntidv, "block_off"); // 
  Value *gtid = B.CreateAdd(tidoff, tidv, "cu_idx");

  // PTX doesn't like .<n> global names, rename them to
  // replace the '.' with an underscore, '_'. 
  for (GlobalVariable &G : KernelModule.globals()) {
    auto name = G.getName().str();
    std::replace(name.begin(), name.end(), '.', '_');
    G.setName(name);
  }

  // Check if there are unresolved sumbbols to see if we might need libdevice
  std::set<std::string> unresolved;
  for (auto &f : KernelModule) {
    if (f.hasExternalLinkage())
      unresolved.insert(f.getName().str());
  }

  if (!unresolved.empty()) {
    // Load libdevice and check for provided functions
    llvm::SMDiagnostic SMD;
    Optional<std::string> path = sys::Process::FindInEnvPath(
        "CUDA_PATH", "nvvm/libdevice/libdevice.10.bc");

    if (!path) {
      report_fatal_error("Cuda ABI transform: failed to find libdevice!");
    }

    std::unique_ptr<llvm::Module> libdevice = parseIRFile(*path, SMD, Ctx);
    if (!libdevice) {
      report_fatal_error("Cuda abi transform: failed to parse libdevice!");
    }

    // We iterate through the provided functions of the moodule and if there are
    // remaining function calls we add them.
    std::set<std::string> provided;
    std::string nvpref = "__nv_";
    for (auto &f : *libdevice) {
      std::string name = f.getName().str();
      auto res = std::mismatch(nvpref.begin(), nvpref.end(), name.begin());
      auto oldName = name.substr(res.second - name.begin());
      if (res.first == nvpref.end() && unresolved.count(oldName) > 0)
        provided.insert(oldName);
    }
    for (auto &fn : provided) {
      if (auto *f = KernelModule.getFunction(fn))
        f->setName(nvpref + fn);
    }

    auto l = Linker(KernelModule);
    l.linkInModule(std::move(libdevice), 2);
  }

  std::vector<Instruction *> tids;
  for (auto &F : KernelModule) {
    for (auto &BB : F) {
      for (auto &I : BB) {
        if (auto *CI = dyn_cast<CallInst>(&I)) {
          if (Function *f = CI->getCalledFunction()) {
            if (f->getName() == "gtid")
              tids.push_back(&I);
          }
        }
      }
    }
  }

  for (auto p : tids) {
    p->replaceAllUsesWith(gtid);
    p->eraseFromParent();
  }

  if (auto *f = KernelModule.getFunction("gtid"))
    f->eraseFromParent();
}

std::string CudaLoop::createPTXFile() {

    assert(PTXTargetMachine && "can't emit PTX w/out machine target");

    // There were a couple of paths here to consider for generating PTX
    // so here's a high-level summary.  The first option is to directly
    // use the Cuda API within the transform (like the JIT-based runtime
    // target does).  The second approach is to use a Cuda toolchain
    // approach via the use of intermediate files produced by the
    // transform.
    //
    // The most significant drawbacks of the first approach are a heavy
    // change in the LLVM CMake config to include all the bells-and-whistles
    // for adding Cuda into the mix.  In addition, use of the API would
    // require an appropriate GPU in the system where the compiler is being
    // run (eliminating/hindering some cross-compilation use cases).  An
    // advantage of the approach is that it eliminates the creation of
    // external files and (perhaps) slows down overall compilation times.
    //
    // Given a heavily used cross-compilation approach in a number of use
    // cases this transform uses the non-API implementation.  This also
    // closely matches the implementation approach used for Cuda by Clang.
    // While it isn't easy to share that code here, it does provide us with
    // a reference implementation.
    std::error_code EC;
    std::string PTXFileName = KernelName + ".ptx";
    std::unique_ptr<ToolOutputFile> PTXFile;
    PTXFile = std::make_unique<ToolOutputFile>(PTXFileName, EC,
                                               sys::fs::OpenFlags::OF_None);

    // TODO: Update when we merge with LLVM >= 13.x to use the
    // PTXFile->outputFilename() so we get the fully qualified
    // path here...
    LLVM_DEBUG(dbgs() << "generating ptx file: " << PTXFile->getFilename()
                      << ".\n");

    LLVMContext &Ctx = KernelModule.getContext();
    Function &F = *KernelModule.getFunction(KernelName.c_str());

    KernelModule.addModuleFlag(llvm::Module::Override, "nvvm-reflect-ftz",
                               true);

    legacy::PassManager PM;
    legacy::FunctionPassManager FPM(&KernelModule);
    PassManagerBuilder Builder;
    Builder.OptLevel = OptLevel;
    Builder.VerifyInput = 1;
    Builder.Inliner = createFunctionInliningPass(Builder.OptLevel, 0, false);
    Builder.populateLTOPassManager(PM);
    Builder.populateFunctionPassManager(FPM);
    Builder.populateModulePassManager(PM);

    bool Fail;
    Fail = PTXTargetMachine->addPassesToEmitFile(PM, PTXFile->os(), 
               nullptr, CodeGenFileType::CGFT_AssemblyFile, false);
    if (Fail)
      report_fatal_error("An unknown error caused the Tapir Cuda-abi "
                         "transform to fail, PTX code could not be emitted.");


    // Run the configured optimization passes and save the PTX file.
    FPM.doInitialization();
    if (PTXTargetMachine)
      PTXTargetMachine->adjustPassManager(Builder);
    for (Function &Fn : KernelModule)
      FPM.run(Fn);
    FPM.doFinalization();
    PM.run(KernelModule);

    // Don't automatically clean up the PTX file -- the next step will be to
    // convert it into a gpu-architecture-target s-code via the 'ptxas'
    // component in the Cuda toolchain...
    PTXFile->keep();
    return PTXFile->getFilename().str();
}


std::string CudaLoop::createFatBinaryFile(const std::string &PTXFileName) {

  LLVM_DEBUG(dbgs() << "creating fat binary file from ptx source file.\n");

  std::error_code EC;
  std::string AsmFileName = KernelName + ".s";
  std::unique_ptr<ToolOutputFile> AsmFile;
  AsmFile = std::make_unique<ToolOutputFile>(AsmFileName, EC,
                                    sys::fs::OpenFlags::OF_None);

  std::string FBFileName = KernelName + ".fatbin";
  std::unique_ptr<ToolOutputFile> FBFile;
  FBFile = std::make_unique<ToolOutputFile>(FBFileName, EC,
                                            sys::fs::OpenFlags::OF_None);
  // TODO: LLVM docs suggest we shouldn't be using findProgramByName()
  // as a fully qualified path is preferred -- punting on this for now.
  auto PTXASExe = sys::findProgramByName("ptxas");
  if (EC = PTXASExe.getError())
    report_fatal_error("'ptxas' not found. "
                       "Is a CUDA installation in your path?");
  LLVM_DEBUG(dbgs() << "ptxas: " << *PTXASExe << "\n");

  opt::ArgStringList PTXASArgList;
  opt::ArgStringList FatBinArgList;

  PTXASArgList.push_back(PTXASExe->c_str());
  PTXASArgList.push_back("--gpu-name");
  PTXASArgList.push_back(GPUArch.c_str());
  PTXASArgList.push_back("--machine");
  PTXASArgList.push_back(HostMArch.c_str());

  if (Verbose)
    PTXASArgList.push_back("--verbose");

  if (Debug)
    PTXASArgList.push_back("--device-debug");
  if (GenLineInfo)
    PTXASArgList.push_back("--generate-line-info");
  if (BoundsCheck) {
    if (OptLevel != 0 || !Debug)
      PTXASArgList.push_back("--sp-bounds-check");
  }

  if (SurpressDBInfo) {
    if (Debug || GenLineInfo) 
      PTXASArgList.push_back("--suppress-debug-info");
    else 
      errs() << "warning: ignoring -tapir-cu-surpress-debug-info as it "
             << "requires debug or generate-line-info option.\n";
  }

  if (OptLevel > 3) {
    errs() << "warning: -tapir-cu-opt-level=" << OptLevel 
           << " is not supported, using level 3 instead.\n";
    OptLevel = 3;
  }      
  PTXASArgList.push_back("--opt-level");
  switch(OptLevel) {
    case 0: 
      PTXASArgList.push_back("0");
      break;
    case 1:
      PTXASArgList.push_back("1");
      break;
    case 2: 
      PTXASArgList.push_back("2");
      break;
    case 3:
      PTXASArgList.push_back("3");
      break;
    default:
      llvm_unreachable_internal("unhandled optimization level",
                                __FILE__, __LINE__);
      break;
  }

  if (AllowExpensiveOpts && OptLevel < 2) {
    PTXASArgList.push_back("--allow-expensive-optimizations");
    PTXASArgList.push_back("true");
  }

  if (DisableFMA) {
    PTXASArgList.push_back("--fmad");
    PTXASArgList.push_back("false");
  }

  if (DisableConstantBank) 
    PTXASArgList.push_back("--disable-optimizer-constants");
  
  if (KernelModule.getNamedMetadata("llvm.dbg.cu")) {
    PTXASArgList.push_back("-g");
    PTXASArgList.push_back("--generate-line-info");
  }

  LLVM_DEBUG(dbgs() << "\tcreating s-code file: "
                    << AsmFile->getFilename().str(); );
  PTXASArgList.push_back("--output-file");
  std::string scode_filename = AsmFile->getFilename().str();
  PTXASArgList.push_back(scode_filename.c_str());
  PTXASArgList.push_back(PTXFileName.c_str());
  PTXASArgList.push_back(nullptr);

    // Run ptxas to convert PTX to a s-code/GPU assembly file.
  SmallVector<const char *, 128> PTXASArgv;
  PTXASArgv.append(PTXASArgList.begin(), PTXASArgList.end());
  PTXASArgv.push_back(nullptr);
  auto PTXASArgs = toStringRefArray(PTXASArgv.data());

  LLVM_DEBUG(dbgs() << "\tptxas command line:\n";
    unsigned c = 0;
    for(auto dbg_arg : PTXASArgs) {
      dbgs() << "\t\t" << c << ": " << dbg_arg << "\n";
      c++;
    }
  );

  std::string errMsg;
  bool execFailed;
  int execStat = sys::ExecuteAndWait(*PTXASExe, PTXASArgs, None, {},
                                     0, /* secs to wait -- 0 --> unlimited */
                                     0, /* memory limit -- 0 --> unlimited */
                                     &errMsg, &execFailed);
  if (execFailed)
    report_fatal_error("unable to execute 'ptxas'.");

  if (execStat != 0)
    // 'ptxas' ran but returned an error state.
    // TODO: Need to check what sort of actual state 'ptxas'
    // returns to the environment -- currently assuming it
    // matches standard practices...
    report_fatal_error("ptxas execution error:" + errMsg);

  FBFile->keep();


  // TODO: LLVM docs suggest we shouldn't be using findProgramByName()...
  auto FatBinExe = sys::findProgramByName("fatbinary");
  if (EC = FatBinExe.getError())
    report_fatal_error("'fatbinary' not found. "
                       "Is a CUDA installation in your path?");
  LLVM_DEBUG(dbgs() << "\tfatbinary: " << *FatBinExe << "\n");

  FatBinArgList.push_back(FatBinExe->c_str());

  if (HostMArch == "32") 
    FatBinArgList.push_back("--32");
  else if (HostMArch == "64")
    FatBinArgList.push_back("--64");

  FatBinArgList.push_back("--create");
  std::string fatbin_filename = FBFile->getFilename().str();
  FatBinArgList.push_back(fatbin_filename.c_str());
  std::string img_scode_args = std::string("--image=profile=") + GPUArch + ",file=" +
      scode_filename;
  FatBinArgList.push_back(img_scode_args.c_str());
  std::string VArch = VirtualArchForCudaArch(GPUArch); 
  if (VArch == "unknown") {
    std::string Msg = GPUArch + ": unsupported CUDA architecture!";
    report_fatal_error(Msg.c_str());
  }
  std::string img_ptx_args = std::string("--image=profile=") +
                             VirtualArchForCudaArch(GPUArch) + ",file=" +
                             PTXFileName.c_str();
  FatBinArgList.push_back(img_ptx_args.c_str());
  FatBinArgList.push_back(nullptr);

  SmallVector<const char *, 128> FatBinArgv;
  FatBinArgv.append(FatBinArgList.begin(), FatBinArgList.end());
  auto FatBinArgs = toStringRefArray(FatBinArgv.data());

  LLVM_DEBUG({
    dbgs() << "\tfatbinary command line:\n";
    unsigned c = 0;
    for(auto dbg_arg : FatBinArgs) {
      dbgs() << "\t\t" << c << ": " << dbg_arg << "\n";
      c++;
    }
  });

  execStat = sys::ExecuteAndWait(*FatBinExe, FatBinArgs, None, {},
                                 0, /* secs to wait -- 0 --> unlimited */
                                 0, /* memory limit -- 0 --> unlimited */
                                 &errMsg, &execFailed);

  if (execFailed)
    report_fatal_error("unable to execute 'fatbinary'.");

  if (execStat != 0)
    // 'fatbinary' ran but returned an error state.
    // TODO: Need to check what sort of actual state 'fatbinary'
    // returns to the environment -- currently assuming it
    // matches standard practices...
    report_fatal_error("'fatbinary' error:" + errMsg);

  FBFile->keep();

  // Take the next phase of steps required to embed the fat binary
  // into the generated code.
  return std::string(FBFile->getFilename().str());
}


// Create a global variable that contains the contents of a fat binary file
// created using the cuda 'ptxas' and 'fatbinary' components from the cuda
// toolchain.
GlobalVariable* CudaLoop::createKernelBuffer() {

  // Before we move along too far check to see if want to save the kernel 
  // module -- this can be helpful if we bomb-out in PTX code generation 
  // below. 
  if (KeepIntermediateFiles) {
    std::error_code EC;
    std::error_condition ok;
    std::unique_ptr<ToolOutputFile> KMFile;
    std::string KMFileName = KernelName + ".ll";
    KMFile = std::make_unique<ToolOutputFile>(KMFileName, EC,
                                              sys::fs::OpenFlags::OF_None);
    if (EC) {
      errs() << "failed to create kernel module IR output file '" << KMFileName
             << "'.\n"
             << EC.message() << ".\n";
    } else {
      KernelModule.print(KMFile->os(), nullptr);
      KMFile->keep();
    }
  }

  // In order to support cross-compilation use cases we use an approach very
  // similar to clang for cuda support -- we transform the loop (kernel) into
  // ptx file, use 'ptxas' to assemble it and then take the result and use
  // 'fatbinary' to create the binary kernel that will be embedded into the
  // executable.
  std::string PTXFileName = createPTXFile();
  if (PTXFileName.empty()) {
    report_fatal_error("PTX file generation failed!");
    // we'll likely never make it here given the fatal error call above...
    return nullptr;
  }

  LLVM_DEBUG(dbgs() << "fatbinary stage: ptx file '"
                    << PTXFileName << "' created.\n");

  std::string FBFileName = createFatBinaryFile(PTXFileName);
  if (FBFileName.empty()) {
    report_fatal_error("Fatbinary file generation failed!");
    // we'll likely never make it here given the fatal error call above...
    return nullptr;
  }

  LLVM_DEBUG(dbgs() << "fatbinary stage: fatbin file '"
                    << FBFileName << "' created.\n");

  // We now have a fatbinary image file on disk.  Start the process of
  // embedding it into the executable by creating a global variable to
  // hold the contents of the fatbin file.  The resulting variable is
  // our return value...
  std::unique_ptr<llvm::MemoryBuffer> FBBuffer = nullptr;
  ErrorOr<std::unique_ptr<MemoryBuffer>> FBBufferOrErr =
          MemoryBuffer::getFile(FBFileName);
  if (std::error_code EC = FBBufferOrErr.getError()) {
    report_fatal_error("failed to load cuda fat binary image: " + EC.message());
    return nullptr;  // no-op. 
  }

  FBBuffer = std::move(FBBufferOrErr.get());
  LLVM_DEBUG(dbgs() << "\tloaded fat binary file size is "
                    << FBBuffer->getBufferSize() 
                    << " bytes.\n");
  LLVMContext &Ctx = M.getContext();
  const char *FatbinConstantName = ".nv_fatbin";
  const char *FatbinSectionName = ".nvFatBinSegment";
  const char *ModuleSectionName = "__nv_module_id";

  Type *Int8Ty = Type::getInt8Ty(Ctx);
  Constant *FBCS = ConstantDataArray::getRaw(
                StringRef(FBBuffer->getBufferStart(), FBBuffer->getBufferSize()), 
                FBBuffer->getBufferSize(), Int8Ty);
  GlobalVariable *GV;
  GV = new GlobalVariable(M, FBCS->getType(), true,
                          GlobalValue::PrivateLinkage, FBCS,
                          getKernelName() + Twine("_fatbin"));
  //GV->setSection(FatbinSectionName);
  //GV->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::None);

  LLVM_DEBUG( dbgs() << "\tcreated fat binary global variable '"
                     << GV->getName() << "'.\n");

  // Clean up temporary PTX and fat binary files now that we've 
  // successfully created the global variable.
  if (!KeepIntermediateFiles) {
    std::error_code EC;
    std::error_condition ok;

    if (EC = llvm::sys::fs::remove(PTXFileName)) 
      errs() << "error removing PTX file: " <<  EC.message() << "\n";

    if (EC = llvm::sys::fs::remove(FBFileName))
      errs() << "error removing fatbinary file: " << EC.message() << "\n";
  }

  return GV;
}


void CudaLoop::processOutlinedLoopCall(TapirLoopInfo &TL,
                                       TaskOutlineInfo &TOI,
                                       DominatorTree &DT) {

  LLVMContext &Ctx = M.getContext();
  Type *Int8Ty = Type::getInt8Ty(Ctx);
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);
  Type *VoidPtrTy = Type::getInt8PtrTy(Ctx);
                                             
  Function *Parent = TOI.ReplCall->getFunction();
  Value *TripCount = OrderedInputs[0];
  BasicBlock* RCBB = TOI.ReplCall->getParent();
  BasicBlock* NBB = RCBB->splitBasicBlock(TOI.ReplCall);
  TOI.ReplCall->eraseFromParent();

  IRBuilder<> B(&NBB->front());
  LLVMContext &KModCtx = KernelModule.getContext();
  ValueToValueMapTy VMap;

  // We recursively add definitions and declarations to the device module
  SmallVector<Function *> todo;
  Function *KF = KernelModule.getFunction(KernelName);
  assert((KF != nullptr) && "No kernel/function found in the module!");
  todo.push_back(KF);

  while (!todo.empty()) {
    auto *F = todo.back();
    todo.pop_back();
    assert((F != nullptr) && "null function in todo list");

    for (auto &BB : *F) {
      for (auto &I : BB) {
        if (auto *CI = dyn_cast<CallInst>(&I)) {
          if (Function *f = CI->getCalledFunction()) {
            if (f->getParent() != &KernelModule) {
              // TODO: improve check for function, could be overloaded
              auto *deviceF = KernelModule.getFunction(f->getName());
              if (!deviceF) {
                if (f->getParent() == &M) {
                  deviceF = Function::Create(f->getFunctionType(), 
                                             f->getLinkage(),
                                             f->getName(),
                                             KernelModule);
                  VMap[f] = deviceF;
                  auto *NewFArgIt = deviceF->arg_begin();
                  for (auto &Arg : f->args()) {
                    auto ArgName = Arg.getName();
                    NewFArgIt->setName(ArgName);
                    VMap[&Arg] = &(*NewFArgIt++);
                  }
                  SmallVector<ReturnInst *, 8> Returns;
                  CloneFunctionInto(deviceF, f, VMap,
                                    CloneFunctionChangeType::DifferentModule,
                                    Returns);
                  // GPU calls are slow, try to force inlining
                  deviceF->addFnAttr(Attribute::AlwaysInline);
                  todo.push_back(deviceF);
                }
              }
              CI->setCalledFunction(deviceF);
            }
          }
        }

        for (auto &op : I.operands()) {
          if (GlobalVariable *GV = dyn_cast<GlobalVariable>(op)) {
            if (GV->getParent() == &M) {
              GlobalVariable *NewGV = new GlobalVariable(
                  KernelModule, GV->getValueType(), GV->isConstant(),
                  GV->getLinkage(), (Constant *)nullptr, GV->getName(),
                  (GlobalVariable *)nullptr, GV->getThreadLocalMode(),
                  GV->getType()->getAddressSpace());
              NewGV->copyAttributesFrom(GV);
              //NewGV->getAttributes().dump();
              VMap[op] = NewGV;
              const Comdat *SC = GV->getComdat();
              if (!SC)
                return;
              Comdat *DC = NewGV->getParent()->getOrInsertComdat(SC->getName());
              DC->setSelectionKind(SC->getSelectionKind());
              NewGV->setComdat(DC);
              NewGV->setLinkage(GV->getLinkage());
              NewGV->setInitializer(GV->getInitializer());
              op = NewGV;
            }
          }
        }
      }
    }
  } 

  runOptimizationPasses(KernelModule, OptLevel);
  transformForPTX();

  BasicBlock &EBB = Parent->getEntryBlock();
  IRBuilder<> EB(&EBB.front());
  EB.CreateCall(KitCudaInitFn, {});

  /*
  if (DefaultBlocksPerGrid != 0) {
    // TODO: At present, the runtime's kernel launch parameters can be
    // overriden in a per compiler invocation right now.  We should
    // probably ponder a way to make this work per loop (via metadata?).
    // 
    // NOTE: These parameters only "stick" for the next kernel launch.
    Value *BPG = ConstantInt::get(Int32Ty, DefaultBlocksPerGrid);
    Value *TPB = ConstantInt::get(Int32Ty, DefaultThreadsPerBlock);
    EB.CreateCall(KitCudaSetDefaultLaunchParamsFn, { BPG, TPB});
  } else {
    Value *TPB = ConstantInt::get(Int32Ty, DefaultThreadsPerBlock);
    EB.CreateCall(KitCudaSetDefaultTBPFn, {TPB});
  }
  */
 
  ArrayType *ArrayTy = ArrayType::get(VoidPtrTy, OrderedInputs.size());
  Value *ArgArray = B.CreateAlloca(ArrayTy);
  unsigned int i = 0;
  for(Value *V : OrderedInputs) {
    Value *VP = B.CreateAlloca(V->getType());
    B.CreateStore(V, VP);
    Value *VoidVPtr = B.CreateBitCast(VP, VoidPtrTy);
    Value *ArgPtr = B.CreateConstInBoundsGEP2_32(ArrayTy, ArgArray, 0, i++);
    B.CreateStore(VoidVPtr, ArgPtr);

    // TODO: This is still experimental and currently just about as simple of 
    // an approach as we can take on issuing prefetch calls.  Is it better 
    // than nothing -- it is obviously very far from ideal placement. 
    if (CodeGenDisablePrefetch == false) {
      Type *VT = V->getType(); 
      if (VT->isPointerTy()) {
        Value *VoidPP = B.CreateBitCast(V, VoidPtrTy);
        B.CreateCall(KitCudaMemPrefetchFn, { VoidPP });
      }
    }
  }

  Value *GrainSize = TL.getGrainsize()
                     ? ConstantInt::get(TripCount->getType(), TL.getGrainsize())
                     : OrderedInputs[2];
  Value *RunSizeQ = B.CreateUDiv(TripCount, GrainSize, "run_size");
  Value *RunRem = B.CreateURem(TripCount, GrainSize, "run_rem");
  Value *isRem = B.CreateICmp(ICmpInst::ICMP_UGT, RunRem,
                              ConstantInt::get(RunRem->getType(), 0));
  Value *isRemAdd = B.CreateZExt(isRem, RunSizeQ->getType());
  Value *RunSize = B.CreateZExt(B.CreateAdd(RunSizeQ, isRemAdd), Int64Ty);
  Value *argsPtr = B.CreateConstInBoundsGEP2_32(ArrayTy, ArgArray, 0, 0);

  // Generate a call to launch the kernel. 
  Constant *KNameCS = ConstantDataArray::getString(Ctx, KernelName);
  GlobalVariable *KNameGV = new GlobalVariable(M, KNameCS->getType(), 
                                               true,
                                               GlobalValue::PrivateLinkage, 
                                               KNameCS, ".str");
  KNameGV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  Type *StrTy = KNameGV->getType();
  const DataLayout &DL = M.getDataLayout();
  Constant *Zeros[] = { ConstantInt::get(DL.getIndexType(StrTy), 0),
                        ConstantInt::get(DL.getIndexType(StrTy), 0) };
  Constant *KNameParam = ConstantExpr::getGetElementPtr(KNameGV->getValueType(),
                                                        KNameGV, Zeros);

  GlobalVariable *FBB = createKernelBuffer();
  Value *FBPtr = B.CreateConstInBoundsGEP2_32(FBB->getValueType(), FBB, 0, 0, "fatbin");
  Value *Stream = B.CreateCall(KitCudaLaunchFn,
                              { FBPtr, KNameParam, argsPtr, RunSize }, "stream");
  B.CreateCall(KitCudaWaitFn, Stream);
}

CudaABI::CudaABI(Module & M)
  : TapirTarget(M)
{ }

CudaABI::~CudaABI() 
{ }

Value *CudaABI::lowerGrainsizeCall(CallInst * GrainsizeCall) {
  // TODO: Some quick checks suggest we are almost always getting
  // called to select this value (at least when trip counts can't 
  // be easily determined?).  More work needs to go into picking 
  // this value that likely strongly correlates with the launch 
  // parameters...  More parameter studies are likely a key piece
  // of this puzzle.
  Value *Grainsize = ConstantInt::get(GrainsizeCall->getType(), 
                                      DefaultGrainSize);
  // Replace uses of grainsize intrinsic call with a computed 
  // grainsize value. 
  GrainsizeCall->replaceAllUsesWith(Grainsize);
  return Grainsize;
}

void CudaABI::lowerSync(SyncInst & SI)
{ /* no-op */ }


void CudaABI::addHelperAttributes(Function & F)
{ /* no-op */ }


void CudaABI::preProcessFunction(Function & F, TaskInfo & TI,
                                   bool OutliningTapirLoops)
{ /* no-op */ }

void CudaABI::postProcessFunction(Function & F, bool OutliningTapirLoops) {
  if (!OutliningTapirLoops)
    return;
}

void CudaABI::postProcessHelper(Function & F)
{ /* no-op */ }

void CudaABI::preProcessOutlinedTask(llvm::Function &, llvm::Instruction *,
                                         llvm::Instruction *, bool,
                                         BasicBlock *)
{ /* no-op */ }


void CudaABI::postProcessOutlinedTask(Function & F,
                                        Instruction *DetachPt,
                                        Instruction * TaskFrameCreate,
                                        bool IsSpawner,
                                        BasicBlock *TFEntry)
{ /* no-op */ }

void CudaABI::postProcessRootSpawner(Function & F,
                                       BasicBlock * TFEntry)
{ /* no-op */ }

void CudaABI::processSubTaskCall(TaskOutlineInfo & TOI,
                                   DominatorTree & DT)
{ /* no-op */ }

void CudaABI::preProcessRootSpawner(llvm::Function &,
                                      BasicBlock * TFEntry)
{ /* no-op */ }

LoopOutlineProcessor*
CudaABI::getLoopOutlineProcessor(const TapirLoopInfo *TL) const {
  std::string ModuleName = M.getName().str();
  // PTX dislikes names containing '.' -- replace them with 
  // underscores. 
  std::replace(ModuleName.begin(), ModuleName.end(), '.', '_');
  std::replace(ModuleName.begin(), ModuleName.end(), '-', '_');
  std::string KN;
  bool makeKNUnique = true;
  if (M.getNamedMetadata("llvm.dbg.cu") || M.getNamedMetadata("llvm.dbg")) {
    unsigned lineNumber = TL->getLoop()->getStartLoc()->getLine();
    KN = CUABI_KERNEL_NAME_PREFIX + ModuleName + "_" + Twine(lineNumber).str();
    makeKNUnique = false;
  } else
    KN = CUABI_KERNEL_NAME_PREFIX + ModuleName;
  CudaLoop *CLOP = new CudaLoop(M, KN, makeKNUnique);
  return CLOP;
}
