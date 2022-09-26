//===- HipABI.cpp - Lower Tapir to the Kitsune GPU back end -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Kitsune+Tapir HIP ABI to convert Tapir
// instructions to calls into the HIP-centric portions of the Kitsune
// runtime for GPUs to produce a fully compiled (not JIT) executable
// that is suitable for a given architecture target.
//
// NOTE: Several aspects of this transform mimic Clang's code generation
// for HIP. Any significant changes to Clang at that level might require
// changes here as well.
//
//===----------------------------------------------------------------------===//
#include "llvm/Transforms/Tapir/HipABI.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SmallVectorMemoryBuffer.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/AggressiveInstCombine/AggressiveInstCombine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/Inliner.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Tapir/Outline.h"
#include "llvm/Transforms/Tapir/TapirGPUUtils.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"
#include "llvm/Transforms/Vectorize.h"

using namespace llvm;

#define DEBUG_TYPE "hip-abi"
static const std::string HIPABI_PREFIX = "__hipabi";
static const std::string HIPABI_KERNEL_NAME_PREFIX = HIPABI_PREFIX + "_kern_";


// ---- HIP transformation-specific command line arguments.
//
// The transform has its own set of command line arguments that provide
// additional functionality, debugging, etc.  As a reminder, these can
// used in the form:
//
//    -mllvm -hipabi-option[...]
//

// Select a specific target AMD GPU architecture.  At present this
// directly matches those available with the AMDGPU backend target.
//
// TODO: Might be nice to allow build-time setting of this default.
//
static cl::opt<std::string>
    GPUArch("hipabi-arch", cl::init("gfx908"), cl::NotHidden,
            cl::desc("Target GPU architecture for HIP ABI transformation."
                     "(default: gfx908 (MI100)"));

/// Enable verbose mode.  Handled internally as well as passed on to
/// any extra toolchain components (when available).
static cl::opt<bool>
    Verbose("hipabi-verbose", cl::init(false), cl::Hidden,
    cl::desc("Provide verbose details as the transformation runs."));


/// Set the optimization level for use within the transformation.  This
/// level is used internally within the transform IR as well handed off
/// to any external toolchain elements (e.g., the clang offload bundler).
static cl::opt<unsigned>
    OptLevel("hipabi-opt-level", cl::init(3), cl::NotHidden,
             cl::desc("Specify the GPU kernel optimization level."));

/// Enable an extra set of passes over the host-side code after the
/// code has been transformed (e.g., loops replaced with kernel launch
/// calls).
static cl::opt<bool>
    RunHostPostOpt("hipabi-run-post-opts", cl::init(false), cl::NotHidden,
    cl::desc("Run a post-transform optimization pass on the host-side code."));

/// Keep the complete set of intermediate files around after compilation.  This
/// includes LLVM IR, GCN, and the fatbinary/bundle file.
static cl::opt<bool>
    KeepIntermediateFiles("hipabi-keep-files", cl::init(false), cl::Hidden,
    cl::desc("Keep all the intermediate files on disk after"
             "successful completion of the transforms "
             "various steps."));

/// Generate code to prefetch data prior to kernel launches.  This is literally
/// in the few lines right before a launch so obviously less than ideal.
static cl::opt<bool>
    CodeGenDisablePrefetch("hipabi-disable-prefetch", cl::init(false),
                           cl::Hidden,
                           cl::desc("Disable insertion of calls to do data "
                                    "prefetching for memory managed kernel  "
                                    "parameters."));

/// Set the HIP ABI's default grain size value.  This is used internally
/// by the transform.
static cl::opt<unsigned>
    DefaultGrainSize("hipabi-default-grainsize", cl::init(1), cl::Hidden,
    cl::desc("The default grainsize used by the transform "
             "when analysis fails to determine one. (default=1)"));


// --- Loop Outliner


// For some clarity (that is documented elsewhere) many fields are
// buried in the AMDGPU/HSA dispatch pointer structure:
//
//    struct hsa_kernel_dispatch_packet_s {
//      uint16_t   header;
//      uint16_t   setup;
//      uint16_t   workgroup_size_x;
//      uint16_t   workgroup_size_y;
//      uint16_t   reserved0;
//      uint16_t   grid_size_x;
//      uint16_t   grid_size_y;
//      uint16_t   grid_size_z;
//      ...
//    };
//
// See the AMDGPU Target source for more details...
//
Value *HipLoop::emitDispatchPtr(IRBuilder<> &Builder) {
  LLVMContext &Ctx = KernelModule.getContext();

  Function *DispatchPtrFn =
        Intrinsic::getDeclaration(&KernelModule,
                               Intrinsic::amdgcn_dispatch_ptr);
  CallInst *DispatchPtr = Builder.CreateCall(DispatchPtrFn, {},
                                             "dispatch_ptr");
  DispatchPtr->addAttribute(AttributeList::ReturnIndex, Attribute::NoAlias);
  DispatchPtr->addAttribute(AttributeList::ReturnIndex, Attribute::NonNull);
  DispatchPtr->addAttribute(AttributeList::ReturnIndex,
                                Attribute::getWithAlignment(Ctx, Align(4)));
  // Size of the dispatch packet struct.
  DispatchPtr->addDereferenceableAttr(AttributeList::ReturnIndex, 64);
  return DispatchPtr;
}

// Index is 0, 1, or 2 for x, y, and z dimensions.
Value *HipLoop::emitWorkGroupSize(IRBuilder<> &Builder, unsigned Index) {
  LLVMContext &Ctx = KernelModule.getContext();
  Type *Int8Ty = Type::getInt8Ty(Ctx);
  Type *Int16Ty = Type::getInt16Ty(Ctx);
  Type *Int32Ty = Type::getInt32Ty(Ctx);

  const unsigned XOffset = 4;
  Value *DispatchPtr = emitDispatchPtr(Builder);
  Constant *Offset = ConstantInt::get(Int32Ty, XOffset + Index * 2);
  Value *GEP = Builder.CreateGEP(Int8Ty, DispatchPtr, Offset);
  auto *DestTy =
        Int16Ty->getPointerTo(GEP->getType()->getPointerAddressSpace());
  auto *Cast = Builder.CreateBitCast(GEP, DestTy);

  auto *LD = Builder.CreateLoad(Int16Ty, Cast);
  LD->setMetadata(LLVMContext::MD_invariant_load, MDNode::get(Ctx, None));
  Value *WGSize = Builder.CreateIntCast(LD, Int32Ty, false);
  return WGSize;
}

// Index is 0, 1, or 2 for x, y, and z dimensions.
Value *HipLoop::emitGridSize(IRBuilder<> &Builder, unsigned Index) {
  LLVMContext &Ctx = M.getContext();
  Type *Int8Ty = Type::getInt8Ty(Ctx);
  Type *Int32Ty = Type::getInt32Ty(Ctx);

  const unsigned XOffset = 12;
  Value *DispatchPtr = emitDispatchPtr(Builder);
  Constant *Offset = ConstantInt::get(Int32Ty, XOffset + Index * 4);
  Value *GEP = Builder.CreateInBoundsGEP(Int32Ty,
                                         DispatchPtr,
                                         Offset);
  auto *DstTy = Int32Ty->getPointerTo(GEP->getType()->getPointerAddressSpace());
  auto *Cast = Builder.CreateBitCast(GEP, DstTy);
  auto *LD = Builder.CreateLoad(DstTy, Cast);
  LD->setMetadata(LLVMContext::MD_invariant_load,
                  MDNode::get(Ctx, None));
  return LD;
}



/// Static ID for kernel naming -- each encountered kernel (loop)
/// during compilation will receive a unique ID.
unsigned HipLoop::NextKernelID = 0;

HipLoop::HipLoop(Module &M, Module &KModule,
                 const std::string &Name,
                 HipABI *LoopTarget)
    : LoopOutlineProcessor(M, KModule),
      TTarget(LoopTarget),
      KernelName(Name),
      KernelModule(KModule) {

  std::string UN = KernelName + "_" + Twine(NextKernelID).str();
  NextKernelID++;
  KernelName = UN;

  LLVM_DEBUG(dbgs() << "hipabi: hip loop outliner creation:\n"
                    << "\tbase kernel name: " << KernelName << "\n"
                    << "\tmodule name     : " << KernelModule.getName()
                    << "\n\n");

  LLVMContext &Ctx = KernelModule.getContext();
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);
  Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *VoidPtrTy = Type::getInt8PtrTy(Ctx);
  PointerType *VoidPtrPtrTy = VoidPtrTy->getPointerTo();
  PointerType *CharPtrTy = Type::getInt8PtrTy(Ctx);

  KitHipWorkItemIdXFn = Intrinsic::getDeclaration(&KernelModule,
                               Intrinsic::amdgcn_workitem_id_x);
  KitHipWorkItemIdYFn = Intrinsic::getDeclaration(&KernelModule,
                               Intrinsic::amdgcn_workitem_id_y);
  KitHipWorkItemIdZFn = Intrinsic::getDeclaration(&KernelModule,
                               Intrinsic::amdgcn_workitem_id_z);

  KitHipWorkGroupIdXFn = Intrinsic::getDeclaration(&KernelModule,
                               Intrinsic::amdgcn_workgroup_id_x);
  KitHipWorkGroupIdYFn = Intrinsic::getDeclaration(&KernelModule,
                               Intrinsic::amdgcn_workgroup_id_y);
  KitHipWorkGroupIdZFn = Intrinsic::getDeclaration(&KernelModule,
                               Intrinsic::amdgcn_workgroup_id_z);

  // Get entry points into the Hip-centric portion of the Kitsune GPU
  // runtime.
  KitHipLaunchFn = M.getOrInsertFunction("__kitrt_hipLaunchFBKernel",
                                         VoidPtrTy, // returns an opaque stream
                                         VoidPtrTy, // fat-binary
                                         VoidPtrTy, // kernel name
                                         VoidPtrPtrTy, // arguments
                                         Int64Ty);     // trip count
  KitHipWaitFn =
      M.getOrInsertFunction("__kitrt_hipStreamSynchronize", VoidTy, VoidPtrTy);
  KitHipMemPrefetchFn =
      M.getOrInsertFunction("__kitrt_hipMemPrefetch", VoidTy, VoidPtrTy);
  KitHipCreateFBModuleFn =
      M.getOrInsertFunction("__kitrt_hipCreateFBModule", VoidPtrTy, VoidPtrTy);
  KitHipGetGlobalSymbolFn =
      M.getOrInsertFunction("__kitrt_hipGetGlobalSymbol",
                            Int64Ty,    // return the device pointer for symbol.
                            CharPtrTy,  // symbol name
                            VoidPtrTy); // HIP module

  KitHipMemcpySymbolToDevFn =
      M.getOrInsertFunction("__kitrt_hipMemcpySymbolToDevice",
                            VoidTy,   // returns
                            Int32Ty,  // host pointer
                            Int64Ty,  // device pointer
                            Int64Ty); // number of bytes to copy
}

HipLoop::~HipLoop() {}

void HipLoop::setupLoopOutlineArgs(Function &F, ValueSet &HelperArgs,
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
  for (Value *V : TLInputsFixed) {
    HelperArgs.insert(V);
    HelperInputs.push_back(V);
  }

  for (Value *V : HelperInputs) {
    OrderedInputs.push_back(V);
  }
}

unsigned HipLoop::getIVArgIndex(const Function &F,
                                const ValueSet &Args) const {
  // The argument for the primary induction variable is the second input.
  return 1;
}

unsigned HipLoop::getLimitArgIndex(const Function &F,
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

void HipLoop::preProcessTapirLoop(TapirLoopInfo &TL,
                                  ValueToValueMapTy &VMap) {

  // TODO: process loop prior to outlining to do GPU/HIP-specific things
  // like capturing global variables, etc.
  LLVM_DEBUG(dbgs() << "hipabi: preprocessing tapir loop for kernel '"
                    << KernelName << "', in module '" << KernelModule.getName()
                    << "'.\n");

  // Collect the top-level entities (Function, GlobalVariable, GlobalAlias
  // and GlobalIFunc) that are used in the outlined loop. Since the outlined
  // loop will live in the KernelModule, any GlobalValue's used in it will
  // need to be cloned into the KernelModule and then register with HIP
  // in the HIP-centric ctor.
  std::set<GlobalValue *> UsedGlobalValues;

  LLVM_DEBUG(dbgs() << "\tgathering and analyzing global values...\n");

  Loop &L = *TL.getLoop();

  for (Loop *SL : L)
    for (BasicBlock *BB : SL->blocks())
      collect(*BB, UsedGlobalValues);

  for (BasicBlock *BB : L.blocks())
    collect(*BB, UsedGlobalValues);

  // Clone global variables (TODO: and aliases).
  for (GlobalValue *V : UsedGlobalValues) {
    if (GlobalVariable *G = dyn_cast<GlobalVariable>(V)) {
      // TODO: Make sure this logic makes sense...
      //
      // We don't necessarily need a GPU-side clone of a
      // global variable -- instead we need a location where
      // we can copy symbol information over from the host.
      GlobalVariable *NewG = new GlobalVariable(
          KernelModule, G->getValueType(), false,
          GlobalValue::ExternalWeakLinkage,
          (Constant *)Constant::getNullValue(G->getValueType()),
          G->getName() + "_devvar", (GlobalVariable *)nullptr);

      VMap[G] = NewG;

      LLVM_DEBUG(dbgs() << "\tcreated kernel-side global variable '"
                        << NewG->getName() << "'.\n");
      TTarget->pushGlobalVariable(G);

    } else if (dyn_cast<GlobalAlias>(V)) {
      llvm_unreachable("kitsune: GlobalAlias not implemented.");
    }
  }

  // Create declarations for all functions first. These may be needed in the
  // global variables and aliases.
  for (GlobalValue *G : UsedGlobalValues) {
    if (Function *F = dyn_cast<Function>(G)) {
      Function *DeviceF = KernelModule.getFunction(F->getName());
      if (not DeviceF) {
        LLVM_DEBUG(dbgs() << "\t\tanalyzing missing kernel function '"
                          << F->getName() << "'...\n");
        Function *LF = resolveDeviceFunction(F);
        if (LF && not KernelModule.getFunction(LF->getName())) {
          LLVM_DEBUG(dbgs() << "\t\t\tcreated *libdevice* function for '"
                            << LF->getName() << "'.\n");
          DeviceF = Function::Create(LF->getFunctionType(), F->getLinkage(),
                                     LF->getName(), KernelModule);
        } else {
          LLVM_DEBUG(dbgs() << "\t\t\tcreated device function '"
                            << F->getName() << "'.\n");
          DeviceF = Function::Create(F->getFunctionType(), F->getLinkage(),
                                     F->getName(), KernelModule);
        }
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
    if (dyn_cast<GlobalIFunc>(V)) {
      llvm_unreachable("kitsune: GlobalIFunc not yet supported.");
    }
  }

  // Now clone any function bodies that need to be cloned. This should be
  // done as late as possible so that the VMap is populated with any other
  // global values that need to be remapped.
  for (GlobalValue *v : UsedGlobalValues) {
    if (Function *F = dyn_cast<Function>(v)) {
      if (F->size()) {
        SmallVector<ReturnInst *, 8> Returns;
        Function *DeviceF = cast<Function>(VMap[F]);
        CloneFunctionInto(DeviceF, F, VMap,
                          CloneFunctionChangeType::DifferentModule, Returns);

        LLVM_DEBUG(dbgs() << "hipabi: cloning device function '"
                          << DeviceF->getName() << "' into kernel module.\n");

        // GPU calls are slow, try to force inlining...
        DeviceF->addFnAttr(Attribute::AlwaysInline);
      }
    }
  }
  LLVM_DEBUG(dbgs() << "\tfinished preprocessing tapir loop.\n");
}

void HipLoop::postProcessOutline(TapirLoopInfo &TLI,
                                 TaskOutlineInfo &Out,
                                 ValueToValueMapTy &VMap) {
  LLVMContext &Ctx = M.getContext();
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
  // Helper->setLinkage(Function::ExternalLinkage);

  // Set the target features for the helper.
  AttrBuilder Attrs;
  Attrs.addAttribute("target-cpu", GPUArch);
  Helper->removeFnAttr("target-cpu");
  Helper->removeFnAttr("target-features");
  Helper->removeFnAttr("personality");
  Helper->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
  Helper->addAttributes(AttributeList::FunctionIndex, Attrs);
  // Verify that the Thread ID corresponds to a valid iteration.  Because
  // Tapir loops use canonical induction variables, valid iterations range
  // from 0 to the loop limit with stride 1.  The End argument encodes the
  // loop limit. Get end and grain size arguments
  Argument *End;
  Value *Grainsize;
  {
    // TODO: We really only want a grain size of 1 for now...
    auto OutlineArgsIter = Helper->arg_begin();
    // End argument is the first LC arg.
    End = &*OutlineArgsIter++;

    // Get the grain size value, which is either constant or the third LC
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
  // This is the classic thread ID calculation:
  //      i = blockDim.x * blockIdx.x + threadIdx.x;
  // For now we only generate 1-D thread IDs.
  Value *ThreadIdx = B.CreateCall(KitHipWorkItemIdXFn);
  Value *BlockIdx = B.CreateCall(KitHipWorkGroupIdXFn);
  Value *BlockDim = emitWorkGroupSize(B, 0);

  Value *ThreadIV = B.CreateIntCast(B.CreateAdd(ThreadIdx,
                            B.CreateMul(BlockIdx, BlockDim, "blk_offset"),
                           "hipthread_id"),
                           PrimaryIV->getType(),
                           false, "thread_iv");

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

Function *HipLoop::resolveDeviceFunction(Function *Fn) {

  if (Function *KernFn = KernelModule.getFunction(Fn->getName())) {
    LLVM_DEBUG(dbgs() << "\tfound device function: "
                      << Fn->getName() << "()\n");
    return KernFn;
  }
  Function *DevFn = nullptr;
  const std::string AMDGCN_PREFIX = "llvm.amdgcn.";
  std::string IntrinsicName = AMDGCN_PREFIX + Fn->getName().str();
  Intrinsic::ID IntrinsicID = Fn->lookupIntrinsicID(IntrinsicName);
  if (IntrinsicID != Intrinsic::not_intrinsic) {
    DevFn = Intrinsic::getDeclaration(&KernelModule, IntrinsicID);
    LLVM_DEBUG(dbgs() << "\tfound device intrinsic: "
                      << IntrinsicName << "()\n");
  }
  return DevFn;
}

void HipLoop::transformForGCN(Function &F) {

  LLVM_DEBUG(dbgs() << "Transforming function '" << F.getName() << "' "
                    << "in preparation for AMDGPU code generation.\n");

  LLVMContext &Ctx = KernelModule.getContext();
  IRBuilder<> B(F.getEntryBlock().getFirstNonPHI());

  // Compute blockDim.x * blockIdx.x + threadIdx.x;
  Value *tidv = B.CreateCall(KitHipWorkItemIdXFn, {}, "thread_idx");
  Value *ntidv = B.CreateCall(KitHipWorkGroupIdXFn, {}, "block_idx");
  Value *ctaidv = emitWorkGroupSize(B, 0);
  LLVM_DEBUG(dbgs() << "launch details:\n";
                    ctaidv->getType()->dump();
                    tidv->getType()->dump());

  Value *tidoff = B.CreateMul(ctaidv, ntidv, "block_off");
  Value *gtid = B.CreateAdd(tidoff, tidv, "t_idx");

  // We now need to walk the kernel (outlined loop) and look for
  // unresolved function calls.  In particular we need to check
  // to see if they can be resolved via hip/rocm provided bitcode
  // files...  This is messy and undefined...
  LLVM_DEBUG(dbgs() << "hipabi: search for unresolved functions...\n");
  std::list<CallInst *> Replaced;
  for (auto I = inst_begin(&F); I != inst_end(&F); I++) {
    if (auto CI = dyn_cast<CallInst>(&*I)) {
      Function *CF = CI->getCalledFunction();
      if (CF->size() == 0 && not CF->isIntrinsic()) {
        Function *DF = resolveDeviceFunction(CF);
        if (DF != nullptr) {
          CallInst *NCI = dyn_cast<CallInst>(CI->clone());
          NCI->insertAfter(CI);
          NCI->setCalledFunction(DF);
          CI->replaceAllUsesWith(NCI);
          Replaced.push_back(CI);
        }
      }
    }
  }

  for (auto CI : Replaced)
    CI->eraseFromParent();

  std::vector<Instruction *> TIDs;
  for (auto &F : KernelModule) {
    for (auto &BB : F) {
      for (auto &I : BB) {
        if (auto *CI = dyn_cast<CallInst>(&I)) {
          if (Function *F = CI->getCalledFunction()) {
            if (F->getName() == "gtid")
              TIDs.push_back(&I);
          }
        }
      }
    }
  }

  for (auto P : TIDs) {
    P->replaceAllUsesWith(gtid);
    P->eraseFromParent();
  }

  if (auto *F = KernelModule.getFunction("gtid"))
    F->eraseFromParent();

  if (KeepIntermediateFiles) {
    std::error_code EC;
    std::unique_ptr<ToolOutputFile> KernelIRFile;
    SmallString<255> IRFileName(Twine(F.getName()).str() + "-amdgcn");
    sys::path::replace_extension(IRFileName, ".ll");
    KernelIRFile = std::make_unique<ToolOutputFile>(
        IRFileName, EC, sys::fs::OpenFlags::OF_None);
    KernelModule.print(KernelIRFile->os(), nullptr);
    KernelIRFile->keep();
  }
}

void HipLoop::processOutlinedLoopCall(TapirLoopInfo &TL,
                                      TaskOutlineInfo &TOI,
                                      DominatorTree &DT) {

  LLVM_DEBUG(dbgs() << "\tprocessing outlined loop call for kernel '"
                    << KernelName << "'.\n");

  LLVMContext &Ctx = M.getContext();
  PointerType *VoidPtrTy = Type::getInt8PtrTy(Ctx);

  Function *Parent = TOI.ReplCall->getFunction();
  Value *TripCount = OrderedInputs[0];
  BasicBlock *RCBB = TOI.ReplCall->getParent();
  BasicBlock *NBB = RCBB->splitBasicBlock(TOI.ReplCall);
  TOI.ReplCall->eraseFromParent();

  IRBuilder<> B(&NBB->front());

  Function &F = *KernelModule.getFunction(KernelName.c_str());
  transformForGCN(F);

  BasicBlock &EBB = Parent->getEntryBlock();
  IRBuilder<> EB(&EBB.front());

  ArrayType *ArrayTy = ArrayType::get(VoidPtrTy, OrderedInputs.size());
  Value *ArgArray = EB.CreateAlloca(ArrayTy);
  unsigned int i = 0;
  for (Value *V : OrderedInputs) {
    Value *VP = EB.CreateAlloca(V->getType());
    B.CreateStore(V, VP);
    Value *VoidVPtr = B.CreateBitCast(VP, VoidPtrTy);
    Value *ArgPtr = B.CreateConstInBoundsGEP2_32(ArrayTy, ArgArray, 0, i);
    B.CreateStore(VoidVPtr, ArgPtr);
    i++;

    // TODO: This is still experimental and obviously lacking any
    // significant heuristics about when to issue a prefetch...
    if (!CodeGenDisablePrefetch) {
      Type *VT = V->getType();
      if (VT->isPointerTy()) {
        Value *VoidPP = B.CreateBitCast(V, VoidPtrTy);
        B.CreateCall(KitHipMemPrefetchFn, {VoidPP});
      }
    }
  }

  const DataLayout &DL = M.getDataLayout();
  Value *argsPtr = B.CreateConstInBoundsGEP2_32(ArrayTy, ArgArray, 0, 0);
  // Generate a call to launch the kernel.
  Constant *KNameCS = ConstantDataArray::getString(Ctx, KernelName);
  GlobalVariable *KNameGV =
      new GlobalVariable(M, KNameCS->getType(), true,
                         GlobalValue::PrivateLinkage, KNameCS, ".str");
  KNameGV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  Type *StrTy = KNameGV->getType();
  Constant *Zeros[] = {ConstantInt::get(DL.getIndexType(StrTy), 0),
                       ConstantInt::get(DL.getIndexType(StrTy), 0)};
  Constant *KNameParam =
      ConstantExpr::getGetElementPtr(KNameGV->getValueType(), KNameGV, Zeros);

  // We can't get to the complete fat binary data until all loops in the
  // input module have been processed (i.e., the complete kernel module is
  // populated, converted to GCN, turned into an assembled binary, etc.).
  // Because of this we create a "stand in" (dummy) here and will replace
  // it later in the ABI's transformation pipeline.
  Constant *DummyFBGV =
      tapir::getOrInsertFBGlobal(M, "_hipabi.dummy_fatbin", VoidPtrTy);
  Value *DummyFBPtr = B.CreateLoad(VoidPtrTy, DummyFBGV);

  Value *Stream;
  Type *Int64Ty = Type::getInt64Ty(Ctx);
  CastInst *TCCI = nullptr;
  if (TripCount->getType() != Int64Ty) {
    TCCI = CastInst::CreateIntegerCast(TripCount, Int64Ty, false);
    B.Insert(TCCI, "tcci");
  }

  if (!TTarget->hasGlobalVariables()) {
    LLVM_DEBUG(dbgs() << "\t\tcreating no-globals kernel launch.\n");
    if (TCCI)
      Stream = B.CreateCall(KitHipLaunchFn,
                            {DummyFBPtr, KNameParam, argsPtr, TCCI}, "stream");

    else
      Stream =
          B.CreateCall(KitHipLaunchFn,
                       {DummyFBPtr, KNameParam, argsPtr, TripCount}, "stream");
  } else {
    LLVM_DEBUG(dbgs() << "\t\tcreating kernel launch w/ globals.\n");
    Value *CM = B.CreateCall(KitHipCreateFBModuleFn, {DummyFBPtr});
    if (TCCI)
      Stream = B.CreateCall(KitHipModuleLaunchFn,
                            {CM, KNameParam, argsPtr, TCCI}, "stream");
    else
      Stream = B.CreateCall(KitHipModuleLaunchFn,
                            {CM, KNameParam, argsPtr, TripCount}, "stream");
  }

  // LLVM_DEBUG(dbgs() << "\t\tfinishing outlined loop with sync call.\n");
  B.CreateCall(KitHipWaitFn, Stream);
}

// ----- Hip Target

// As is the pattern with the GPU targets, the HipABI is setup to process
// all Tapir constructs within a given input Module (M).  It then creates
// a corresponding module that contains the transformed device-side code.
// This is the KernelModule that is created below in the target
// constructor.

HipABI::HipABI(Module &InputModule)
    : TapirTarget(InputModule), // This becomes 'M' inside the Target.
      KernelModule(HIPABI_PREFIX + InputModule.getName().str(),
                   InputModule.getContext()) {

  LLVM_DEBUG(dbgs() << "hipabi: creating target for module: '"
                    << M.getName() << "'\n");

  // Build the details we need for the LLVM-side target.
  std::string ArchString = "amdgcn";
  Triple TargetTriple(ArchString, "amd", "amdhsa");
  std::string Error;
  const Target *AMDGPUTarget = TargetRegistry::lookupTarget("",
                                                TargetTriple, Error);
  if (not AMDGPUTarget) {
    errs() << "hipabi: target lookup failed! '" << Error << "'\n";
    report_fatal_error("hipabi: unable to find registered HIP target. "
                       "Was LLVM built with the AMDGPU target enabled?");
  }

  AMDTargetMachine = AMDGPUTarget->createTargetMachine(
      TargetTriple.getTriple(), GPUArch, "", TargetOptions(), Reloc::PIC_,
      CodeModel::Large, CodeGenOpt::Aggressive);

  KernelModule.setTargetTriple(TargetTriple.str());
  KernelModule.setDataLayout(AMDTargetMachine->createDataLayout());

  LLVM_DEBUG(dbgs() << "\tcreated target: "
                    << TargetTriple.getTriple() << "\n");
}

HipABI::~HipABI() {
  LLVM_DEBUG(dbgs() << "hipabi: destroying target.\n");
}

Value *HipABI::lowerGrainsizeCall(CallInst *GrainsizeCall) {
  // TODO: The grain size on the GPU is a completely different beast
  // than the CPU cases Tapir was originally designed for.  At present
  // keeping the grain size at 1 has almost always shown to yield the
  // best results in terms of performance but we should take a closer
  // look...  We have some tweaks for experimenting with this via the
  // command line but it remains unexplored.
  Value *Grainsize;
  Grainsize = ConstantInt::get(GrainsizeCall->getType(), DefaultGrainSize);
  // Replace uses of grain size intrinsic call with a computed
  // grain size value.
  GrainsizeCall->replaceAllUsesWith(Grainsize);
  GrainsizeCall->eraseFromParent();
  return Grainsize;
}

// We can't create a correct launch sequence until all the kernels
// within a (LLVM) module are generated.  When post-processing the
// module we create the fatbinary and then to revisit the kernel
// launch calls we created at the loop level and replace the fat
// binary pointer/handle with the completed version.
//
// In addition, we must copy data for global variables from the
// host to the device prior to kernel launches.  This requires
// digging some additional details out of the fat binary.
void HipABI::finalizeLaunchCalls(Module &M, GlobalVariable *BundleBin) {

  LLVM_DEBUG(dbgs() << "\tpatching kernel launch calls...\n");

  LLVMContext &Ctx = M.getContext();
  const DataLayout &DL = M.getDataLayout();
  Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *VoidPtrTy = Type::getInt8PtrTy(Ctx);
  PointerType *CharPtrTy = Type::getInt8PtrTy(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);

  // Look up a global (device-side) symbol via a module
  // created from the fat binary.
  FunctionCallee KitHipGetGlobalSymbolFn =
      M.getOrInsertFunction("__kitrt_hipGetGlobalSymbol",
                            Int64Ty,    // device pointer
                            CharPtrTy,  // symbol name
                            VoidPtrTy); // HIP "module"

  FunctionCallee KitHipMemcpyToDeviceFn =
      M.getOrInsertFunction("__kitrt_hipMemcpySymbolToDevice",
                            VoidTy,    // returns
                            VoidPtrTy, // host ptr
                            Int64Ty,   // device ptr
                            Int64Ty);  // num bytes

  // There are two forms of kernel launch we need to search for.  The first
  // is a kernel launch without any global variables in use.  In this case
  // we have a simple replacement of the first parameter with the now complete
  // fat binary.
  //
  // The second case is a kernel launch with globals.  In this case, we need to
  // find the corresponding global within the fat binary and then issue a copy
  // of the host side data to the device (prior to the kernel launch).
  // Therefore this path is bit more complex as we have to find the creation of
  // the module that requires the updated fat binary, then fetch the
  // device pointer for each global, issue a corresponding memcpy, and then
  // launch the kernel.
  auto &FnList = M.getFunctionList();
  for (auto &Fn : FnList) {
    for (auto &BB : Fn) {
      for (auto &I : BB) {
        if (CallInst *CI = dyn_cast<CallInst>(&I)) {
          if (Function *CFn = CI->getCalledFunction()) {
            if (CFn->getName().startswith("__kitrt_hipLaunchFBKernel")) {
              Value *CBundleBin;
              CBundleBin = CastInst::CreateBitOrPointerCast(BundleBin,
                                                         VoidPtrTy,
                                                         "_hipbin.fatbin",
                                                         CI);
              CI->setOperand(0, CBundleBin);
            } else if (CFn->getName().startswith("__kitrt_hipCreateFBModule")) {
              Value *CBundleBin;
              CBundleBin = CastInst::CreateBitOrPointerCast(BundleBin,
                                                            VoidPtrTy,
                                                            "_hipbin.fatbin",
                                                            CI);
              CI->setOperand(0, CBundleBin);

              Instruction *NI = CI->getNextNonDebugInstruction();
              // Unless something else has monkeyed with our generated code
              // NI should be the launch call...  However, that's not critical
              // but we do need the following instructions to codegen between
              // CI and the launch...
              assert(NI && "unexpected null instruction!");
              for (auto &HostGV : GlobalVars) {
                std::string DevVarName = HostGV->getName().str() + "_devvar";
                Value *SymName = tapir::createConstantStr(DevVarName, M, DevVarName);
                Value *DevPtr =
                    CallInst::Create(KitHipGetGlobalSymbolFn, {SymName, CI},
                                     ".hipabi_devptr", NI);
                Value *VGVPtr =
                    CastInst::CreatePointerCast(HostGV, VoidPtrTy, "", NI);
                uint64_t NumBytes = DL.getTypeAllocSize(HostGV->getValueType());
                CallInst::Create(
                    KitHipMemcpyToDeviceFn,
                    {VGVPtr, DevPtr, ConstantInt::get(Int64Ty, NumBytes)}, "",
                    NI);
              }
            }
          }
        }
      }
    }
  }

  GlobalVariable *DummyFB = M.getGlobalVariable("_hipabi.dummy_fatbin", true);
  if (DummyFB) {
    Constant *CFB =
        ConstantExpr::getPointerCast(BundleBin, VoidPtrTy->getPointerTo());
    LLVM_DEBUG(dbgs() << "\tcleaning up dummy fatbin global.\n");
    DummyFB->replaceAllUsesWith(CFB);
    DummyFB->eraseFromParent();
  } else {
    LLVM_DEBUG(dbgs() << "\twarning! "
                      << "Unable to find dummy fatbin for clean-up!.\n");
  }
}

// Create a AMD equivalent of a "fat binary" object file for
// inclusion in the final generated executable.  This is done
// per module.
HipABIOutputFile HipABI::createBundleFile() {
  LLVM_DEBUG(dbgs() << "hipabi: generating AMD GCN file.\n");

  // At this point we assume the kernel module has all the
  // necessary code for kernels, device side functions,
  // globals, etc. to generate the object file to embed in
  // the final executable.  We have to create a number of
  // files to work with the offload bundler to accomplish
  // this.
  std::error_code EC;

  SmallString<255> ObjFileName(Twine(HIPABI_PREFIX + M.getName()).str());
  sys::path::replace_extension(ObjFileName, ".hip.o");
  std::unique_ptr<ToolOutputFile> ObjFile;
  ObjFile = std::make_unique<ToolOutputFile>(ObjFileName,
                        EC, sys::fs::OpenFlags::OF_None);

  SmallString<255> LinkedObjFileName(ObjFileName.str());
  sys::path::replace_extension(LinkedObjFileName, ".ld.o");
  std::unique_ptr<ToolOutputFile> LinkedObjFile;
  LinkedObjFile = std::make_unique<ToolOutputFile>(LinkedObjFileName,
                        EC, sys::fs::OpenFlags::OF_None);

  if (KeepIntermediateFiles) {
    ObjFile->keep();
    LinkedObjFile->keep();
  }

  SmallString<255> BundleFileName(ObjFileName.str());
  sys::path::replace_extension(BundleFileName, ".bndl.o");
  std::unique_ptr<ToolOutputFile> BundleFile;
  BundleFile =
        std::make_unique<ToolOutputFile>(BundleFileName, EC,
                        sys::fs::OpenFlags::OF_None);
  // TODO: Check EC!

  LLVM_DEBUG(dbgs() << "\tgenerating HIP bundle:\n"
                    << "\t\tobject file: " << ObjFile->getFilename() << "\n"
                    << "\t\tlinked obj file: " << LinkedObjFile->getFilename()<< "\n"
                    << "\t\tbundle file: " << BundleFile->getFilename()
                    << "\n");

  // Note that we created the (LLVM) target details at construction.

  legacy::PassManager PassMgr;
  legacy::FunctionPassManager FnPassMgr(&KernelModule);
  PassManagerBuilder PMBuilder;
  PMBuilder.OptLevel = OptLevel;
  PMBuilder.VerifyInput = 1;
  PMBuilder.Inliner = createFunctionInliningPass(PMBuilder.OptLevel, 0, false);
  PMBuilder.DisableUnrollLoops = false;
  PMBuilder.LoopVectorize = false;  // TODO: to vectorize or not to vectorize?
  PMBuilder.SLPVectorize = false;   // TODO: to vectorize or not to vectorize?
  PMBuilder.populateFunctionPassManager(FnPassMgr);
  PMBuilder.populateModulePassManager(PassMgr);

  bool Fail;
  Fail = AMDTargetMachine->addPassesToEmitFile(PassMgr, ObjFile->os(),
                   nullptr, CodeGenFileType::CGFT_ObjectFile,
                   false);
  if (Fail)
    report_fatal_error("hipabi: emit failed for GCN target!");

  FnPassMgr.doInitialization();
  AMDTargetMachine->adjustPassManager(PMBuilder);
  for(Function &Fn : KernelModule)
    FnPassMgr.run(Fn);
  PassMgr.run(KernelModule);

  // We should now have an object file for the next steps of
  // getting the linked and bundled files created with the
  // clang bundler...

  auto LLDExe = sys::findProgramByName("ld.lld");
  if ((EC = LLDExe.getError()))
    report_fatal_error("'ld.lld' not found! "
                       "check your path?");

  opt::ArgStringList LDDArgList;

  LDDArgList.push_back(LLDExe->c_str());
  std::string mcpu_arg = "-plugin-opt=mcpu=" + GPUArch + ":xnack+";
  LDDArgList.push_back(mcpu_arg.c_str());
  LDDArgList.push_back("-shared");
  LDDArgList.push_back("-plugin-opt=-amdgpu-internalize-symbols");

  std::string optlevel_arg = "-plugin-opt=";
  switch (OptLevel) {
    case 0:
      optlevel_arg += "O0";
      break;
    case 1:
      optlevel_arg += "O1";
      break;
    case 2:
      optlevel_arg += "O2";
      break;
    case 3:
      optlevel_arg += "O3";
      break;
    default:
      llvm_unreachable_internal("unhandled/unexpected optimization level",
                                __FILE__, __LINE__);
  }
  LDDArgList.push_back(optlevel_arg.c_str());
  LDDArgList.push_back("-plugin-opt=-amdgpu-early-inline-all=true");
  LDDArgList.push_back("-plugin-opt=-amdgpu-function-calls=false");
  LDDArgList.push_back("-o");
  std::string linked_objfile(LinkedObjFile->getFilename().str().c_str());
  LDDArgList.push_back(linked_objfile.c_str());
  std::string objfilename (ObjFile->getFilename().str().c_str());
  LDDArgList.push_back(objfilename.c_str());
  LDDArgList.push_back(nullptr);

  auto LDDArgs = toStringRefArray(LDDArgList.data());
  LLVM_DEBUG(dbgs() << "hipabi: ld.lld command line:\n";
             unsigned c = 0;
             for(auto dbg_arg : LDDArgs) {
               dbgs() << "\t" << c << ". " << dbg_arg << "\n";
               c++;
             }
             dbgs() << "\n\n";
  );

  std::string ErrMsg;
  bool ExecFailed;
  int ExecStat = sys::ExecuteAndWait(*LLDExe, LDDArgs, None, {},
                                     0, // secs to wait -- 0 --> unlimited.
                                     0, // memory limit -- 0 --> unlimited.
                                     &ErrMsg, &ExecFailed);
  if (ExecFailed)
    report_fatal_error("hipabi: 'ldd' execution failed!");
  if (ExecStat != 0)
    report_fatal_error("hipabi: 'ldd' failure - " + ErrMsg);

  auto Bundler = sys::findProgramByName("clang-offload-bundler");
  if ((EC = Bundler.getError()))
    report_fatal_error("'clang-offload-bundler' not found! "
                       "check your path?");
  opt::ArgStringList BundleArgList;
  BundleArgList.push_back(Bundler->c_str());
  BundleArgList.push_back("-type=o");
  std::string input_args = "-inputs=/dev/null," +
                           LinkedObjFile->getFilename().str();
  BundleArgList.push_back(input_args.c_str());

  std::string target_arg = "-targets=host-" + M.getTargetTriple() +
                           ",hipv4-" + KernelModule.getTargetTriple() +
                           "--" + GPUArch.c_str();
  BundleArgList.push_back(target_arg.c_str());

  std::string output_arg = "--outputs=" + BundleFile->getFilename().str();
  BundleArgList.push_back(output_arg.c_str());
  BundleArgList.push_back(nullptr);

  auto BundleArgs = toStringRefArray(BundleArgList.data());
  LLVM_DEBUG(dbgs() << "hipabi: clang offload bundler command line:\n";
             unsigned c = 0;
             for(auto dbg_arg : BundleArgs) {
               dbgs() << "\t" << c << ". " << dbg_arg << "\n";
               c++;
             }
             dbgs() << "\n\n";
  );

  ExecStat = sys::ExecuteAndWait(*Bundler, BundleArgs, None, {},
                                 0, // secs to wait. 0 is unlimited.
                                 0, // memory limit. 0 is unlimited.
                                 &ErrMsg, &ExecFailed);
  if (ExecFailed)
    report_fatal_error("hipabi: 'clang-offload-bundler' execution failed!");

  if (ExecStat != 0)
    report_fatal_error("hipabi: 'clang-offload-bundler' failure - " + ErrMsg);

  BundleFile->keep();
  return BundleFile;
}

GlobalVariable *
HipABI::embedBundle(HipABIOutputFile &BundleFile) {
  std::unique_ptr<llvm::MemoryBuffer> Bundle = nullptr;
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
      MemoryBuffer::getFile(BundleFile->getFilename());

  if (std::error_code EC = BufferOrErr.getError()) {
    report_fatal_error("hipabi: failed to load bundle file: " +
                       EC.message());
  }

  Bundle = std::move(BufferOrErr.get());
  LLVM_DEBUG(dbgs() << "read binary bundle file, "
                    << Bundle->getBufferSize() << " bytes.\n");

  LLVMContext &Ctx = M.getContext();
  Type *Int8Ty = Type::getInt8Ty(Ctx);
  Constant *BundleArray = ConstantDataArray::getRaw(
                  StringRef(Bundle->getBufferStart(),
                            Bundle->getBufferSize()),
                  Bundle->getBufferSize(), Int8Ty);
  GlobalVariable *BundleGV;
  BundleGV = new GlobalVariable(M, BundleArray->getType(),
                                true, GlobalValue::PrivateLinkage,
                                BundleArray, "_hipabi_bundle_ptr");
  return BundleGV;
}

void HipABI::bindGlobalVariables(Value *Handle, IRBuilder<> &B) {
  LLVMContext &Ctx = M.getContext();
  const DataLayout &DL = M.getDataLayout();
  Type *IntTy = Type::getInt32Ty(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);
  Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *VoidPtrTy = Type::getInt8PtrTy(Ctx);
  PointerType *VoidPtrPtrTy = VoidPtrTy->getPointerTo();
  Type *VarSizeTy = Int64Ty;
  PointerType *CharPtrTy = Type::getInt8PtrTy(Ctx);

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

Function *HipABI::createCtor(GlobalVariable *Bundle,
                             GlobalVariable *Wrapper) {
  LLVMContext &Ctx = M.getContext();
  Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *VoidPtrTy = Type::getInt8PtrTy(Ctx);
  PointerType *VoidPtrPtrTy = VoidPtrTy->getPointerTo();
  Type *IntTy = Type::getInt32Ty(Ctx);

  Function *CtorFn = Function::Create(
      FunctionType::get(VoidTy, VoidPtrTy, false),
      GlobalValue::InternalLinkage,
      HIPABI_PREFIX + ".ctor." + M.getName(), &M);

  BasicBlock *CtorEntryBB = BasicBlock::Create(Ctx, "entry", CtorFn);
  IRBuilder<> CtorBuilder(CtorEntryBB);
  const DataLayout &DL = M.getDataLayout();

  // Tuck the call to initialize the Kitsune runtime into the constructor;
  // this in turn will initialize HIP.
  FunctionCallee KitRTInitFn = M.getOrInsertFunction("__kitrt_hipInit", VoidTy);
  CtorBuilder.CreateCall(KitRTInitFn, {});

  FunctionCallee RegisterFatbinaryFn =
      M.getOrInsertFunction("__hipRegisterFatBinary",
                            FunctionType::get(VoidPtrPtrTy,
                                              VoidPtrTy,
                                              false));
  CallInst *RegFatbin = CtorBuilder.CreateCall(RegisterFatbinaryFn,
                            CtorBuilder.CreateBitCast(Wrapper, VoidPtrTy));

  GlobalVariable *Handle = new GlobalVariable(M, VoidPtrPtrTy,
                                    false, GlobalValue::InternalLinkage,
                                    ConstantPointerNull::get(VoidPtrPtrTy),
                                    HIPABI_PREFIX + ".fbhand");
  Handle->setAlignment(Align(DL.getPointerABIAlignment(0)));
  CtorBuilder.CreateAlignedStore(RegFatbin, Handle,
                                 DL.getPointerABIAlignment(0));
  Handle->setUnnamedAddr(GlobalValue::UnnamedAddr::None);

  Value *HandlePtr = CtorBuilder.CreateLoad(VoidPtrPtrTy, Handle,
                                            HIPABI_PREFIX + ".fbhand.ptr");

  if (!GlobalVars.empty()) {
    LLVM_DEBUG(dbgs() << "\tbinding host and device global variables...\n");
    bindGlobalVariables(HandlePtr, CtorBuilder);
  }

  #if 0 // do we need this for HIP?
  // Wrap up bundle/fatbinary registration steps...
  FunctionCallee EndFBRegistrationFn =
      M.getOrInsertFunction("__hipRegisterFatBinaryEnd",
                  FunctionType::get(VoidTy,
                                    VoidPtrPtrTy, // cubin handle.
                                    false));
  CtorBuilder.CreateCall(EndFBRegistrationFn, RegFatbin);
  #endif

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

Function *HipABI::createDtor(GlobalVariable *BundleHandle) {
  LLVMContext &Ctx = M.getContext();
  const DataLayout &DL = M.getDataLayout();
  Type *VoidTy = Type::getVoidTy(Ctx);
  Type *VoidPtrTy = Type::getInt8PtrTy(Ctx);
  Type *VoidPtrPtrTy = VoidPtrTy->getPointerTo();

  FunctionCallee UnregisterFatbinFn =
      M.getOrInsertFunction("__hipUnregisterFatBinary",
                            FunctionType::get(VoidTy, VoidPtrPtrTy, false));

  Function *DtorFn = Function::Create(
      FunctionType::get(VoidTy, VoidPtrTy, false), GlobalValue::InternalLinkage,
      HIPABI_PREFIX + ".dtor", &M);

  // TODO: Do we call into this too many times???
  BasicBlock *DtorEntryBB = BasicBlock::Create(Ctx, "entry", DtorFn);
  IRBuilder<> DtorBuilder(DtorEntryBB);
  Value *HandleValue = DtorBuilder.CreateAlignedLoad(
      VoidPtrPtrTy, BundleHandle, DL.getPointerABIAlignment(0));
  DtorBuilder.CreateCall(UnregisterFatbinFn, HandleValue);

  FunctionCallee KitRTDestroyFn =
      M.getOrInsertFunction("__kitrt_hipDestroy", VoidTy);
  DtorBuilder.CreateCall(KitRTDestroyFn, {});

  DtorBuilder.CreateRetVoid();
  return DtorFn;
}

void HipABI::registerBundle(GlobalVariable *Bundle) {
  const int BundleMagicID = 0x48495046;
  const char *BundleConstantName = ".hip_fatbin";
  const char *BundleSectionName = ".hipFatBinSegment";
  const char *ModuleIDSectionName = "__hip_module_id";
  StringRef ModuleIDPrefix = "__hip_";

  const DataLayout &Layout = M.getDataLayout();
  Type *BundleStrTy = Bundle->getType();
  Constant *Zeros[] = {
        ConstantInt::get(Layout.getIndexType(BundleStrTy), 0),
        ConstantInt::get(Layout.getIndexType(BundleStrTy), 0)
  };

  Constant *BundlePtr =  ConstantExpr::getGetElementPtr(Bundle->getValueType(),
                                                     Bundle,
                                                     Zeros);
  LLVMContext &Ctx = M.getContext();
  const DataLayout &DL = M.getDataLayout();
  Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *VoidPtrTy = Type::getInt8PtrTy(Ctx);
  Type *IntTy = Type::getInt32Ty(Ctx);

  StructType *WrapperTy = StructType::get(IntTy,      // magic #
                                          IntTy,      // version
                                          VoidPtrTy,  // data
                                          VoidPtrTy); // unused for now.
  Constant *WrapperS = ConstantStruct::get(WrapperTy,
      ConstantInt::get(IntTy, 0x48495046),
      ConstantInt::get(IntTy, 1),
      BundlePtr,
      ConstantPointerNull::get(VoidPtrTy));
  Bundle->setSection(".hipFatBinSegment ");

  GlobalVariable *Wrapper = new GlobalVariable(M, WrapperTy, true,
                                    GlobalValue::InternalLinkage,
                                    WrapperS, "_hipabi_wrapper");
  Wrapper->setSection("__hip_module_id");
  // TODO: There are some odd alignment details for HIP inside Clang's
  //lowering -- not sure what the details are for that...
  Wrapper->setAlignment(Align(DL.getPrefTypeAlignment(Wrapper->getType())));

  // The rest of the registration details are tucked into a constructor
  // entry...
  Function *CtorFn = createCtor(Bundle, Wrapper);
  if (CtorFn) {
    FunctionType *CtorFnTy = FunctionType::get(VoidTy, false);
    Type *CtorFnPtrTy =
        PointerType::get(CtorFnTy, M.getDataLayout().getProgramAddressSpace());
    tapir::appendToGlobalCtors(M,
                               ConstantExpr::getBitCast(CtorFn, CtorFnPtrTy),
                               65536, nullptr);
  }
}

void HipABI::postProcessModule() {
  LLVM_DEBUG(dbgs() << "hipabi: post processing and finalizing module '"
                    << M.getName() << "'.\n");

  // At this point we know all tapir constructs in the
  // input module (M) have been processed and the kernel
  // module is populated with the corresponding transformed
  // code.
  // TODO: This is not entirely accurate...  Right now the
  // process of mixing non-GPU parallel constructs (e.g.,
  // spawn + sync) is not well defined for this
  // transformation.
  HipABIOutputFile BundleFile = createBundleFile();
  GlobalVariable *Bundle = embedBundle(BundleFile);
  finalizeLaunchCalls(M, Bundle);
  registerBundle(Bundle);

  // TODO: Is it worth doing any of this????
  //
  // EXPERIMENTAL:  (re)run a series of optimization passes
  // over the original input module now that we've outlined
  // the parallel constructs.  It is not at all clear there
  // are any benefits to this at all.
  if (RunHostPostOpt) {
    LLVM_DEBUG(dbgs() << "\trunning post-outlining optimization passes.\n");
    legacy::PassManager PassMgr;
    legacy::FunctionPassManager FnPassMgr(&M);
    PassManagerBuilder Builder;
    Builder.Inliner = createFunctionInliningPass(OptLevel, 0, false);
    Builder.OptLevel = OptLevel;
    Builder.SizeLevel = 0;
    Builder.VerifyInput = 1;
    Builder.DisableUnrollLoops = false;
    Builder.LoopVectorize = true;
    Builder.SLPVectorize = true;
    Builder.populateFunctionPassManager(FnPassMgr);
    Builder.populateModulePassManager(FnPassMgr);
    FnPassMgr.doInitialization();
    for(Function &Fn : M)
      FnPassMgr.run(Fn);
    FnPassMgr.doFinalization();
    PassMgr.run(M);
  }

  if (not KeepIntermediateFiles) {
    sys::fs::remove(BundleFile->getFilename());
    // TODO: Check the bundling code for cleanup details...
  }
}

LoopOutlineProcessor *
HipABI::getLoopOutlineProcessor(const TapirLoopInfo *TL) {
  std::string ModuleName = M.getName().str();
  std::string KernelName;

  if (M.getNamedMetadata("llvm.dbg")) {
    // If we have debug info in the module, use a line number-based
    // naming scheme for kernels.
    unsigned LineNumber = TL->getLoop()->getStartLoc()->getLine();
    KernelName = HIPABI_KERNEL_NAME_PREFIX + ModuleName +
                 "_" + Twine(LineNumber).str();
  } else
    KernelName = HIPABI_KERNEL_NAME_PREFIX + ModuleName;

  HipLoop *Outliner = new HipLoop(M, KernelModule, KernelName, this);
  return Outliner;
}

void HipABI::pushGlobalVariable(GlobalVariable *GV) {
  GlobalVars.push_back(GV);
}
