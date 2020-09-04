//===- OpenCLABI.cpp - Lower Tapir to the Kitsune OpenCL back end -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Kitsune OpenCL ABI to convert Tapir instructions to
// calls into the Kitsune runtime system for OpenCL SPIRV code.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Tapir/OpenCLABI.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Tapir/Outline.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Vectorize.h"
#include "llvm/Support/TargetRegistry.h"
#include <LLVMSPIRVLib/LLVMSPIRVLib.h>
#include <sstream>

using namespace llvm;

#define DEBUG_TYPE "openclabi"

Value *OpenCLABI::lowerGrainsizeCall(CallInst *GrainsizeCall) {
  Value *Grainsize = ConstantInt::get(GrainsizeCall->getType(), 8);

  // Replace uses of grainsize intrinsic call with this grainsize value.
  GrainsizeCall->replaceAllUsesWith(Grainsize);
  return Grainsize;
}

void OpenCLABI::lowerSync(SyncInst &SI) {
  // currently a no-op...
}

void OpenCLABI::preProcessFunction(Function &F, TaskInfo &TI,
                                 bool OutliningTapirLoops) {
}

void OpenCLABI::postProcessFunction(Function &F, bool OutliningTapirLoops) {
}

void OpenCLABI::postProcessHelper(Function &F) {
}

void OpenCLABI::processOutlinedTask(Function &F) {
}

void OpenCLABI::processSpawner(Function &F) {
}

void OpenCLABI::processSubTaskCall(TaskOutlineInfo &TOI, DominatorTree &DT) {
}

LoopOutlineProcessor *OpenCLABI::getLoopOutlineProcessor(
    const TapirLoopInfo *TL) const {
  return new SPIRVLoop(M);
}

// Static counter for assigning IDs to kernels.
unsigned SPIRVLoop::NextKernelID = 0;

SPIRVLoop::SPIRVLoop(Module &M)
    : LoopOutlineProcessor(M, SPIRVM), SPIRVM("spirvModule", M.getContext()) {
  // Assign an ID to this kernel.
  MyKernelID = NextKernelID++;

  // Setup an SPIRV triple.
  Triple SPIRVTriple("spir64-unknown-unknown");
  SPIRVM.setTargetTriple(SPIRVTriple.str());

  // Insert runtime-function declarations in SPIRV host modules.
  Type *SPIRVInt32Ty = Type::getInt32Ty(SPIRVM.getContext());
  GetThreadIdx = SPIRVM.getOrInsertFunction("get_global_id",
                                          SPIRVInt32Ty, SPIRVInt32Ty);
  GetBlockIdx = SPIRVM.getOrInsertFunction("get_local_id",
                                         SPIRVInt32Ty, SPIRVInt32Ty);
  GetBlockDim = SPIRVM.getOrInsertFunction("get_local_size",
                                         SPIRVInt32Ty, SPIRVInt32Ty);

  Type *VoidTy = Type::getVoidTy(M.getContext());
  Type *VoidPtrTy = Type::getInt8PtrTy(M.getContext());
  Type *Int8Ty = Type::getInt8Ty(M.getContext());
  Type *Int32Ty = Type::getInt32Ty(M.getContext());
  Type *Int64Ty = Type::getInt64Ty(M.getContext());
  KitsuneOpenCLInit = M.getOrInsertFunction("__kitsune_opencl_init", VoidTy);
  KitsuneGPUInitKernel = M.getOrInsertFunction("__kitsune_opencl_init_kernel",
                                               VoidTy, Int32Ty, Int32Ty, VoidPtrTy);
  KitsuneGPUSetArg = M.getOrInsertFunction("__kitsune_opencl_set_arg",
                                              VoidTy, Int32Ty, Int32Ty, VoidPtrTy, Int32Ty, Int8Ty);
  KitsuneGPUSetRunSize = M.getOrInsertFunction("__kitsune_opencl_set_run_size",
                                               VoidTy, Int32Ty, Int64Ty);
  KitsuneGPURunKernel = M.getOrInsertFunction("__kitsune_opencl_run_kernel",
                                              VoidTy, Int32Ty);
  KitsuneGPUFinish = M.getOrInsertFunction("__kitsune_opencl_finish", VoidTy);
}

void SPIRVLoop::setupLoopOutlineArgs(
    Function &F, ValueSet &HelperArgs, SmallVectorImpl<Value *> &HelperInputs,
    ValueSet &InputSet, const SmallVectorImpl<Value *> &LCArgs,
    const SmallVectorImpl<Value *> &LCInputs, const ValueSet &TLInputsFixed) {
  // Add the loop control inputs.

  // The first parameter defines the extent of the index space, i.e., the number
  // of threads to launch.
  {
    Argument *EndArg = cast<Argument>(LCArgs[1]);
    EndArg->setName("runSize");
    HelperArgs.insert(EndArg);

    Value *InputVal = LCInputs[1];
    HelperInputs.push_back(InputVal);
    // Add loop-control input to the input set.
    InputSet.insert(InputVal);
  }
  // The second parameter defines the start of the index space.
  {
    Argument *StartArg = cast<Argument>(LCArgs[0]);
    StartArg->setName("runStart");
    HelperArgs.insert(StartArg);

    Value *InputVal = LCInputs[0];
    HelperInputs.push_back(InputVal);
    // Add loop-control input to the input set.
    InputSet.insert(InputVal);
  }
  // The third parameter defines the grainsize, if it is not constant.
  if (!isa<ConstantInt>(LCInputs[2])) {
    Argument *GrainsizeArg = cast<Argument>(LCArgs[2]);
    GrainsizeArg->setName("runStride");
    HelperArgs.insert(GrainsizeArg);

    Value *InputVal = LCInputs[2];
    HelperInputs.push_back(InputVal);
    // Add loop-control input to the input set.
    InputSet.insert(InputVal);
  }

  // Add the loop control inputs.
  for (Value *V : TLInputsFixed) {
    assert(!HelperArgs.count(V));
    HelperArgs.insert(V);
    HelperInputs.push_back(V);
  }

  for(Value *V : HelperInputs){
    OrderedInputs.push_back(V);
  }
}

unsigned SPIRVLoop::getIVArgIndex(const Function &F, const ValueSet &Args) const {
  // The argument for the primary induction variable is the second input.
  return 1;
}

unsigned SPIRVLoop::getLimitArgIndex(const Function &F, const ValueSet &Args)
  const {
  // The argument for the loop limit is the first input.
  return 0;
}

void SPIRVLoop::postProcessOutline(TapirLoopInfo &TL, TaskOutlineInfo &Out,
                                 ValueToValueMapTy &VMap) {
  LLVMContext &Ctx = M.getContext();
  Type *Int8Ty = Type::getInt8Ty(Ctx);
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);
  Type *VoidPtrTy = Type::getInt8PtrTy(Ctx);
  Task *T = TL.getTask();
  Loop *L = TL.getLoop();

  Function *Helper = Out.Outline;
  BasicBlock *Entry = cast<BasicBlock>(VMap[L->getLoopPreheader()]);
  BasicBlock *Header = cast<BasicBlock>(VMap[L->getHeader()]);
  BasicBlock *Exit = cast<BasicBlock>(VMap[TL.getExitBlock()]);
  PHINode *PrimaryIV = cast<PHINode>(VMap[TL.getPrimaryInduction().first]);
  Value *PrimaryIVInput = PrimaryIV->getIncomingValueForBlock(Entry);
  Instruction *ClonedSyncReg = cast<Instruction>(
      VMap[T->getDetach()->getSyncRegion()]);

  // We no longer need the cloned sync region.
  ClonedSyncReg->eraseFromParent();

  // Set the helper function to have external linkage.
  Helper->setLinkage(Function::ExternalLinkage);

  // Get the thread ID for this invocation of Helper.
  IRBuilder<> B(Entry->getTerminator());
  Value *ThreadIdx = B.CreateCall(GetThreadIdx, ConstantInt::get(Int32Ty, 0));
  Value *BlockIdx = B.CreateCall(GetBlockIdx, ConstantInt::get(Int32Ty, 0));
  Value *BlockDim = B.CreateCall(GetBlockDim, ConstantInt::get(Int32Ty, 0));
  Value *ThreadID = B.CreateIntCast(
      B.CreateAdd(ThreadIdx, B.CreateMul(BlockIdx, BlockDim), "threadId"),
      PrimaryIV->getType(), false);

  // Verify that the Thread ID corresponds to a valid iteration.  Because Tapir
  // loops use canonical induction variables, valid iterations range from 0 to
  // the loop limit with stride 1.  The End argument encodes the loop limit.
  // Get end and grainsize arguments
  Argument *End;
  Value *Grainsize;
  {
    auto OutlineArgsIter = Helper->arg_begin();
    // End argument is the first LC arg.
    End = &*OutlineArgsIter;

    // Get the grainsize value, which is either constant or the third LC arg.
    if (unsigned ConstGrainsize = TL.getGrainsize())
      Grainsize = ConstantInt::get(PrimaryIV->getType(), ConstGrainsize);
    else
      // Grainsize argument is the third LC arg.
      Grainsize = &*++(++OutlineArgsIter);
  }
  ThreadID = B.CreateMul(ThreadID, Grainsize);
  Value *ThreadEnd = B.CreateAdd(ThreadID, Grainsize);
  Value *Cond = B.CreateICmpUGE(ThreadID, End);

  ReplaceInstWithInst(Entry->getTerminator(), BranchInst::Create(Exit, Header,
                                                                 Cond));
  // Use the thread ID as the start iteration number for the primary IV.
  PrimaryIVInput->replaceAllUsesWith(ThreadID);

  // Update cloned loop condition to use the thread-end value.
  unsigned TripCountIdx = 0;
  ICmpInst *ClonedCond = cast<ICmpInst>(VMap[TL.getCondition()]);
  if (ClonedCond->getOperand(0) != End)
    ++TripCountIdx;
  assert(ClonedCond->getOperand(TripCountIdx) == End &&
         "End argument not used in condition");
  ClonedCond->setOperand(TripCountIdx, ThreadEnd);

  LLVMContext &SPIRVCtx = SPIRVM.getContext();

  SmallVector<Metadata *, 3> AV;
  AV.push_back(ValueAsMetadata::get(Helper));
  AV.push_back(MDString::get(SPIRVCtx, "kernel"));
  AV.push_back(ValueAsMetadata::get(ConstantInt::get(Type::getInt32Ty(SPIRVCtx),
                                                     1)));
  //Annotations->addOperand(MDNode::get(Ctx, AV));

  legacy::PassManager *PassManager = new legacy::PassManager;

  PassManager->add(createVerifierPass());

  // Add in our optimization passes

  //PassManager->add(createInstructionCombiningPass());
  PassManager->add(createReassociatePass());
  PassManager->add(createGVNPass());
  PassManager->add(createCFGSimplificationPass());
  PassManager->add(createSLPVectorizerPass());
  //PassManager->add(createBreakCriticalEdgesPass());
  PassManager->add(createConstantPropagationPass());
  PassManager->add(createDeadInstEliminationPass());
  PassManager->add(createDeadStoreEliminationPass());
  //PassManager->add(createInstructionCombiningPass());
  PassManager->add(createCFGSimplificationPass());
  PassManager->add(createDeadCodeEliminationPass());
  PassManager->run(SPIRVM);

  delete PassManager;

  LLVM_DEBUG(dbgs() << "SPIRV Module: " << SPIRVM);


  // generate spirv kernel code
  std::ostringstream str; 
  std::string ErrMsg; 
  bool success = writeSpirv(&SPIRVM, str, ErrMsg); 
  if(!success){
    std::cerr << "Failed to compile to spirv: " << ErrMsg << std::endl; 
    exit(1); 
  }
  auto s = str.str(); 
  Constant *SPIRV = ConstantDataArray::getRaw(s, s.length(), Int8Ty);
  SPIRVGlobal = new GlobalVariable(M, SPIRV->getType(), true,
                                 GlobalValue::PrivateLinkage, SPIRV,
                                 "spirv_" + Twine(Helper->getName()));

}

void SPIRVLoop::processOutlinedLoopCall(TapirLoopInfo &TL, TaskOutlineInfo &TOI,
                                      DominatorTree &DT) {
  LLVMContext &Ctx = M.getContext();
  Type *Int8Ty = Type::getInt8Ty(Ctx);
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);
  Type *VoidPtrTy = Type::getInt8PtrTy(Ctx);

  Task *T = TL.getTask();
  CallBase *ReplCall = cast<CallBase>(TOI.ReplCall);
  Function *Parent = ReplCall->getFunction();
  Value *RunStart = ReplCall->getArgOperand(getIVArgIndex(*Parent,
                                                          TOI.InputSet));
  Value *TripCount = ReplCall->getArgOperand(getLimitArgIndex(*Parent,
                                                              TOI.InputSet));
  IRBuilder<> B(ReplCall);

  Value *KernelID = ConstantInt::get(Int32Ty, MyKernelID);
  Value *SPIRVPtr = B.CreateBitCast(SPIRVGlobal, VoidPtrTy);

  Constant *kernelSize = ConstantInt::get(Int32Ty, 
    SPIRVGlobal->getInitializer()->getType()->getArrayNumElements()); 

  B.CreateCall(KitsuneOpenCLInit, {});
  B.CreateCall(KitsuneGPUInitKernel, { KernelID, kernelSize, SPIRVPtr });

  int ArgID = 0; 
  for (Value *V : OrderedInputs) {
    Value *ElementSize = nullptr;
    Value *FieldName;
    LLVM_DEBUG(dbgs() << "Input set value: " << *V << "\n"); 
    unsigned m = 0;
    for (const User *U : V->users()) {
      if (const Instruction *I = dyn_cast<Instruction>(U)) {
        // TODO: Properly restrict this check to users within the cloned loop
        // body.  Checking the dominator tree doesn't properly check
        // exception-handling code, although it's not clear we should see such
        // code in these loops.
        if (!DT.dominates(T->getEntry(), I->getParent()))
          continue;

        if (isa<LoadInst>(U))
          m |= 1;
        else if (isa<StoreInst>(U))
          m |= 2;
      }
    }
    Value *Mode = ConstantInt::get(Int8Ty, m);
    Value *VPtr = B.CreateAlloca(V->getType()); 
    Value *VoidVPtr = B.CreateBitCast(VPtr, VoidPtrTy);
    B.CreateStore(V, VPtr); 
    DataLayout DL(Parent->getParent());
    Value *VSize = B.getInt32(DL.getTypeAllocSize(V->getType())); 
    Value *ArgIDV = ConstantInt::get(Int32Ty, ArgID++); 
    B.CreateCall(KitsuneGPUSetArg, { KernelID, ArgIDV, VoidVPtr, VSize, Mode });
  }


  Value *RunSize = B.CreateSub(TripCount, ConstantInt::get(TripCount->getType(),
                                                           1));
  B.CreateCall(KitsuneGPUSetRunSize, { KernelID, RunSize });

  B.CreateCall(KitsuneGPURunKernel, { KernelID });

  B.CreateCall(KitsuneGPUFinish, {});

  ReplCall->eraseFromParent();
}

/*
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"
#include "llvm/Transforms/Tapir/LoopSpawningTI.h"
#include "llvm/Transforms/Tapir/OpenCLABI.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/IR/IntrinsicInst.h"
#include <LLVMSPIRVLib/LLVMSPIRVLib.h>
#include <sstream>

using namespace llvm;

void RuntimeCilkFor::processOutlinedLoopCall(TapirLoopInfo &TL,
                                             TaskOutlineInfo &TOI,
                                             DominatorTree &DT) {
  Function *Outlined = TOI.Outline;
  Instruction *ReplStart = TOI.ReplStart;
  Instruction *ReplCall = TOI.ReplCall;
  CallSite CS(ReplCall);
  BasicBlock *CallCont = TOI.ReplRet;
  BasicBlock *UnwindDest = TOI.ReplUnwind;
  Function *Parent = ReplCall->getFunction();
  Module &M = *Parent->getParent();
  unsigned IVArgIndex = getIVArgIndex(*Parent, TOI.InputSet);
  Type *PrimaryIVTy =
      CS.getArgOperand(IVArgIndex)->getType();
  Value *TripCount = CS.getArgOperand(IVArgIndex + 1);
  Value *GrainsizeVal = CS.getArgOperand(IVArgIndex + 2);

  // TODO: calls into external functions
  // TODO: captured values
  
  // Types
  LLVMContext &Ctx = M.getContext();
  Type *VoidTy = Type::getVoidTy(M.getContext());
  Type *Int8Ty = Type::getInt8Ty(Ctx);
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);
  Type *VoidPtrTy = Type::getInt8PtrTy(Ctx);
  IRBuilder<> B(Out.ReplCall); 

  // spirv function
  Function *OL = Out.Outline; 
  Function *CLF = Function::Create(OL->getFunctionType(), OL->getLinkage(), OL->getName()); 
  CLM.getFunctionList().push_back(CLF);
  CLM.getOrInsertFunction(CLF->getName(), CLF->getFunctionType(),
                          CLF->getAttributes());
  

void OpenCL::postProcessOutline(TapirLoopInfo &TL, TaskOutlineInfo &Out,
                                 ValueToValueMapTy &VMap) {
  
  // spirv module
  Module CLM(M.getModuleIdentifier() + "_SPIRV", M.getContext()); 
  CLM.setTargetTriple("spir64-unknown-unknown");

  // spirv function
  Function *OL = Out.Outline; 
  Function *CLF = Function::Create(OL->getFunctionType(), OL->getLinkage(), OL->getName()); 
  CLM.getFunctionList().push_back(CLF);
  CLM.getOrInsertFunction(CLF->getName(), CLF->getFunctionType(),
                          CLF->getAttributes());
  SmallVector<ReturnInst*, 8> Returns;
  ValueToValueMapTy spirv_vmap;
  auto spirvArg = CLF->arg_begin();
  for(auto &arg : OL->args()) 
    spirv_vmap[&arg] = &*spirvArg++;
  
  CloneFunctionInto(CLF, OL, spirv_vmap, false, Returns); 
  for(BasicBlock &BB : *CLF){
    for(Instruction &I : BB){
      IntrinsicInst *II = dyn_cast<IntrinsicInst>(&I);
      if(II){
        if(II->getIntrinsicID() == Intrinsic::syncregion_start){
          II->eraseFromParent(); 
          break; 
        }
      }
    }
  }

  // remove old outlined loop
  // OL->eraseFromParent();  // causes issues with later hint removal pass

  // generate spirv kernel code
  std::ostringstream str; 
  std::string ErrMsg; 
  bool success = writeSpirv(&CLM, str, ErrMsg); 
  if(!success){
    std::cerr << "Failed to compile to spirv: " << ErrMsg << std::endl; 
    exit(1); 
  }
  auto s = str.str(); 
  Constant *SPIRV = ConstantDataArray::getRaw(s, s.length(), Int8Ty);
  SPIRVKernel = new GlobalVariable(M, SPIRV->getType(), true,
                                 GlobalValue::PrivateLinkage, SPIRV,
                                 "spirv_" + Twine(OL->getName()));
  Value *SPIRVPtr = B.CreateBitCast(SPIRVKernel, VoidPtrTy);
  Constant *kernelSize = ConstantInt::get(Int32Ty, 
    SPIRVKernel->getInitializer()->getType()->getArrayNumElements()); 

  // insert runtime call
  auto KitsuneOpenCLCall = M.getOrInsertFunction("_kitsune_cl_call", VoidTy, VoidPtrTy, Int32Ty, Int64Ty);
  auto NewReplCall = B.CreateCall(KitsuneOpenCLCall, { SPIRVPtr, kernelSize, TL.getTripCount() }); 
  Loop *L = TL.getLoop(); 
  //BasicBlock *Entry = L->getLoopPreheader();
  //BasicBlock *Exit = TL.getExitBlock();
  //ReplaceInstWithInst(Entry->getTerminator(), BranchInst::Create(Exit));
  Out.replaceReplCall(NewReplCall); 
}
*/

