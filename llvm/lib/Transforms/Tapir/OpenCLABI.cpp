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
  Type *SPIRVInt64Ty = Type::getInt64Ty(SPIRVM.getContext());
  GetThreadIdx = SPIRVM.getOrInsertFunction("_Z13get_global_idj", // not sure why this mangled name is the key, but it is
                                          SPIRVInt64Ty, SPIRVInt32Ty);
  Function* getid = SPIRVM.getFunction("_Z13get_global_idj"); 
  getid->setCallingConv(CallingConv::SPIR_FUNC); 
  //GetBlockIdx = SPIRVM.getOrInsertFunction("get_local_id",
   //                                      SPIRVInt32Ty, SPIRVInt32Ty);
  //GetBlockDim = SPIRVM.getOrInsertFunction("get_local_size",
  //                                      SPIRVInt32Ty, SPIRVInt32Ty);

  Type *VoidTy = Type::getVoidTy(M.getContext());
  Type *VoidPtrTy = Type::getInt8PtrTy(M.getContext());
  Type *Int8Ty = Type::getInt8Ty(M.getContext());
  Type *Int32Ty = Type::getInt32Ty(M.getContext());
  Type *Int64Ty = Type::getInt64Ty(M.getContext());
  KitsuneOpenCLInit = M.getOrInsertFunction("__kitsune_opencl_init", VoidTy);
  KitsuneGPUInitKernel = M.getOrInsertFunction("__kitsune_opencl_init_kernel",
                                               VoidTy, Int32Ty, Int32Ty, VoidPtrTy);
  KitsuneGPUMemWrite = M.getOrInsertFunction("__kitsune_opencl_mem_write",
                                              VoidPtrTy, Int32Ty, VoidPtrTy, Int64Ty, Int8Ty);
  KitsuneGPUMemRead = M.getOrInsertFunction("__kitsune_opencl_mem_read",
                                              VoidTy, Int32Ty, VoidPtrTy, VoidPtrTy, Int64Ty);
  KitsuneGPUSetArg = M.getOrInsertFunction("__kitsune_opencl_set_arg",
                                              VoidTy, Int32Ty, Int32Ty, VoidPtrTy, Int32Ty);
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
    if (auto *VTy = dyn_cast<PointerType>(V->getType())){
      auto gty = PointerType::get(VTy->getPointerElementType(), 1); 
      auto *arg = new Argument(gty, V->getName()); 
      //arg->addAttr(Attribute::NoCapture); 
      HelperArgs.insert(arg); 
    } else {
      HelperArgs.insert(V); 
    }
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

void changeAddressSpace(Value* V, unsigned AS){
  if(isa<PointerType>(V->getType())){
    V->mutateType(PointerType::get(V->getType()->getPointerElementType(), AS)); 
  }
}

void fixAddressSpaces(Function *F){
  // CloneFunctionInto doesn't propagate address space changes to arguments
  // TODO: Should this be done in CloneFunctionInto?
  // Update all instruction types that use pointers that have been moved into an
  // address spaces
  // We cannot use RAUW, as it requires the types to be the same
  SetVector<std::pair<Value*, unsigned>> users;
  SetVector<Value*> processed; 
  for(auto &arg : F->args())
    if(isa<PointerType>(arg.getType()))
      users.insert(std::make_pair(&arg, arg.getType()->getPointerAddressSpace())); 
  
  while(!users.empty()){
    // Hack to handle GEP  
    std::pair<Value*, unsigned> u = *users.begin(); 
    LLVM_DEBUG(dbgs() << "Processing : " << *u.first << "\n"); 
    if(auto *pty = dyn_cast<PointerType>(u.first->getType())){
      LLVM_DEBUG(dbgs() << "Pointer should be global type, is of type: " << *pty << "\n"); 
      changeAddressSpace(u.first, u.second); 
      for(auto *uu : u.first->users()){
        if(!processed.count(uu)){
          users.insert(std::make_pair(uu, u.second)); 
        }
      }
    }
    processed.insert(u.first); 
    users.remove(u); 
  }
}


void SPIRVLoop::postProcessOutline(TapirLoopInfo &TL, TaskOutlineInfo &Out,
                                 ValueToValueMapTy &VMap) {
  LLVMContext &Ctx = M.getContext();
  Type *Int8Ty = Type::getInt8Ty(Ctx);
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  //Type *Int64Ty = Type::getInt64Ty(Ctx);
  //Type *VoidPtrTy = Type::getInt8PtrTy(Ctx);
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
  //Helper->setLinkage(Function::ExternalLinkage);
  Helper->setCallingConv(CallingConv::SPIR_KERNEL); 
  Helper->setName("kitsune_spirv_kernel"); 

  // Get the thread ID for this invocation of Helper.
  IRBuilder<> B(Entry->getTerminator());
  Value *ThreadIdx = B.CreateCall(GetThreadIdx, ConstantInt::get(Int32Ty, 0));
  //Value *BlockIdx = B.CreateCall(GetBlockIdx, ConstantInt::get(Int32Ty, 0));
  //Value *BlockDim = B.CreateCall(GetBlockDim, ConstantInt::get(Int32Ty, 0));
  Value *ThreadID = B.CreateIntCast(ThreadIdx, PrimaryIV->getType(), false);

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
    // ReplaceInstWithInst(gep, GetElementPtrInst::Create(
    if (unsigned ConstGrainsize = TL.getGrainsize())
      Grainsize = ConstantInt::get(PrimaryIV->getType(), ConstGrainsize);
    else
      // Grainsize argument is the third LC arg.
      Grainsize = &*++(++OutlineArgsIter);
  }
  ThreadID = B.CreateMul(ThreadID, Grainsize);
  Value *ThreadEndGrain = B.CreateAdd(ThreadID, Grainsize);
  Value *Cmp = B.CreateICmp(ICmpInst::ICMP_ULT, ThreadEndGrain, End); 
  Value *ThreadEnd = B.CreateSelect(Cmp, ThreadEndGrain, End); 
  Value *Cond = B.CreateICmpUGE(ThreadID, ThreadEnd);

  ReplaceInstWithInst(Entry->getTerminator(), BranchInst::Create(Exit, Header,
                                                                 Cond));
  // Use the thread ID as the start iteration number for the primary IV.
  PrimaryIVInput->replaceAllUsesWith(ThreadID);

  // Update cloned loop condition to use the thread-end value.
  /*
  unsigned TripCountIdx = 0;
  ICmpInst *ClonedCond = cast<ICmpInst>(VMap[TL.getCondition()]);
  if (ClonedCond->getOperand(0) != ThreadEnd)
    ++TripCountIdx;
  assert(ClonedCond->getOperand(TripCountIdx) == ThreadEnd &&
         "End argument not used in condition");
  ClonedCond->setOperand(TripCountIdx, ThreadEnd);
  */
  // Fix argument pointer types to global, nocapture
  // TODO: read/write attributes?
  for(auto &arg : Helper->args()){
    if (auto *apty = dyn_cast<PointerType>(arg.getType())){
      arg.mutateType(PointerType::get(apty->getPointerElementType(), 1)); 
      arg.addAttr(Attribute::NoCapture); 
    }
  }
  SmallVector<Type*, 8> paramTys; 
  for(auto &arg : Helper->args()){
    paramTys.push_back(arg.getType()); 
  }
  ArrayRef<Type*> newParams(paramTys); 
  if(auto *fpty = dyn_cast<PointerType>(Helper->getType())){
    LLVM_DEBUG(dbgs() << "Helper is pointer" << *Helper->getType() << "\n"); 
    if(auto *fty = dyn_cast<FunctionType>(fpty->getPointerElementType())){
      LLVM_DEBUG(dbgs() << "Helper is pointer to function" << *Helper->getType() << "\n"); 
      Helper->mutateType(PointerType::get(FunctionType::get(fty->getReturnType(), newParams, false), 0)); 
    }
  }
  LLVM_DEBUG(dbgs() << "Function type after globalization of argument pointers << " << *Helper->getType() << "\n"); 
  LLVM_DEBUG(dbgs() << "SPIRVM after globalization of argument pointers << " << *Helper->getParent() << "\n"); 

  fixAddressSpaces(Helper); 

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
  //Type *Int8Ty = Type::getInt8Ty(Ctx);
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  //Type *Int64Ty = Type::getInt64Ty(Ctx);
  Type *VoidPtrTy = Type::getInt8PtrTy(Ctx);

  //Task *T = TL.getTask();
  //Instruction *ReplCall = cast<CallBase>(TOI.ReplCall);
  Function *Parent = TOI.ReplCall->getFunction();

  //Value* TripCount = isSRetInput(TOI.InputSet[0]) ? TOI.InputSet[1] : TOI.InputSet[0]; 
  //Value *RunStart = ReplCall->getArgOperand(getIVArgIndex(*Parent,
  //                                                        TOI.InputSet));
  //Value *TripCount = ReplCall->getArgOperand(getLimitArgIndex(*Parent,
  //                                                            TOI.InputSet));
  Value *TripCount = OrderedInputs[0];
  IRBuilder<> B(TOI.ReplCall);

  Value *KernelID = ConstantInt::get(Int32Ty, MyKernelID);
  Value *SPIRVPtr = B.CreateBitCast(SPIRVGlobal, VoidPtrTy);

  Constant *kernelSize = ConstantInt::get(Int32Ty, 
    SPIRVGlobal->getInitializer()->getType()->getArrayNumElements()); 
  BasicBlock &EBB = Parent->getEntryBlock(); 
  IRBuilder<> EB(&EBB.front()); 
  EB.CreateCall(KitsuneOpenCLInit, {});
  EB.CreateCall(KitsuneGPUInitKernel, { KernelID, kernelSize, SPIRVPtr });

  DataLayout DL(Parent->getParent());
  IRBuilder<> RB(TOI.ReplCall->getParent()->getTerminator()); 
  int ArgID = 0; 
  for (Value *V : OrderedInputs) {
    //Value *ElementSize = nullptr;
    LLVM_DEBUG(dbgs() << "Input set value: " << *V << "\n"); 
    Type *VTy = V->getType();
    Value *ArgIDV = ConstantInt::get(Int32Ty, ArgID++); 
    Value *VoidVPtr, *VSize; 
    if (auto VPTy = dyn_cast<PointerType>(VTy)){
      unsigned m = 0;
      for (const User *U : V->users()) {
        if (const GetElementPtrInst *Ind = dyn_cast<GetElementPtrInst>(U)) {
          for(const User *U : Ind->users()){
            // TODO: What about pointers to pointers?
            // TODO: Less hacky solution
            if (isa<LoadInst>(U))
              m |= 1;
            else if (isa<StoreInst>(U))
              m |= 2;
          }
        }
      }
      CallInst *mmap_marker = nullptr;  
      for(auto &BB : *Parent) if(DT.dominates(&BB, TOI.ReplCall->getParent())){
        for(auto &I : BB){
          if(auto *CI = dyn_cast<CallInst>(&I)){
            if(Function *fun = CI->getCalledFunction()){
              if(fun->getName() == "__kitsune_opencl_mmap_marker"){
                Value *OPtr = CI->getArgOperand(0); 
                if(auto *BCI = dyn_cast<CastInst>(OPtr)){
                  if(BCI->getOperand(0) == V){
                    mmap_marker = CI; 
                  }
                }
                if(auto *VCI = dyn_cast<CastInst>(V)){
                  if(VCI->getOperand(0) == OPtr){
                    mmap_marker = CI;
                  }
                }
              }
            }
          }
        }
      }
      assert(mmap_marker && "Couldn't find mmap for opencl parallel for"); 
      IRBuilder<> MMB(mmap_marker); 
      VoidVPtr = mmap_marker->getArgOperand(0);  
      Value *N = mmap_marker->getArgOperand(1);  
      Value *mval = B.getInt8(m); 
      Type *VElemType = VPTy->getElementType(); 
      Value *VElemSize = MMB.getInt64(DL.getTypeAllocSize(VElemType)); 
      Value *VArrSize = MMB.CreateMul(VElemSize, N); 
      Value *VPtr = B.CreateBitCast(V, VoidPtrTy);
      V = MMB.CreateCall(KitsuneGPUMemWrite, { B.getInt32(MyKernelID), VPtr, VArrSize, mval }); 
      if(m & 2) RB.CreateCall(KitsuneGPUMemRead, {B.getInt32(MyKernelID), VPtr, V, VArrSize }); 
    }
    Value *VPtr = B.CreateAlloca(V->getType()); 
    B.CreateStore(V, VPtr); 
    VoidVPtr = B.CreateBitCast(VPtr, VoidPtrTy);
    VSize = B.getInt32(DL.getTypeAllocSize(VTy)); 
    B.CreateCall(KitsuneGPUSetArg, { KernelID, ArgIDV, VoidVPtr, VSize });
  }


  Value *RunSize = B.CreateSub(TripCount, ConstantInt::get(TripCount->getType(), 1));
  
  B.CreateCall(KitsuneGPUSetRunSize, { KernelID, RunSize });

  B.CreateCall(KitsuneGPURunKernel, { KernelID });

  RB.CreateCall(KitsuneGPUFinish, {});

  //TOI.ReplCall->eraseFromParent();
}

