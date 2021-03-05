//===- RealmABI.cpp - Lower Tapir into Realm runtime system calls -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the RealmABI interface, which is used to convert Tapir
// instructions -- detach, reattach, and sync -- to calls into the Realm
// runtime system.  
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Tapir/RealmABI.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Tapir/Outline.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/TapirUtils.h"
#include <vector>
#include <iostream>

using namespace llvm;

#define DEBUG_TYPE "realmabi"

FunctionCallee RealmABI::get_realmGetNumProcs() {
  if(RealmGetNumProcs)
    return RealmGetNumProcs;

  LLVMContext &C = M.getContext(); 
  const DataLayout &DL = M.getDataLayout();
  AttributeList AL;

  std::vector<Type*> TypeArray;
  // TODO: Set appropriate function attributes.
  FunctionType *FTy = FunctionType::get(DL.getIntPtrType(C),
					TypeArray, 
					false);
  RealmGetNumProcs = M.getOrInsertFunction("realmGetNumProcs", FTy, AL);
  return RealmGetNumProcs;
}

static StructType* getBarrierType(LLVMContext &C){
  auto eventTy = StructType::get(Type::getInt64Ty(C));
  return StructType::get(eventTy, Type::getInt64Ty(C));
}

FunctionCallee RealmABI::get_createRealmBarrier(){
  if(CreateBar) 
    return CreateBar; 
  LLVMContext &C = M.getContext(); 

  AttributeList AL; 
  FunctionType *FTy = FunctionType::get(
    getBarrierType(C), {}, false);
  CreateBar = M.getOrInsertFunction("createRealmBarrier", FTy, AL);
  return CreateBar;
}

FunctionCallee RealmABI::get_destroyRealmBarrier(){
  if(DestroyBar) 
    return DestroyBar; 
  LLVMContext &C = M.getContext(); 

  AttributeList AL; 
  FunctionType *FTy = FunctionType::get(
    Type::getInt8Ty(C), {getBarrierType(C)}, false);
  DestroyBar = M.getOrInsertFunction("destroyRealmBarrier", FTy, AL);
  return DestroyBar;
}

FunctionCallee RealmABI::get_realmSpawn() {
  if(RealmSpawn)
    return RealmSpawn;

  LLVMContext &C = M.getContext(); 
  const DataLayout &DL = M.getDataLayout();
  AttributeList AL;

  Type* TypeArray[] = { RealmFTy,              // RealmFTy fxn
			Type::getInt8PtrTy(C), // void *args
			DL.getIntPtrType(C)};  // size_t argsize

  // TODO: Set appropriate function attributes.
  FunctionType *FTy = FunctionType::get(
      Type::getInt32Ty(C),     // returns int
      TypeArray,
      false);
  RealmSpawn = M.getOrInsertFunction("realmSpawn", FTy, AL);
  return RealmSpawn;
}

FunctionCallee RealmABI::get_realmSync() {
  if(RealmSync)
    return RealmSync;

  LLVMContext &C = M.getContext(); 
  AttributeList AL;
  
  Type* TypeArray[] = { getBarrierType(C) };
  // TODO: Set appropriate function attributes.
  FunctionType *FTy = FunctionType::get(Type::getInt8Ty(C), 
					TypeArray, 
					false);
  RealmSync = M.getOrInsertFunction("realmSync", FTy, AL);
  return RealmSync;
}

FunctionCallee RealmABI::get_realmInitRuntime() {
  if(RealmInitRuntime)
    return RealmInitRuntime;

  LLVMContext &C = M.getContext(); 
  AttributeList AL;
  
  Type* TypeArray[] = { Type::getInt32Ty(C),                            // int argc
			PointerType::getUnqual(Type::getInt8PtrTy(C))}; // char **argv

  // TODO: Set appropriate function attributes.
  FunctionType *FTy = FunctionType::get(
      Type::getInt32Ty(C),                            // returns int
      TypeArray,
      false);

  RealmInitRuntime = M.getOrInsertFunction("realmInitRuntime", FTy, AL);
  return RealmInitRuntime;
}

FunctionCallee RealmABI::get_realmFinalize() {
  if(RealmFinalize)
    return RealmFinalize;

  LLVMContext &C = M.getContext(); 
  AttributeList AL;

  std::vector<Type*> TypeArray;
  // TODO: Set appropriate function attributes.
  FunctionType *FTy = FunctionType::get(Type::getInt8PtrTy(C), 
					TypeArray, 
					false);
  RealmFinalize = M.getOrInsertFunction("realmFinalize", FTy, AL);
  return RealmFinalize;
}

#define REALM_FUNC(name) get_##name()

RealmABI::RealmABI(Module &M) : TapirTarget(M) {
  LLVMContext &C = M.getContext();
  const DataLayout &DL = M.getDataLayout();
  // Initialize any types we need for lowering.
  // NOTE: RealmFTy is NOT the same as a Realm::Processor::TaskFuncPtr
  RealmFTy = PointerType::getUnqual(
      FunctionType::get(Type::getInt64Ty(C), { Type::getInt8PtrTy(C) }, false));
}

RealmABI::~RealmABI() {
  //call something that deletes the context struct
}

/// Lower a call to get the grainsize of this Tapir loop.
///
/// The grainsize is computed by the following equation:
///
///     Grainsize = min(2048, ceil(Limit / (8 * workers)))
///
/// This computation is inserted into the preheader of the loop.
Value *RealmABI::lowerGrainsizeCall(CallInst *GrainsizeCall) {
  Value *Limit = GrainsizeCall->getArgOperand(0);
  IRBuilder<> Builder(GrainsizeCall);

  // Get 8 * workers
  Value *Workers = Builder.CreateCall(get_realmGetNumProcs());
  Value *WorkersX8 = Builder.CreateIntCast(
      Builder.CreateMul(Workers, ConstantInt::get(Workers->getType(), 8)),
      Limit->getType(), false);
  // Compute ceil(limit / 8 * workers) =
  //           (limit + 8 * workers - 1) / (8 * workers)
  Value *SmallLoopVal =
    Builder.CreateUDiv(Builder.CreateSub(Builder.CreateAdd(Limit, WorkersX8),
                                         ConstantInt::get(Limit->getType(), 1)),
                       WorkersX8);
  // Compute min
  Value *LargeLoopVal = ConstantInt::get(Limit->getType(), 2048);
  Value *Cmp = Builder.CreateICmpULT(LargeLoopVal, SmallLoopVal);
  Value *Grainsize = Builder.CreateSelect(Cmp, LargeLoopVal, SmallLoopVal);

  // Replace uses of grainsize intrinsic call with this grainsize value.
  GrainsizeCall->replaceAllUsesWith(Grainsize);
  return Grainsize;
}

Value *RealmABI::getOrCreateBarrier(Value *SyncRegion, Function *F) {
  LLVMContext &C = M.getContext();
  Value* barrier;
  if((barrier = SyncRegionToBarrier[SyncRegion]))
    return barrier;
  else {
    IRBuilder<> builder(F->getEntryBlock().getTerminator());
    AllocaInst* ab = builder.CreateAlloca(getBarrierType(C)); 
    barrier = ab; 
    Value *barrierVal = builder.CreateCall(get_createRealmBarrier(), {}, "");
    builder.CreateAlignedStore(barrierVal, barrier, ab->getAlignment()); 
    SyncRegionToBarrier[SyncRegion] = barrier;

    // Make sure we destroy the barrier at all exit points to prevent memory leaks
    for(BasicBlock &BB : *F) {
      if(isa<ReturnInst>(BB.getTerminator())){
        CallInst::Create(get_destroyRealmBarrier(), {barrier}, "",
                         BB.getTerminator());
      }
    }
    
    return barrier;
  }
}

#if 0
//old sync (non-barrier)
void RealmABI::lowerSync(SyncInst &SI) {
  IRBuilder<> builder(&SI); 

  std::vector<Value *> args;  //realmSync takes no arguments
  auto sincwait = REALM_FUNC(realmSync);
  //auto sincwait = get_realmSync();  // why don't we just do this? no macro
  builder.CreateCall(sincwait, args);

  BranchInst *PostSync = BranchInst::Create(SI.getSuccessor(0));
  ReplaceInstWithInst(&SI, PostSync);
  return;
}
#endif //old sync

void RealmABI::lowerSync(SyncInst &SI) {
  IRBuilder<> builder(&SI); 
  auto F = SI.getParent()->getParent(); 
  auto& C = M.getContext(); 
  Value* SR = SI.getSyncRegion(); 
  auto barrier = getOrCreateBarrier(SR, F); 
  std::vector<Value *> args = {barrier}; 
  builder.CreateCall(get_realmSync(), args);

  BranchInst *PostSync = BranchInst::Create(SI.getSuccessor(0));
  ReplaceInstWithInst(&SI, PostSync);
  return;
}

void RealmABI::processSubTaskCall(TaskOutlineInfo &TOI, DominatorTree &DT) {
  Function *Outlined = TOI.Outline;
  Instruction *ReplStart = TOI.ReplStart;
  CallBase *ReplCall = cast<CallBase>(TOI.ReplCall);
  BasicBlock *CallBlock = ReplStart->getParent();

  LLVMContext &C = M.getContext();
  const DataLayout &DL = M.getDataLayout();

  // At this point, we have a call in the parent to a function containing the
  // task body.  That function takes as its argument a pointer to a structure
  // containing the inputs to the task body.  This structure is initialized in
  // the parent immediately before the call.

  // To match the kitsune-rt Realm wrapper, we replace the existing call with
  // a call to realmSpawn
  IRBuilder<> CallerIRBuilder(ReplCall);
  Value *OutlinedFnPtr = CallerIRBuilder.CreatePointerBitCastOrAddrSpaceCast(
      Outlined, RealmFTy);
  AllocaInst *CallerArgStruct = cast<AllocaInst>(ReplCall->getArgOperand(0));
  Type *ArgsTy = CallerArgStruct->getAllocatedType();
  Value *ArgStructPtr = CallerIRBuilder.CreateBitCast(CallerArgStruct,
                                                      Type::getInt8PtrTy(C));
  ConstantInt *ArgSize = ConstantInt::get(DL.getIntPtrType(C),
                                          DL.getTypeAllocSize(ArgsTy));
  CallInst *Call = CallerIRBuilder.CreateCall(
      get_realmSpawn(), { OutlinedFnPtr, ArgStructPtr, ArgSize});
  Call->setDebugLoc(ReplCall->getDebugLoc());
  TOI.replaceReplCall(Call);
  ReplCall->eraseFromParent();

  // Add lifetime intrinsics for the argument struct.  TODO: Move this logic
  // into underlying LoweringUtils routines?
  CallerIRBuilder.SetInsertPoint(ReplStart);
  CallerIRBuilder.CreateLifetimeStart(CallerArgStruct, ArgSize);
  CallerIRBuilder.SetInsertPoint(CallBlock, ++Call->getIterator());
  CallerIRBuilder.CreateLifetimeEnd(CallerArgStruct, ArgSize);

  if (TOI.ReplUnwind)
    // We assume that realmSpawn dealt with the exception.  But
    // replacing the invocation of the helper function with the call to
    // realmSpawn will remove the terminator from CallBlock.  Restore
    // that terminator here.
    BranchInst::Create(TOI.ReplRet, CallBlock);

  // VERIFY: If we're using realmSpawn, we don't need a separate helper
  // function to manage the allocation of the argument structure.
}

void RealmABI::preProcessFunction(Function &F, TaskInfo &TI,
				  bool OutliningTapirLoops) {
#if 0  
  // NOTE: the code from here through the end of preProcessFunction is only
  // retained for reference purposes and because things are undergoing
  // development.  It should be deleted if not used once debugging is complete.
  if (OutliningTapirLoops)
    // Don't do any preprocessing when outlining Tapir loops.
    return;

  LLVMContext &C = M.getContext();
  Task *T = TI.getRootTask();
  Spindle *endSpindle = *(std::prev(T->spindle_end()));
  BasicBlock *endBlock = *(std::prev(endSpindle->block_end())); //(T->spindle_end())->block_end();
  IRBuilder<> footerB(endBlock->getTerminator());
  std::vector<Value*> submitArgs; //realmSync takes no args
  footerB.CreateCall(REALM_FUNC(realmSync), submitArgs);

#if 0
  for (Task *T : post_order(TI.getRootTask())) {
    if (T->isRootTask()) {
      BasicBlock *Spawned = T->getEntry();
      //Value *SR = Detach->getSyncRegion(); 

      // Add a submit to end of task body
      //
      // TB: I would interpret the above comment to mean we want qt_sinc_submit()
      // before the task terminates.  But the code I see for inserting
      // qt_sinc_submit just inserts the call at the end of the entry block of the
      // task, which is not necessarily the end of the task.  I kept the code I
      // found, but I'm not sure if it is correct.
      IRBuilder<> footerB(Spawned->getTerminator());
      std::vector<Value*> submitArgs; //realmSync takes no args
      footerB.CreateCall(REALM_FUNC(realmSync), submitArgs);
    }
    else
      continue;
  }
#endif
#endif
}

void RealmABI::postProcessFunction(Function &F, bool OutliningTapirLoops) {
  if (OutliningTapirLoops)
    // Don't do any postprocessing when outlining Tapir loops.
    return;

  Module *M = F.getParent();
  LLVMContext &C = M->getContext();
  IRBuilder<> builder(F.getEntryBlock().getFirstNonPHIOrDbg());

  //default values of 0 and nullptr
  //TODO: handle the case where main actually has an argc and argv
  Value* zero = ConstantInt::get(Type::getInt32Ty(C), 0);
  Value* null = Constant::getNullValue(PointerType::getUnqual(Type::getInt8PtrTy(C)));
  
  Value* initArgs[2];
  initArgs[0] = zero;
  initArgs[1] = null;
  
  builder.CreateCall(REALM_FUNC(realmInitRuntime), initArgs);
 
  return;
}

void RealmABI::postProcessHelper(Function &F) {}
