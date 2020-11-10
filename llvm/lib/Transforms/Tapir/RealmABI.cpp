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
  if(CreateBar) return CreateBar; 
  LLVMContext &C = M.getContext(); 

  AttributeList AL; 
  FunctionType *FTy = FunctionType::get(
    getBarrierType(C), {}, false);
  CreateBar = M.getOrInsertFunction("createRealmBarrier", FTy, AL);
}

FunctionCallee RealmABI::get_destroyRealmBarrier(){
  if(DestroyBar) return DestroyBar; 
  LLVMContext &C = M.getContext(); 

  AttributeList AL; 
  FunctionType *FTy = FunctionType::get(
    Type::getInt8Ty(C), {getBarrierType(C)}, false);
  DestroyBar = M.getOrInsertFunction("destroyRealmBarrier", FTy, AL);
}

FunctionCallee RealmABI::get_realmSpawn() {
  if(RealmSpawn)
    return RealmSpawn;

  LLVMContext &C = M.getContext(); 
  const DataLayout &DL = M.getDataLayout();
  AttributeList AL;

  Type* TypeArray[] = { TaskFuncPtrTy,         // TaskFuncPtr fxn
			Type::getInt8PtrTy(C), // const void *args
			DL.getIntPtrType(C),   // size_t arglen
			Type::getInt8PtrTy(C), // void *user_data
			DL.getIntPtrType(C)};  // size_t user_data_len
  
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
  Type* TypeArray[] = { Type::getInt8PtrTy(C),   // const void *args
			DL.getIntPtrType(C),     // size_t arglen 
			Type::getInt8PtrTy(C),   // const void *user_data
			DL.getIntPtrType(C),     // size_t user_data_len
			DL.getIntPtrType(C)};    // unsigned long long proc

  TaskFuncPtrTy = PointerType::getUnqual(
      FunctionType::get(Type::getInt8Ty(C),      // returns void 
			TypeArray,
			false));
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
  Value *Workers = Builder.CreateCall(REALM_FUNC(realmGetNumProcs));
  //Value *Workers = Builder.CreateCall(get_realmGetNumProcs()); // no macro
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
    barrier = CallInst::Create(get_createRealmBarrier(), {}, "",
                            F->getEntryBlock().getTerminator());
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
#endif

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

  //Create the canonical TaskFuncPtr
  //trying int64 as stand-in for Realm::Processor because a ::realm_id_t is ultimately an unsigned long long
  Type* typeArray[] = {Type::getInt8PtrTy(C), // const void*
		       DL.getIntPtrType(C),   // size_t
		       Type::getInt8PtrTy(C), // const void*
		       DL.getIntPtrType(C),   // size_t
		       DL.getIntPtrType(C)};  // unsigned long long

  FunctionType *RealmFnTy = FunctionType::get(
      Type::getInt8Ty(C), 
      typeArray,
      false);

  Function *RealmFn = Function::Create(
      RealmFnTy, GlobalValue::InternalLinkage, ".realm_outlined.", M);
  RealmFn->addFnAttr(Attribute::AlwaysInline);
  RealmFn->addFnAttr(Attribute::NoUnwind);
  RealmFn->addFnAttr(Attribute::UWTable);

  //get the argument types
  std::vector<Type*> FnParams = Outlined->getFunctionType()->params();
  StructType *ArgsTy = StructType::create(FnParams, "realm_arg_struct");
  PointerType *ArgsPtrTy = PointerType::getUnqual(ArgsTy);

  std::vector<Value*> out_args;
  for (auto &Arg : RealmFn->args()) {
    Arg.setName("");
    out_args.push_back(&Arg);
  }

  // Entry Code for newly-minted Function
  BasicBlock *EntryBB = BasicBlock::Create(C, "entry", RealmFn, nullptr);
  IRBuilder<> EntryBuilder(EntryBB);
  Value *argStructPtr = EntryBuilder.CreateBitCast(out_args[0], ArgsPtrTy); 
  ValueToValueMapTy valmap;

  unsigned int argc = 0;
  for (auto& arg : Outlined->args()) {
    auto *DataAddrEP = EntryBuilder.CreateStructGEP(ArgsTy, argStructPtr, argc); 
    auto *DataAddr = EntryBuilder.CreateAlignedLoad(
        DataAddrEP,
        DL.getTypeAllocSize(DataAddrEP->getType()->getPointerElementType()));
    valmap.insert(std::pair<Value*,Value*>(&arg,DataAddr));
    argc++;
  }

  // Replace return values with return zero 
  SmallVector< ReturnInst *,5> retinsts;
  CloneFunctionInto(RealmFn, Outlined, valmap, true, retinsts);
  EntryBuilder.CreateBr(RealmFn->getBasicBlockList().getNextNode(*EntryBB));

  for (auto& ret : retinsts) {
    auto retzero = ReturnInst::Create(C, ConstantInt::get(Type::getInt8Ty(C), 0));
    ReplaceInstWithInst(ret, retzero);
  }

  // Caller code
  IRBuilder<> CallerIRBuilder(ReplCall);
  AllocaInst *CallerArgStruct = CallerIRBuilder.CreateAlloca(ArgsTy); 
  std::vector<Value*> LoadedCapturedArgs;
  CallInst *cal = dyn_cast<CallInst>(ReplCall); //could also be TOI.ReplCall

  for(auto& a:cal->arg_operands()) {
    LoadedCapturedArgs.push_back(a);
  }

  unsigned int cArgc = 0;
  for (auto& arg : LoadedCapturedArgs) {
    auto *DataAddrEP2 = CallerIRBuilder.CreateStructGEP(ArgsTy, CallerArgStruct, cArgc); 
    CallerIRBuilder.CreateAlignedStore(
        LoadedCapturedArgs[cArgc], DataAddrEP2,
        DL.getTypeAllocSize(arg->getType()));
    cArgc++;
  }

  assert(argc == cArgc && "Wrong number of arguments passed to outlined function"); 

  Value *RealmFnPtr = CallerIRBuilder.CreatePointerBitCastOrAddrSpaceCast(
                                        RealmFn, TaskFuncPtrTy);
  ConstantInt *ArgSize = ConstantInt::get(DL.getIntPtrType(C), ArgsTy->getNumElements()); 
  ConstantInt *ArgDataSize = ConstantInt::get(DL.getIntPtrType(C), DL.getTypeAllocSize(ArgsTy)); 
  Value *ArgsStructVoidPtr = CallerIRBuilder.CreateBitCast(CallerArgStruct, Type::getInt8PtrTy(C)); 

  CallerIRBuilder.SetInsertPoint(ReplStart);
  CallerIRBuilder.CreateLifetimeStart(CallerArgStruct, ArgDataSize);

  std::vector<Value*> callerArgs = { RealmFnPtr, 
				     ArgsStructVoidPtr, 
				     ArgSize, 
				     ArgsStructVoidPtr, 
				     ArgDataSize}; 

  CallInst *Call = CallerIRBuilder.CreateCall(REALM_FUNC(realmSpawn), callerArgs); 
  Call->setDebugLoc(ReplCall->getDebugLoc());

  CallerIRBuilder.SetInsertPoint(CallBlock, ++Call->getIterator());
  CallerIRBuilder.CreateLifetimeEnd(CallerArgStruct, ArgDataSize);

  TOI.replaceReplCall(Call);
  ReplCall->eraseFromParent();

  if (TOI.ReplUnwind)
    // We assume that realmSpawn dealt with the exception.  But
    // replacing the invocation of the helper function with the call to
    // realmSpawn will remove the terminator from CallBlock.  Restore
    // that terminator here.
    BranchInst::Create(TOI.ReplRet, CallBlock);

  //LLVM_DEBUG(RealmFn->dump()); 

#if 0
  // NOTE: the code from here through the end of processSubTaskCall is only
  // retained for reference purposes and because things are undergoing
  // development.  It should be deleted if not used once debugging is complete.
  Function *RealmFnPtr = formatFunctionToRealmF(Outlined, ReplStart);

  // At this point, we have a call in the parent to a function containing the
  // task body.  That function takes as its argument a pointer to a structure
  // containing the inputs to the task body.  This structure is initialized in
  // the parent immediately before the call.

  // To match the Realm ABI, we replace the existing call with a call to
  // realmSync from the kitsune-rt realm wrapper.
  IRBuilder<> CallerIRBuilder(ReplCall);
  Value *OutlinedFnPtr = CallerIRBuilder.CreatePointerBitCastOrAddrSpaceCast(
      Outlined, TaskFuncPtrTy);
  AllocaInst *CallerArgStruct = cast<AllocaInst>(ReplCall->getArgOperand(0));
  Type *ArgsTy = CallerArgStruct->getAllocatedType();

  Value *ArgStructPtr = CallerIRBuilder.CreateBitCast(CallerArgStruct,
  						      Type::getInt8PtrTy(C));
  ConstantInt *ArgSize = ConstantInt::get(DL.getIntPtrType(C),
                                          DL.getTypeAllocSize(ArgsTy));
  ConstantInt *ArgNum = ConstantInt::get(DL.getIntPtrType(C),
    					 ArgsTy->getNumContainedTypes());

  CallerIRBuilder.SetInsertPoint(ReplStart);
  CallerIRBuilder.CreateLifetimeStart(CallerArgStruct, ArgSize);

  Value* CallArgs[] = { OutlinedFnPtr, 
			ArgStructPtr, 
			ArgNum,
			ArgStructPtr,
			ArgSize};

  CallInst *Call = CallerIRBuilder.CreateCall(REALM_FUNC(realmSpawn), CallArgs); 
  Call->setDebugLoc(ReplCall->getDebugLoc());

  // Add lifetime intrinsics for the argument struct.  TODO: Move this logic
  // into underlying LoweringUtils routines?
  //CallerIRBuilder.SetInsertPoint(ReplStart);
  //CallerIRBuilder.CreateLifetimeStart(CallerArgStruct, ArgSize);
  CallerIRBuilder.SetInsertPoint(CallBlock, ++Call->getIterator());
  CallerIRBuilder.CreateLifetimeEnd(CallerArgStruct, ArgSize);

  TOI.replaceReplCall(Call);
  ReplCall->eraseFromParent();

  if (TOI.ReplUnwind)
    // We assume that realmSpawn dealt with the exception.  But
    // replacing the invocation of the helper function with the call to
    // realmSpawn will remove the terminator from CallBlock.  Restore
    // that terminator here.
    BranchInst::Create(TOI.ReplRet, CallBlock);

  // VERIFY: If we're using realmSpawn, we don't need a separate helper
  // function to manage the allocation of the argument structure.
#endif
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
