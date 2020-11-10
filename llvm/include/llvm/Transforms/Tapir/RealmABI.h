//===- RealmABI.h - Interface to the Realm runtime ----*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass is a simple pass wrapper around the PromoteMemToReg function call
// exposed by the Utils library.
//
//===----------------------------------------------------------------------===//
#ifndef REALM_ABI_H_
#define REALM_ABI_H_

#define REALM_ENABLE_C_BINDINGS TRUE

//#include "llvm/Transforms/Tapir/LoopSpawning.h"
#include "llvm/Transforms/Tapir/LoweringUtils.h"

namespace llvm {

class RealmABI : public TapirTarget {

  Type *TaskFuncPtrTy = nullptr;

  //Opaque Realm RTS functions
  FunctionCallee RealmGetNumProcs = nullptr;
  FunctionCallee RealmSpawn = nullptr;
  FunctionCallee RealmSync = nullptr;
  FunctionCallee RealmInitRuntime = nullptr;
  FunctionCallee RealmFinalize = nullptr;
  FunctionCallee CreateBar = nullptr;
  FunctionCallee DestroyBar = nullptr; 

  //Accessors for opaque Realm RTS functions
  FunctionCallee get_realmGetNumProcs();
  FunctionCallee get_realmSpawn();
  FunctionCallee get_realmSync();
  FunctionCallee get_realmInitRuntime();
  FunctionCallee get_realmFinalize();
  FunctionCallee get_createRealmBarrier();
  FunctionCallee get_destroyRealmBarrier();

public:
  RealmABI(Module &M);
  ~RealmABI();

  Value * lowerGrainsizeCall(CallInst *GrainsizeCall) override final;
  Value *getOrCreateBarrier(Value *SyncRegion, Function *F); 
  void lowerSync(SyncInst &inst) override final;
  void processSubTaskCall(TaskOutlineInfo &TOI, DominatorTree &DT)
    override final;

  void preProcessFunction(Function &F, TaskInfo &TI,
			  bool OutliningTapirLoops) override final;
  void postProcessFunction(Function &F, bool OutliningTapirLoops) 
    override final;
  void postProcessHelper(Function &F) override final;
  Function* formatFunctionToRealmF(Function* extracted, Instruction* ical);

  // not used
  //void processOutlinedTask(Function &F) override final {}
  //void processSpawner(Function &F) override final {}
  
  //void lowerTaskFrameAddrCall(CallInst *TaskFrameAddrCall) override final;
  //bool shouldProcessFunction(const Function &F) const override final;
  
  ArgStructMode getArgStructMode() const override final {
    return ArgStructMode::Static;
  }
  
  Type *getReturnType() const override final {
    return Type::getInt32Ty(M.getContext());
  }
  void preProcessOutlinedTask(Function &F, Instruction *DetachPt,
                              Instruction *TaskFrameCreate,
                              bool IsSpawner) override final {}
  void postProcessOutlinedTask(Function &F, Instruction *DetachPt,
                               Instruction *TaskFrameCreate,
                               bool IsSpawner) override final {}
  void preProcessRootSpawner(Function &F) override final {}
  void postProcessRootSpawner(Function &F) override final {}


};
}  // end of llvm namespace

#endif
