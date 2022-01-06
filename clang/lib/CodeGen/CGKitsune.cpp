/**
 ***************************************************************************
 * TODO: Need to update LANL/Triad Copyright notice... 
 * 
 * Copyright (c) 2017, Los Alamos National Security, LLC.
 * All rights reserved.
 *
 *  Copyright 2010. Los Alamos National Security, LLC. This software was
 *  produced under U.S. Government contract DE-AC52-06NA25396 for Los
 *  Alamos National Laboratory (LANL), which is operated by Los Alamos
 *  National Security, LLC for the U.S. Department of Energy. The
 *  U.S. Government has rights to use, reproduce, and distribute this
 *  software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY,
 *  LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY
 *  FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
 *  derivative works, such modified software should be clearly marked,
 *  so as not to confuse it with the version available from LANL.
 *
 *  Additionally, redistribution and use in source and binary forms,
 *  with or without modification, are permitted provided that the
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
 ***************************************************************************/
#include "clang/AST/Attr.h"
#include "clang/Basic/Attributes.h"
#include "clang/Basic/AttrKinds.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/CodeGen/CGFunctionInfo.h"
#include "CodeGenFunction.h"
#include "CGCleanup.h"
#include "clang/AST/StmtKitsune.h"
#include "llvm/IR/ValueMap.h"


using namespace clang;
using namespace CodeGen;


LoopAttributes::LSStrategy CodeGenFunction::GetTapirStrategyAttr(
       ArrayRef<const Attr *> Attrs) {


    LoopAttributes::LSStrategy Strategy = LoopAttributes::SEQ;
  
  auto curAttr = Attrs.begin();

  while(curAttr != Attrs.end()) {
    
    const attr::Kind AttrKind = (*curAttr)->getKind();

    if (AttrKind == attr::TapirStrategy) {
      const auto *SAttr = cast<const TapirStrategyAttr>(*curAttr);
      
      switch(SAttr->getTapirStrategyType()) {

      case TapirStrategyAttr::SEQ:
        Strategy = LoopAttributes::SEQ;
        break;

      case TapirStrategyAttr::DAC:
        Strategy = LoopAttributes::DAC;
        break;

      case TapirStrategyAttr::GPU:
        Strategy = LoopAttributes::GPU;
        break;

      default:
        llvm_unreachable("all strategies should be handled before this!");
        break;
      }
    }
  }
  return Strategy;
}

LoopAttributes::LTarget CodeGenFunction::GetTapirRTTargetAttr(
      ArrayRef<const Attr *> Attrs) {

  auto curAttr = Attrs.begin();

  // TODO: Need to make sure the default matches build parameters! 
  LoopAttributes::LTarget Target = LoopAttributes::CilkRT; 

  while(curAttr != Attrs.end()) {
    
    const attr::Kind AttrKind = (*curAttr)->getKind();

    if (AttrKind == attr::TapirRTTarget) {
      const auto *TAttr = cast<const TapirRTTargetAttr>(*curAttr);
      
      switch(TAttr->getTapirRTTargetType()) {

      case TapirRTTargetAttr::CheetahRT:
        Target = LoopAttributes::CheetahRT;
        break;
      case TapirRTTargetAttr::CilkRT: 
        Target = LoopAttributes::CilkRT;
        break;        
      case TapirRTTargetAttr::CudaRT:
        Target = LoopAttributes::CudaRT;
        break;
      case TapirRTTargetAttr::HipRT:
        Target = LoopAttributes::HipRT;
        break;
      case TapirRTTargetAttr::OmpRT:
        Target = LoopAttributes::OmpRT;
        break;
      case TapirRTTargetAttr::QthreadsRT:
        Target = LoopAttributes::QthreadsRT;
        break;
      case TapirRTTargetAttr::RealmRT:
        Target = LoopAttributes::RealmRT;
        break;
      case TapirRTTargetAttr::RocmRT:
        Target = LoopAttributes::RocmRT;
        break;
      case TapirRTTargetAttr::SequentialRT:
        Target = LoopAttributes::SequentialRT;
        break;
      case TapirRTTargetAttr::ZeroRT:
        Target = LoopAttributes::ZeroRT;
        break;
      default:
        llvm_unreachable("All target attributes should be handled here!");
        break;
      }
    }
    curAttr++;
  }
  return Target;
}


// Stolen from CodeGenFunction.cpp
static void EmitIfUsed(CodeGenFunction &CGF, llvm::BasicBlock *BB) {
  if (!BB) return;
  if (!BB->use_empty())
    return CGF.CurFn->getBasicBlockList().push_back(BB);
  delete BB;
}

llvm::Instruction *CodeGenFunction::EmitLabeledSyncRegionStart(StringRef SV) {
  // Start the sync region.  To ensure the syncregion.start call dominates all
  // uses of the generated token, we insert this call at the alloca insertion
  // point.
  llvm::Instruction *SRStart = llvm::CallInst::Create(
      CGM.getIntrinsic(llvm::Intrinsic::syncregion_start), SV, AllocaInsertPt);
  return SRStart;
}

/// EmitSyncStmt - Emit a sync node.
void CodeGenFunction::EmitSyncStmt(const SyncStmt &S) {
  llvm::BasicBlock *ContinueBlock = createBasicBlock("sync.continue");

  // If this code is reachable then emit a stop point (if generating
  // debug info). We have to do this ourselves because we are on the
  // "simple" statement path.
  if (HaveInsertPoint())
    EmitStopPoint(&S);

  Builder.CreateSync(ContinueBlock, 
    getOrCreateLabeledSyncRegion(S.getSyncVar())->getSyncRegionStart());
  EmitBlock(ContinueBlock);
}

void CodeGenFunction::EmitSpawnStmt(const SpawnStmt &S) {
  // Set up to perform a detach.
  // PushDetachScope();
  SyncRegion* SR = getOrCreateLabeledSyncRegion(S.getSyncVar());
  //StartLabeledDetach(SR);

  llvm::BasicBlock* DetachedBlock = createBasicBlock("det.achd");
  llvm::BasicBlock* ContinueBlock = createBasicBlock("det.cont");

  auto OldAllocaInsertPt = AllocaInsertPt; 
  llvm::Value *Undef = llvm::UndefValue::get(Int32Ty);
  AllocaInsertPt = new llvm::BitCastInst(Undef, Int32Ty, "",
                                             DetachedBlock);

  Builder.CreateDetach(DetachedBlock, ContinueBlock,
                           SR->getSyncRegionStart());


  EmitBlock(DetachedBlock);
  EmitStmt(S.getSpawnedStmt());

  Builder.CreateReattach(ContinueBlock,
                             SR->getSyncRegionStart());

  llvm::Instruction* ptr = AllocaInsertPt; 
  AllocaInsertPt = OldAllocaInsertPt; 
  ptr->eraseFromParent(); 

  EmitBlock(ContinueBlock);
}

void CodeGenFunction::SetAllocaInsertPoint(llvm::Value* v, llvm::BasicBlock* bb){
  AllocaInsertPt = new llvm::BitCastInst(v, Int32Ty, "", bb);
}

// Emit a load of the induction variable
// It has a side effect of erasing the mapping in the 
// LocalDeclMap but keeping track of the original mapping 
// as well as the new RValue after the load. This is all
// a precursor to capturing the IV by value in the body emission.
void CodeGenFunction::EmitIVLoad(const VarDecl* LoopVar, 
                                DeclMapByValueTy& IVDeclMap) {

  // The address corresponding to the IV
  Address IVAddress = LocalDeclMap.find(LoopVar)->second;

  // Remove the IV mapping from the LocalDeclMap 
  LocalDeclMap.erase(LoopVar);

  // Get the type
  QualType type = LoopVar->getType();

  // Create the vector of values
  llvm::SmallVector<llvm::Value *, 4> ValueVec;

  // Emit all the shallow copy loads and update 
  switch (getEvaluationKind(type)) {
    case TEK_Scalar: {
      LValue IVLV = MakeAddrLValue(IVAddress, type);
      RValue IVRV = EmitLoadOfLValue(IVLV, LoopVar->getBeginLoc()); 
      ValueVec.push_back(IVRV.getScalarVal());
      break;
    }
    case TEK_Complex: {
      ComplexPairTy Val =
        EmitLoadOfComplex(MakeAddrLValue(IVAddress, type), LoopVar->getBeginLoc());
      ValueVec.append({Val.first, Val.second});
      break;
    }
    case TEK_Aggregate: {
      if (const llvm::StructType *STy = dyn_cast<llvm::StructType>(IVAddress.getElementType())) {
        for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i) {
          Address EltPtr = Builder.CreateStructGEP(IVAddress, i);
          llvm::Value *Elt = Builder.CreateLoad(EltPtr);
          ValueVec.push_back(Elt);
        }
      } else {
        LValue IVLV = MakeAddrLValue(IVAddress, type);
        RValue IVRV = EmitLoadOfLValue(IVLV, LoopVar->getBeginLoc()); 
        ValueVec.push_back(IVRV.getScalarVal());
      }
      break;
    }
  }

  // Capture the mapping from LoopVar to the old address and new vector of Value*'s
  IVDeclMap.insert({LoopVar, {IVAddress, ValueVec}});
}

// Emit a thread safe copy of the induction variable and set it's value
// to the current value of the induction variable
void CodeGenFunction::EmitThreadSafeIV(const VarDecl* IV, const llvm::SmallVector<llvm::Value*,4>& Values){

  // emit the thread safe induction variable and cleanups
  AutoVarEmission LVEmission = EmitAutoVarAlloca(*IV);
  EmitAutoVarCleanups(LVEmission);

  // get the address of the emission
  Address Loc = LVEmission.getObjectAddress(*this);

  // turn the address into an LValue
  LValue LV = MakeAddrLValue(Loc, IV->getType());	

  // Make sure the LValue isn't garbage collected
  LV.setNonGC(true);

  // Store the IV RValue into the newly created thread safe induction variable
  EmitStoreOfScalar(Values.back(), LV, true);
  //EmitStoreThroughLValue(Values.back(), LV, true);
}

// Restore the original mapping between the Vardecl and its address
void CodeGenFunction::RestoreDeclMap(const VarDecl* IV, const Address IVAddress){

  // remove the mapping to the thread safe induction variable
  LocalDeclMap.erase(IV);

  // restore the original mapping
  LocalDeclMap.insert({IV, IVAddress}); 
}

void CodeGenFunction::EmitForallStmt(const ForallStmt &S,
                                  ArrayRef<const Attr *> ForAttrs) {

  /////////////////////////////////////////////////////////////////////////////
  // <KITSUNE>

  // new basic blocks and jump destinations with Tapir terminators
  llvm::BasicBlock* Detach = createBasicBlock("forall.detach");
  JumpDest Reattach = getJumpDestInCurrentScope("forall.reattach");
  JumpDest Sync = getJumpDestInCurrentScope("forall.sync");
  
  // declarations
  DeclMapByValueTy IVDeclMap; // map from Vardecl to {IV, thread safe IV vector}
  llvm::AssertingVH<llvm::Instruction> OldAllocaInsertPt = AllocaInsertPt;
  llvm::Value *Undef = llvm::UndefValue::get(Int32Ty);

  // emit the sync region
  PushSyncRegion();
  llvm::Instruction *SRStart = EmitSyncRegionStart();
  CurSyncRegion->setSyncRegionStart(SRStart);
  LoopStack.setSpawnStrategy(LoopAttributes::DAC);

  // </KITSUNE>
  /////////////////////////////////////////////////////////////////////////////

  JumpDest LoopExit = getJumpDestInCurrentScope("forall.end");

  LexicalScope ForScope(*this, S.getSourceRange());

  // Evaluate the initialization before the loop.
  EmitStmt(S.getInit());

  // Start the loop with a block that tests the condition.
  // <Kitsune/>
  JumpDest Condition = getJumpDestInCurrentScope("forall.cond"); 
  llvm::BasicBlock *CondBlock = Condition.getBlock();
  EmitBlock(CondBlock);

  const SourceRange &R = S.getSourceRange();
  LoopStack.push(CondBlock, CGM.getContext(), CGM.getCodeGenOpts(), ForAttrs,
                 SourceLocToDebugLoc(R.getBegin()),
                 SourceLocToDebugLoc(R.getEnd()));

  // <KITSUNE/>
  // We always have an increment and continue to it
  JumpDest Increment = getJumpDestInCurrentScope("forall.inc");

  // Store the blocks to use for break and continue.
  BreakContinueStack.push_back(BreakContinue(LoopExit, Reattach));

  // Create a cleanup scope for the condition variable cleanups.
  // <KITSUNE/> Don't need this unless we allow condition scope variables
  LexicalScope ConditionScope(*this, S.getSourceRange());

  // If the for statement has a condition scope, emit the local variable
  // declaration.
  // <KITSUNE/> Presently, we don't support condition variables, but we should :-)
  if (S.getConditionVariable()) {
    EmitDecl(*S.getConditionVariable());
  }

  llvm::BasicBlock *ExitBlock = LoopExit.getBlock();
  // If there are any cleanups between here and the loop-exit scope,
  // create a block to stage a loop exit along.
  if (ForScope.requiresCleanups())
    ExitBlock = createBasicBlock("forall.cond.cleanup");

  // As long as the condition is true, iterate the loop.
  llvm::BasicBlock *ForBody = createBasicBlock("forall.body");

  // C99 6.8.5p2/p4: The first substatement is executed if the expression
  // compares unequal to 0.  The condition must be a scalar type.
  llvm::Value *BoolCondVal = EvaluateExprAsBool(S.getCond());
  Builder.CreateCondBr(
      BoolCondVal, Detach, Sync.getBlock(),
      createProfileWeightsForLoop(S.getCond(), getProfileCount(S.getBody())));

  if (ExitBlock != LoopExit.getBlock()) {
    EmitBlock(ExitBlock);
    EmitBranchThroughCleanup(Sync);
  }

  /////////////////////////////////////////////////////////////////////////////
  // <KITSUNE>

  EmitBlock(Detach);

  // Extract the DeclStmt from the statement init
  const DeclStmt *DS = cast<DeclStmt>(S.getInit());
  
  // Create threadsafe induction variables before the detach and put them in IVDeclMap
  for (auto *DI : DS->decls()) 
    EmitIVLoad(dyn_cast<VarDecl>(DI), IVDeclMap);

  // create the detach terminator
  Builder.CreateDetach(ForBody, Increment.getBlock(), SRStart);
  
  // </KITSUNE>
  /////////////////////////////////////////////////////////////////////////////

  EmitBlock(ForBody);

  incrementProfileCounter(&S);

  {
    // Create a separate cleanup scope for the body, in case it is not
    // a compound statement.
    RunCleanupsScope BodyScope(*this);

    ///////////////////////////////////////////////////////////////////////////
    // <KITSUNE>

    // change the alloca insert point to the body block
    SetAllocaInsertPoint(Undef, ForBody);

    // emit the thread safe induction variables and initialize them by value
    for (const auto &ivp : IVDeclMap) 
      EmitThreadSafeIV(ivp.first, ivp.second.second);

    // </KITSUNE>
    ///////////////////////////////////////////////////////////////////////////

    EmitStmt(S.getBody());
  }

  /////////////////////////////////////////////////////////////////////////////
  // <KITSUNE>

  // Restore induction variable mappings after emitting body, and before
  // the increment
  for (const auto &ivp : IVDeclMap) 
    RestoreDeclMap(ivp.first, ivp.second.first);

  // emit the reattach block
  EmitBlock(Reattach.getBlock());
  Builder.CreateReattach(Increment.getBlock(), SRStart);

  // reset the alloca insertion point
  AllocaInsertPt->removeFromParent();
  AllocaInsertPt = OldAllocaInsertPt; 

  // </KITSUNE>
  /////////////////////////////////////////////////////////////////////////////

  // Emit the increment.
  EmitBlock(Increment.getBlock());
  EmitStmt(S.getInc());

  BreakContinueStack.pop_back();

  ConditionScope.ForceCleanup();

  EmitStopPoint(&S);
  EmitBranch(CondBlock);

  ForScope.ForceCleanup();

  LoopStack.pop();

  // <KITSUNE/> Emit the Sync block and terminator
  EmitBlock(Sync.getBlock());
  Builder.CreateSync(LoopExit.getBlock(), SRStart);

  // Emit the fall-through block.
  EmitBlock(LoopExit.getBlock(), true);
}

void CodeGenFunction::EmitCXXForallRangeStmt(const CXXForallRangeStmt &S,
                                             ArrayRef<const Attr *> ForAttrs) {

  /////////////////////////////////////////////////////////////////////////////
  // <KITSUNE>

  // new basic blocks and jump destinations with Tapir terminators
  llvm::BasicBlock* Detach = createBasicBlock("forall.detach");
  JumpDest Reattach = getJumpDestInCurrentScope("forall.reattach");
  JumpDest LoopExit = getJumpDestInCurrentScope("forall.sync");
  
  // declarations
  //DeclMapByValueTy IVDeclMap; // map from Vardecl to {IV, thread safe IV}
  llvm::AssertingVH<llvm::Instruction> OldAllocaInsertPt = AllocaInsertPt;
  llvm::Value *Undef = llvm::UndefValue::get(Int32Ty);

  // emit the sync region
  PushSyncRegion();
  llvm::Instruction *SRStart = EmitSyncRegionStart();
  CurSyncRegion->setSyncRegionStart(SRStart);
  LoopStack.setSpawnStrategy(LoopAttributes::DAC);

  // </KITSUNE>
  /////////////////////////////////////////////////////////////////////////////


  llvm::BasicBlock *End = createBasicBlock("forall.end");

  LexicalScope ForScope(*this, S.getSourceRange());

  // Evaluate the first pieces before the loop.
  if (S.getInit())
    EmitStmt(S.getInit());
  EmitStmt(S.getRangeStmt());
  EmitStmt(S.getBeginStmt());
  EmitStmt(S.getEndStmt());
  EmitStmt(S.getIndexStmt());
  EmitStmt(S.getIndexEndStmt());

  // Start the loop with a block that tests the condition.
  // If there's an increment, the continue scope will be overwritten
  // later.
  llvm::BasicBlock *CondBlock = createBasicBlock("forall.cond");
  EmitBlock(CondBlock);

  const SourceRange &R = S.getSourceRange();
  LoopStack.push(CondBlock, CGM.getContext(), CGM.getCodeGenOpts(), ForAttrs,
                 SourceLocToDebugLoc(R.getBegin()),
                 SourceLocToDebugLoc(R.getEnd()));

  // If there are any cleanups between here and the loop-exit scope,
  // create a block to stage a loop exit along.
  llvm::BasicBlock *ExitBlock = LoopExit.getBlock();
  if (ForScope.requiresCleanups())
    ExitBlock = createBasicBlock("forall.cond.cleanup");
  
  // The loop body, consisting of the specified body and the loop variable.
  llvm::BasicBlock *ForBody = createBasicBlock("forall.body");
  
  // The body is executed if the expression, contextually converted
  // to bool, is true.
  llvm::Value *BoolCondVal = EvaluateExprAsBool(S.getCond());
  llvm::MDNode *Weights = createProfileOrBranchWeightsForLoop(
      S.getCond(), getProfileCount(S.getBody()), S.getBody());
  Builder.CreateCondBr(BoolCondVal, Detach, LoopExit.getBlock(), Weights);

  if (ExitBlock != LoopExit.getBlock()) {
    EmitBlock(ExitBlock);
    EmitBranchThroughCleanup(LoopExit);
  }






  /////////////////////////////////
  // Create the detach block
  /////////////////////////////////

  // Emit the (currently empty) detach block
  EmitBlock(Detach);

  // Extract the DeclStmt from the statement init
  const DeclStmt *DS = cast<DeclStmt>(S.getIndexStmt());
  
  DeclMapTy IVDeclMap; 
  llvm::SmallVector<
      std::pair<VarDecl *,
                std::unique_ptr<llvm::SmallVector<llvm::Value *, 4>>>,
      4>
      ivs;
  for (auto *DI : DS->decls()) {
    auto *LoopVar = dyn_cast<VarDecl>(DI);
    Address OuterLoc = LocalDeclMap.find(LoopVar)->second; 
    LocalDeclMap.erase(LoopVar);
    IVDeclMap.insert({LoopVar, OuterLoc}); 
    QualType type = LoopVar->getType();
    ivs.push_back(
          {LoopVar, std::make_unique<llvm::SmallVector<llvm::Value *, 4>>()});
    switch (getEvaluationKind(type)) {
      case TEK_Scalar: {
        LValue OuterLV = MakeAddrLValue(OuterLoc, type);
        RValue OuterRV = EmitLoadOfLValue(OuterLV, DI->getBeginLoc());
        ivs.back().second->push_back(OuterRV.getScalarVal());
        break;
      }
      case TEK_Complex: {
        ComplexPairTy Val =
          EmitLoadOfComplex(MakeAddrLValue(OuterLoc, type), DI->getBeginLoc());
        ivs.back().second->push_back(Val.first);
        ivs.back().second->push_back(Val.second);
        break;
      }
      case TEK_Aggregate: {
        if (const llvm::StructType *STy =
          dyn_cast<llvm::StructType>(OuterLoc.getElementType())) {
          // Load each element of the structure
          for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i) {
            Address EltPtr = Builder.CreateStructGEP(OuterLoc, i);
            llvm::Value *Elt = Builder.CreateLoad(EltPtr);
            ivs.back().second->push_back(Elt);
          }
        } else {
          LValue OuterLV = MakeAddrLValue(OuterLoc, type);
          RValue OuterRV = EmitLoadOfLValue(OuterLV, DI->getBeginLoc());
          ivs.back().second->push_back(OuterRV.getScalarVal());
        }
        break;
      }
    }
  }

  // Create a block for the increment. In case of a 'continue', we jump there.
  llvm::BasicBlock *Increment = createBasicBlock("forall.inc");

  // create the detach terminator
  Builder.CreateDetach(ForBody, Increment, SRStart);
  
  // </KITSUNE>
  /////////////////////////////////////////////////////////////////////////////





  EmitBlock(ForBody);
  incrementProfileCounter(&S);

  // Store the blocks to use for break and continue.
  BreakContinueStack.push_back(BreakContinue(LoopExit, Reattach));

  {
    // Create a separate cleanup scope for the loop variable and body.
    LexicalScope BodyScope(*this, S.getSourceRange());

    ///////////////////////////////////////////////////////////////////////////
    // <KITSUNE>

    // change the alloca insert point to the body block
    SetAllocaInsertPoint(Undef, ForBody);


    for (auto &ivp : ivs) {
      auto *LoopVar = ivp.first;
      auto &LoadedVals = ivp.second;
      AutoVarEmission LVEmission = EmitAutoVarAlloca(*LoopVar);
      EmitAutoVarCleanups(LVEmission);
      QualType type = LoopVar->getType();
      Address Loc = LVEmission.getObjectAddress(*this);
      LValue LV = MakeAddrLValue(Loc, type);
      LV.setNonGC(true);
      switch (getEvaluationKind(type)) {
      case TEK_Scalar: {
        EmitStoreOfScalar(LoadedVals->back(), LV, /*isInit*/ true);
        break;
      }
      case TEK_Complex: {
	ComplexPairTy Val = {(*LoadedVals)[0], (*LoadedVals)[1]};
	EmitStoreOfComplex(Val, LV, /*isInit*/ true);
        break;
      }
      case TEK_Aggregate: {
        if (const llvm::StructType *STy =
                dyn_cast<llvm::StructType>(Loc.getElementType())) {
	  // Store the previously-loaded value into the new structure
          for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i) {
            Address EltPtr = Builder.CreateStructGEP(Loc, i);
            llvm::Value *Elt = (*LoadedVals)[i];
            Builder.CreateStore(Elt, EltPtr);
          }
        } else {
          EmitStoreOfScalar(LoadedVals->back(), LV, /*isInit*/ true);
        }
        break;
      }
      }

    }
    EmitStmt(S.getLoopVarStmt());
    EmitStmt(S.getBody());
  }

  /////////////////////////////////////////////////////////////////////////////
  // <KITSUNE>

  // Restore induction variable mappings after emitting body, and before
  // the increment
  for (const auto &p : IVDeclMap){
    LocalDeclMap.erase(p.first);
    LocalDeclMap.insert(p); 
  }

  EmitBlock(Reattach.getBlock());
  Builder.CreateReattach(Increment, SRStart);

  // reset the alloca insertion point
  AllocaInsertPt->removeFromParent();
  AllocaInsertPt = OldAllocaInsertPt; 

  // </KITSUNE>
  /////////////////////////////////////////////////////////////////////////////

  EmitBlock(Increment);

  EmitStmt(S.getInc());

  BreakContinueStack.pop_back();

  EmitStopPoint(&S);
  
  EmitBranch(CondBlock);

  ForScope.ForceCleanup();

  LoopStack.pop();

  // <KITSUNE/> Emit the Sync block and terminator
  EmitBlock(LoopExit.getBlock());
  Builder.CreateSync(End, SRStart);

  EmitBlock(End, true);
}

