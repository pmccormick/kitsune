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

void CodeGenFunction::EmitForallStmt(const ForallStmt &S,
                                             ArrayRef<const Attr *> ForAttrs) {

                                        
  assert(S.getInit() && "forall loop has no init");
  assert(S.getCond() && "forall loop has no condition");
  assert(S.getInc() && "forall loop has no increment");

  // Create all jump destinations and blocks in the order they appear in the IR
  // some are jump destinations, some are basic blocks
  JumpDest Condition = getJumpDestInCurrentScope("forall.cond");
  llvm::BasicBlock *Detach = createBasicBlock("forall.detach");
  llvm::BasicBlock *ForBody = createBasicBlock("forall.body");
  JumpDest Reattach = getJumpDestInCurrentScope("forall.reattach");
  llvm::BasicBlock *Increment = createBasicBlock("forall.inc");
  JumpDest Cleanup = getJumpDestInCurrentScope("forall.cond.cleanup");
  JumpDest Sync = getJumpDestInCurrentScope("forall.sync");
  llvm::BasicBlock *End = createBasicBlock("forall.end");

  // Extract a convenience block
  llvm::BasicBlock *ConditionBlock = Condition.getBlock();

  const SourceRange &R = S.getSourceRange();
  LexicalScope ForScope(*this, R);

  // Evaluate the first part before the loop.
  EmitStmt(S.getInit());

  // get the current insert block (e.g. 'entry'), this will be the basic block
  // where the induction variable is allocated/defined
  //llvm::BasicBlock *EntryBlock = Builder.GetInsertBlock();

  // create the sync region
  PushSyncRegion();
  llvm::Instruction *SRStart = EmitSyncRegionStart();
  CurSyncRegion->setSyncRegionStart(SRStart);

  // TODO: Need to check attributes for spawning strategy. 
  LoopStack.setSpawnStrategy(LoopAttributes::DAC);

  EmitBlock(ConditionBlock);

  LoopStack.push(ConditionBlock, CGM.getContext(), ForAttrs,
                 SourceLocToDebugLoc(R.getBegin()),
                 SourceLocToDebugLoc(R.getEnd()));

  // Store the blocks to use for break and continue.
  BreakContinueStack.push_back(BreakContinue(Reattach, Reattach));

  // Create a cleanup scope for the condition variable cleanups.
  LexicalScope ConditionScope(*this, R);

  // If the for statement has a condition scope, emit the local variable
  // declaration.
  if (S.getConditionVariable()) {
    EmitDecl(*S.getConditionVariable());
  }

  // C99 6.8.5p2/p4: The first substatement is executed if the expression
  // compares unequal to 0.  The condition must be a scalar type.
  llvm::Value *BoolCondVal = EvaluateExprAsBool(S.getCond());
  Builder.CreateCondBr(
      BoolCondVal, Detach, Sync.getBlock(),
      createProfileWeightsForLoop(S.getCond(), getProfileCount(S.getBody())));

  if (ForScope.requiresCleanups()) {
    EmitBlock(Cleanup.getBlock());
    EmitBranchThroughCleanup(Sync);
  }

  //////////////////////////////
  // Handle the Detach block
  //////////////////////////////

  // Emit the (currently empty) detach block
  EmitBlock(Detach);

  // Extract the DeclStmt from the statement init
  const DeclStmt *DS = cast<DeclStmt>(S.getInit());
  
  // create the value map between the induction variables and their corresponding detach mirrors
  llvm::ValueMap<llvm::Value*, llvm::AllocaInst *> InductionDetachMap;

  // Emit the detach block
  EmitDetachBlock(DS, InductionDetachMap);

  // create the detach terminator
  Builder.CreateDetach(ForBody, Increment, SRStart);  

  //////////////////////////////
  // End the Detach block
  //////////////////////////////

  EmitBlock(ForBody);

  incrementProfileCounter(&S);

  {
    // Create a separate cleanup scope for the body, in case it is not
    // a compound statement.
    RunCleanupsScope BodyScope(*this);
    EmitStmt(S.getBody());
  }


  /////////////////////////////////////////////////////////////////
  // Modify the body block to use the detach block variable mirror.
  // At this point in the codegen, the body block has been emitted
  // and we can safely replace the induction variable with the detach
  // block mirror in the entire function, since the increment block
  // (a valid use of the induction variable) has not been emitted yet.
  /////////////////////////////////////////////////////////////////

  ReplaceAllUsesInCurrentBlock(InductionDetachMap);

  EmitBlock(Reattach.getBlock());
  Builder.CreateReattach(Increment, SRStart);

  EmitBlock(Increment);

  EmitStmt(S.getInc());

  BreakContinueStack.pop_back();

  ConditionScope.ForceCleanup();

  EmitStopPoint(&S);

  EmitBranch(ConditionBlock);

  ForScope.ForceCleanup();

  LoopStack.pop();

  EmitBlock(Sync.getBlock());
  Builder.CreateSync(End, SRStart);

  EmitBlock(End, true);
}

void CodeGenFunction::EmitCXXForallRangeStmt(const CXXForallRangeStmt &S,
                                             ArrayRef<const Attr *> ForAttrs) {
  // Create all jump destinations and blocks in the order they appear in the IR
  // some are jump destinations, some are basic blocks
  JumpDest Condition = getJumpDestInCurrentScope("forall.cond");
  llvm::BasicBlock *Detach = createBasicBlock("forall.detach");
  llvm::BasicBlock *ForBody = createBasicBlock("forall.body");
  JumpDest Reattach = getJumpDestInCurrentScope("forall.reattach");
  llvm::BasicBlock *Increment = createBasicBlock("forall.inc");
  JumpDest Cleanup = getJumpDestInCurrentScope("forall.cond.cleanup");
  JumpDest Sync = getJumpDestInCurrentScope("forall.sync");
  llvm::BasicBlock *End = createBasicBlock("forall.end");

  // Extract a convenience block
  llvm::BasicBlock *ConditionBlock = Condition.getBlock();

  const SourceRange &R = S.getSourceRange();
  LexicalScope ForScope(*this, S.getSourceRange());

  // Evaluate the first pieces before the loop.
  if (S.getInit())
    EmitStmt(S.getInit());
  EmitStmt(S.getRangeStmt());
  EmitStmt(S.getBeginStmt());
  EmitStmt(S.getEndStmt());

  // create the sync region
  PushSyncRegion();
  llvm::Instruction *SRStart = EmitSyncRegionStart();
  CurSyncRegion->setSyncRegionStart(SRStart);

  // FIXME: Need to get attributes for spawning strategy from
  // code versus this hard-coded route...
  LoopStack.setSpawnStrategy(LoopAttributes::DAC);

  EmitBlock(ConditionBlock);

  LoopStack.push(ConditionBlock, CGM.getContext(), ForAttrs,
                 SourceLocToDebugLoc(R.getBegin()),
                 SourceLocToDebugLoc(R.getEnd()));

  // Store the blocks to use for break and continue.
   BreakContinueStack.push_back(BreakContinue(Reattach, Reattach));

  // The body is executed if the expression, contextually converted
  // to bool, is true.
  llvm::Value *BoolCondVal = EvaluateExprAsBool(S.getCond());
  Builder.CreateCondBr(
      BoolCondVal, Detach, Sync.getBlock(),
      createProfileWeightsForLoop(S.getCond(), getProfileCount(S.getBody())));

  if (ForScope.requiresCleanups()) {
    EmitBlock(Cleanup.getBlock());
    EmitBranchThroughCleanup(Sync);
  }

  /////////////////////////////////
  // Create the detach block
  /////////////////////////////////

  // Emit the (currently empty) detach block
  EmitBlock(Detach);

  // Extract the DeclStmt from the statement init
  const DeclStmt *DS = cast<DeclStmt>(S.getBeginStmt());

  // create the value map between the induction variables and their corresponding detach mirrors
  llvm::ValueMap<llvm::Value*, llvm::AllocaInst *> InductionDetachMap;

  // Emit the detach block
  EmitDetachBlock(DS, InductionDetachMap);


  // create the detach terminator
  Builder.CreateDetach(ForBody, Increment, SRStart);

  /////////////////////////////////
  // Finished the detach block
  /////////////////////////////////
  
  EmitBlock(ForBody);

  incrementProfileCounter(&S);

  {
    // Create a separate cleanup scope for the loop variable and body.
    LexicalScope BodyScope(*this, R);
    EmitStmt(S.getLoopVarStmt());
    EmitStmt(S.getBody());
  }

  /////////////////////////////////////////////////////////////////
  // Modify the body block to use the detach block variable mirror.
  // At this point in the codegen, the body block has been emitted
  // and we can safely replace the induction variable with the detach
  // block mirror in the entire function, since the increment block
  // (a valid use of the induction variable) has not been emitted yet.
  /////////////////////////////////////////////////////////////////

  ReplaceAllUsesInCurrentBlock(InductionDetachMap);

  EmitBlock(Reattach.getBlock());
  Builder.CreateReattach(Increment, SRStart);

  EmitBlock(Increment);

  EmitStmt(S.getInc());

  BreakContinueStack.pop_back();

  EmitStopPoint(&S);
  
  EmitBranch(ConditionBlock);

  ForScope.ForceCleanup();

  LoopStack.pop();

  EmitBlock(Sync.getBlock());
  Builder.CreateSync(End, SRStart);

  EmitBlock(End, true);
}

void CodeGenFunction::ReplaceAllUsesInCurrentBlock(llvm::ValueMap<llvm::Value*, llvm::AllocaInst *> &InductionDetachMap){

  // get the current basic block
  llvm::BasicBlock *CurrentBlock = Builder.GetInsertBlock();

  for (auto &&kv: InductionDetachMap){
  /*
    // At the moment, I believe the following block of code should be equivalent
    // to the one following, but there is an llvm bug? that makes them inequivalent 
    for (auto &U : InductionVar->uses()) {
      auto I = cast<llvm::Instruction>(U.getUser());
      if (I->getParent() == ForBody) U.set(DetachVar);
    }
  */
  
    // this code is pulled essentially verbatim from ReplaceNonLocalUsesWith
    // FYI: A Use is an operand to an instruction, A User is an instruction. 
    llvm::Instruction *IV = dyn_cast<llvm::Instruction>(kv.first);

    // this code is pulled essentially verbatim from ReplaceNonLocalUsesWith
    for (llvm::Value::use_iterator UI = IV->use_begin(), UE = IV->use_end();
        UI != UE;) {
      llvm::Use &U = *UI++;
      llvm::Instruction *I = cast<llvm::Instruction>(U.getUser());
      if (I->getParent() == CurrentBlock) U.set(kv.second);
    }

  }
}

void CodeGenFunction::EmitDetachBlock(const DeclStmt *DS, llvm::ValueMap<llvm::Value*, llvm::AllocaInst *> &InductionDetachMap){

  // iterate over all VarDecl's in the DeclStmt
  for (auto *DI : DS->decls()){

    // convert the Decl iterator into a VarDecl
    const VarDecl *InductionVarDecl=dyn_cast<VarDecl>(DI);

    // Get the induction variable (e.g. %i)
    llvm::Value *InductionVar = GetAddrOfLocalVar(InductionVarDecl).getPointer();

    // Use the Clang::CodeGen::CGBuilderTy::CreateLoad to load the induction variable.
    // This is the current value of the induction variable (e.g. %1 below)
    // e.g. %1 = load i32, i32* %i, align 4
    // In principle, the following line could be put in the #ifndef block and
    // the load instruction in TB's variant could be derived directly from the
    // condition block. But by keeping the load here I have a known handle without
    // doing any searching. It will likely get optimized away anyway.
    llvm::Value *InductionVal = Builder.CreateLoad(GetAddrOfLocalVar(InductionVarDecl));

    #ifndef __TB__
    // Codegen a local copy in the detach block of the induction variable and 
    // store the induction variable value

    // Get the Clang QualType for the Induction Variable
    QualType RefType = InductionVarDecl->getType();

    // Use the LLVM Builder to create the detach variable alloca
    // e.g. %i.detach = alloca i32
    // At the moment, I don't know how to force an alignment into the alloca
    llvm::AllocaInst *DetachVar = Builder.CreateAlloca(
        getTypes().ConvertType(RefType), nullptr, InductionVar->getName() + ".detach");

    // Use the Clang::CodeGen::CGBuilderTy to store the induction variable
    // into the detach variable mirror (e.g. store i32 %1, i32* %i.detach, align 4)
    Builder.CreateAlignedStore(InductionVal, DetachVar,
                              getContext().getTypeAlignInChars(RefType));

    // Add the mapping from induction variable to detach variable
    InductionDetachMap[InductionVar]=DetachVar;         

    #endif
  }



}
