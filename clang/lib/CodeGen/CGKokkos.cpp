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
#include <cstdio>
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/CodeGen/CGFunctionInfo.h"
#include "CodeGenFunction.h"

using namespace clang;
using namespace CodeGen;

namespace {

  // Part of the challenge in unwinding all the Kokkos
  // C++ details is getting through a bunch of implicit 
  // goop in the AST. Fortunately Clang can help here 
  // a lot as it has mechanisms to toss out implicit 
  // constructs and get us closer to the underlying 
  // expression.  Pulled out of inline code to allow 
  // for more agressive simplications/modifications 
  // if needed (and reuse across different constructs).
  static const Expr *SimplifyExpr(const Expr *E) {
    return E->IgnoreImplicit()->IgnoreImpCasts();
  }

  // Break apart the various components of a Kokkos 
  // parallel_for.  This boils down to tearing apart 
  // a CallExpr.  It helps that at this point we've 
  // already passed all the "goodness" in semantic 
  // analysis and other frontend stages.  However, 
  // there are aspects of what you can do generally 
  // in Kokkos and what we can tear apart here (both
  // big picture and in terms of the state of the 
  // current implementation).  A current set of 
  // limitations are:
  // 
  //   - Functors are not supported (primarily given
  //     challenges around separate compilation units)
  //   - 'Named constructs', which Kokkos uses for profiling,
  //     are extracted but unused. 
  //   - There are likely many holes in types that are legal 
  //     constructs but not yet captured/covered in the 
  //     current set of transformations currently implemented.
  //     The code tries to avoid crashing/asserting and instead
  //     falls back to the standard C++ mechanisms. We will start
  //     ramp up testing as we get more exposure to use cases. 
  // 
  static void 
  ExtractParallelForComponents(const CallExpr* CE,
					   std::string &CN, const Expr *& BE, 
					   const LambdaExpr *& LE)
  {
    // Recongized constructs: 
    // 
    //   1. parallel_for(N, lambda_expr...);
    //
    //   2. parallel_for("name", N, lambda_expr...);

    unsigned int curArgIndex = 0;

    // Check for a 'named' construct.  NOTE: we assume 
    // this only comes in the form of a string literal. 
    const Expr *OE = CE->getArg(curArgIndex); // Original expression 
    const Expr *SE = SimplifyExpr(OE);        // Simplified expression. 

    if (SE->getStmtClass() == Expr::CXXConstructExprClass) {
      const CXXConstructExpr *CXXCE = dyn_cast<CXXConstructExpr>(SE);
      SE = CXXCE->getArg(0)->IgnoreImplicit();
      if (SE->getStmtClass() == Expr::StringLiteralClass) {
        CN = dyn_cast<StringLiteral>(SE)->getString().str();
        curArgIndex++;
        OE = CE->getArg(curArgIndex);
        SE = SimplifyExpr(OE);
      } 
    } 

    // Check details of the bounds portion of the parallel_for. 
    // Note: This can take several forms and it is likely we've 
    // missed some cases... 
    if (SE->getStmtClass() == Expr::IntegerLiteralClass) {
      BE = OE;
      curArgIndex++;
      OE = CE->getArg(curArgIndex);
      SE = SimplifyExpr(OE);
    } else if (SE->getStmtClass() == Expr::BinaryOperatorClass) {
      BE = OE;
      curArgIndex++;
      OE = CE->getArg(curArgIndex);
      SE = SimplifyExpr(OE);
    } else if (SE->getStmtClass() == Expr::DeclRefExprClass) {
      BE = OE;
      curArgIndex++;
      OE = CE->getArg(curArgIndex);
      SE = SimplifyExpr(OE);
    } else if (SE->getStmtClass() == Expr::CallExprClass) { 
      BE = OE;
      curArgIndex++;
      OE = CE->getArg(curArgIndex);
      SE = SimplifyExpr(OE);
    } else if (SE->getStmtClass() == Expr::CXXTemporaryObjectExprClass) {
      BE = SE;
      curArgIndex++;
      OE = CE->getArg(curArgIndex);
      SE = SimplifyExpr(OE);
    } else {
      SE->dump();
      BE = nullptr;
      LE = nullptr;
      return;
    }

    if (SE->getStmtClass() == Expr::LambdaExprClass) {
      LE = dyn_cast<LambdaExpr>(SE);
    } else {
      LE = nullptr;
      return;
    }
  }


}

// Sort through what sort of Kokkos construct we're looking at 
// and work on transforming it into a Tapir-centric lowering.  
// 
bool CodeGenFunction::EmitKokkosConstruct(const CallExpr *CE,
                ArrayRef<const Attr *> Attrs) {
  assert(CE != 0 && "CodeGenFunction::EmitKokkosConstruct: null callexpr passed!");

  const FunctionDecl *Func = CE->getDirectCallee();
  assert(Func != 0 && "Kokkos construct doesn't have a function declaration!");

  if (Func->getQualifiedNameAsString() == "Kokkos::parallel_for") {
    return EmitKokkosParallelFor(CE, Attrs);
  } else if (Func->getQualifiedNameAsString() == "Kokkos::parallel_reduce") {
    return EmitKokkosParallelReduce(CE, Attrs);
  } else {
    return false;
  }
}  // hidden/local namespace 


std::vector<const ParmVarDecl*>
CodeGenFunction::EmitKokkosParallelForInductionVar(const LambdaExpr *Lambda) {
  const CXXMethodDecl *MD = Lambda->getCallOperator();
  assert(MD && "EmitKokkosParallelFor() -- bad method decl from labmda call.");
  
  std::vector<const ParmVarDecl*> params;
  
  for (unsigned int i = 0; i<MD->getNumParams(); i++) {
    const ParmVarDecl *InductionVarDecl = MD->getParamDecl(i);
    assert(InductionVarDecl && "EmitKokkosParallelFor() -- bad loop variable decl!");
    
    //EmitVarDecl(*InductionVarDecl);
    params.push_back(InductionVarDecl);
  }
  
  return params;
}

void CodeGenFunction::EmitKokkosParallelForCond(const Expr *BoundsExpr, 
     const ParmVarDecl *InductionVarDecl, 
     llvm::BasicBlock *DetachBlock,
     llvm::BasicBlock *ExitBlock,
     JumpDest &Sync) {

  llvm::Value *LoopEnd = nullptr;
  if (BoundsExpr->getStmtClass() == Expr::BinaryOperatorClass) {
    RValue RV = EmitAnyExpr(BoundsExpr);
    LoopEnd = RV.getScalarVal();
  } else { 
    LoopEnd = EmitScalarExpr(BoundsExpr);
  }

  llvm::Type  *InductionVarTy = ConvertType(InductionVarDecl->getType());
  unsigned NBits  = LoopEnd->getType()->getPrimitiveSizeInBits();
  unsigned LVBits = InductionVarTy->getPrimitiveSizeInBits();

  // We may need to truncate/extend the range to get it to match 
  // the type of loop variable. 
  if (NBits > LVBits) {
    LoopEnd = Builder.CreateTrunc(LoopEnd, InductionVarTy);
  } else if (NBits < LVBits) {
    LoopEnd = Builder.CreateZExt(LoopEnd, InductionVarTy);
  } else {
    // bit count matches, nothing to do...
  }

  llvm::Value *InductionVal = Builder.CreateLoad(GetAddrOfLocalVar(InductionVarDecl));
  llvm::Value *CondVal = Builder.CreateICmpULT(InductionVal, LoopEnd);
  Builder.CreateCondBr(CondVal, DetachBlock, Sync.getBlock());
}

bool CodeGenFunction::EmitKokkosParallelFor(const CallExpr *CE,
              ArrayRef<const Attr *> ForallAttrs) {
    
  // TODO: Need to add code to process any attributes (ForallAttrs).

  // Tease apart the parallel_for into its various components. 
  std::string      PFName; // construct name (for kokkos profiling)  
  const Expr       *BE = nullptr; // "bounds" expression
  const LambdaExpr *Lambda = nullptr; // the lambda  
  ExtractParallelForComponents(CE, PFName, BE, Lambda);
  
  if (Lambda == nullptr) { 
    // The parallel_for doesn't have a (recognizable) lambda expression.  
    // 
    // Functor support is problematic as it can live in a different 
    // compilaton unit.  We always punt and go the pure C++ route in
    // these cases. 
    DiagnosticsEngine &Diags = CGM.getDiags();
    // TODO: should reword this warning terminology. 
    Diags.Report(CE->getExprLoc(), diag::warn_kokkos_no_functor);
    return false;
  }

  if (BE == nullptr) {
    // We didn't get a known bounds expression back -- this is most likely 
    // due to some type of expression that we have yet to deal with.  We're
    // going to have to uncover these on a case-by-cases basis.
    DiagnosticsEngine &Diags = CGM.getDiags();
    Diags.Report(CE->getExprLoc(), diag::warn_kokkos_unknown_bounds_expr);
    return false;
  }
  
  // Check to see if we have an MDRange present
  if (BE->getStmtClass() == Expr::CXXTemporaryObjectExprClass) {
    const CXXTemporaryObjectExpr *CXXTO = dyn_cast<CXXTemporaryObjectExpr>(BE);
    std::string className = CXXTO->getBestDynamicClassType()->getNameAsString();
    
    if (className == "MDRangePolicy") {
      return EmitKokkosParallelForMD(CE, PFName, BE, Lambda, ForallAttrs);
    } else {
      // What should we do here?
    }
  }

  // Create all jump destinations and basic blocks in the order they 
  // appear in the IR. 
  JumpDest Condition = getJumpDestInCurrentScope("kokkos.forall.cond");
  llvm::BasicBlock *Detach = createBasicBlock("kokkos.forall.detach");
  llvm::BasicBlock *PForBody = createBasicBlock("kokkos.forall.body");
  JumpDest Reattach = getJumpDestInCurrentScope("kokkos.forall.reattach");
  llvm::BasicBlock *Increment = createBasicBlock("kokkos.forall.inc");
  JumpDest Cleanup = getJumpDestInCurrentScope("kokkos.forall.cond.cleanup");
  JumpDest Sync = getJumpDestInCurrentScope("kokkos.forall.sync");
  llvm::BasicBlock *End = createBasicBlock("kokkos.forall.end");

  // Extract a conveince block and setup the lexical scope based on 
  // the lambda's source range. 
  llvm::BasicBlock *ConditionBlock = Condition.getBlock();
  
  const SourceRange &R = CE->getSourceRange();
  LexicalScope PForScope(*this, R);

  // Now we can start the dirty work of transforming the lambda into a 
  // for loop.  


  // The first step is to extract the argument to the lambda and transform it into 
  // the loop induction variable.  As part of this we assume the following are true
  // about the parallel_for:
  //    1. The iterator can be assigned a value of zero. 
  //    2. We ignore the details of what is captured by the lambda.
  // 
  // TODO: Do we need to "relax" these assumptions to support broader code coverage?
  // This is 'equivalent' to the Init statement in a traditional for loop (e.g. int i = 0). 
  const ParmVarDecl *InductionVarDecl; 
  InductionVarDecl = EmitKokkosParallelForInductionVar(Lambda).at(0);
  EmitVarDecl(*InductionVarDecl);

   // Create the sync region. 
  PushSyncRegion();
  llvm::Instruction *SRStart = EmitSyncRegionStart();
  CurSyncRegion->setSyncRegionStart(SRStart);

  // TODO: Need to check attributes for spawning strategy. 
  LoopStack.setSpawnStrategy(LoopAttributes::DAC);
  
  EmitBlock(ConditionBlock);
  
  LoopStack.push(ConditionBlock, CGM.getContext(), ForallAttrs,
                 SourceLocToDebugLoc(R.getBegin()),
                 SourceLocToDebugLoc(R.getEnd()));

  // Store the blocks to use for break and continue. 
  BreakContinueStack.push_back(BreakContinue(Reattach, Reattach));

  // Create a scope for the condition variable cleanup. 
  LexicalScope ConditionScope(*this, R);

  // Create the conditional.
  EmitKokkosParallelForCond(BE, InductionVarDecl, Detach, End, Sync);

  if (PForScope.requiresCleanups()) {
    EmitBlock(Cleanup.getBlock());
    EmitBranchThroughCleanup(Sync);
  }

  // Handle the detach block...
  EmitBlock(Detach);

  auto OldAllocaInsertPt = AllocaInsertPt;
  llvm::Value *Undef = llvm::UndefValue::get(Int32Ty);
  AllocaInsertPt = new llvm::BitCastInst(Undef, Int32Ty, "", PForBody);

  llvm::Value *GInductionVar = GetAddrOfLocalVar(InductionVarDecl).getPointer();
  llvm::Value *GInductionVal = Builder.CreateLoad(GetAddrOfLocalVar(InductionVarDecl));

  QualType RefType = InductionVarDecl->getType();
  
  // Create the detach terminator 
  Builder.CreateDetach(PForBody, Increment, SRStart);

  EmitBlock(PForBody);
  incrementProfileCounter(CE);

  llvm::AllocaInst *TLInductionVar =
      Builder.CreateAlloca(getTypes().ConvertType(RefType), nullptr,
                           InductionVarDecl->getName() + ".detach");
  Builder.CreateAlignedStore(GInductionVal, TLInductionVar,
                             getContext().getTypeAlignInChars(RefType));
  {
    // Create a separate cleanup scope for the body, in case it is not
    // a compound statement.
    InKokkosConstruct = true;
    RunCleanupsScope BodyScope(*this);
    EmitStmt(Lambda->getBody());
    InKokkosConstruct = false;
  }

  auto tmp = AllocaInsertPt; 
  AllocaInsertPt = OldAllocaInsertPt; 
  tmp->removeFromParent(); 

  // Modify the body to use the ''detach''-local induction variable.
  // At this point in the codegen, the body block has been emitted 
  // and we can safely replace the ''sequential`` induction variable 
  // within the detach basic block.
  llvm::BasicBlock *CurrentBlock = Builder.GetInsertBlock();
  for(llvm::Value::use_iterator UI = GInductionVar->use_begin(), UE = GInductionVar->use_end(); 
      UI != UE; ) {
    llvm::Use &U = *UI++;
    llvm::Instruction *I = cast<llvm::Instruction>(U.getUser());
    if (I->getParent() == CurrentBlock) 
      U.set(TLInductionVar);
  }

  EmitBlock(Reattach.getBlock());
  Builder.CreateReattach(Increment, SRStart);

  EmitBlock(Increment);
  llvm::Value *IncVal = Builder.CreateLoad(GetAddrOfLocalVar(InductionVarDecl));
  llvm::Value *One = llvm::ConstantInt::get(ConvertType(InductionVarDecl->getType()), 1);
  IncVal = Builder.CreateAdd(IncVal, One);
  Builder.CreateStore(IncVal, GetAddrOfLocalVar(InductionVarDecl));

  BreakContinueStack.pop_back();
  ConditionScope.ForceCleanup();
  EmitStopPoint(CE);

  EmitBranch(ConditionBlock);
  PForScope.ForceCleanup();
  LoopStack.pop();

  EmitBlock(Sync.getBlock());
  Builder.CreateSync(End, SRStart);
  EmitBlock(End, true);
  return true;
}

// The builds an MDRange parallel_for (basically, a loop with multiple dimensions).
// The function is rather simple. We basically break down the MDRange object, and then call a recursive function
// that does the fun work of creating a loop
//
bool CodeGenFunction::EmitKokkosParallelForMD(const CallExpr *CE, std::string PFName, const Expr *BE, const LambdaExpr *Lambda,
            ArrayRef<const Attr *> ForallAttrs) {
    
  // TODO: Need to add code to process any attributes (ForallAttrs).
  
  // Build the queue of dimensions
  std::vector<const Expr *> DimQueue;
  std::vector<const Expr *> StartQueue;
  
  const CXXTemporaryObjectExpr *CXXTO = dyn_cast<CXXTemporaryObjectExpr>(BE);
  const InitListExpr *StartingBounds = dyn_cast<InitListExpr>(CXXTO->getArg(0)->IgnoreImplicit());
  const InitListExpr *UpperBounds = dyn_cast<InitListExpr>(CXXTO->getArg(1)->IgnoreImplicit());
  
  for (unsigned int i = 0; i<StartingBounds->getNumInits(); i++) {
      const Expr *val = StartingBounds->getInit(i)->IgnoreImplicit();
      StartQueue.push_back(val);
  }
  
  for (unsigned int i = 0; i<UpperBounds->getNumInits(); i++) {
    const Expr *val = UpperBounds->getInit(i)->IgnoreImplicit();
    DimQueue.push_back(val);
  }
  
  // Get the induction variables
  std::vector<const ParmVarDecl*> params = EmitKokkosParallelForInductionVar(Lambda);
  
  // Build the inner loops, and eventually the body
  std::vector<std::pair<llvm::Value*, llvm::AllocaInst*>> TLIVarList;
  return EmitKokkosInnerLoop(CE, Lambda, nullptr, DimQueue, StartQueue, params, TLIVarList, ForallAttrs);
}

// This is in charge of building an inner loop. It works as a recursive function to allow the loops
// to actually end up being nested
//
// This should be usuable by any function that requires inner loops
//
bool CodeGenFunction::EmitKokkosInnerLoop(const CallExpr *CE, const LambdaExpr *Lambda,
            llvm::BasicBlock *TopBlock,
            std::vector<const Expr*> DimQueue,
            std::vector<const Expr*> StartQueue,
            std::vector<const ParmVarDecl*> params,
            std::vector<std::pair<llvm::Value*, llvm::AllocaInst*>> TLIVarList,
            ArrayRef<const Attr *> ForallAttrs) {
  // Load the data we need
  int pos = DimQueue.size();
  const Expr *BE = DimQueue.front();
  DimQueue.erase(DimQueue.begin());

  const Expr *SE = StartQueue.front();
  StartQueue.erase(StartQueue.begin());

  const ParmVarDecl *InductionVarDecl = params.front();
  params.erase(params.begin());
  
  llvm::BasicBlock *InductionSet = createBasicBlock("kokkos.forall.set" + std::to_string(pos));
  JumpDest Condition = getJumpDestInCurrentScope("kokkos.forall.cond" + std::to_string(pos));
  llvm::BasicBlock *Detach = createBasicBlock("kokkos.forall.detach" + std::to_string(pos));
  llvm::BasicBlock *PForBody = createBasicBlock("kokkos.forall.body" + std::to_string(pos));
  JumpDest Reattach = getJumpDestInCurrentScope("kokkos.forall.reattach" + std::to_string(pos));
  llvm::BasicBlock *Increment = createBasicBlock("kokkos.forall.inc" + std::to_string(pos));
  JumpDest Cleanup = getJumpDestInCurrentScope("kokkos.forall.cond.cleanup" + std::to_string(pos));
  JumpDest Sync = getJumpDestInCurrentScope("kokkos.forall.sync" + std::to_string(pos));
  llvm::BasicBlock *End = createBasicBlock("kokkos.forall.end" + std::to_string(pos));

  /*// Set the induction variable's starting point
  EmitBlock(InductionSet);
  EmitVarDecl(*InductionVarDecl);
  llvm::Value *LoopStart = EmitScalarExpr(SE);
  Builder.CreateStore(LoopStart, GetAddrOfLocalVar(InductionVarDecl));*/
  
  // Extract a conveince block and setup the lexical scope based on 
  // the lambda's source range. 
  llvm::BasicBlock *ConditionBlock = Condition.getBlock();
  
  const SourceRange &R = CE->getSourceRange();
  LexicalScope PForScope(*this, R);

  // Now we can start the dirty work of transforming the lambda into a 
  // for loop.  


  // The first step is to extract the argument to the lambda and transform it into 
  // the loop induction variable.  As part of this we assume the following are true
  // about the parallel_for:
  //    1. The iterator can be assigned a value of zero. 
  //    2. We ignore the details of what is captured by the lambda.
  // 
  // TODO: Do we need to "relax" these assumptions to support broader code coverage?
  // This is 'equivalent' to the Init statement in a traditional for loop (e.g. int i = 0). 

   // Create the sync region. 
  PushSyncRegion();
  llvm::Instruction *SRStart = EmitSyncRegionStart();
  CurSyncRegion->setSyncRegionStart(SRStart);
  
  // Set the induction variable's starting point
  EmitBlock(InductionSet);
  EmitVarDecl(*InductionVarDecl);
  llvm::Value *LoopStart = EmitScalarExpr(SE);
  Builder.CreateStore(LoopStart, GetAddrOfLocalVar(InductionVarDecl));

  // TODO: Need to check attributes for spawning strategy. 
  LoopStack.setSpawnStrategy(LoopAttributes::DAC);
  
  EmitBlock(ConditionBlock);
  
  LoopStack.push(ConditionBlock, CGM.getContext(), ForallAttrs,
                 SourceLocToDebugLoc(R.getBegin()),
                 SourceLocToDebugLoc(R.getEnd()));

  // Store the blocks to use for break and continue. 
  BreakContinueStack.push_back(BreakContinue(Reattach, Reattach));

  // Create a scope for the condition variable cleanup. 
  LexicalScope ConditionScope(*this, R);

  // Create the conditional.
  EmitKokkosParallelForCond(BE, InductionVarDecl, Detach, End, Sync);

  if (PForScope.requiresCleanups()) {
    EmitBlock(Cleanup.getBlock());
    EmitBranchThroughCleanup(Sync);
  }

  // Handle the detach block...
  EmitBlock(Detach);

  auto OldAllocaInsertPt = AllocaInsertPt;
  llvm::Value *Undef = llvm::UndefValue::get(Int32Ty);
  AllocaInsertPt = new llvm::BitCastInst(Undef, Int32Ty, "", PForBody);

  llvm::Value *GInductionVar = GetAddrOfLocalVar(InductionVarDecl).getPointer();
  llvm::Value *GInductionVal = Builder.CreateLoad(GetAddrOfLocalVar(InductionVarDecl));

  QualType RefType = InductionVarDecl->getType();
  
  // Create the detach terminator 
  Builder.CreateDetach(PForBody, Increment, SRStart);

  EmitBlock(PForBody);
  incrementProfileCounter(CE);

  llvm::AllocaInst *TLInductionVar =
      Builder.CreateAlloca(getTypes().ConvertType(RefType), nullptr,
                           InductionVarDecl->getName() + ".detach");
  Builder.CreateAlignedStore(GInductionVal, TLInductionVar,
                             getContext().getTypeAlignInChars(RefType));
                             
  std::pair<llvm::Value*, llvm::AllocaInst*> pair(GInductionVar, TLInductionVar);
  TLIVarList.push_back(pair);
  {
    if (DimQueue.size() == 0) {
      // Create a separate cleanup scope for the body, in case it is not
      // a compound statement.
      InKokkosConstruct = true;
      RunCleanupsScope BodyScope(*this);
      EmitStmt(Lambda->getBody());
      InKokkosConstruct = false;
      
      // Modify the body to use the ''detach''-local induction variable.
      // At this point in the codegen, the body block has been emitted 
      // and we can safely replace the ''sequential`` induction variable 
      // within the detach basic block.
      //
      // When Kokkos::Views (and likely some other structures) are used, we end up with extra blocks
      // between the original for loop body block and the reattach block. Without the loop, it will
      // only modify the last of these block. The loop iterates and updates all the blocks back to the
      // original for loop body block to use the thread-local induction variables
      //
      llvm::BasicBlock *CurrentBlock = Builder.GetInsertBlock();
      for (;;) {
        for (unsigned int i = 0; i<TLIVarList.size(); i++) {
          auto TLVar = TLIVarList.at(i).second;
          auto GInductionVar = TLIVarList.at(i).first;
          
          for(llvm::Value::use_iterator UI = GInductionVar->use_begin(), UE = GInductionVar->use_end(); 
              UI != UE; ) {
            llvm::Use &U = *UI++;
            llvm::Instruction *I = cast<llvm::Instruction>(U.getUser());
            if (I->getParent() == CurrentBlock) 
              U.set(TLVar);
          }
        }
        
        if (CurrentBlock == PForBody) break;
        else CurrentBlock = CurrentBlock->getSinglePredecessor();
      }
    } else {
      EmitKokkosInnerLoop(CE, Lambda, ConditionBlock, DimQueue, StartQueue, params, TLIVarList, ForallAttrs);
    }
  }
  
  auto tmp = AllocaInsertPt; 
  AllocaInsertPt = OldAllocaInsertPt; 
  tmp->removeFromParent(); 

  EmitBlock(Reattach.getBlock());
  Builder.CreateReattach(Increment, SRStart);

  EmitBlock(Increment);
  llvm::Value *IncVal = Builder.CreateLoad(GetAddrOfLocalVar(InductionVarDecl));
  llvm::Value *One = llvm::ConstantInt::get(ConvertType(InductionVarDecl->getType()), 1);
  IncVal = Builder.CreateAdd(IncVal, One);
  Builder.CreateStore(IncVal, GetAddrOfLocalVar(InductionVarDecl));

  BreakContinueStack.pop_back();
  ConditionScope.ForceCleanup();
  EmitStopPoint(CE);

  EmitBranch(ConditionBlock);
  PForScope.ForceCleanup();
  LoopStack.pop();

  EmitBlock(Sync.getBlock());
  Builder.CreateSync(End, SRStart);
  EmitBlock(End, true);
  return true;
}

bool CodeGenFunction::EmitKokkosParallelReduce(const CallExpr *CE,
                    ArrayRef<const Attr *> Attrs) {
  DiagnosticsEngine &Diags = CGM.getDiags();
  Diags.Report(CE->getExprLoc(), diag::warn_kokkos_reduce_unsupported);
  return false;
}

