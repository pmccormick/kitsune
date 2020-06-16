/**
 ***************************************************************************
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

static const char* KITSUNE_MD_NAME = "kitsune.semantics";

using namespace clang;
using namespace CodeGen;

namespace {

  static const Expr *SimplifyExpr(const Expr *E) {
    return E->IgnoreImplicit()->IgnoreImpCasts();
  }


  static bool ExtractConstructName(const Expr* E, 
				   std::string& ConstructName) {

    if (E->getStmtClass() == Expr::CXXConstructExprClass) {
      const CXXConstructExpr *CXXCE = dyn_cast<CXXConstructExpr>(E);
      const Expr *Arg = CXXCE->getArg(0)->IgnoreImplicit();
      if (Arg->getStmtClass() == Expr::StringLiteralClass) {
        ConstructName = dyn_cast<StringLiteral>(Arg)->getString().str();
        return true;
      } 
    } 

    return false;
  }

  static void ExtractParallelForComponents(const CallExpr* CE,
					   std::string &ConstructName, 
					   const Expr *&PolicyExpr, 
					   const LambdaExpr *&LE) {
    unsigned int curArgIndex = 0;
    const Expr*  OrigExpr    = CE->getArg(curArgIndex);
    const Expr*  SimpExpr    = SimplifyExpr(OrigExpr);

    if (ExtractConstructName(SimpExpr, ConstructName)) {
      curArgIndex++;
      OrigExpr = CE->getArg(curArgIndex);
      SimpExpr = SimplifyExpr(OrigExpr);
    } else {
      ConstructName.clear();
    }

    switch(SimpExpr->getStmtClass()) {
      case Expr::IntegerLiteralClass:
      case Expr::BinaryOperatorClass:
      case Expr::DeclRefExprClass:
      case Expr::CallExprClass:
        PolicyExpr = OrigExpr;
        curArgIndex++;
        OrigExpr = CE->getArg(curArgIndex);
        SimpExpr = SimplifyExpr(OrigExpr);
        break;
      default:
        //SE->dump();
        PolicyExpr = nullptr;
        LE = nullptr;
        return;
    }

    if (SimpExpr->getStmtClass() == Expr::LambdaExprClass) {
      LE = dyn_cast<LambdaExpr>(SimpExpr);
    } else {
      LE = nullptr;
      return;
    }
  }


  // FIXME: This should probably be moved out of the kokkos-centric implementation.  
  /// \brief Cleanup to ensure parent stack frame is synced.
  struct RethrowCleanup : public EHScopeStack::Cleanup {
    llvm::BasicBlock *InvokeDest;
  public:
    RethrowCleanup(llvm::BasicBlock *InvokeDest = nullptr)
      : InvokeDest(InvokeDest) {}
    virtual ~RethrowCleanup() {}
    void Emit(CodeGenFunction &CGF, Flags F) {
      llvm::BasicBlock *DetRethrowBlock = CGF.createBasicBlock("det.rethrow");
      if (InvokeDest) {
        CGF.Builder.CreateInvoke(CGF.CGM.getIntrinsic(llvm::Intrinsic::detached_rethrow),
				 DetRethrowBlock, InvokeDest);
      } else {
        CGF.Builder.CreateBr(DetRethrowBlock);
      }
      CGF.EmitBlock(DetRethrowBlock);
    }
  };

  // FIXME: This should probably be moved out of the kokkos-centric implementation.  
  // Helper routine copied from CodeGenFunction.cpp
  static void EmitIfUsed(CodeGenFunction &CGF, llvm::BasicBlock *BB) {
    if (!BB) return;
    if (!BB->use_empty())
      return CGF.CurFn->getBasicBlockList().push_back(BB);
    delete BB;
  }
}

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
}


// Kokkos parallel_for details: 
//  
//  Interface: 
// 
//   * template<class ExecPolicy, class FunctorType>
//     Kokkos::parallel_for(const std::string& name, 
//                          const ExecPolicy& policy, 
//                          const FunctorType& functor);
//
//   * template<class ExecPolicy, class FunctorType>
//     Kokkos::parallel_for(const ExecPolicy& policy,
//                          const FunctorType& functor);
// 
// 
//  Parameters: 
// 
//   * name: a string used in profiling and debugging via Kokkos Profiling Hooks. 
// 
//   * ExecPolicy: defines the iteration space and other execution properties. 
//     Valid policies are:
//
//     - IntegerType: a one-dimensional iteration range starting from 0 and going
//       to a count. 
//     - RangePolicy: defines a one-dimensional iteration range.
//     - MDRangePolicy: defines a multi-dimensional iteration space.
//     - TeamPolicy: define a one-dimensional iteration range to be executed by
//       a thread team (only valid inside a parallel region executed through a 
//       TeamPolicy or TaskTeam.
//
//   * FunctorType: A valid functor having an operator() with a matching signature
//     for ExecPolicy.  The functor can be defined using a C++ class/struct or lambda.
//
// 
//  Semantics 
//
//   * Neither concurrency nor order of execution of loop iterations are guaranteed.
// 
//   * This call can potentially be asynchrnous (i.e. return the caller immediately). 
//
// 
// Kitsune details: 
// 
//   - Functors (due to potentially living in separate compilation units) are not
//     supported. 
// 
//   - The 'name' parameter form is recognized and processed but information is 
//     incorporated into the IR metadata vs. being registered with the Kokkos 
//     profiling hooks. 
// 
//   - Execution policy details are still a work-in-progress.  Certain details 
//     will potentially move into attributes vs. trying to tear apart C++ gore 
//     as we lower... 
// 
bool CodeGenFunction::EmitKokkosParallelFor(const CallExpr *KokkosExpr, 
					    ArrayRef<const Attr *> ForallAttrs) {

  std::string ConstructName;
  const Expr *PolicyExpr = nullptr;  
  const LambdaExpr *Lambda = nullptr; 

  ExtractParallelForComponents(KokkosExpr, ConstructName, PolicyExpr, Lambda);

  // If there were any issues extracting the various components we'll
  // have a nullptr value for the corresponding expression(s).  We
  // issue a warning about failure and return false -- indicating we
  // should fall back to standard C++ mechanisms.
  DiagnosticsEngine &Diags = CGM.getDiags();
  if (PolicyExpr == nullptr) {
    // FIXME: need to change naming of warning to match kokkos terminology.
    Diags.Report(KokkosExpr->getExprLoc(), diag::warn_kokkos_unknown_bounds_expr);
    return false;
  }

  if (Lambda == nullptr) { 
    Diags.Report(KokkosExpr->getExprLoc(), diag::warn_kokkos_no_functor);
    return false;
  }

  // Create some metadata entries that we can use (1) as a way to help
  // sort through the generated IR and (2) communicate some semantics
  // to the transformation and code gen stages. 
  llvm::LLVMContext &Ctx = CGM.getModule().getContext();
  llvm::Metadata *MDVals[] = {
    llvm::MDString::get(Ctx, "kokkos.parallel_for"),
    llvm::MDString::get(Ctx, "kokkos.loop_index")
  };
  llvm::NamedMDNode *DeclMD = CGM.getModule().getOrInsertNamedMetadata(KITSUNE_MD_NAME);  
  DeclMD->addOperand(llvm::MDNode::get(Ctx, MDVals[0]));
  DeclMD->addOperand(llvm::MDNode::get(Ctx, MDVals[1]));


  // Start the process of converting the parallel_for construct into a 
  // parallel loop structure... 
  JumpDest LoopExit = getJumpDestInCurrentScope("kokkos.forall.end");
  PushSyncRegion();
  llvm::Instruction *SyncRegionStart = EmitSyncRegionStart();
  CurSyncRegion->setSyncRegionStart(SyncRegionStart);

  LexicalScope ForallScope(*this, KokkosExpr->getSourceRange());

  const CXXMethodDecl *MD = Lambda->getCallOperator();
  assert(MD && "EmitKokkosParallelFor() -- bad method decl!");
  const ParmVarDecl *LoopVarD = MD->getParamDecl(0);
  assert(LoopVarD && "EmitKokkosParallelFor() -- bad loop variable!");
  EmitVarDecl(*LoopVarD);
  Address Addr = GetAddrOfLocalVar(LoopVarD);
  llvm::Value *Zero = llvm::ConstantInt::get(ConvertType(LoopVarD->getType()), 0);
  Builder.CreateStore(Zero, Addr);


  // Next, work towards determining the end of the loop range.
  llvm::Value *LoopEnd = nullptr;
  if (PolicyExpr->getStmtClass() == Expr::BinaryOperatorClass) {
    RValue RV = EmitAnyExpr(PolicyExpr);
    LoopEnd = RV.getScalarVal();
  } else { 
    LoopEnd = EmitScalarExpr(PolicyExpr);
  }

  llvm::Type  *LoopVarTy = ConvertType(LoopVarD->getType());
  unsigned NBits  = LoopEnd->getType()->getPrimitiveSizeInBits();
  unsigned LVBits = LoopVarTy->getPrimitiveSizeInBits();
  // We may need to truncate/extend the range to get it to match 
  // the type of loop variable. 
  if (NBits > LVBits) {
    LoopEnd = Builder.CreateTrunc(LoopEnd, LoopVarTy);
  } else if (NBits < LVBits) {
    LoopEnd = Builder.CreateZExt(LoopEnd, LoopVarTy);
  } else {
    // bit count matches, nothing to do... 
  }

  JumpDest Continue = getJumpDestInCurrentScope("kokkos.forall.cond");
  llvm::BasicBlock *CondBlock = Continue.getBlock();
  EmitBlock(CondBlock);

  const SourceRange &R = KokkosExpr->getSourceRange();

  LoopStack.setSpawnStrategy(GetTapirStrategyAttr(ForallAttrs));
  LoopStack.push(CondBlock, CGM.getContext(), ForallAttrs,
		 SourceLocToDebugLoc(R.getBegin()),
		 SourceLocToDebugLoc(R.getEnd()));

  JumpDest Preattach = getJumpDestInCurrentScope("kokkos.forall.preattach");
  Continue = getJumpDestInCurrentScope("kokkos.forall.inc");

  // Store the blocks to use for break and continue. 
  // 
  // FIXME?: Why is the code below BreakContinue(Preattach, Preattach)
  // versus BreakContinue(Preattach, Continue)?  
  BreakContinueStack.push_back(BreakContinue(Preattach, Preattach));

  // Create a clean up scope for the condition variable. 
  LexicalScope ConditionalScope(*this, R);

  // Save the old alloca insertion point. 
  llvm::AssertingVH<llvm::Instruction> OldAllocaInsertPt = AllocaInsertPt;
  // Save the old exception handling state. 
  llvm::BasicBlock *OldEHResumeBlock  = EHResumeBlock;
  llvm::Value      *OldExceptionSlot  = ExceptionSlot;
  llvm::AllocaInst *OldEHSelectorSlot = EHSelectorSlot;

  llvm::BasicBlock *SyncContinueBlock = createBasicBlock("kokkos.end.continue");
  bool madeSync = false;

  llvm::BasicBlock  *DetachBlock;
  llvm::BasicBlock  *ForallBodyEntry;
  llvm::BasicBlock  *ForallBody;

  {
    llvm::BasicBlock *ExitBlock = LoopExit.getBlock();
    // If there is any cleanup between here and the loop-exit scope
    // we need to create a block to stage the loop exit. 
    if (ForallScope.requiresCleanups()) {
      ExitBlock = createBasicBlock("kokkos.cond.cleanup");
    }

    // As long as the conditional is true we continue looping... 
    DetachBlock = createBasicBlock("kokkos.forall.detach");
    // Emit extra entry block for the detached body, this ensures 
    // that the detached block has only one predecessor. 
    ForallBodyEntry = createBasicBlock("kokkos.forall.body.entry");
    ForallBody      = createBasicBlock("kokkos.forall.body");

    llvm::Value *LoopVal     = Builder.CreateLoad(Addr);
    llvm::Value *BoolCondVal = Builder.CreateICmpULT(LoopVal, LoopEnd);
    Builder.CreateCondBr(BoolCondVal, DetachBlock, ExitBlock);

    if (ExitBlock != LoopExit.getBlock()) {
      EmitBlock(ExitBlock);
      Builder.CreateSync(SyncContinueBlock, SyncRegionStart);
      EmitBlock(SyncContinueBlock);
      PopSyncRegion();
      madeSync = true;
      EmitBranchThroughCleanup(LoopExit);
    }

    EmitBlock(DetachBlock);

    llvm::DetachInst *Detach = Builder.CreateDetach(ForallBodyEntry, Continue.getBlock(), SyncRegionStart);
    llvm::LLVMContext &Ctx = CGM.getModule().getContext();

    llvm::NamedMDNode *MD = CGM.getModule().getOrInsertNamedMetadata("kitsune.semantics");
    llvm::Metadata *MDVals[] = {
      llvm::MDString::get(Ctx, "kokkos.parallel_for")
    };
    MD->addOperand(llvm::MDNode::get(Ctx, MDVals));
    Detach->setMetadata("kitsune.semantics", MD->getOperand(0));

    // Create a new alloca insertion point.
    llvm::Value *Undef = llvm::UndefValue::get(Int32Ty);
    AllocaInsertPt = new llvm::BitCastInst(Undef, Int32Ty, 
       "", ForallBodyEntry);
    // Set up nested exception handling state. 
    EHResumeBlock  = nullptr;
    ExceptionSlot  = nullptr;
    EHSelectorSlot = nullptr;
    EmitBlock(ForallBodyEntry);
  }

  // Create a scope for the loop-variable cleanup.
  RunCleanupsScope DetachCleanupScope(*this);
  EHStack.pushCleanup<RethrowCleanup>(EHCleanup);

  Builder.CreateBr(ForallBody);
  EmitBlock(ForallBody);
  incrementProfileCounter(KokkosExpr);

  {
    // Create a separate cleanup scope for the forall body
    // (in case it is not a compound statement).
    RunCleanupsScope BodyScope(*this);

    // Emit the lambda expression as the body of the forall 
    // loop.  Given this is a lambda it may have special wrapped 
    // AST for handling captured variables -- to address this we 
    // have to flag it so we handle it as a special case... 
    InKokkosConstruct = true;
    EmitStmt(Lambda->getBody());
    InKokkosConstruct = false;
    Builder.CreateBr(Preattach.getBlock());
  }
  
  {
    EmitBlock(Preattach.getBlock());
    DetachCleanupScope.ForceCleanup();
    Builder.CreateReattach(Continue.getBlock(), SyncRegionStart);
  }

  {
    llvm::Instruction *Ptr = AllocaInsertPt;
    AllocaInsertPt = OldAllocaInsertPt;
    Ptr->eraseFromParent();

    // Restore the exception handling state. 
    EmitIfUsed(*this, EHResumeBlock);
    EHResumeBlock  = OldEHResumeBlock;
    ExceptionSlot  = OldExceptionSlot;
    EHSelectorSlot = OldEHSelectorSlot;
  }

  // Emit the increment next. 
  EmitBlock(Continue.getBlock());

  // Emit the loop variable increment. 
  llvm::Value *IncVal = Builder.CreateLoad(Addr);
  llvm::Value *One    = llvm::ConstantInt::get(ConvertType(LoopVarD->getType()), 1);
  IncVal = Builder.CreateAdd(IncVal, One);
  Builder.CreateStore(IncVal, Addr);
  BreakContinueStack.pop_back();
  ConditionalScope.ForceCleanup();

  EmitStopPoint(KokkosExpr);
  EmitBranch(CondBlock);
  ForallScope.ForceCleanup();
  LoopStack.pop();

  // Emit the fall-through block. 
  EmitBlock(LoopExit.getBlock(), true);
  if (!madeSync) {
    Builder.CreateSync(SyncContinueBlock, SyncRegionStart);
    EmitBlock(SyncContinueBlock);
    PopSyncRegion();
  }

  return true;
}


namespace {

  static void ExtractParallelReduceComponents(const CallExpr *CE,
					      std::string &ConstructName,
					      const Expr *&BoundsExpr,
					      const VarDecl *&LocalReduceVar,
					      const DeclRefExpr *&ReducedValExpr,
					      const LambdaExpr *&LE)
  {
    unsigned int curArgIndex = 0;
    const Expr *OrigExpr = CE->getArg(curArgIndex);
    const Expr *SimpExpr = SimplifyExpr(OrigExpr); 

    // Check to see if we have a named construct. 
    if (ExtractConstructName(SimpExpr, ConstructName)) {
      curArgIndex++;
      OrigExpr = CE->getArg(curArgIndex);
      SimpExpr = SimplifyExpr(OrigExpr);
    }

    if (SimpExpr->getStmtClass() == Expr::CXXConstructExprClass) {
      const CXXConstructExpr *CXXCE = dyn_cast<CXXConstructExpr>(SimpExpr);
      SimpExpr = CXXCE->getArg(0)->IgnoreImplicit();
      if (SimpExpr->getStmtClass() == Expr::StringLiteralClass) {
        ConstructName = dyn_cast<StringLiteral>(SimpExpr)->getString().str();
      } 
    } 

    // Check for various forms of "trip count".
    switch(SimpExpr->getStmtClass()) {
      case Expr::IntegerLiteralClass:
      case Expr::BinaryOperatorClass:
      case Expr::DeclRefExprClass:
      case Expr::CallExprClass:
        BoundsExpr = OrigExpr;
        curArgIndex++;
        OrigExpr = CE->getArg(curArgIndex);
        SimpExpr = SimplifyExpr(OrigExpr);
        break;
      default: 
        //SimpExpr->dump();
        BoundsExpr = nullptr;
        LE = nullptr;
        return;
    }

    if (SimpExpr->getStmtClass() == Expr::LambdaExprClass) {
      LE = dyn_cast<LambdaExpr>(SimpExpr);
      curArgIndex++;
    } else {
      LE = nullptr;
      return;
    }

    const CXXMethodDecl* MD = LE->getCallOperator();
    LocalReduceVar = MD->getParamDecl(1);

    const FunctionDecl *FD = CE->getDirectCallee();
    if (FD != nullptr) {
      ReducedValExpr = dyn_cast<DeclRefExpr>(CE->getArg(curArgIndex));
    }
  }
}


bool 
CodeGenFunction::EmitKokkosParallelReduce(const CallExpr *CE, 
					  ArrayRef<const Attr *> ReduceAttrs) {

  std::string ConstructName;          // debug/profile name. 
  const Expr  *BE = nullptr;          // "bounds" expression. 
  const VarDecl *LocalVD = nullptr;   // local reduction variable.  
  const DeclRefExpr *RE = nullptr;    // final reduction expression. 
  const LambdaExpr *Lambda = nullptr; // the lambda 
  DiagnosticsEngine &Diags = CGM.getDiags();
  llvm::LLVMContext &Ctx = CGM.getModule().getContext();
  
  llvm::Metadata *MDVals[] = {
    llvm::MDString::get(Ctx, "kokkos.parallel_reduce"),
    llvm::MDString::get(Ctx, "local.reduction_index"),
    llvm::MDString::get(Ctx, "local.reduction_var_ref"), 
    llvm::MDString::get(Ctx, "local.reduction_var"),
    llvm::MDString::get(Ctx, "result.gather"),    
    llvm::MDString::get(Ctx, "result.reduced_val_ref"),
    llvm::MDString::get(Ctx, "result.reduced_var")
  };


  llvm::NamedMDNode *DeclMD = CGM.getModule().getOrInsertNamedMetadata(KITSUNE_MD_NAME);
  DeclMD->addOperand(llvm::MDNode::get(Ctx, MDVals[0]));
  DeclMD->addOperand(llvm::MDNode::get(Ctx, MDVals[1]));
  DeclMD->addOperand(llvm::MDNode::get(Ctx, MDVals[2]));
  DeclMD->addOperand(llvm::MDNode::get(Ctx, MDVals[3]));
  DeclMD->addOperand(llvm::MDNode::get(Ctx, MDVals[4]));
  DeclMD->addOperand(llvm::MDNode::get(Ctx, MDVals[5]));

  ExtractParallelReduceComponents(CE, ConstructName, BE, LocalVD, RE, Lambda);

  if (Lambda == nullptr) {
    Diags.Report(CE->getExprLoc(), diag::warn_kokkos_no_functor);
    return false;
  }

  if (BE == nullptr) { 
    Diags.Report(CE->getExprLoc(), diag::warn_kokkos_unknown_bounds_expr);
    return false;
  }

  if (LocalVD == nullptr) {
    Diags.Report(CE->getExprLoc(), diag::warn_kokkos_reduce_bad_intermediate_vardecl);
    return false;
  }

  if (RE == nullptr) {
    Diags.Report(CE->getExprLoc(), diag::warn_kokkos_reduce_bad_final_vardecl);
    return false;
  }


  JumpDest LoopExit = getJumpDestInCurrentScope("kokkos.reduce.end");
  PushSyncRegion();
  llvm::Instruction *SyncRegionStart = EmitSyncRegionStart();
  CurSyncRegion->setSyncRegionStart(SyncRegionStart);

  LexicalScope ReductionScope(*this, CE->getSourceRange());

  const CXXMethodDecl *MD = Lambda->getCallOperator();
  assert(MD && "EmitKokkosParallelReduce() -- bad method decl!");
  const ParmVarDecl *LoopVarD = MD->getParamDecl(0);
  assert(LoopVarD && "EmitKokkosParallelReduce() -- bad loop variable!");
  EmitVarDecl(*LoopVarD);

  Address LoopVarAddr = GetAddrOfLocalVar(LoopVarD);
  llvm::Value *Zero = llvm::ConstantInt::get(ConvertType(LoopVarD->getType()), 0);
  Builder.CreateStore(Zero, LoopVarAddr);
  if (llvm::Instruction *I = dyn_cast<llvm::Instruction>(LoopVarAddr.getPointer())) {
    I->setMetadata(KITSUNE_MD_NAME, DeclMD->getOperand(/*local.loop_index*/1));
  }

  // Kokkos' reduction call has an internally implemented reduction value that 
  // is hidden from the developer but passed as a reference to the lambda.  To
  // generate correct code we have to tease out the reference and then
  // Emit the reduction variable (gathered value) and initialize it to zero... 
  const ParmVarDecl *LocalReduceVarD = MD->getParamDecl(1);
  assert(LocalReduceVarD && "EmitKokkosParallelReduce() -- bad local reduction variable!");
  assert(LocalReduceVarD->getType()->isReferenceType() && 
          "EmitKokkosParallelReduce -- expected reference type");

  EmitVarDecl(*LocalReduceVarD);
  Address ReduceRefAddr = GetAddrOfLocalVar(LocalReduceVarD);
  if (llvm::Instruction *I = dyn_cast<llvm::Instruction>(ReduceRefAddr.getPointer())) {
    I->setMetadata(KITSUNE_MD_NAME, DeclMD->getOperand(/*reduce_var*/2));
  }
  QualType RefType        = LocalReduceVarD->getType();
  QualType ReductionVarType  = RefType.getNonReferenceType();
  llvm::Type *LReduceType = getTypes().ConvertType(ReductionVarType);
  llvm::AllocaInst *RInst = Builder.CreateAlloca(LReduceType, nullptr, ".reduce_var");
  RInst->setMetadata(KITSUNE_MD_NAME, DeclMD->getOperand(/*reduce_var*/3));
  Address RVarAddr = Address(RInst, getContext().getTypeAlignInChars(ReductionVarType));

  llvm::Value *RZero = llvm::ConstantInt::get(ConvertType(ReductionVarType), 0);
  Builder.CreateStore(RZero, RVarAddr);

  llvm::Instruction *RefStore = Builder.CreateStore(RVarAddr.getPointer(), ReduceRefAddr);
  RefStore->setMetadata(KITSUNE_MD_NAME, DeclMD->getOperand(/*reduce_var*/3));

  // Next, work towards determining the end of the loop range.
  llvm::Value *LoopEnd = nullptr;
  if (BE->getStmtClass() == Expr::BinaryOperatorClass) {
    RValue RV = EmitAnyExpr(BE);
    LoopEnd = RV.getScalarVal();
  } else { 
    LoopEnd = EmitScalarExpr(BE);
  }

  llvm::Type  *LoopVarTy = ConvertType(LoopVarD->getType());
  unsigned NBits  = LoopEnd->getType()->getPrimitiveSizeInBits();
  unsigned LVBits = LoopVarTy->getPrimitiveSizeInBits();
  // We may need to truncate/extend the range to get it to match 
  // the type of loop variable. 
  if (NBits > LVBits) {
    LoopEnd = Builder.CreateTrunc(LoopEnd, LoopVarTy);
  } else if (NBits < LVBits) {
    LoopEnd = Builder.CreateZExt(LoopEnd, LoopVarTy);
  } else {
    // bit count matches, nothing to do... 
  }

  LValue FinalReductionLValue = EmitDeclRefLValue(RE);
  llvm::Value *FinalReductionVal = FinalReductionLValue.getPointer(*this);
  if (llvm::Instruction *I = dyn_cast<llvm::Instruction>(FinalReductionVal)) {
    I->setMetadata(KITSUNE_MD_NAME, DeclMD->getOperand(/*local.loop_index*/4));
  }
  
  JumpDest Continue = getJumpDestInCurrentScope("kokkos.reduce.cond");
  llvm::BasicBlock *CondBlock = Continue.getBlock();
  EmitBlock(CondBlock);

  const SourceRange &R = CE->getSourceRange();

  LoopStack.setSpawnStrategy(GetTapirStrategyAttr(ReduceAttrs));
  LoopStack.push(CondBlock, CGM.getContext(), ReduceAttrs,
		 SourceLocToDebugLoc(R.getBegin()),
		 SourceLocToDebugLoc(R.getEnd()));

  JumpDest Preattach = getJumpDestInCurrentScope("kokkos.reduce.preattach");
  Continue = getJumpDestInCurrentScope("kokkos.reduce.inc");
  BreakContinueStack.push_back(BreakContinue(Preattach, Preattach));

  // Create a clean up scope for the condition variable. 
  LexicalScope ConditionalScope(*this, R);

  // Save the old alloca insertion point. 
  llvm::AssertingVH<llvm::Instruction> OldAllocaInsertPt = AllocaInsertPt;
  // Save the old exception handling state. 
  llvm::BasicBlock *OldEHResumeBlock  = EHResumeBlock;
  llvm::Value      *OldExceptionSlot  = ExceptionSlot;
  llvm::AllocaInst *OldEHSelectorSlot = EHSelectorSlot;

  llvm::BasicBlock *SyncContinueBlock = createBasicBlock("kokkos.end.continue");
  bool madeSync = false;

  llvm::BasicBlock  *DetachBlock;
  llvm::BasicBlock  *ReduceBodyEntry;
  llvm::BasicBlock  *ReduceBody;

  {
    llvm::BasicBlock *ExitBlock = LoopExit.getBlock();
    // If there is any cleanup between here and the loop-exit scope
    // we need to create a block to stage the loop exit. 
    if (ReductionScope.requiresCleanups()) {
      ExitBlock = createBasicBlock("kokkos.cond.cleanup");
    }

    // As long as the conditional is true we continue looping... 
    DetachBlock = createBasicBlock("kokkos.reduce.detach");
    // Emit extra entry block for the detached body, this ensures 
    // that the detached block has only one predecessor. 
    ReduceBodyEntry = createBasicBlock("kokkos.reduce.body.entry");
    ReduceBody      = createBasicBlock("kokkos.reduce.body");

    llvm::Value *LoopVal     = Builder.CreateLoad(LoopVarAddr);
    llvm::Value *BoolCondVal = Builder.CreateICmpULT(LoopVal, LoopEnd);
    Builder.CreateCondBr(BoolCondVal, DetachBlock, ExitBlock
			 /* FIXME: createProfileWeights()... */);

    if (ExitBlock != LoopExit.getBlock()) {
      EmitBlock(ExitBlock);
      Builder.CreateSync(SyncContinueBlock, SyncRegionStart);
      EmitBlock(SyncContinueBlock);
      PopSyncRegion();
      madeSync = true;
      EmitBranchThroughCleanup(LoopExit);
    }

    EmitBlock(DetachBlock);
    llvm::DetachInst *Detach = Builder.CreateDetach(ReduceBodyEntry, 
						    Continue.getBlock(), 
						    SyncRegionStart);
    Detach->setMetadata(KITSUNE_MD_NAME, DeclMD->getOperand(0));

    // Create a new alloca insertion point.
    llvm::Value *Undef = llvm::UndefValue::get(Int32Ty);
    AllocaInsertPt = new llvm::BitCastInst(Undef, Int32Ty, 
       "", ReduceBodyEntry);
    // Set up nested exception handling state. 
    EHResumeBlock  = nullptr;
    ExceptionSlot  = nullptr;
    EHSelectorSlot = nullptr;
    EmitBlock(ReduceBodyEntry);
  }

  // Create a scope for the loop-variable cleanup.
  RunCleanupsScope DetachCleanupScope(*this);
  EHStack.pushCleanup<RethrowCleanup>(EHCleanup);


  if (LoopVarD) {
    AutoVarEmission LVEmission = EmitAutoVarAlloca(*LoopVarD);
    QualType type = LoopVarD->getType();
    Address Loc = LVEmission.getObjectAddress(*this);
    LValue LV = MakeAddrLValue(Loc, type);
    LV.setNonGC(true);
    EmitStoreThroughLValue(RValue::get(LoopVarAddr.getPointer()), LV, true);

    EmitAutoVarCleanups(LVEmission);
  }

  Builder.CreateBr(ReduceBody);
  EmitBlock(ReduceBody);
  incrementProfileCounter(CE);

  // Create a separate cleanup scope for the reduce body and emit 
  // the lambda as the body of the reduction loop. 
  {
    RunCleanupsScope BodyScope(*this);
    // Given this is a lambda it may have special wrapped AST for
    // handling captured variables -- to address this we have to flag
    // that we're in Kokkos-mode so when we emit the body it is
    // handled correctly...
    InKokkosConstruct = true;
    EmitStmt(Lambda->getBody());
    InKokkosConstruct = false;
    Builder.CreateBr(Preattach.getBlock());
  }
  
  // Finish the detached body and emit the reattach. 
  {
    EmitBlock(Preattach.getBlock());

    DetachCleanupScope.ForceCleanup();
    Builder.CreateReattach(Continue.getBlock(), SyncRegionStart);
  }

  // Restore CFG state after the detached region... 
  {
    // Restore alloca insertion point. 
    llvm::Instruction *Ptr = AllocaInsertPt;
    AllocaInsertPt = OldAllocaInsertPt;
    Ptr->eraseFromParent();

    // Restore EH state. 
    EmitIfUsed(*this, EHResumeBlock);
    EHResumeBlock  = OldEHResumeBlock;
    ExceptionSlot  = OldExceptionSlot;
    EHSelectorSlot = OldEHSelectorSlot;
  }

  // Emit the increment next. 
  EmitBlock(Continue.getBlock());
  llvm::Value *IncVal = Builder.CreateLoad(LoopVarAddr);
  llvm::Value *One    = llvm::ConstantInt::get(ConvertType(LoopVarD->getType()), 1);
  IncVal = Builder.CreateAdd(IncVal, One);
  Builder.CreateStore(IncVal, LoopVarAddr);

  BreakContinueStack.pop_back();
  ConditionalScope.ForceCleanup();

  EmitStopPoint(CE);
  EmitBranch(CondBlock);

  ReductionScope.ForceCleanup();

  LoopStack.pop();

  EmitBlock(LoopExit.getBlock(), true);

  llvm::LoadInst *LI = Builder.CreateLoad(RVarAddr, "local.lsum");
  LI->setMetadata(KITSUNE_MD_NAME, DeclMD->getOperand(/*reduce_var*/4));
  Builder.CreateStore(LI, FinalReductionLValue.getAddress(*this));

  // Emit the fall-through block. 
  if (!madeSync) {
    Builder.CreateSync(SyncContinueBlock, SyncRegionStart);
    EmitBlock(SyncContinueBlock);
    PopSyncRegion();
  }

  return true;
}
