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
 *z
 ***************************************************************************/
#include <cstdio>
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/CodeGen/CGFunctionInfo.h"
#include "CodeGenFunction.h"

using namespace clang;
using namespace CodeGen;

// Worth discussing why we want an anonymous namespace and what belongs here
// as opposed to putting in CodeGenFunction.h
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
} // hidden/local namespace


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
bool CodeGenFunction::ParseAndValidateParallelFor(const CallExpr* CE,
            std::string &CN, 
            SmallVector<std::pair<const ParmVarDecl*,std::pair<const Expr*, const Expr*>>,6> &IVInfos,
            const LambdaExpr *&LE,
            DiagnosticsEngine &Diags)
{
  // Recognized constructs:
  //
  //   1. parallel_for(["name"], N, lambda_expr...);
  //
  //   2. parallel_for(["name"], Kokkos::MDRangePolicy<Kokkos::Rank<DIM>>({0,0,...},{N,K,...}), lambda_expr...);

  unsigned int curArgIndex = 0;
  const Expr *SE = SimplifyExpr(CE->getArg(curArgIndex));        // Simplified expression.

  // Parse a 'named' construct. We will capture the name, but don't do
  // anything with it. NOTE: we assume this only comes in the form of a string
  // literal. This is ridiculously fragile. Only parses string literals as the
  // first argument e.g. parallel_for("name",...). It will not parse variables
  // e.g. parallel_for(name,...)
  if (auto* CXXCE=dyn_cast<CXXConstructExpr>(SE))
    if (auto* NameExpr=dyn_cast<StringLiteral>(CXXCE->getArg(0)->IgnoreImplicit())){
      CN = NameExpr->getString().str();
      SE = SimplifyExpr(CE->getArg(++curArgIndex));;
    };
    
  // Parse a vector of IV bounds, can be either an *Expr or an MDRangePolicy
  SmallVector<std::pair<const Expr*, const Expr*>,6> BoundsList;

  if (const CXXTemporaryObjectExpr *CXXTO = dyn_cast<CXXTemporaryObjectExpr>(SE);
    CXXTO && CXXTO->getBestDynamicClassType()->getNameAsString() == "MDRangePolicy") {
    // The first non-name argument is an MDRangePolicy, extract both lower and upper bounds
    // for multiple induction variables

    // extract the initializer lists of lower and upper bounds
    const InitListExpr *LowerBounds = dyn_cast<InitListExpr>(CXXTO->getArg(0)->IgnoreImplicit());
    const InitListExpr *UpperBounds = dyn_cast<InitListExpr>(CXXTO->getArg(1)->IgnoreImplicit());

    // zip the bounds
    for (unsigned int i = 0; i<LowerBounds->getNumInits(); i++) 
      BoundsList.push_back(std::pair<const Expr*, const Expr*>(
        LowerBounds->getInit(i)->IgnoreImplicit(), 
        UpperBounds->getInit(i)->IgnoreImplicit())
      );      
  } else  { 

    // No MDRangePolicy, in which case, this is an upper bound for a single IVInfo
    // Rather than try and parse, we will just assume this is a valid
    // expression as there are just too many ways to use or get in trouble.
    // Note we use the full expression for the upper bound, not the simplified
    // expression
    BoundsList.push_back(std::pair<const Expr*, const Expr*>(nullptr, CE->getArg(curArgIndex)));        
  }; 

  // Move to the next argument of the parallel_for, which should be the lambda
  SE = SimplifyExpr(CE->getArg(++curArgIndex));

  // Parse the lambda expression in the parallel_for
  LE = dyn_cast<LambdaExpr>(SE);

  // Extract the ParmVarDecl's from the lambda
  ArrayRef<ParmVarDecl*> Params{LE->getCallOperator()->parameters()};

  // Perform semantic analysis of the derived constructs

  // Check the bounds expression
  // if (IVInfos == nullptr) {
  //   // We didn't get a known bounds expression back -- this is most likely
  //   // due to some type of expression that we have yet to deal with.  We're
  //   // going to have to uncover these on a case-by-cases basis.
  //   Diags.Report(CE->getExprLoc(), diag::warn_kokkos_unknown_bounds_expr);
  //   return false;
  // }

  // Check the lambda expression is at least well formed
  if (LE == nullptr) {
    // The parallel_for doesn't have a (recognizable) lambda expression.
    Diags.Report(CE->getExprLoc(), diag::warn_kokkos_no_functor);
    return false;
  }

  // Maker sure the lambda has defined arguments
  if (LE->getCallOperator()->getNumParams()==0){
    // The lambda has no arguments, and gets all variables by capture
    Diags.Report(CE->getExprLoc(), diag::warn_kokkos_no_functor_args);
    return false; 
  }

  // DO WAY MORE ERROR CHECKING...

  // Pack everything up
  for (int i=0; i<Params.size(); ++i)
    IVInfos.push_back({Params[i],BoundsList[i]});

  // Everything was parsed correctly
  return true;
}


// Emit the ParmVarDecl defined in a Kokkos lambda and initialize
// the resulting value
void CodeGenFunction::EmitAndInitializeKokkosIV(
  const std::pair<const ParmVarDecl*,std::pair<const Expr*, const Expr*>> &IVInfo){

  // Convenience variable for the actual ParmVarDecl
  const ParmVarDecl *IV = IVInfo.first;

  // Just emit the induction variable from the ParmVarDecl
  EmitVarDecl(*IV);
  
  // Determine the initial value (default is 0)
  llvm::Value *InitValue = IVInfo.second.first ? 
    EmitScalarExpr(IVInfo.second.first): 
    InitValue = llvm::ConstantInt::get(ConvertType(IV->getType()), 0);

  // Store the initial value into the address
  Builder.CreateStore(InitValue, GetAddrOfLocalVar(IV));
}

// Emit a Kokkos parallel for condition 
llvm::Value* CodeGenFunction::EmitKokkosParallelForCond(
  const std::pair<const ParmVarDecl*,std::pair<const Expr*, const Expr*>> &IVInfo) {

  // Convenience variables for the actual ParmVarDecl and upper bound
  const ParmVarDecl *IV = IVInfo.first;
  const Expr *BoundsExpr = IVInfo.second.second;

  llvm::Value *LoopEnd = nullptr;
  if (BoundsExpr->getStmtClass() == Expr::BinaryOperatorClass) {
    RValue RV = EmitAnyExpr(BoundsExpr);
    LoopEnd = RV.getScalarVal();
  } else { 
    LoopEnd = EmitScalarExpr(BoundsExpr);
  }

  llvm::Type  *InductionVarTy = ConvertType(IV->getType());
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

  llvm::Value *InductionVal = Builder.CreateLoad(GetAddrOfLocalVar(IV));
  return Builder.CreateICmpULT(InductionVal, LoopEnd);
}

// Emit a Kokkos parallel for increment
void CodeGenFunction::EmitKokkosIncrement(const ParmVarDecl *IV){
  // load the induction variable
  llvm::Value *IncVal = Builder.CreateLoad(GetAddrOfLocalVar(IV));

  // create the number "1" with the correct type
  llvm::Value *One = llvm::ConstantInt::get(ConvertType(IV->getType()), 1);

  // emit the add one
  IncVal = Builder.CreateAdd(IncVal, One);

  // emit the store back into the induction variable
  Builder.CreateStore(IncVal, GetAddrOfLocalVar(IV));
}

// Emit the whole Kokkos parallel for
bool CodeGenFunction::EmitKokkosParallelFor(const CallExpr *CE,
              ArrayRef<const Attr *> KokkosAttrs) {
  /////////////////////////////////////////////////////////////////////////////
  // Code Modifications necessary for implementing parallel loops not required
  // by serial loops.

  // New basic blocks and jump destinations with Tapir terminators
  // Note that we only need one of each of these regardless of the number of
  // nested loops. 
  llvm::BasicBlock* Detach = createBasicBlock("kokkos.detach");
  JumpDest Reattach = getJumpDestInCurrentScope("kokkos.reattach");
  JumpDest Sync = getJumpDestInCurrentScope("kokkos.sync");
  
  // Declarations for capturing the IV vardecl to old and new llvm Values as
  // well as the alloca insertion point which we need to change and change back
  DeclMapByValueTy IVDeclMap; // map from Vardecl to {IV, thread safe IV vector}
  llvm::AssertingVH<llvm::Instruction> OldAllocaInsertPt = AllocaInsertPt;
  llvm::Value *Undef = llvm::UndefValue::get(Int32Ty);

  // emit the sync region
  PushSyncRegion();
  llvm::Instruction *SRStart = EmitSyncRegionStart();
  CurSyncRegion->setSyncRegionStart(SRStart);
  LoopStack.setSpawnStrategy(LoopAttributes::DAC);

  // Parse and validate the parallel for
  std::string      PFName; // construct name (for kokkos profiling)
  SmallVector<std::pair<const ParmVarDecl*,std::pair<const Expr*, const Expr*>>,6> IVInfos; 
  const LambdaExpr *Lambda = nullptr; // the lambda
  DiagnosticsEngine &Diags = CGM.getDiags();
  if (!ParseAndValidateParallelFor(CE, PFName, IVInfos, Lambda, Diags)) return true;
  const int numIVs = IVInfos.size();

  // End of parallel modification code block
  /////////////////////////////////////////////////////////////////////////////

  JumpDest LoopExit=getJumpDestInCurrentScope("kokkos.end");

  // In the case of nested loops, we need to have independent 
  // condition and Increment basic blocks for each induction variable
  // I don't think we need ForScope's unless we have structure type IVs,
  // but hopefully we will one day, so leaving in for now
  SmallVector<JumpDest, 6> Condition;
  SmallVector<llvm::BasicBlock*, 6> CondBlock;
  SmallVector<JumpDest, 6> Increment;
  SmallVector<LexicalScope*, 6> ForScope; 

  // Break from precedent and create all the basic blocks first because in the
  // codegen we need to link to the *next* increment block. Could probably move
  // the Increment creation to the code, but since I already have the loop...
  for (int i=0; i<numIVs; ++i){
    Condition.push_back(getJumpDestInCurrentScope("kokkos.cond")); 
    CondBlock.push_back(Condition.back().getBlock());
    Increment.push_back(getJumpDestInCurrentScope("kokkos.inc"));
  }
    

  // Emit and initialize the outermost IV initialization before the loop. This is the analog of
  // EmitStmt(S.getInit());
  EmitAndInitializeKokkosIV(IVInfos[0]);

  // Get the source range of the parallel_for once  
  const SourceRange &R = CE->getSourceRange();

  // loop over induction variables to create nested loop conditions and increments
  for (int i=0; i<numIVs; ++i){
  
    // Create a lexical scope for each induction variable. Using new, allows us
    // defer creation to here
    ForScope.push_back(new LexicalScope(*this, R));

    // In a kokkos parallel for there will always be a condition block
    EmitBlock(CondBlock[i]);
  
    // Emit and initialize nested IVs. This is the analog of
    // EmitStmt(S.getInit()); 
    // The allocas all go to the top, but the variable reset of the next IV will
    // correctly happen in the nested condition blocks.   
    if (i<numIVs-1) EmitAndInitializeKokkosIV(IVInfos[i+1]);

    LoopStack.push(CondBlock[i], CGM.getContext(), CGM.getCodeGenOpts(), KokkosAttrs,
                  SourceLocToDebugLoc(R.getBegin()),
                  SourceLocToDebugLoc(R.getEnd()));


    // Store the blocks to use for break and continue. Since we are emitting all
    // the loop code, we have control over the code except for the body block.
    // We will never encounter a break (anyway in parallel code) or continue
    // except for within the innermost loop body. Therefore any continue will be
    // encountered in the body and will jump to the Reattach. If we move the detach
    // to one of the outer loops, then we will need to move the reattach accordingly
    BreakContinueStack.push_back(BreakContinue(LoopExit, Reattach));

    // Note: There is need for condition scopes since we create the conditions and they are
    // simple inequalities
    
    // C99 6.8.5p2/p4: The first substatement is executed if the expression
    // compares unequal to 0.  The condition must be a scalar type.
    // Create the conditional.
    llvm::Value *BoolCondVal = EmitKokkosParallelForCond(IVInfos[i]);
    Builder.CreateCondBr(
        BoolCondVal, (i<numIVs-1 ? CondBlock[i+1] : Detach), (i==0 ? Sync.getBlock() : Increment[i-1].getBlock()));
    // DWS fix profile weights
    // ,createProfileWeightsForLoop(S.getCond(), getProfileCount(S.getBody()))

  }

  /////////////////////////////////////////////////////////////////////////////
  // The following block of code emits the detach block for parallel execution
  // along with its Tapir terminator. This is where we capture the induction 
  // variable by value and store it on the stack of the calling thread.

  EmitBlock(Detach);

  // Create threadsafe induction variables before the detach and put them in IVInfoDeclMap
  for (const auto& IVInfo : IVInfos) 
    EmitIVLoad(dyn_cast<VarDecl>(IVInfo.first), IVDeclMap);

  // As long as the condition is true, iterate the loop.
  llvm::BasicBlock *ForBody = createBasicBlock("kokkos.body");

  // create the detach terminator
  Builder.CreateDetach(ForBody, Increment.back().getBlock(), SRStart);
  
  // End of parallel modification code block
  /////////////////////////////////////////////////////////////////////////////

  EmitBlock(ForBody);

  incrementProfileCounter(CE);

  {
    // Need to explain this, because right now I have no idea, but 
    // is absolutely necessary not to segfault during codegen :-)
    InKokkosConstruct=true;

    // Create a separate cleanup scope for the body, in case it is not
    // a compound statement.
    RunCleanupsScope BodyScope(*this);

    ///////////////////////////////////////////////////////////////////////////
    // In this block of code, we change the alloca insert point so that the
    // alloca's happen after the detach and within the body block. This makes
    // sure each thread has its own local copy of the induction variable. We
    // also need to store the thread safe value from the calling thread into
    // this local copy. In EmitThreadSafeIV, we use AutoVarAlloca so any codegen
    // in the body automatically and correctly mapped to the local thread
    // safe copy of the induction variable.

    // change the alloca insert point to the body block
    SetAllocaInsertPoint(Undef, ForBody);

    // emit the thread safe induction variables and initialize them by value
    for (const auto &ivp : IVDeclMap) 
      EmitThreadSafeIV(ivp.first, ivp.second.second);

    // End of parallel modification code block
    ///////////////////////////////////////////////////////////////////////////

    // emit the body of the lambda expression
    EmitStmt(Lambda->getBody());

    InKokkosConstruct=false;
  }

  /////////////////////////////////////////////////////////////////////////////
  // In this block of code, we need to unwind the codegen of the induction
  // variable from the current local thread safe copy back to the original
  // induction variable. We also need to emit the reattach block and reset the
  // alloca insertion point.

  // Restore induction variable mappings after emitting body, and before
  // the increment
  for (const auto &ivp : IVDeclMap) 
    RestoreDeclMap(ivp.first, ivp.second.first);

  // emit the reattach block
  EmitBlock(Reattach.getBlock());
  Builder.CreateReattach(Increment.back().getBlock(), SRStart);

  // reset the alloca insertion point
  AllocaInsertPt->removeFromParent();
  AllocaInsertPt = OldAllocaInsertPt; 

  // End of parallel modification code block
  /////////////////////////////////////////////////////////////////////////////


  // Emit the increments, which is the Kokkos analog of 
  // EmitStmt(S.getInc());

  for (int i=numIVs-1; i>=0; --i) {

    // Emit the increment basic block
    EmitBlock(Increment[i].getBlock());

    // Emit the actual increment code
    EmitKokkosIncrement(IVInfos[i].first);

    BreakContinueStack.pop_back();

    EmitStopPoint(CE);
    EmitBranch(CondBlock[i]);

    ForScope[i]->ForceCleanup();

    LoopStack.pop();
}
  // Emit the Sync block and terminator
  EmitBlock(Sync.getBlock());
  Builder.CreateSync(LoopExit.getBlock(), SRStart);

  // Emit the fall-through block.
  EmitBlock(LoopExit.getBlock(), true);

  // Clean up the ForScope new's
  for (int i=0; i<numIVs; ++i) delete ForScope[i];

  // DWS remove after type change???
  return true;
}


bool CodeGenFunction::EmitKokkosParallelReduce(const CallExpr *CE,
                    ArrayRef<const Attr *> Attrs) {
  DiagnosticsEngine &Diags = CGM.getDiags();
  Diags.Report(CE->getExprLoc(), diag::warn_kokkos_reduce_unsupported);
  return false;
}
