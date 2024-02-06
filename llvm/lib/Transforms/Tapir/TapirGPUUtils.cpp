//===- TapirGPUUtils.cpp - Lower Tapir to the Kitsune GPU back end --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Kitsune+Tapir HIP ABI to convert Tapir
// instructions to calls into the HIP-centric portions of the Kitsune
// runtime for GPUs to produce a fully compiled (not JIT) executable
// that is suitable for a given architecture target.
//
// NOTE: Several aspects of this transform mimic Clang's code generation
// for HIP. Any significant changes to Clang at that level might require
// changes here as well.
//
//===----------------------------------------------------------------------===//
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SmallVectorMemoryBuffer.h"
#include "llvm/Transforms/Tapir/TapirGPUUtils.h"

using namespace llvm;

namespace tapir {

Constant *getOrInsertFBGlobal(Module &M, StringRef Name, Type *Ty) {
  return M.getOrInsertGlobal(Name, Ty, [&] {
    return new GlobalVariable(M, Ty, true, GlobalValue::InternalLinkage,
                              nullptr, Name, nullptr);
  });
}

Constant *createConstantStr(const std::string &Str, Module &M,
                            const std::string &Name,
                            const std::string &SectionName,
                            unsigned Alignment) {
  LLVMContext &Ctx = M.getContext();
  Constant *CSN = ConstantDataArray::getString(Ctx, Str);
  GlobalVariable *GV = new GlobalVariable(
      M, CSN->getType(), true, GlobalVariable::PrivateLinkage, CSN, Name);
  Type *StrTy = GV->getType();

  const DataLayout &DL = M.getDataLayout();
  Constant *Zeros[] = {ConstantInt::get(DL.getIndexType(StrTy), 0),
                       ConstantInt::get(DL.getIndexType(StrTy), 0)};
  if (!SectionName.empty()) {
    GV->setSection(SectionName);
    // Mark the address as used which make sure that this section isn't
    // merged and we will really have it in the object file.
    GV->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::None);
  }

  if (Alignment)
    GV->setAlignment(llvm::Align(Alignment));

  Constant *CS = ConstantExpr::getGetElementPtr(GV->getValueType(), GV, Zeros);
  return CS;
}

// Adapted from Transforms/Utils/ModuleUtils.cpp
void appendToGlobalCtors(Module &M, Constant *C, int Priority, Constant *Data) {
  IRBuilder<> IRB(M.getContext());
  FunctionType *FnTy = FunctionType::get(IRB.getVoidTy(), false);

  // Get the current set of static global constructors and add
  // the new ctor to the list.
  SmallVector<Constant *, 16> CurrentCtors;
  StructType *EltTy = StructType::get(
      IRB.getInt32Ty(), PointerType::getUnqual(FnTy), IRB.getInt8PtrTy());
  if (GlobalVariable *GVCtor = M.getNamedGlobal("llvm.global_ctors")) {
    if (Constant *Init = GVCtor->getInitializer()) {
      unsigned N = Init->getNumOperands();
      CurrentCtors.reserve(N + 1);
      for (unsigned i = 0; i != N; ++i)
        CurrentCtors.push_back(cast<Constant>(Init->getOperand(i)));
    }
    GVCtor->eraseFromParent();
  }

  // Build a 3 field global_ctor entry.
  // We don't take a comdat key.
  Constant *CSVals[3];
  CSVals[0] = IRB.getInt32(Priority);
  CSVals[1] = C;
  CSVals[2] = Data ? ConstantExpr::getPointerCast(Data, IRB.getInt8PtrTy())
                   : Constant::getNullValue(IRB.getInt8PtrTy());
  Constant *RuntimeCtorInit =
      ConstantStruct::get(EltTy, makeArrayRef(CSVals, EltTy->getNumElements()));

  CurrentCtors.push_back(RuntimeCtorInit);

  // Create a new initializer.
  ArrayType *AT = ArrayType::get(EltTy, CurrentCtors.size());
  Constant *NewInit = ConstantArray::get(AT, CurrentCtors);

  // Create the new global variable and replace all uses of
  // the old global variable with the new one.
  (void)new GlobalVariable(M, NewInit->getType(), false,
                           GlobalValue::AppendingLinkage, NewInit,
                           "llvm.global_ctors");
}

void getKernelInstructionMix(const Function *F, KernelInstMixData &InstMix) {
  InstMix.num_memory_ops = 0;
  InstMix.num_flops = 0;
  InstMix.num_iops = 0;

  std::set<const Function *> CalledFuncs;
  for (auto I = inst_begin(F); I != inst_end(F); I++) {
    if (I->mayReadOrWriteMemory()) {
      InstMix.num_memory_ops++;
    } else if (I->isBinaryOp()) {
      Type *Ty = I->getType();
      if (Ty->isHalfTy() || Ty->isFloatTy() || Ty->isDoubleTy() ||
          Ty->isX86_FP80Ty())
        InstMix.num_flops++;
      else
        InstMix.num_iops++;
    } else if (I->isUnaryOp()) {
      Type *Ty = I->getType();
      if (Ty->isHalfTy() || Ty->isFloatTy() || Ty->isDoubleTy() ||
          Ty->isX86_FP80Ty())
        InstMix.num_flops++;
      else
        InstMix.num_iops++;
    } else {
      if (auto CI = dyn_cast<CallInst>(&*I)) {
        CalledFuncs.insert(CI->getCalledFunction());
      }
    }
  }

  for (auto F : CalledFuncs) {
    KernelInstMixData localInstMix;
    getKernelInstructionMix(F, localInstMix);
    InstMix.num_memory_ops += localInstMix.num_memory_ops;
    InstMix.num_flops += localInstMix.num_flops;
    InstMix.num_iops += localInstMix.num_iops;
  }
}

} // namespace tapir
