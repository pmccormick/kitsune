//===- OpenCLABI.cpp - Spawn parallel loops efficiently using opencl  -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Modify Tapir loops to spawn their iterations efficiently.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Tapir/TapirLoopInfo.h"
#include "llvm/Transforms/Tapir/LoopSpawningTI.h"
#include "llvm/Transforms/Tapir/OpenCLABI.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/IR/IntrinsicInst.h"
#include <LLVMSPIRVLib/LLVMSPIRVLib.h>
#include <sstream>

using namespace llvm;

void RuntimeCilkFor::processOutlinedLoopCall(TapirLoopInfo &TL,
                                             TaskOutlineInfo &TOI,
                                             DominatorTree &DT) {
  Function *Outlined = TOI.Outline;
  Instruction *ReplStart = TOI.ReplStart;
  Instruction *ReplCall = TOI.ReplCall;
  CallSite CS(ReplCall);
  BasicBlock *CallCont = TOI.ReplRet;
  BasicBlock *UnwindDest = TOI.ReplUnwind;
  Function *Parent = ReplCall->getFunction();
  Module &M = *Parent->getParent();
  unsigned IVArgIndex = getIVArgIndex(*Parent, TOI.InputSet);
  Type *PrimaryIVTy =
      CS.getArgOperand(IVArgIndex)->getType();
  Value *TripCount = CS.getArgOperand(IVArgIndex + 1);
  Value *GrainsizeVal = CS.getArgOperand(IVArgIndex + 2);

  // TODO: calls into external functions
  // TODO: captured values
  
  // Types
  LLVMContext &Ctx = M.getContext();
  Type *VoidTy = Type::getVoidTy(M.getContext());
  Type *Int8Ty = Type::getInt8Ty(Ctx);
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);
  Type *VoidPtrTy = Type::getInt8PtrTy(Ctx);
  IRBuilder<> B(Out.ReplCall); 

  // spirv function
  Function *OL = Out.Outline; 
  Function *CLF = Function::Create(OL->getFunctionType(), OL->getLinkage(), OL->getName()); 
  CLM.getFunctionList().push_back(CLF);
  CLM.getOrInsertFunction(CLF->getName(), CLF->getFunctionType(),
                          CLF->getAttributes());
  

void OpenCL::postProcessOutline(TapirLoopInfo &TL, TaskOutlineInfo &Out,
                                 ValueToValueMapTy &VMap) {
  
  // spirv module
  Module CLM(M.getModuleIdentifier() + "_SPIRV", M.getContext()); 
  CLM.setTargetTriple("spir64-unknown-unknown");

  // spirv function
  Function *OL = Out.Outline; 
  Function *CLF = Function::Create(OL->getFunctionType(), OL->getLinkage(), OL->getName()); 
  CLM.getFunctionList().push_back(CLF);
  CLM.getOrInsertFunction(CLF->getName(), CLF->getFunctionType(),
                          CLF->getAttributes());
  SmallVector<ReturnInst*, 8> Returns;
  ValueToValueMapTy spirv_vmap;
  auto spirvArg = CLF->arg_begin();
  for(auto &arg : OL->args()) 
    spirv_vmap[&arg] = &*spirvArg++;
  
  CloneFunctionInto(CLF, OL, spirv_vmap, false, Returns); 
  for(BasicBlock &BB : *CLF){
    for(Instruction &I : BB){
      IntrinsicInst *II = dyn_cast<IntrinsicInst>(&I);
      if(II){
        if(II->getIntrinsicID() == Intrinsic::syncregion_start){
          II->eraseFromParent(); 
          break; 
        }
      }
    }
  }

  // remove old outlined loop
  // OL->eraseFromParent();  // causes issues with later hint removal pass

  // generate spirv kernel code
  std::ostringstream str; 
  std::string ErrMsg; 
  bool success = writeSpirv(&CLM, str, ErrMsg); 
  if(!success){
    std::cerr << "Failed to compile to spirv: " << ErrMsg << std::endl; 
    exit(1); 
  }
  auto s = str.str(); 
  Constant *SPIRV = ConstantDataArray::getRaw(s, s.length(), Int8Ty);
  SPIRVKernel = new GlobalVariable(M, SPIRV->getType(), true,
                                 GlobalValue::PrivateLinkage, SPIRV,
                                 "spirv_" + Twine(OL->getName()));
  Value *SPIRVPtr = B.CreateBitCast(SPIRVKernel, VoidPtrTy);
  Constant *kernelSize = ConstantInt::get(Int32Ty, 
    SPIRVKernel->getInitializer()->getType()->getArrayNumElements()); 

  // insert runtime call
  auto KitsuneOpenCLCall = M.getOrInsertFunction("_kitsune_cl_call", VoidTy, VoidPtrTy, Int32Ty, Int64Ty);
  auto NewReplCall = B.CreateCall(KitsuneOpenCLCall, { SPIRVPtr, kernelSize, TL.getTripCount() }); 
  Loop *L = TL.getLoop(); 
  //BasicBlock *Entry = L->getLoopPreheader();
  //BasicBlock *Exit = TL.getExitBlock();
  //ReplaceInstWithInst(Entry->getTerminator(), BranchInst::Create(Exit));
  Out.replaceReplCall(NewReplCall); 
}

