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

#include "llvm/Transforms/Tapir/LoopSpawningTI.h"
#include "llvm/Transforms/Tapir/OpenCLABI.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/IR/IntrinsicInst.h"
#include <LLVMSPIRVLib/LLVMSPIRVLib.h>
#include <sstream>

using namespace llvm;

/// The opencl loop-outline processor leaves the outlined Tapir loop as is.
void OCLSpawning::postProcessOutline(TapirLoopInfo &TL, TaskOutlineInfo &Out,
                        ValueToValueMapTy &VMap) {
  LoopOutlineProcessor::postProcessOutline(TL, Out, VMap);
  Module CLM(M.getModuleIdentifier() + "_CL", M.getContext()); 

  CLM.setTargetTriple("spir64-unknown-unknown");

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
  // TODO: Handle all calls into functions

  
  std::ostringstream str; 
  std::string ErrMsg; 
  bool success = writeSpirv(&CLM, str, ErrMsg); 
  if(!success){
    std::cerr << "Failed to compile to spirv: " << ErrMsg << std::endl; 
    exit(1); 
  }

  // Create a global string to hold the PTX code
  Constant *SPIRV = ConstantDataArray::getString(M.getContext(), str.str());
  SPIRVKernel = new GlobalVariable(M, SPIRV->getType(), true,
                                 GlobalValue::PrivateLinkage, SPIRV,
                                 "spirv_" + Twine(OL->getName()));
}

