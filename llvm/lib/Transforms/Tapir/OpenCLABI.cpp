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
#include <LLVMSPIRVLib/LLVMSPIRVLib.h>
#include <sstream>

using namespace llvm;

/// The opencl loop-outline processor leaves the outlined Tapir loop as is.
  void OCLSpawning::postProcessOutline(TapirLoopInfo &TL, TaskOutlineInfo &Out,
                          ValueToValueMapTy &VMap) {
    LoopOutlineProcessor::postProcessOutline(TL, Out, VMap);
    Module CLM(M.getModuleIdentifier() + "_CL", M.getContext()); 
    Function *OL = Out.Outline; 
    Function *CLF = Function::Create(OL->getFunctionType(), OL->getLinkage(), OL->getName()); 
    CLM.getFunctionList().push_back(CLF);
    CLM.getOrInsertFunction(CLF->getName(), CLF->getFunctionType(),
                          CLF->getAttributes());
    SmallVector<ReturnInst*, 8> Returns;
    CloneFunctionInto(OL, CLF, VMap, /*ModuleLevelChanges=*/false, Returns);
    Out.Outline = CLF; 
    
    std::ostringstream str; 
    std::string ErrMsg; 
    bool Fail = writeSpirv(&CLM, str, ErrMsg); 
    assert(!Fail && "Failed to emit spirv");

    // Create a global string to hold the PTX code
    Constant *SPIRV = ConstantDataArray::getString(M.getContext(), str.str());
    GlobalVariable(M, SPIRV->getType(), true,
                                   GlobalValue::PrivateLinkage, SPIRV,
                                   "spirv_" + Twine(OL->getName()));
}

