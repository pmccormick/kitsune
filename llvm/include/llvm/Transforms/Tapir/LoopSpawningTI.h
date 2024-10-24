//===- LoopSpawningTI.h - Spawn loop iterations efficiently -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass modifies Tapir loops to spawn their iterations efficiently.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_TAPIR_LOOPSPAWNING_H
#define LLVM_TRANSFORMS_TAPIR_LOOPSPAWNING_H

#include "llvm/IR/PassManager.h"
#include "llvm/Passes/OptimizationLevel.h"

namespace llvm {

/// The LoopSpawning Pass.
struct LoopSpawningPass : public PassInfoMixin<LoopSpawningPass> {
  LoopSpawningPass(OptimizationLevel OptLevel = OptimizationLevel::O2)
    : Level(OptLevel) { }
  
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

  OptimizationLevel Level;
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_TAPIR_LOOPSPAWNING_H
