//
//===- TapirGPUUtils.h - Helpers for GPU targets ---------------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//
#ifndef TapirGPUUtils_H_
#define TapirGPUUtils_H_

#include "llvm/IR/Constant.h"
#include "llvm/IR/Module.h"

namespace tapir {
  extern void appendToGlobalCtors(llvm::Module &M,
                                  llvm::Constant *C,
                                  int Priority,
                                  llvm::Constant *Data);
}

#endif

