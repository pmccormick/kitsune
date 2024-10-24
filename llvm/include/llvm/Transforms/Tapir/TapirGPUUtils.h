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
using namespace llvm;

extern Constant *getOrInsertFBGlobal(Module &M, StringRef Name, Type *Ty);

Constant *createConstantStr(const std::string &Str, Module &M,
                            const std::string &Name = "",
                            const std::string &SectionName = "",
                            unsigned Alignment = 0);

extern void appendToGlobalCtors(llvm::Module &M, llvm::Constant *C,
                                int Priority, llvm::Constant *Data);

// NOTE: This needs to be kept up to date with the structure in
// the kitsune runtime!  We currently have avoided including files
// between the two but perhaps we should...  ????
struct KernelInstMixData {
  uint64_t numMemoryOps;
  uint64_t numFlops;
  uint64_t numIntOps;
  uint64_t numOtherOps;
};

extern void getKernelInstructionMix(const llvm::Function *F,
                                    KernelInstMixData &InstMix);
} // namespace tapir

#endif
