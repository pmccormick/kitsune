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

struct KernelInstMixData {
  uint64_t num_memory_ops;
  uint64_t num_flops;
  uint64_t num_iops;
};

extern void getKernelInstructionMix(const llvm::Function *F,
                                    KernelInstMixData &InstMix);
} // namespace tapir

#endif
