//===- CudaABI.h - Tapir to the Kitsune runtime CUDA target -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
//
// Copyright (c) 2021, 2023 Los Alamos National Security, LLC.
//  All rights reserved.
//
// Copyright 2021, 2023. Los Alamos National Security, LLC. This
//  software was produced under U.S. Government contract
//  DE-AC52-06NA25396 for Los Alamos National Laboratory (LANL), which
//  is operated by Los Alamos National Security, LLC for the
//  U.S. Department of Energy. The U.S. Government has rights to use,
//  reproduce, and distribute this software.  NEITHER THE GOVERNMENT
//  NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS
//  OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.
//  If software is modified to produce derivative works, such modified
//  software should be clearly marked, so as not to confuse it with
//  the version available from LANL.
//
//  Additionally, redistribution and use in source and binary forms,
//  with or without modification, are permitted provided that the
//  following conditions are met:
//
// Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above
//      copyright notice, this list of conditions and the following
//      disclaimer in the documentation and/or other materials provided
//      with the distribution.
//
//    * Neither the name of Los Alamos National Security, LLC, Los
//      Alamos National Laboratory, LANL, the U.S. Government, nor the
//      names of its contributors may be used to endorse or promote
//      products derived from this software without specific prior
//      written permission.
//
//  THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
//  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
//  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
//  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//  DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR
//  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
//  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
//  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
//  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
//  AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
//  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
//  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
//
//===---------------------------------------------------------------------===//
//
#ifndef TapirCuda_ABI_H_
#define TapirCuda_ABI_H_

#include "llvm/Transforms/Tapir/LoweringUtils.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"
#include "llvm/Support/ToolOutputFile.h"

namespace llvm {

class TargetMachine;
class CudaLoop;

typedef std::unique_ptr<ToolOutputFile> CudaABIOutputFile;

class CudaABI : public TapirTarget {

public:
  CudaABI(Module &M);
  ~CudaABI();

  Value *lowerGrainsizeCall(CallInst *GrainsizeCall) override final;
  void lowerSync(SyncInst &SI) override final;

  void addHelperAttributes(Function &F) override final;
  void preProcessFunction(Function &F, TaskInfo &TI,
                          bool OutliningTapirLoops) override final;
  void postProcessFunction(Function &F, bool OutliningTapirLoops) override final;
  void postProcessHelper(Function &F) override final;

  void preProcessOutlinedTask(Function &F,
                              Instruction *DetachPt,
                              Instruction *TaskFrameCreate,
                              bool IsSpawner,
                              BasicBlock *TFEntry) override final;

  void postProcessOutlinedTask(Function &F,
                               Instruction *DetachPt,
                               Instruction *TaskFrameCreate,
                               bool IsSpawner,
                               BasicBlock *TFEntry) override final;

  void preProcessRootSpawner(Function &F,
                             BasicBlock *TFEntry) override final;
  void postProcessRootSpawner(Function &F,
                              BasicBlock *TFEntry) override final;

  void processSubTaskCall(TaskOutlineInfo &TOI,
                          DominatorTree &DT) override final;

  void postProcessModule() override final;

  LoopOutlineProcessor *getLoopOutlineProcessor(const TapirLoopInfo *TL)
                          override final;

  void pushPTXFilename(const std::string &PTXFilename);

  std::unique_ptr<Module>& getLibDeviceModule();

  void pushGlobalVariable(GlobalVariable *GV);
  bool hasGlobalVariables() const {
    return !GlobalVars.empty();
  }
  int globalVarCount() const {
    return GlobalVars.size();
  }
  void pushSR(Value *SR) {
    SyncRegList.insert(SR);
  }

  private:
    CudaABIOutputFile generatePTX();
    CudaABIOutputFile assemblePTXFile(CudaABIOutputFile &PTXFile);
    CudaABIOutputFile createFatbinaryFile(CudaABIOutputFile &AsmFile);
    GlobalVariable *embedFatbinary(CudaABIOutputFile &FatbinaryFile);
    void registerFatbinary(GlobalVariable *RawFatbinary);
    void finalizeLaunchCalls(Module &M, GlobalVariable *Fatbin);
    void bindGlobalVariables(Value *CM, IRBuilder<> &B);
    Function *createCtor(GlobalVariable *Fatbinary, GlobalVariable *Wrapper);
    Function *createDtor(GlobalVariable *FBHandle);

    std::unique_ptr<Module> LibDeviceModule;

    typedef std::list<std::string> StringListTy;
    StringListTy ModulePTXFileList;
    typedef std::list<GlobalVariable *> GlobalVarListTy;
    GlobalVarListTy GlobalVars;

    typedef std::set<Value*> SyncRegionListTy;
    SyncRegionListTy SyncRegList;

    Module   KernelModule;
    TargetMachine *PTXTargetMachine;
};

/// The loop outline process for transforming a Tapir parallel loop
/// representation into a Cuda runtime and PTX --> fat binary kernel
/// execution.
///
///  * The loop processor requires a CUDA install and that the 'ptxas'
///    and 'fatbinary' executables are in the user's path.  While it
///    is tempting to inline direct CUDA calls into the transform this
///    has two drawbacks:
///
///      1. CMake dependencies on Cuda would need to be added.
///      2. GPU would be required at compile time (i.e., no cross
///         compilation support).
///
class CudaLoop : public LoopOutlineProcessor {
  friend class CudaABI;

private:
  CudaABI *TTarget = nullptr;
  static unsigned NextKernelID;    // Give the generated kernel a unique ID.
  unsigned KernelID;               // Unique ID for this transformed loop.
  std::string KernelName;          // A unique name for the kernel.
  Module  &KernelModule;           // PTX module holds the generated kernel(s).

  // Cuda/PTX thread index access.
  Function *CUThreadIdxX  = nullptr,
           *CUThreadIdxY  = nullptr,
           *CUThreadIdxZ  = nullptr;
  // Cuda/PTX block index and dimensions access.
  Function *CUBlockIdxX   = nullptr,
           *CUBlockIdxY   = nullptr,
           *CUBlockIdxZ   = nullptr;
  Function *CUBlockDimX   = nullptr,
           *CUBlockDimY   = nullptr,
           *CUBlockDimZ   = nullptr;
  // Cuda/PTX grid dimensions access.
  Function *CUGridDimX    = nullptr,
           *CUGridDimY    = nullptr,
           *CUGridDimZ    = nullptr;

  // Cuda thread synchronize
  Function *CUSyncThreads = nullptr;

  StructType *KernelInstMixTy;

  FunctionCallee KitCudaLaunchFn = nullptr;
  FunctionCallee KitCudaSyncFn = nullptr;

  // Runtime prefetch support entry points.
  FunctionCallee KitCudaMemPrefetchFn = nullptr;
  FunctionCallee KitCudaMemPrefetchOnStreamFn = nullptr;
  FunctionCallee KitCudaStreamMemPrefetchFn = nullptr;
  FunctionCallee KitCudaStreamSetMemPrefetchFn = nullptr;

  FunctionCallee KitCudaCreateFBModuleFn = nullptr;
  FunctionCallee KitCudaGetGlobalSymbolFn = nullptr;
  FunctionCallee KitCudaMemcpySymbolToDeviceFn = nullptr;
  SmallVector<Value *, 5> OrderedInputs;

public:
  CudaLoop(Module &M,   // Input module (host side)
           Module &KM,  // Target module for CUDA code
           const std::string &KernelName, // CUDA kernel name
           CudaABI *TT, // Target
           bool MakeUniqueName = true);
  ~CudaLoop();

  void setupLoopOutlineArgs(Function &F, ValueSet &HelperArgs,
                            SmallVectorImpl<Value *> &HelperInputs,
                            ValueSet &InputSet,
                            const SmallVectorImpl<Value *> &LCArgs,
                            const SmallVectorImpl<Value *> &LCInputs,
                            const ValueSet &TLInputsFixed) override final;

  unsigned getIVArgIndex(const Function &F, const ValueSet &Args)
                         const override final;

  unsigned getLimitArgIndex(const Function &F, const ValueSet &Args)
                            const override final;

  std::string getKernelName() const { return KernelName; }

  unsigned getKernelID() const {
    return KernelID;
  }

  void preProcessTapirLoop(TapirLoopInfo &TL,
                           ValueToValueMapTy &VMap) override;
  void postProcessOutline(TapirLoopInfo &TL, TaskOutlineInfo & Out,
                          ValueToValueMapTy &VMap) override final;
  void processOutlinedLoopCall(TapirLoopInfo &TL, TaskOutlineInfo & TOI,
                               DominatorTree &DT) override final;
  void transformForPTX(Function &F);

  Function *resolveLibDeviceFunction(Function *F, bool enableFastMode);
};

}

#endif
