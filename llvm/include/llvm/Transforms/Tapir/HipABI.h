//===- HipABI.h - Tapir to Kitsune runtime HIP target -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
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
#ifndef TapirHip_ABI_H_
#define TapirHip_ABI_H_

#include "llvm/Transforms/Tapir/LoweringUtils.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"
#include "llvm/Support/ToolOutputFile.h"

namespace llvm {

class TargetMachine;
class HipLoop;

typedef std::unique_ptr<ToolOutputFile> HipABIOutputFile;

class HipABI : public TapirTarget {

public:
  HipABI(Module &InputModule);
  ~HipABI();

  // ----- Core Tapir code transform callbacks.

  /// Lower a call to the tapir.loop.grainsize intrinsic into a grain size
  /// (coarsening) value.  For GPU codes we currently limit this to a
  /// value of 1.
  Value *lowerGrainsizeCall(CallInst *GrainsizeCall) override final;

  /// Lower the given Tapir sync instruction (SI).
  void lowerSync(SyncInst &SI) override final;

  /// Process Function F before any function outlining is performed.  This
  /// routine should not modify the CFG structure.
  virtual void preProcessFunction(Function &F, TaskInfo &TI,
                                  bool ProcessingTapirLoops) override;

  // Add attributes to the Function Helper produced from outlining a task.
  void addHelperAttributes(Function &F) override;

  // Pre-process the Function F that has just been outlined from a task.  This
  // routine is executed on each outlined function by traversing in post-order
  // the tasks in the original function.
  void preProcessOutlinedTask(Function &F,
                              Instruction *DetachPt,
                              Instruction *TaskFrameCreate,
                              bool isSpawner,
                              BasicBlock *BB) override
  { /* no-op */ }

  // Post-process the Function F that has just been outlined from a task.  This
  // routine is executed on each outlined function by traversing in post-order
  // the tasks in the original function.
  void postProcessOutlinedTask(Function &F, Instruction *DetachPtr,
                               Instruction *TaskFrameCreate, bool IsSpawner,
                               BasicBlock *TFEntry) override
  { /* no-op */ }

  // Pre-process the root Function F as a function that can spawn subtasks.
  void preProcessRootSpawner(Function &F, BasicBlock *TFEntry) override
  { /* no-op */ }

  // Post-process the root Function F as a function that can spawn subtasks.
  void postProcessRootSpawner(Function &F, BasicBlock *TFEntry) override
  { /* no-op */ }

  // Process the invocation of a task for an outlined function.  This routine is
  // invoked after processSpawner once for each child subtask.
  void processSubTaskCall(TaskOutlineInfo &TOI, DominatorTree &DT) override
  { /* no-op */ }

  // Process Function F at the end of the lowering process.
  void postProcessFunction(Function &F, bool OutliningTapirLoops) override;

  std::unique_ptr<Module>& getLibDeviceModule();

  /// @brief  Add a global var of those that need a host-to-device connection.
  /// @param GV: The global variable to add to the set.
  void pushGlobalVariable(GlobalVariable *GV);

  /// @brief Any global variables to handle for host-to-device code gen?
  /// @return True if there are globals to process, false otherwise.
  bool hasGlobalVariables() const { return !GlobalVars.empty(); }

  int globalVarCount() const { return GlobalVars.size(); }

  void pushSR(Value *SR) { SyncRegList.insert(SR); }

  /// @brief Save a kernel for post-processing.
  /// @param KF - the kernel function to save.
  /// @return void
  void saveKernel(Function *KF) {
    KernelFunctions.push_back(KF);
  }

  void transformConstants(Function *M);

  void transformArguments(Function *Fn);

  // Process the host-side module at the end of lowering all functions //
  // within the module.
  void postProcessModule() override final;

  // Process a generated helper Function F produced via outlining, at the end of
  // the lowering process.
  void postProcessHelper(Function &F) override
  { /* no-op */ }

  // Return the HIP outline processor associated with this target.
  LoopOutlineProcessor *getLoopOutlineProcessor(const TapirLoopInfo *TL)
                                                override final;

  private:
  // ----- Hip-centric transformation support.

  /// @brief Generate a AMDGPU (GCN) object file for the kernel module.
  /// @return The created object file.
  HipABIOutputFile createTargetObj(const StringRef &ObjFileName);

  /// @brief Link the object file created by the AMDGPU backend.
  /// @param ObjFile - the object file created by the AMDGPU target.
  /// @param SOFileName - output file name of linker output.
  /// @return The linked output file.
  HipABIOutputFile linkTargetObj(const HipABIOutputFile &ObjFile,
                                 const StringRef &SOFileName);

  /// @brief Generate a bundle/GCN file for the (kernels) module.
  /// @return The file containing the GCN for the kernel.
  HipABIOutputFile createBundleFile();

  /// @brief  Embed the given bundle file in the generated code.
  /// @param BundleFile: The bundle file.
  /// @return A global variable containing the fat binary.
  GlobalVariable *embedBundle(HipABIOutputFile &BundleFile);

  /// @brief Load the given ROCM-centric bitcode file and return a module.
  /// @param BCFileName: The file name for the bitcode file (not a full path)
  /// @return  A module holding the bitcode.
  std::unique_ptr<Module> loadBCFile(const std::string& BCFileName);

  /// @brief Load (link) the given module into the generated kernel module.
  /// @param M - the module to load/link into the generated kernel module.
  /// @return  True on success, false otherwise.
  bool linkInModule(std::unique_ptr<Module>& M);

  /// @brief Register all the create kernels (device entry points) with HIP runtime.
  /// @param Handle - HIP handle for fat binary.
  /// @param B - the IR builder to use for code gen.
  /// @return void
  void registerKernels(Value *HandlePtr, IRBuilder<> &B);

  /// @brief Establish a host-to-device registration of the global vars.
  /// @param Handle: The GPU-side module (not llvm) that contains the kernels.
  /// @param B: The builder to use for codegen.
  void bindGlobalVariables(Value *Handle, IRBuilder<> &B);

  /// @brief  Take the necessary steps to register the bundle w/ runtime.
  /// @param Bundle -- the HIP bundle (fat binary) global variable.
  void registerBundle(GlobalVariable *Bundle);

  /// @brief  Add global constructor to initialize and register the runtime.
  /// @param Bundle The HIP "bundle" (fat binary).
  /// @param Wrapper The structure wrapper around the fat binary.
  /// @return The created ctor.function.
  Function *createCtor(GlobalVariable *Bundle, GlobalVariable *Wrapper);

  /// @brief Add global destructor to cleanup the runtime details.
  /// @param BundleHandle Handle to the HIP bundle (fat binary).
  /// @return The created dtor function.
  Function *createDtor(GlobalVariable *BundleHandle);

  std::unique_ptr<Module> LibDeviceModule;
  std::string BaseModuleName;


  /// @brief  Make a final pass and 'bind' launch calls to fat binary image.
  /// @param M -- the module containing the launch calls.
  /// @param BundleBin -- the fat binary image that contains the kernels.
  void finalizeLaunchCalls(Module &M, GlobalVariable *BundleBin);

  typedef std::list<GlobalVariable*> GlobalVarListTy;
  GlobalVarListTy GlobalVars;
  typedef std::set<Value *> SyncRegionListTy;
  SyncRegionListTy SyncRegList;
  typedef std::list<Function*> KernelListTy;
  KernelListTy KernelFunctions;

  Module KernelModule;
  bool ROCmModulesLoaded;
  TargetMachine *AMDTargetMachine;

  FunctionCallee   KitHipGetGlobalSymbolFn = nullptr;
  FunctionCallee   KitHipMemcpySymbolToDevFn = nullptr;
  FunctionCallee   KitHipSyncFn = nullptr;
};

/// The loop outline process for transforming a Tapir parallel loop
/// representing into a Hip runtime and PTX --> fat binary kernel
/// execution.
///
///  * The loop processor requires a CUDA install and that the 'ptxas'
///    and 'fatbinary' executables are in the user's path.  While it
///    is tempting to inline direct CUDA calls into the transform this
///    has two drawbacks:
///
///      1. CMake dependencies on Hip would need to be added.
///      2. GPU would be required at compile time (i.e., no cross
///         compilation support).
///
class HipLoop : public LoopOutlineProcessor {
  friend class HipABI;

public:
  /// @brief Build the HipLoop outline processor.
  /// @param M: Module containing the input code.
  /// @param KM: The module that will contain the generated kernel.
  /// @param KernelName: The name of the kernel function that is generated.
  /// @param TT: The "parent" tapir target.
  HipLoop(Module &M, Module &KM, const std::string &KernelName, HipABI *Target);
  ~HipLoop();

  /// Prepares the set HelperArgs of function arguments for the outlined helper
  /// function Helper for a Tapir loop.  Also prepares the list HelperInputs of
  /// input values passed to a call to Helper.  HelperArgs and HelperInputs are
  /// derived from the loop-control arguments LCArgs and loop-control inputs
  /// LCInputs for the Tapir loop, as well the set TLInputsFixed of arguments to
  /// the task underlying the Tapir loop.
  void setupLoopOutlineArgs(Function &F, ValueSet &HelperArgs,
                            SmallVectorImpl<Value *> &HelperInputs,
                            ValueSet &InputSet,
                            const SmallVectorImpl<Value *> &LCArgs,
                            const SmallVectorImpl<Value *> &LCInputs,
                            const ValueSet &TLInputsFixed) override;

  /// Returns an integer identifying the index of the helper-function argument
  /// in Args that specifies the starting iteration number.  This return value
  /// must complement the behavior of setupLoopOutlineArgs().
  unsigned getIVArgIndex(const Function &F,
		         const ValueSet &Args) const override;

  /// Returns an integer identifying the index of the helper-function argument
  /// in Args that specifies the ending iteration number.  This return value
  /// must complement the behavior of setupLoopOutlineArgs().
  unsigned getLimitArgIndex(const Function &F,
                            const ValueSet &Args) const override;

  /// Process the TapirLoop before it is outlined -- just prior to the
  /// outlining occurs.  This allows the VMap and related details to be
  /// customized prior to outlining related operations (e.g. cloning of
  /// LLVM constructs).
  void preProcessTapirLoop(TapirLoopInfo &TL,
                           ValueToValueMapTy &VMap) override;

  /// Processes an outlined Function Helper for a Tapir loop, just after the
  /// function has been outlined.
  void postProcessOutline(TapirLoopInfo &TL, TaskOutlineInfo &Out,
                          ValueToValueMapTy &VMap) override;

  /// Processes a call to an outlined Function Helper for a Tapir loop.
  void processOutlinedLoopCall(TapirLoopInfo &TL, TaskOutlineInfo &TOI,
                               DominatorTree &DT) override;

  std::string getKernelName() const { return KernelName; }
  unsigned getKernelID() const { return KernelID; }

private:
  // ----- Hip-centric loop code generation support.


  Value *emitWorkItemId(IRBuilder<> &Builder, int ItemIndex, int Low, int High);
  Value *emitWorkGroupId(IRBuilder<> &Builder, int ItemIndex);
  Value *emitWorkGroupSize(IRBuilder<> &Builder, int ItemIndex);

  /// @brief  Emit code to access the GCN dispatch pointer for V4 ABI.
  /// @param Builder: The IR builder to use.
  /// @return The dispatch structure pointer.
  Value *emitDispatchPtr(IRBuilder<> &Builder);

  /// @brief  Emit code to access the GCN dispatch pointer for v5 ABI.
  /// @param Builder: The IR builder to use.
  /// @return The dispatch structure pointer.
  Value *emitImplicitArgPtr(IRBuilder<> &Builder);

  /// @brief Resolve a call on the device side.
  /// @param Fn: The function to resolve on the device side.
  /// @return  The new Function for the device side call.
  //Function *resolveDeviceFunction(Function *Fn);

  /// @brief Transform the given Function so it is ready for GCN generation.
  /// @param F The function to transform.
  //void transformForGCN(Function &F);

  HipABI *TTarget = nullptr;
  static unsigned NextKernelID; // Give the generated kernel a unique ID.
  unsigned KernelID;            // Unique ID for this transformed loop.
  std::string KernelName;       // A unique name for the kernel.
  Module &KernelModule;         // PTX module holds the generated kernel(s).

  // AMDGCN intrinsics.  TODO: These should probably not be
  // prefixed with the kitsune runtime...
  FunctionCallee   KitHipWorkItemIdFn;
  FunctionCallee   KitHipWorkItemIdXFn,
                   KitHipWorkItemIdYFn,
                   KitHipWorkItemIdZFn;
  FunctionCallee   KitHipWorkGroupIdFn;
  FunctionCallee   KitHipWorkGroupIdXFn,
                   KitHipWorkGroupIdYFn,
                   KitHipWorkGroupIdZFn;
  FunctionCallee   KitHipBlockDimFn;

  // Kitsune runtime entry points.
  FunctionCallee   KitHipLaunchFn = nullptr;
  FunctionCallee   KitHipModuleLoadDataFn = nullptr;
  FunctionCallee   KitHipModuleLaunchFn = nullptr;

  // Runtime prefetch support entry points.
  FunctionCallee   KitHipStreamSetMemPrefetchFn =  nullptr;  
  FunctionCallee   KitHipMemPrefetchFn =  nullptr;
  FunctionCallee   KitHipMemPrefetchOnStreamFn = nullptr;
  FunctionCallee   KitHipStreamMemPrefetchFn = nullptr;

  FunctionCallee   KitHipCreateFBModuleFn = nullptr;


  SmallVector<Value *, 5> OrderedInputs;
};

}

#endif
