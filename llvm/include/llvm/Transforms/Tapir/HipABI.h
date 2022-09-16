//
//===- HipABI.h - Interface to the Kitsune Hip back end -----*- C++ -*--===//
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
  void lowerSync(SyncInst &SI) override final { }

  /// Process Function F before any function outlining is performed.  This
  /// routine should not modify the CFG structure.
  virtual void preProcessFunction(Function &F, TaskInfo &TI,
                                  bool ProcessingTapirLoops) { }

  // Add attributes to the Function Helper produced from outlining a task.
  void addHelperAttributes(Function &F) { /* no-op */ }

  // Pre-process the Function F that has just been outlined from a task.  This
  // routine is executed on each outlined function by traversing in post-order
  // the tasks in the original function.
  void preProcessOutlinedTask(Function &F,
                              Instruction *I,
                              bool isSpawner,
                              BasicBlock *BB)
  { /* no-op */ }

  // Post-process the Function F that has just been outlined from a task.  This
  // routine is executed on each outlined function by traversing in post-order
  // the tasks in the original function.
  void postProcessOutlinedTask(Function &F, Instruction *DetachPtr,
                               Instruction *TaskFrameCreate, bool IsSpawner,
                               BasicBlock *TFEntry)
  { /* no-op */ }

  // Pre-process the root Function F as a function that can spawn subtasks.
  void preProcessRootSpawner(Function &F, BasicBlock *TFEntry)
  { /* no-op */ }

  // Post-process the root Function F as a function that can spawn subtasks.
  void postProcessRootSpawner(Function &F, BasicBlock *TFEntry)
  { /* no-op */ }

  // Process the invocation of a task for an outlined function.  This routine is
  // invoked after processSpawner once for each child subtask.
  void processSubTaskCall(TaskOutlineInfo &TOI, DominatorTree &DT)
  { /* no-op */ }

  // Process Function F at the end of the lowering process.
  void postProcessFunction(Function &F, bool OutliningTapirLoops)
  { /* no-op */ }

  /// @brief  Add a global var of those that need a host-to-device connection.
  /// @param GV: The global variable to add to the set.
  void pushGlobalVariable(GlobalVariable *GV);

  /// @brief Any global variables to handle for host-to-device code gen?
  /// @return True if there are globals to process, false otherwise.
  bool hasGlobalVariables() const { return !GlobalVars.empty(); }

  // Process the host-side module at the end of lowering all functions //
  // within the module.
  void postProcessModule() override final;

  // Process a generated helper Function F produced via outlining, at the end of
  // the lowering process.
  void postProcessHelper(Function &F) { /* no-op */ }

  // Return the HIP outline processor associated with this target.
  LoopOutlineProcessor *getLoopOutlineProcessor(const TapirLoopInfo *TL)
                                                override final;

  private:
  // ----- Hip-centric transformation support.

  /// @brief Generate a bundle/GCN file for the (kernels) module.
  /// @return The file containing the GCN for the kernel.
  HipABIOutputFile createBundleFile();

  /// @brief  Embed the given bundle file in the generated code.
  /// @param BundleFile: The bundle file.
  /// @return A global variable containing the fat binary.
  GlobalVariable *embedBundle(HipABIOutputFile &BundleFile);



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

  /// @brief  Make a final pass and 'bind' launch calls to fat binary image.
  /// @param M -- the module containing the launch calls.
  /// @param BundleBin -- the fat binary image that contains the kernels.
  void finalizeLaunchCalls(Module &M, GlobalVariable *BundleBin);

  //
  //std::unique_ptr<Module> &getLibDeviceModule();
  //std::unique_ptr<Module> LibDeviceModule;
  typedef std::list<GlobalVariable *> GlobalVarListTy;
  GlobalVarListTy GlobalVars;
  Module KernelModule;
  TargetMachine *AMDTargetMachine;
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
                            const ValueSet &TLInputsFixed);

  /// Returns an integer identifying the index of the helper-function argument
  /// in Args that specifies the starting iteration number.  This return value
  /// must complement the behavior of setupLoopOutlineArgs().
  unsigned getIVArgIndex(const Function &F, const ValueSet &Args) const;

  /// Returns an integer identifying the index of the helper-function argument
  /// in Args that specifies the ending iteration number.  This return value
  /// must complement the behavior of setupLoopOutlineArgs().
  unsigned getLimitArgIndex(const Function &F,
                            const ValueSet &Args) const;

  /// Process the TapirLoop before it is outlined -- just prior to the
  /// outlining occurs.  This allows the VMap and related details to be
  /// customized prior to outlining related operations (e.g. cloning of
  /// LLVM constructs).
  void preProcessTapirLoop(TapirLoopInfo &TL, ValueToValueMapTy &VMap);

  /// Processes an outlined Function Helper for a Tapir loop, just after the
  /// function has been outlined.
  void postProcessOutline(TapirLoopInfo &TL, TaskOutlineInfo &Out,
                          ValueToValueMapTy &VMap);

  /// Processes a call to an outlined Function Helper for a Tapir loop.
  void processOutlinedLoopCall(TapirLoopInfo &TL, TaskOutlineInfo &TOI,
                               DominatorTree &DT);

  std::string getKernelName() const { return KernelName; }
  unsigned getKernelID() const { return KernelID; }

private:
  // ----- Hip-centric loop code generation support.

  /// @brief  Emit code to access the GCN dispatch pointer.
  /// @param Builder: The IR builder to use.
  /// @return The dispatch structure pointer.
  Value *emitDispatchPtr(IRBuilder<> &Builder);

  /// @brief  Emit code to get the HIP workgroup size.
  /// @param Builder: The IR builder to use.
  /// @param Index: The workgroup dimension to return the size of.
  /// @return The LLVM value corresponding to the given workgroup size.
  Value *emitWorkGroupSize(IRBuilder<> &Builder, unsigned Index);

  /// @brief  Emit code to get the HIP grid size.
  /// @param Builder: The IR builder to use.
  /// @param Index: The grid dimension to return the size of.
  /// @return The LLVM value corresponding to the size of given grid dimension.
  Value *emitGridSize(IRBuilder<> &Builder, unsigned Index);

  /// @brief Resolve a call on the device side.
  /// @param Fn: The function to resolve on the device side.
  /// @return  The new Function for the device side call.
  Function *resolveDeviceFunction(Function *Fn);

  /// @brief Transform the given Function so it is ready for GCN generation.
  /// @param F The function to transform.
  void transformForGCN(Function &F);

  HipABI *TTarget = nullptr;
  static unsigned NextKernelID; // Give the generated kernel a unique ID.
  unsigned KernelID;            // Unique ID for this transformed loop.
  std::string KernelName;       // A unique name for the kernel.
  Module &KernelModule;         // PTX module holds the generated kernel(s).

  // AMDGCN intrinsics.  TODO: These should probably not be
  // prefixed with the kitsune runtime...
  FunctionCallee   KitHipWorkItemIdXFn,
                   KitHipWorkItemIdYFn,
                   KitHipWorkItemIdZFn;
  FunctionCallee   KitHipWorkGroupIdXFn,
                   KitHipWorkGroupIdYFn,
                   KitHipWorkGroupIdZFn;

  // Kitsune runtime entry points.
  FunctionCallee   KitHipLaunchFn;
  FunctionCallee   KitHipModuleLaunchFn;
  FunctionCallee   KitHipWaitFn;
  FunctionCallee   KitHipMemPrefetchFn;
  FunctionCallee   KitHipCreateFBModuleFn;
  FunctionCallee   KitHipGetGlobalSymbolFn;
  FunctionCallee   KitHipMemcpySymbolToDevFn;

  SmallVector<Value *, 5> OrderedInputs;
};

}

#endif
