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

  void finalizeLaunchCalls(Module &M, GlobalVariable *BundleBin);
  HipABIOutputFile createGCNFile();
  Value *lowerGrainsizeCall(CallInst *GrainsizeCall) override final;
  void lowerSync(SyncInst &SI) override final;
  GlobalVariable *embedBundle(HipABIOutputFile &BundleFile);
  void bindGlobalVariables(Value *Handle, IRBuilder<> &B);
  Function *createCtor(GlobalVariable *Bundle, GlobalVariable *Wrapper);
  Function *createDtor(GlobalVariable *BundleHandle);
  void registerBundle(GlobalVariable *Bundle);
  std::unique_ptr<Module> &getLibDeviceModule();
  void pushGlobalVariable(GlobalVariable *GV);
  bool hasGlobalVariables() const { return !GlobalVars.empty(); }
  int globalVarCount() const { return GlobalVars.size(); }

  void postProcessModule() override final;
  LoopOutlineProcessor *getLoopOutlineProcessor(const TapirLoopInfo *TL)
                        override final;
  Value *lowerGrainsizeCall(CallInst *GrainsizeCall);
  void lowerSync(SyncInst &SI);
  void addHelperAttributes(Function &F);
  void preProcessFunction(Function &F, TaskInfo &TI,
                          bool OutliningTapirLoops);
  void postProcessFunction(Function &F, bool OutliningTapirLoops);
  void postProcessHelper(Function &F);
  void preProcessOutlinedTask(Function &F, Instruction *I,
                              Instruction* I, bool, BasicBlock *BB);
  void postProcessOutlinedTask(Function &F, Instruction *DetachPtr,
                               Instruction *TaskFrameCreate,
                               bool IsSpawner, BasicBlock *TFEntry);
  void preProcessRootSpawner(Function &F, BasicBlock *TFEntry);
  void postProcessRootSpawner(Function &F, BasicBlock *TFEntry);
  void processSubTaskCall(TaskOutlineInfo &TOI, DominatorTree &DT);

  private:
    HipABIOutputFile generatePTX();
    HipABIOutputFile assemblePTXFile(HipABIOutputFile &PTXFile);
    HipABIOutputFile createFatbinaryFile(HipABIOutputFile &AsmFile);
    GlobalVariable *embedFatbinary(HipABIOutputFile &FatbinaryFile);
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

    Module   KernelModule;
    TargetMachine *AMDTargetMachine;
};

/// The loop outline process for transforming a Tapir parallel loop
/// represention into a Hip runtime and PTX --> fat binary kernel
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

private:
  HipABI *TTarget = nullptr;
  static unsigned NextKernelID;    // Give the generated kernel a unique ID.
  unsigned KernelID;               // Unique ID for this transformed loop.
  std::string KernelName;          // A unique name for the kernel.
  Module  &KernelModule;           // PTX module holds the generated kernel(s).

  FunctionCallee GetThreadIdx = nullptr;

  // Hip/PTX thread index access.
  Function *CUThreadIdxX  = nullptr,
           *CUThreadIdxY  = nullptr,
           *CUThreadIdxZ  = nullptr;
  // Hip/PTX block index and dimensions access.
  Function *CUBlockIdxX   = nullptr,
           *CUBlockIdxY   = nullptr,
           *CUBlockIdxZ   = nullptr;
  Function *CUBlockDimX   = nullptr,
           *CUBlockDimY   = nullptr,
           *CUBlockDimZ   = nullptr;
  // Hip/PTX grid dimensions access.
  Function *CUGridDimX    = nullptr,
           *CUGridDimY    = nullptr,
           *CUGridDimZ    = nullptr;

  // Hip thread synchronize
  Function *CUSyncThreads = nullptr;

  FunctionCallee KitHipLaunchFn = nullptr;
  FunctionCallee KitHipLaunchModuleFn = nullptr;
  FunctionCallee KitHipWaitFn   = nullptr;
  FunctionCallee KitHipMemPrefetchFn = nullptr;
  FunctionCallee KitHipCreateFBModuleFn = nullptr;
  FunctionCallee KitHipGetGlobalSymbolFn = nullptr;
  FunctionCallee KitHipMemcpySymbolToDeviceFn = nullptr;
  SmallVector<Value *, 5> OrderedInputs;

public:
  HipLoop(Module &M,   // Input module (host side)
           Module &KM,  // Target module for CUDA code
           const std::string &KernelName, // CUDA kernel name
           HipABI *TT, // Target
           bool MakeUniqueName = true);
  ~HipLoop();

  Value *emitDispatchPtr(IRBuilder<> &Builder);
  Value *emitWorkGroupSize(IRBuilder<> &Builder, unsigned Index);
  Value *emitGridSize(IRBuilder<> &Builder, unsigned Index);

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
  void transformForGCN(Function &F);

  Function *resolveLibDeviceFunction(Function *F);
};

}

#endif
