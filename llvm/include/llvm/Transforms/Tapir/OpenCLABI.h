#include "llvm/Transforms/Tapir/LoopSpawningTI.h"
#include "llvm/Transforms/Tapir/LoweringUtils.h"
#include "llvm/ADT/DenseMap.h"

using namespace llvm; 

class OCLSpawning : public LoopOutlineProcessor {
public:
  OCLSpawning(Module &M) : LoopOutlineProcessor(M) {}
  void postProcessOutline(TapirLoopInfo &TL, TaskOutlineInfo &Out,
                          ValueToValueMapTy &VMap) override final;  
};
