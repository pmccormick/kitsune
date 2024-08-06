find_package(CUDAToolkit)

add_llvm_component_library(LLVMTapirOpts
  CudaABI.cpp
  DRFScopedNoAliasAA.cpp
  HipABI.cpp
  LambdaABI.cpp
  LoopSpawningTI.cpp
  LoopStripMine.cpp
  LoopStripMinePass.cpp
  LoweringUtils.cpp
  OMPTaskABI.cpp
  OpenCilkABI.cpp
  OpenMPABI.cpp
  Outline.cpp
  QthreadsABI.cpp
  RealmABI.cpp
  GPUABI.cpp
  SerialABI.cpp
  SerializeSmallTasks.cpp
  Tapir.cpp
  TapirGPUUtils.cpp
  TapirToTarget.cpp
  TapirLoopInfo.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms/Tapir

  DEPENDS
  intrinsics_gen
  LLVMLinker
  LLVMBitWriter

  COMPONENT_NAME
  TapirOpts

  LINK_COMPONENTS
  Analysis
  Core
  IRReader
  Linker
  MC
  Scalar
  Support
  TransformUtils
  Vectorize
  Passes
  )

if (CUDAToolkit_FOUND)
  message(STATUS "Tapir ABI has CUDA version info: ${CUDAToolkit_VERSION}")
  add_compile_definitions(
    CUDATOOLKIT_VERSION=${CUDAToolkit_VERSION}
    CUDATOOLKIT_VERSION_MAJOR=${CUDAToolkit_VERSION_MAJOR}
    CUDATOOLKIT_VERSION_MINOR=${CUDAToolkit_VERSION_MINOR}
  )
  set(TAPIR_CUDAABI_DEFAULT_TARGET "sm_80" CACHE STRING
	 "The default CUDA device target for Tapir-transformed code")
  add_compile_definitions(
    _CUDAABI_DEFAULT_TARGET_ARCH="${TAPIR_CUDAABI_DEFAULT_TARGET}"
  )
endif()
