# Copyright (c) 2021 Triad National Security, LLC
#                         All rights reserved.
#
# This file is part of the kitsune/llvm project.  It is released under
# the LLVM license.
#
#
find_package(CUDAToolkit)

if (CUDAToolkit_FOUND)
  message(STATUS "Kitsune: LLVM-GPU ABI -- enabling CUDA Toolkit support.")
  set(KITSUNE_ENABLE_CUDA_ABI TRUE)
else()
  message(STATUS "Kitsune: LLVM-GPU ABI -- the CUDA Toolkit was not found, disabling support...")
  set(KITSUNE_ENABLE_CUDA_ABI FALSE)
endif()


# TODO: For now we'll disable HIP and SPIRV support until we can
# sort out the best cmake approach to take.
set(KITSUNE_ENABLE_HIP_ABI FALSE)
set(KITSUNE_ENABLE_SPIRV_ABI FALSE)


if (NOT KITSUNE_ENABLE_CUDA_ABI AND
    NOT KITSUNE_ENABLE_HIP_ABI AND
    NOT KITSUNE_ENABLE_SPIRV_ABI)
  message(FATAL_ERROR "Kitsune: The LLVM-GPU ABI was enabled but no supporting runtimes were found!")
endif()

set(KITSUNE_LLVM_GPU_ABI_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(KITSUNE_LLVM_GPU_ABI_INCLUDE_DIR ${KITSUNE_LLVM_GPU_ABI_DIR})

include_directories(${KITSUNE_SOURCE_DIR}/include)
include_directories(${KITSUNE_LLVM_GPU_ABI_INCLUDE_DIR})

set(KITSUNE_LLVM_GPU_ABI_SRCS
  kitrt-debug.cpp
  llvm-gpu.cpp)

set(KITSUNE_LLVM_GPU_ABI_HDRS
  kitrt-debug.h
  llvm-gpu.h)

if (KITSUNE_ENABLE_CUDA_ABI)
  message(STATUS "Kitsune: LLVM-GPU ABI enabling CUDA support.")
  list(APPEND KITSUNE_LLVM_GPU_ABI_SRCS
    cuda.cpp
    llvm-cuda.cpp)
  list(APPEND KITSUNE_LLVM_GPU_ABI_HDRS
    cuda.h
    llvm-cuda.h)
  include_directories(${CUDAToolkit_INCLUDE_DIRS})
  link_directories(${CUDAToolkit_LIBRARY_DIR})
  set(KITSUNE_LLVM_GPU_LINK_LIBS "-lnvptxcompiler_static -lcudart")
else()
  message(STATUS "Kitsune: LLVM-GPU ABI no CUDA support.")
  list(APPEND KITSUNE_LLVM_GPU_ABI_SRCS
    nocuda.cpp)
endif()


if (KITSUNE_ENABLE_HIP_ABI)
  message(STATUS "Kitsune: LLVM-GPU ABI enabling HIP support.")
  list(APPEND KITSUNE_LLVM_GPU_ABI_SRCS
    llvm-hip.cpp)
  list(APPEND KITSUNE_LLVM_GPU_ABI_HDRS
    llvm-hip.h)
else()
  message(STATUS "Kitsune: LLVM-GPU ABI no HIP support.")
  list(APPEND KITSUNE_LLVM_GPU_ABI_SRCS
    nohip.cpp)
endif()


if (KITSUNE_ENABLE_SPIRV_ABI)
  message(STATUS "Kitsune: LLVM-GPU ABI enabling SPIRV support.")
  list(APPEND KITSUNE_LLVM_GPU_ABI_SRCS
    llvm-spirv.cpp)
  list(APPEND KITSUNE_LLVM_GPU_ABI_HDRS
    llvm-spirv.h)
else()
  message(STATUS "Kitsune: LLVM-GPU ABI no SPIRV support.")
    list(APPEND KITSUNE_LLVM_GPU_ABI_SRCS
    nospirv.cpp)
endif()


set(KITSUNE_LLVM_LIBRARY_PATH ${LLVM_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX})
set(KITSUNE_LLVM_DYLIB ${KITSUNE_LLVM_LIBRARY_PATH}/libLLVM-${LLVM_VERSION_MAJOR}.so)

add_library(llvm-gpu-abi SHARED ${KITSUNE_LLVM_GPU_ABI_SRCS})
target_include_directories(llvm-gpu-abi BEFORE
  PRIVATE
  ${LLVM_MAIN_INCLUDE_DIR}
  ${LLVM_INCLUDE_DIR}
  ${KITSUNE_SOURCE_DIR}/include
  ${KITSUNE_LLVM_GPU_ABI_INCLUDE_DIR}
  )
target_link_libraries(llvm-gpu-abi
  PUBLIC
  LLVM
  -lrt
  -ldl
  -lpthread
  ${KITSUNE_LLVM_GPU_LINK_LIBS})

install(TARGETS llvm-gpu-abi
  LIBRARY DESTINATION lib COMPONENT llvm-gpu-abi
  ARCHIVE DESTINATION lib COMPONENT llvm-gpu-abi)

install(FILES ${KITSUNE_LLVM_GPU_ABI_HDRS}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/kitsune/llvm-gpu-abi/
  )

#if (KITSUNE_INCLUDE_TESTS)
#  add_subdirectory(tests)
#endif()
