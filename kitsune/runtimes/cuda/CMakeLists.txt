# Copyright (c) 2020 Triad National Security, LLC
#                         All rights reserved.
#
# This file is part of the kitsune/llvm project.  It is released under
# the LLVM license.
#
#
find_package(CUDAToolkit REQUIRED)
set(CMAKE_C_STANDARD 11)

# work-around an apparent discrepancy between the cmake docs and
# behavior of the cuda modules and language support.
if (NOT CUDAToolkit_LIBRARY_ROOT)
  cmake_path(GET CUDAToolkit_BIN_DIR PARENT_PATH CUDAToolkit_LIBRARY_ROOT)
endif()

set(cuabi_srcs
  driver_api.cpp
  llvm_support.cpp
  utils.cpp)

option(CUABI_ENABLE_NVVM_SUPPORT "Enable NVVM support within the Cuda ABI runtime layer." ON)
if (CUABI_ENABLE_NVVM_SUPPORT)
  set(CUDAToolkit_NVVM_DIR ${CUDAToolkit_LIBRARY_ROOT}/nvvm)
  set(CUDAToolkit_NVVM_INCLUDE_DIR ${CUDAToolkit_NVVM_DIR}/include)
  set(CUDAToolkit_NVVM_LIB_DIR ${CUDAToolkit_NVVM_DIR}/lib${LLVM_LIBDIR_SUFFIX})
  include_directories(BEFORE ${CUDAToolkit_NVVM_INCLUDE_DIR})
  add_compile_definitions(CUABI_ENABLE_NVVM)
#  list(APPEND cuabi_srcs nvvm_support.c)
endif()

set(KITSUNE_CUABI_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(KITSUNE_CUABI_INCLUDE_DIR ${KITSUNE_CUABI_DIR})

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIRECTORY}
  ${CUDAToolkit_NVVM_INCLUDE_DIR})

add_library(cu-abi STATIC ${cuabi_srcs})
target_include_directories(cu-abi PRIVATE
  ${KITSUNE_SOURCE_DIR}/include
  ${KITSUNE_CUABI_DIR}
  ${CUDAToolkit_INCLUDE_DIRS})

set_target_properties(cu-abi
  PROPERTIES
  POSITION_INDEPENDENT_CODE   TRUE   # do we need this?
  COMPILE_FLAGS   "-fPIC")
add_library(CudaABI ALIAS cu-abi)

install(TARGETS cu-abi
  LIBRARY DESTINATION lib COMPONENT cu-abi
  ARCHIVE DESTINATION lib COMPONENT cu-abi)

if (KITSUNE_INCLUDE_TESTS)
  add_subdirectory(tests)
endif()
