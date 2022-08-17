# Copyright (c) 2021 Triad National Security, LLC
#                         All rights reserved.
#
# This file is part of the kitsune/llvm project.  It is released under
# the LLVM license.
#
#

find_package(CUDAToolkit REQUIRED)

# CUDA support. 
# 
if (CUDAToolkit_FOUND)
  message(ERROR "Kitsune runtime: CUDA tool was found...")
endif()

# TODO: For now we'll force-disable everything but CUDA and LLVM. 
set(KITSUNE_RT_ENABLE_HIP FALSE)
set(KITSUNE_RT_ENABLE_SPIRV FALSE)
set(KITSUNE_RT_ENABLE_REALM FALSE)

set(KITSUNE_RUNTIME_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(KITSUNE_RUNTIME_INCLUDE_DIR ${KITSUNE_RUNTIME_DIR})

include_directories(${KITSUNE_SOURCE_DIR}/include) 
include_directories(${KITSUNE_RUNTIME_INCLUDE_DIR})

set(KITSUNE_RUNTIME_SRCS
  kitrt-debug.cpp
  llvm-gpu.cpp)

set(KITSUNE_RUNTIME_HDRS
  kitrt-debug.h
  llvm-gpu.h)

if (KITSUNE_RT_ENABLE_CUDA)
  message(STATUS "@@@@@@@@@ Kitsune runtime: add CUDA source files to build.")
  list(APPEND KITSUNE_RUNTIME_SRCS
    cuda.cpp
    llvm-cuda.cpp)
  list(APPEND KITSUNE_RUNTIME_HDRS
    kitrt-cuda.h
    llvm-cuda.h)
  include_directories(${CUDAToolkit_INCLUDE_DIRS})
  link_directories(${CUDAToolkit_LIBRARY_DIR})
  set(KITSUNE_RUNTIME_LINK_LIBS "-lnvptxcompiler_static -lcudart -lcuda")
else()
  message(STATUS "Kitsune runtime: No CUDA support.")
  list(APPEND KITSUNE_RUNTIME_SRCS
    nocuda.cpp)
endif()

if (KITSUNE_RT_ENABLE_HIP)
  message(STATUS "Kitsune runtime: Enabling HIP support.")
  list(APPEND KITSUNE_RUNTIME_SRCS
    llvm-hip.cpp)
  list(APPEND KITSUNE_RUNTIME_HDRS
    llvm-hip.h)
else()
  message(STATUS "Kitsune runtime: No HIP support.")
  list(APPEND KITSUNE_RUNTIME_SRCS
    nohip.cpp)
endif()


if (KITSUNE_RT_ENABLE_SPIRV)
  message(STATUS "Kitsune runtime: Enabling SPIRV support.")
  list(APPEND KITSUNE_RUNTIME_SRCS
    llvm-spirv.cpp)
  list(APPEND KITSUNE_RUNTIME_HDRS
    llvm-spirv.h)
else()
  message(STATUS "Kitsune runtime: No SPIRV support.")
    list(APPEND KITSUNE_RUNTIME_SRCS
    nospirv.cpp)
endif()

set(KITSUNE_LLVM_LIBRARY_PATH ${LLVM_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX})
set(KITSUNE_LLVM_DYLIB ${KITSUNE_LLVM_LIBRARY_PATH}/libLLVM-${LLVM_VERSION_MAJOR}.so)

add_library(kitrt SHARED ${KITSUNE_RUNTIME_SRCS})
target_include_directories(kitrt BEFORE
  PRIVATE
  ${LLVM_MAIN_INCLUDE_DIR}
  ${LLVM_INCLUDE_DIR}
  ${KITSUNE_SOURCE_DIR}/include
  ${KITSUNE_RUNTIME_INCLUDE_DIR}
  )
  
target_link_libraries(kitrt
  PUBLIC
  LLVM
  -lrt
  -ldl
  -lpthread
  ${KITSUNE_RUNTIME_LINK_LIBS})

install(TARGETS kitrt
  LIBRARY DESTINATION lib COMPONENT kitrt
  ARCHIVE DESTINATION lib COMPONENT kitrt)

install(FILES ${KITSUNE_RUNTIME_HDRS}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/kitsune/kitrt
  )

#if (KITSUNE_INCLUDE_TESTS)
#  add_subdirectory(tests)
#endif()
