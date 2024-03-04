# Copyright (c) 2021 Triad National Security, LLC
#                         All rights reserved.
#
# This file is part of the kitsune/llvm project.  It is released under
# the LLVM license.
#
#

set(KITRT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(KITRT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(KITRT_INCLUDE_DIR ${KITSUNE_RUNTIME_DIR})

# There is support for two different levels of debugging.  The general
# debug mode (KITRT_ENABLE_DEBUG) will enable runtime assertion checks
# that will cause the runtime to both check and abort on any error
# conditions from the underlying runtimes (e.g., CUDA, HIP, etc.).  The
# second option (KITRT_ENABLE_VERBOSE) will report runtime details and
# give a general feel for what is going on (memory allocations, kernel
# launches, etc.).
option(KITRT_ENABLE_DEBUG "Enable debug mode features for the runtime." OFF)
option(KITRT_ENABLE_VERBOSE "Enable verbose execution mode for debuggin." OFF)
set(KITSUNE_BITCODE_INSTALL_DIR  "lib/bitcode/" CACHE STRING
  "Location to install Kitsune's runtime bitcode files.")

include_directories(${KITSUNE_SOURCE_DIR}/include)
include_directories(${KITSUNE_RUNTIME_INCLUDE_DIR})

if (KITRT_ENABLE_DEBUG)
  add_definitions(-D_KITRT_DEBUG_)
endif()

if (KITRT_ENABLE_VERBOSE)
  add_definitions(-D_KITRT_VERBOSE_)
endif()

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH True)

# Common headers and source files.
set(KITRT_HDRS
  kitrt.h
  debug.h
  memory.h
  memory_map.h)

set(KITRT_SRCS
  kitrt.cpp
  debug.cpp
  memory.cpp
  memory_map.cpp)

set(KITRT_LINK_LIBS "")

if (KITRT_ENABLE_CUDA)
  message(STATUS "Kitsune runtime adding CUDA components to build...")
  find_package(CUDAToolkit REQUIRED)
  
  option(KITCUDA_ENABLE_NVTX "Enable NVTX profiling within the runtime" OFF)
  if (KITCUDA_ENABLE_NVTX)
    add_definitions(-DKITCUDA_ENABLE_NVTX)
  endif()
  add_definitions(-DKITRT_CUDA_ENABLED)

  list(APPEND KITRT_HDRS
    cuda/kitcuda.h
    cuda/kitcuda_dylib.h)
  list(APPEND KITRT_SRCS
     cuda/kitcuda.cpp
     cuda/dylib_support.cpp
     cuda/launching.cpp
     cuda/memory.cpp
     cuda/streams.cpp)
  link_directories(${CUDAToolkit_LIBRARY_DIR})
  include_directories(${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda)
  list(APPEND KITRT_LINK_LIBS
	  "-lnvptxcompiler_static -lcudart -lcuda -lnvToolsExt")

  #add_subdirectory(cuda/numpy)

endif()

if (KITRT_ENABLE_HIP)
  message(STATUS "Kitsune runtime adding HIP components to build...")
  # TODO: This probably needs external settings in order to
  # manage moudles and other ways to install rocm...
  list(APPEND CMAKE_PREFIX_PATH
       $ENV{ROCM_PATH}/hip $ENV{ROCM_PATH}
       /opt/rocm/hip /opt/rocm)
  find_package(hip REQUIRED)
  add_definitions(-DKITRT_HIP_ENABLED)
  list(APPEND KITRT_HDRS
    hip/kithip.h
    hip/kithip_dylib.h)
  list(APPEND KITRT_SRCS
     hip/kithip.cpp
     hip/dylib_support.cpp
     hip/launching.cpp
     hip/memory.cpp
     hip/streams.cpp)  
  list(APPEND KITRT_LINK_LIBS hip::host)
  add_subdirectory(hip)
endif()

set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)
message(STATUS "kitsune runtime library source files: ${KITRT_SRCS}")
add_library(kitrt SHARED ${KITRT_SRCS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(kitrt
  BEFORE PRIVATE
    ${LLVM_MAIN_INCLUDE_DIR}
    ${LLVM_INCLUDE_DIR}
    ${KITSUNE_SOURCE_DIR}/include
    ${KITSUNE_RUNTIME_INCLUDE_DIR})
target_link_libraries(kitrt
  PUBLIC
    ${KITRT_LINK_LIBS}
    LLVM -lpthread -lrt -ldl)
set_property(TARGET kitrt PROPERTY CXX_STANDARD 17)

install(TARGETS kitrt
  LIBRARY DESTINATION lib COMPONENT kitrt
  ARCHIVE DESTINATION lib COMPONENT kitrt)
install(FILES ${KITRT_HDRS}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/kitrt)
install(DIRECTORY cuda hip realm
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/kitrt/
  FILES_MATCHING PATTERN "*.h")

