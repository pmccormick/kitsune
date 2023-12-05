#
# Copyright (c) 2021 Triad National Security, LLC
#                         All rights reserved.
#
# This file is part of the kitsune+tapir project.  It is released under
# the LLVM license.
#
cmake_minimum_required(VERSION 3.19)
if (KITRT_ENABLE_CUDA)
  project(Kitsune C CXX CUDA)
else()
  project(Kitsune C CXX)
endif()

include(FetchContent)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
include(KitsuneUtils)

set(KITSUNE_SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR})
set(KITSUNE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(KITSUNE_BINARY_DIR  ${CMAKE_CURRENT_BINARY_DIR})

if (KITSUNE_ENABLE_KOKKOS_SUPPORT)
  set(KITSUNE_KOKKOS_SRC_DIR  ${LLVM_BINARY_DIR}/kokkos)
  set(KITSUNE_KOKKOS_BUILD_ARGS
      -DKokkos_ENABLE_SERIAL=ON -DBUILD_TESTING=OFF
      -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release)
  #
  # TODO: Note that we have to patch kokkos to disable some sections of code
  # that are using exceptions.  We do this alongside the 'fetch content'
  # mechanisms in cmake. Some aspect of this continually wants to run the
  # patch when it has already been previosuly run...  So...  The patch
  # command has been hacked to always return success to avoid a failed
  # application of the patch from halting the build.
  #
  # TODO: Look for newer Kokkos updates.
  #
  message(STATUS "kokkos fetch content: ${KITSUNE_KOKKOS_SRC_DIR}")
  FetchContent_Declare(
    kokkos
    GIT_REPOSITORY      https://github.com/kokkos/kokkos.git
    GIT_TAG             606866d # 4.0.00
    SOURCE_DIR          ${KITSUNE_KOKKOS_SRC_DIR}
    CMAKE_ARGS          ${KITSUNE_KOKKOS_BUILD_ARGS}
    UPDATE_DISCONNECTED FALSE
    # Kokkos has exceptions that we have to disable to currently work with kitsune+tapir. 
    PATCH_COMMAND patch -p0 --verbose --input=${KITSUNE_SOURCE_DIR}/patches/Kokkos_Exceptions.patch)
  #
  FetchContent_MakeAvailable(kokkos)
endif()

option(KITSUNE_ENABLE_KITRT "Build and install the Kitsune runtime library (kitrt)." ON)
if (KITSUNE_ENABLE_KITRT)
  # TODO: Need to update to add JIT GPU target and Realm back into the mix.
  set(KITSUNE_ALL_KITRT_TARGETS "cuda;hip" CACHE STRING "List of all Kitsune runtime targets.")
  set(KITSUNE_ENABLE_KITRT_TARGETS "" CACHE STRING
    "Semicolon-separated list of runtime targets to build (or \"all\".)")
  # When building the full suite (LLVM + runtime) we use the
  # compiler-side flags to also enable runtime target support.
  message(STATUS "kitsune: checking compiler build settings to enable runtime components...")

  if (KITSUNE_ENABLE_KITRT_TARGETS STREQUAL "all")
    set(KITSUNE_ENABLE_KITRT_TARGETS ${KITSUNE_ALL_KITRT_TARGETS})
  else()
    if (KITSUNE_ENABLE_CUDA_ABI_TARGET)
      message(STATUS "kitsune: cuda compiler target enabled -- enabling runtime support.")
      list(APPEND KITSUNE_ENABLE_KITRT_TARGETS "cuda")
    endif()

    if (KITSUNE_ENABLE_HIP_ABI_TARGET)
      message(STATUS "kitsune: hip compiler target enabled -- enabling runtime support.")
      list(APPEND KITSUNE_ENABLE_KITRT_TARGETS "hip")
    endif()

    if (KITSUNE_ENABLE_LLVM_GPU_ABI_TARGET)
      message(STATUS "kitsune: llvm/jit compiler target enabled -- enabling runtime support.")
      list(APPEND KITSUNE_ENABLE_KITRT_TARGETS "llvm")
    endif()

    if (KITSUNE_ENABLE_REALM_ABI_TARGET)
      message(STATUS "kitsune: realm compiler target enabled -- enabling runtime support.")
      list(APPEND KITSUNE_ENABLE_KITRT_TARGETS "realm")
    endif()

    if (KITSUNE_ENABLE_SPIRV_ABI_TARGET)
      message(STATUS "kitsune: spirv compiler target enabled -- enabling runtime support.")
      list(APPEND KITSUNE_ENABLE_KITRT_TARGETS "spirv")
    endif()
  endif()

  if (NOT KITSUNE_ENABLE_KITRT_TARGETS STREQUAL "'")
    message(STATUS "Kitsune runtime targets list: ${KITSUNE_ENABLE_KITRT_TARGETS}")
    foreach(rt_target IN LISTS KITSUNE_ENABLE_KITRT_TARGETS)
      message(STATUS "  runtime target: ${rt_target}")
      string(TOUPPER "${rt_target}" upper_target)
      if (rt_target IN_LIST KITSUNE_ENABLE_KITRT_TARGETS)
        set(KITRT_ENABLE_${upper_target} ON CACHE BOOL
            "Build kitsune runtime support for ${upper_target} target.")
      endif()
      message(STATUS "Kitsune runtime target KITRT_ENABLE_${upper_target} is ${KITRT_ENABLE_${upper_target}}.")
    endforeach()
    add_subdirectory(runtime)
  endif()
endif()

add_subdirectory(include)

#option(KITSUNE_BUILD_EXAMPLES "Build the Kitsune example programs." OFF)
#if (KITSUNE_BUILD_EXAMPLES)
#  add_subdirectory(examples)
#endif()


