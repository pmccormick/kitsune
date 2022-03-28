#
# Copyright (c) 2021 Triad National Security, LLC
#                         All rights reserved.
#
# This file is part of the kitsune+tapir project.  It is released under
# the LLVM license.
#
cmake_minimum_required(VERSION 3.19)
project(Kitsune)
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
  # TODO: Note that we have to patch kokkos to disable some sections of code
  # that are using exceptions.  In general, this is a bit messy with the
  # use of the 'fetch content' mechanisms in cmake -- in particular, the
  # goofy system appears to continually want to run the patch even though
  # it has already been run previously.  This is a pain when developing.
  # So...  the patch command has been hacked to always return success to
  # avoid failed application of the patch from stopping the build.  Ideally
  # we need to deal with the exception handling in kitsune+tapir to address
  # this...
  FetchContent_Declare(kokkos
    GIT_REPOSITORY      https://github.com/kokkos/kokkos.git
    GIT_TAG             3.5.00
    SOURCE_DIR          ${KITSUNE_KOKKOS_SRC_DIR}
    UPDATE_DISCONNECTED TRUE
    CMAKE_ARGS          ${KITSUNE_KOKKOS_BUILD_ARGS}
    PATCH_COMMAND patch -p0 --verbose --input=${KITSUNE_SOURCE_DIR}/patches/Kokkos_MemorySpace.patch)
  FetchContent_MakeAvailable(kokkos)
endif()

option(KITSUNE_ENABLE_RUNTIME_ABIS "Enable the various Kitsune ABI libraries." ON)
if (KITSUNE_ENABLE_RUNTIME_ABIS)
  # Following llvm style we use a list to pick the set of runtime abi libraries (i.e., wrappers
  # around various runtime systems that aim to simplify code generation details in the Tapir
  # transforms within LLVM.
  set(KITSUNE_ALL_RT_ABIS "realm;llvm-gpu" CACHE STRING "List of all Kitsune ABI libraries.")
  set(KITSUNE_ENABLE_ABI_LIBRARIES "" CACHE STRING
    "Semicolon-separated list of runtime ABI targets to build (or \"all\".)")
  if (KITSUNE_ENABLE_ABI_LIBRARIES STREQUAL "all")
    set(KITSUNE_ENABLE_ABI_LIBRARIES ${KITSUNE_ALL_RT_ABIS})
  endif()

  if (NOT KITSUNE_ENABLE_ABI_LIBRARIES STREQUAL "'")
    foreach(lib ${KITSUNE_ENABLE_ABI_LIBRARIES})
      string(TOUPPER "${lib}" upper_lib)
      string(REGEX REPLACE "-" "_" upper_lib ${upper_lib})
      if ("${lib}" IN_LIST KITSUNE_ENABLE_ABI_LIBRARIES)
        message(STATUS "Kitsune: ${lib} abi target library is enabled")
        set(ABI_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/runtimes/${lib}")
        if (NOT EXISTS "${ABI_LIB_DIR}" OR NOT IS_DIRECTORY "${ABI_LIB_DIR}")
          message(FATAL_ERROR "Requested abi library source not found: ${ABI_LIB_DIR}")
        else()
          message(STATUS "Kitsune: Kitsune_ENABLE_${upper_lib} ON")
          set(KITSUNE_ENABLE_${upper_lib}_ABI ON CACHE
            BOOL "Wether to build ${upper_lib} ABI library.")
          add_subdirectory(runtimes/${lib})
        endif()
      endif()
    endforeach()
  endif()
endif()

option(KITSUNE_BUILD_EXAMPLES "Build the Kitsune example programs." OFF)
if (KITSUNE_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

add_subdirectory(include)

