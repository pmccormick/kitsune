#
# Copyright (c) 2020 Triad National Security, LLC
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

if (KITSUNE_ENABLE_KOKKOS_SUPPORT)
  set(KITSUNE_KOKKOS_SRC_DIR  ${LLVM_BINARY_DIR}/kokkos)
  # TODO: Note that we have to patch Kokkos to disable some sections of code 
  # that are using exceptions.  In general, this is a bit messy with the
  # use of the 'fetch content' mechanisms in CMake -- in particular, the 
  # goofy system appears to continually want to run the patch even though
  # it has already been run previously.  This is a pain when developing. 
  # So...  We have hacked the patch command to always return success to 
  # avoid failed application of the patch from stopping the build.  Need to
  # find a safe way to accomplish this... 
  FetchContent_Declare(kokkos
    GIT_REPOSITORY      https://github.com/kokkos/kokkos.git
    GIT_TAG             3.3.01
    SOURCE_DIR          ${KITSUNE_KOKKOS_SRC_DIR}
    UPDATE_DISCONNECTED TRUE 
    CMAKE_ARGS -D Kokkos_ENABLE_SERIAL=ON -D BUILD_TESTING=OFF 
    PATCH_COMMAND patch -p0 --verbose --input=${KITSUNE_SOURCE_DIR}/patches/Kokkos_MemorySpace.patch)
  FetchContent_MakeAvailable(kokkos)
endif()

option(KITSUNE_ENABLE_RUNTIME_ABIS "Enable the various Kitsune ABI libraries." ON)
if (KITSUNE_ENABLE_RUNTIME_ABIS)
  message(STATUS "kitsune: enabling target runtime ABI library selection.")
  if (KITSUNE_ENABLE_REALM_TARGET)
    set(KITSUNE_REALM_SOURCE_DIR ${LLVM_BINARY_DIR}/legion)
    set(KITSUNE_LEGION_INCLUDE_DIR ${KITSUNE_REALM_SOURCE_DIR}/runtime)
    set(KITSUNE_REALM_INCLUDE_DIR ${KITSUNE_LEGION_SRC_DIR}/realm)
    set(KITSUNE_REALM_BINARY_DIR ${LLVM_BINARY_DIR}/runtimes/runtimes-bins/_deps/realm-build)
    FetchContent_Declare(realm
      GIT_REPOSITORY   https://github.com/StanfordLegion/legion.git
      GIT_TAG          master
      SOURCE_DIR       ${KITSUNE_REALM_SOURCE_DIR}
      UPDATE_DISCONNECTED TRUE)
    set(Legion_BUILD_REALM_ONLY ON)
    set(Legion_USE_LVVM OFF)
    set(Legion_LINK_LLVM_LIBS OFF)
    set(Legion_USE_CUDA OFF)
    set(Legion_USE_HIP OFF)
    FetchContent_MakeAvailable(realm)
  endif()

  # Following llvm style we use a list to pick the set of runtime abi libraries (i.e., wrappers
  # around various runtime systems that aim to simplify code generation details in the Tapir
  # transforms within LLVM.
  set(KITSUNE_ALL_RT_ABIS "realm")
  set(KITSUNE_ENABLE_ABI_LIBRARIES "" CACHE STRING
    "Semicolon-separated list of runtime ABI target libraries to build (${KITSUNE_ALL_ABI_LIBS}), or \"all\".")
  if (KITSUNE_ENABLE_ABI_LIBRARIES STREQUAL "all")
    set(KITSUNE_ENABLE_ABI_LIBRARIES ${KITSUNE_ALL_RT_ABIS})
  endif()

  if (NOT KITSUNE_ENABLE_ABI_LIBRARIES STREQUAL "'")
    foreach(lib ${KITSUNE_ENABLE_ABI_LIBRARIES})
      string(TOUPPER "${lib}" upper_lib)
      string(REGEX REPLACE "-" "_" upper_lib ${upper_lib})
      if ("${lib}" IN_LIST KITSUNE_ENABLE_ABI_LIBRARIES)
        message(STATUS "${lib} runtime abi target library is enabled")
        set(ABI_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/runtimes/${lib}")
        if (NOT EXISTS "${ABI_LIB_DIR}" OR NOT IS_DIRECTORY "${ABI_LIB_DIR}")
          message(FATAL_ERROR "KITSUNE_ENABLE_ABI_LIBRARIES requests library ${lib} but directory not found: ${ABI_LIB_DIR}")
        else()
          set(KITSUNE_ENABLE_${upper_lib}_ABI ON CACHE BOOL "Wether to build ${upper_lib} ABI library.")
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

option(KITSUNE_INCLUDE_TESTS "Enable Kitsune test suite." OFF)
