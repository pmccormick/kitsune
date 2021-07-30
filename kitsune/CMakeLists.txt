#
# Copyright (c) 2020 Triad National Security, LLC
#                         All rights reserved.
#
# This file is part of the kitsune+tapir project.  It is released under
# the LLVM license.
#
cmake_minimum_required(VERSION 3.19)
include(FetchContent)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
include(AddKitsune)

set(KITSUNE_SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR})
set(KITSUNE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

option(KITSUNE_ENABLE_WERROR "Kitsune build fails on warnings." OFF)
option(KITSUNE_INCLUDE_TESTS "Enable Kitsune test suite." OFF)
option(KITSUNE_BUILD_EXAMPLES "Build the Kitsune example programs." OFF)

if (KITSUNE_INCLUDE_TESTS)
  add_subdirectory(tests)
endif()

if (KITSUNE_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
