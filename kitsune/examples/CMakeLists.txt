#
# Copyright (c) 2020 Triad National Security, LLC
#                         All rights reserved.
#
# This file is part of the kitsune/llvm project.  It is released under 
# the LLVM license.
#
if (NOT KITSUNE_BUILD_EXAMPLES)
  set_property(DIRECTORY PROPERTY EXCLUDE_FROM_ALL ON)
  set(EXCLUDE_FROM_ALL ON)
endif()

list(APPEND KITSUNE_EXAMPLE_DEPS
  clang
)

add_kitsune_subdirectory(forall)

if (KITSUNE_ENABLE_KOKKOS) 
  add_subdirectory(kokkos)
  add_subdirectory(stats)
endif()


