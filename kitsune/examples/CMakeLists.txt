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

include(FetchContent)
include(AddKitsune)

if (KITSUNE_ENABLE_KOKKOS_SUPPORT)
  message(DEBUG "Kitsune: Adding Kokkos examples.")
  # For a (small) bit of sanity we fetch a version of Kokkos for 
  # use within our examples -- this is just to minimize 
  # configuration woes and the like that are commonplace as 
  # we stand things up. 
  set(KITSUNE_KOKKOS_SRC_DIR  ${LLVM_BINARY_DIR}/kokkos)
  FetchContent_Declare(kokkos
    GIT_REPOSITORY    https://github.com/kokkos/kokkos.git
    GIT_TAG           3.3.01
    SOURCE_DIR        ${KITSUNE_KOKKOS_SRC_DIR})

  set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "")
  FetchContent_MakeAvailable(kokkos)
  add_subdirectory(kokkos)
endif()

add_subdirectory(forall)


