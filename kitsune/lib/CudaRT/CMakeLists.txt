# Copyright (c) 2020 Triad National Security, LLC
#                         All rights reserved.
#
# This file is part of the kitsune/llvm project.  It is released under
# the LLVM license.
#
#
# project (kitcudaart)

include_directories(SYSTEM ${CUDAToolkit_INCLUDE_DIRS})
include_directories(${KITSUNE_SOURCE_DIR}/include)
include_directories(${LLVM_BINARY_DIR}/include})

add_kitsune_library(kitcudart
  cudart.cpp
  LINK_LIBS
  cuda
)

set_target_properties(kitcudart
  PROPERTIES
  POSITION_INDEPENDENT_CODE   1
  COMPILE_FLAGS  "-fPIC"
)
