# Copyright (c) 2021 Triad National Security, LLC
#                         All rights reserved.
#
# This file is part of the kitsune/llvm project.  It is released under
# the LLVM license.
#
#
if (NOT KITRT_ENABLE_HIP)
  message(FATAL_ERROR "File should not be included when HIP is disabled!")
endif()

#set(KITRT_HIPMATH_SRCS
#  hipmath.cpp)

#add_library(kitrt-hipmath
#  OBJECT ${KITRT_HIPMATH_SRCS})
#target_compile_options(kitrt-hipmath PUBLIC -emit-llvm -O0 -x hip --offload-device-only)

#install(FILES $<TARGET_OBJECTS:kitrt-hipmath>
#  DESTINATION ${KITSUNE_BITCODE_INSTALL_DIR}
#  RENAME kit-$<PATH:REMOVE_EXTENSION,$<PATH:GET_FILENAME,$<TARGET_OBJECTS:kitrt-hipmath>>>.bc)
