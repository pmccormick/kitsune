# Copyright (c) 2020 Triad National Security, LLC
#                         All rights reserved.
#
# This file is part of the kitsune/llvm project.  It is released under
# the LLVM license.
#
#
# project (kitsunerealm)


# Realm uses RTTI -- shutdown LLVM's global configuration of enabling
# -fno-rtti...
set(LLVM_REQUIRES_RTTI 1)

include_directories(${KITSUNE_SOURCE_DIR}/include)
include_directories(${Realm_INCLUDE_DIR})

file(GLOB realm_srcs *.cpp)
file(GLOB realm_incs Realm/*.h)

add_kitsune_library(kitsunerealm
  ${realm_srcs}
  LINK_LIBS
  realm pthread dl rt)

set_target_properties(kitsunerealm
  PROPERTIES
  POSITION_INDEPENDENT_CODE   1
  COMPILE_FLAGS "-fPIC"
  )
