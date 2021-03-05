#
# Copyright (c) 2020 Triad National Security, LLC
#                         All rights reserved.
#
# This file is part of the kitsune/llvm project.  It is released under
# the LLVM license.
#
if (NOT KITSUNE_BUILD_TESTS)
  set_property(DIRECTORY PROPERTY EXCLUDE_FROM_ALL ON)
  set(EXCLUDE_FROM_ALL ON)
endif()

if (KITSUNE_ENABLE_REALM_TARGET)
  add_subdirectory(Realm)
endif()
