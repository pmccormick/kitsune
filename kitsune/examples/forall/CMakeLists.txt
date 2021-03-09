
#
# Copyright (c) 2020 Triad National Security, LLC
#                         All rights reserved.
#
# This file is part of the kitsune/llvm project.  It is released under
# the LLVM license.
#

# Note: Since we borrow LLVM functionality here we
# need to use LLVM_OPTIONAL_SOURCES here to avoid
# a error from the LLVM cmake infrastructure...
# FIXME: This is annoying...
set(LLVM_OPTIONAL_SOURCES
  vecadd.cpp
  std_vecadd.cpp
)

get_kitsune_tapir_rt_flags(tapir_args)

foreach(src_file IN ITEMS ${LLVM_OPTIONAL_SOURCES})
  message(STATUS "  forall example: ${src_file}")
  get_filename_component(base ${src_file} NAME_WLE)
  string(TOLOWER ${base} ${src_file})
  # Add a kitsune-centric executable per enabled runtime.
  foreach(rt IN ITEMS ${tapir_args})
    set(target_name forall-${base}-${rt})
    add_kitsune_example(${target_name} ${src_file}
      DEPENDS kitsunerealm)
    set_target_properties(${target_name}
      PROPERTIES
      COMPILE_FLAGS "-ftapir=${rt} -Rpass-analysis=loop-spawning-ti"
      LINK_FLAGS "-ftapir=${rt}"
      )
    install(TARGETS ${target_name} RUNTIME DESTINATION ${LLVM_BINARY_DIR}/kitsune/examples/forall)
  endforeach()

endforeach()
