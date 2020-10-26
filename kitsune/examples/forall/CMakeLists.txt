
#
# Copyright (c) 2020 Triad National Security, LLC
#                         All rights reserved.
#
# This file is part of the kitsune/llvm project.  It is released under 
# the LLVM license.
#

message("  adding forall examples...")

set(forall_examples 
  vecadd.cpp 
)

get_kitsune_tapir_rt_flags(tapir_args)

foreach(src_file IN ITEMS ${forall_examples}) 
  message(STATUS "    example: ${src_file}")
  get_filename_component(base ${src_file} NAME_WLE)
  string(TOLOWER ${base} ${src_file})

  # Add a kitsune-centric executable per enabled runtime. 
  foreach(rt IN ITEMS ${tapir_args})
    set(target_name forall-${base}-${rt})
    message(STATUS "    example: ${target_name}")
    add_executable(${target_name} ${src_file})
    set_target_properties(${target_name} 
      PROPERTIES
        COMPILE_FLAGS "-ftapir=${rt}"
        LINK_FLAGS "-ftapir=${rt}"
	    INSTALL_NAME_DIR ${CMAKE_INSTALL_PREFIX}/kitsune/examples/forall
    )
    add_dependencies(${target_name} clang)
  endforeach()
  
endforeach()


