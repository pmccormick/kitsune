
#
# Copyright (c) 2020 Triad National Security, LLC
#                         All rights reserved.
#
# This file is part of the kitsune/llvm project.  It is released under 
# the LLVM license.
#

set(forall_examples 
  vecadd.cpp 
)

get_kitsune_tapir_rt_flags(tapir_args)

foreach(src_file IN ITEMS ${forall_examples})
  message(STATUS "  forall example: ${src_file}")
  get_filename_component(base ${src_file} NAME_WLE)
  string(TOLOWER ${base} ${src_file})

  # Add a kitsune-centric executable per enabled runtime. 
  foreach(rt IN ITEMS ${tapir_args})
    set(target_name forall-${base}-${rt})
    message(STATUS "    ${rt} example: ${target_name}")
    add_kitsune_example(${target_name} ${src_file})
    set_target_properties(${target_name} 
      PROPERTIES
        COMPILE_FLAGS "-ftapir=${rt}"
        LINK_FLAGS "-ftapir=${rt}"
    )
  endforeach()
  
endforeach()


