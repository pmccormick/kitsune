
#
# Copyright (c) 2020 Triad National Security, LLC
#                         All rights reserved.
#
# This file is part of the kitsune/llvm project.  It is released under 
# the LLVM license.
#

set(examples 
  vecadd.cpp 
)

get_kitsune_tapir_rt_flags(tapir_args)

foreach(src_file IN ITEMS ${examples}) 
  message(STATUS "  kokkos example: ${src_file}")
  get_filename_component(base ${src_file} NAME_WLE)
  string(TOLOWER ${base} ${src_file})

  add_kitsune_example(${base}-kokkos
      ${src_file}
      )

  set_target_properties(${base}-kokkos 
      PROPERTIES
        COMPILE_FLAGS "-frtti"
        INCLUDE_DIRECTORIES ${Kokkos_INCLUDE_DIR}
        LINK_DIRECTORIES ${Kokkos_LIBRARY_DIR}
        LINK_LIBRARIES ${Kokkos_LINK_LIBS}      
  )

  # Add a executable per enabled runtime target. 
  foreach(rt IN ITEMS ${tapir_args})
    set(target_name kokkos-${base}-${rt})
    message(STATUS "    ${rt} example: ${target_name}")
    add_kitsune_example(${target_name}
      ${src_file}
      )
      
    set_target_properties(${target_name}       
      PROPERTIES
        INCLUDE_DIRECTORIES ${Kokkos_INCLUDE_DIR}
        COMPILE_FLAGS "-fkokkos -fkokkos-no-init -ftapir=${rt}"
        LINK_FLAGS "-ftapir=${rt}"		
        LINK_DIRECTORIES ${Kokkos_LIBRARY_DIR}
        LINK_LIBRARIES ${Kokkos_LINK_LIBS}      
      )
  endforeach()
  
endforeach()


