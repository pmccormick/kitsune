
#
# Copyright (c) 2020 Triad National Security, LLC
#                         All rights reserved.
#
# This file is part of the kitsune/llvm project.  It is released under
# the LLVM license.
#

# A collection of examples/tests of using the cuabi runtime interface.
#
project(CUABITests C)

set(_example_srcs
  vecadd.c)

foreach(src_file IN ITEMS ${_example_srcs})
  get_filename_component(exec_name ${src_file} NAME_WLE)
  add_executable(${exec_name} ${src_file})
  target_include_directories(${exec_name}
    BEFORE
    PRIVATE
    ${KITSUNE_CUABI_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS})
  target_compile_definitions(${exec_name} PUBLIC SOURCE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\")
  link_directories(${exec_name} ${CUDAToolkit_NVVM_LIB_DIR} ${CUDAToolkit_LIBRARY_DIR})
  target_link_libraries(${exec_name} cu-abi nvvm CUDA::cudart CUDA::cuda_driver)
  set_target_properties(${exec_name}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY  ${LLVM_BINARY_DIR}/examples/kitsune/runtimes/cuda/tests)
endforeach()

# Do some clean up...
unset(_example_srcs)
