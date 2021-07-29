#
# Copyright (c) 2020 Triad National Security, LLC
#                         All rights reserved.
#
# This file is part of the kitsune+tapir project.  It is released under
# the LLVM license.
#
cmake_minimum_required(VERSION 3.19)

# RPATH settings on macOS do not affect INSTALL_NAME.
if (POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

# Include file check macros honor CMAKE_REQUIRED_LIBRARIES.
if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

# option() honors normal variables.
if (POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

project(kitsune)
message(STATUS "kitsune: llvm runtime/subproject is enabled.")

include(FetchContent)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
include(AddKitsune)

set(KITSUNE_SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR})
set(KITSUNE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(KITSUNE_BINARY_DIR  ${CMAKE_CURRENT_BINARY_DIR})

option(KITSUNE_ENABLE_WERROR "Kitsune build fails on warnings." OFF)
option(KITSUNE_INCLUDE_TESTS "Enable Kitsune test suite." FALSE)
option(KITSUNE_BUILD_EXAMPLES "Build the Kitsune example programs." FALSE)

# Create a share directory for kitsune configuration files, etc.
file(MAKE_DIRECTORY ${LLVM_BINARY_DIR}/kitsune/share)

set(clang_option_files "")

# add a set of compiler options to the given configuration file.
# this file can be used via clang's --config command line argument.
function(create_cfg_file OUTFILE )
  message(STATUS "creating clang configure file: ${OUTFILE}")
  file(WRITE ${OUTFILE} ${ARGN})
  if (NOT ${OUTFILE} IN_LIST option_files)
    set(option_files ${option_files} ${OUT_FILE} PARENT_SCOPE)
  endif()
endfunction()

# TODO: much of the code below can use some functions to clean it and avoid 
# so much redundancy... 

option(KITSUNE_ENABLE_KOKKOS "Enable Kokkos-aware compilation." ON)
if (KITSUNE_ENABLE_KOKKOS)
  # NOTE: We assume that the serial version of Kokkos is being used in this case, it is 
  # best not to mix a parallel install of Kokkos w/ Kitsune+Tapir given potential runtime
  # conflicts at initialization and related issues...
  if (DEFINED EVN{KOKKOS_PATH})
    message("kitsune: Kokkos variables found in user's environment.")
    set(Kokkos_INCLUDE_DIR $ENV{KOKKOS_PATH}/include CACHE STRING "Kokkos include path.")
    set(Kokkos_LIBRARY_DIR $ENV{KOKKOS_PATH}/lib CACHE STRING "Kokkos library path.")
    set(Kokkos_LINK_LIBS   "-lkokkoscore -ldl -lrt" CACHE STRING "Kokkos link libraries.")
  else() 
    message(STATUS "kitsune: Looking for a Kokkos installation...")
    find_package(Kokkos)
    if (NOT Kokkos_FOUND)
      message(STATUS "kitsune: Kokkos was not found -- creating a local version.")
      FetchContent_Declare(kokkos
        GIT_REPOSITORY    https://github.com/kokkos/kokkos.git
        GIT_TAG           3.3.01
        SOURCE_DIR        ${CMAKE_CURRENT_SOURCE_DIR}/kokkos
      )
      file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/kokkos)
      FetchContent_GetProperties(kokkos)
      if (NOT kokkos_POPULATED)
        message(STATUS "     fetching kokkos...")
        FetchContent_Populate(kokkos)
      endif()
      set(Kokkos_ENABLE_SERIAL ON)   # NOTE: Only use serial configurations of Kokkos! 
      set(Kokkos_CXX_STANDARD  17)
      add_subdirectory(${kokkos_SOURCE_DIR})
      set(Kokkos_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/include/kokkos/)
      set(Kokkos_LIBRARY_DIR ${CMAKE_INSTALL_PREFIX}/lib)
      set(Kokkos_LINK_LIBS   "-lkokkoscore -ldl -lrt")
    else()
      message(STATUS "kitsune: A Kokkos installation was found.")
    endif()

    # Generate a Clang configure (.cfg) file specifically tailored to
    # support the -fkokkos flag.
    set(_kokkos_cfg_filename ${LLVM_BINARY_DIR}/share/kitsune/kokkos.cfg)
    create_cfg_file(${_kokkos_cfg_filename}
      "# compile flags:\n"
      "-I${Kokkos_INCLUDE_DIR} -I${CMAKE_INSTALL_PREFIX}/include\n"
      "# link flags:\n"
      "-L${Kokkos_LIBRARY_DIR} ${Kokkos_LINK_LIBS}\n"
    )
    install(FILES
      ${_kokkos_cfg_filename}
      DESTINATION 
      ${CMAKE_INSTALL_PREFIX}/share/kitsune/
    )
    unset(_kokkos_cfg_filename)
  endif()  
else()
  message(STATUS "kitsune: Kokkos support disabled.")
endif()

#
# CUDA runtime support. 
#
option(Kitsune_ENABLE_CUDATK  "Enable the CUDA support (codegen runtime + ABI transform)." ON)
if (Kitsune_ENABLE_CUDAToolkit)
  message(STATUS "kitsune: Cuda Toolkit support enabled. Configuring...")
  # TODO: Enable CUDA language. 
  find_package(CUDAToolkit 11 REQUIRED)

  


    else()
      set(_cuda_cfg_filename ${LLVM_BINARY_DIR}/share/kitsune/cuda.cfg)
      create_cfg_file(${_cuda_cfg_filename}
        "# NOTE: given cuda is an ABI target, we only need link-time flags.\n"
        "-L${CUDAToolkit_LIBRARY_DIR} -lcudart\n"
      )
      install(FILES
        ${_cuda_cfg_filename}
        DESTINATION ${CMAKE_INSTALL_PREFIX}/share/kitsune/
      )
      unset(_cuda_cfg_filename)
    endif()
  endif()
endif()

#
# Realm runtime support. 
#
option(KITSUNE_ENABLE_REALM  "Enable the Realm runtime code generation target library." OFF)
if (KITSUNE_ENABLE_REALM)
  if (DEFINED ENV{REALM_PATH})
    message("kitsune: Realm runtime variables found in user's environment.")
    set(Realm_INCLUDE_DIR $ENV{REALM_PATH}/include CACHE STRING "Realm runtime include path.")
    set(Realm_LIBRARY_DIR $ENV{REALM_PATH}/lib CACHE STRING "Realm runtime library path.")
    set(Realm_LINK_LIBS "-lrealm -ldl -lpthread" CACHE STRING "Link libraries for Realm runtime and supporting libraries.")
  else() 
    message(STATUS "kitsune: Looking for a Realm installation...")
    find_package(Realm)
    if (NOT Realm_FOUND)
      message(SEND_ERROR "kitsune: Realm was not found -- but KITSUNE_ENABLE_REALM is set.")
      # TODO: right now legion doesn't play well in a fetch-configure-and-build mode for some reason. 
      # we will just disable Realm if we can't find it. 
      #FetchContent_Declare(legion
      #  GIT_REPOSITORY    https://github.com/StanfordLegion/legion.git
      #  GIT_TAG           legion-21.06.0
      #SOURCE_DIR        ${CMAKE_CURRENT_SOURCE_DIR}/legion
      #)
      # 
      #file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/legion)
      #FetchContent_GetProperties(legion)
      #if (NOT legion_POPULATED)
      #  message(STATUS "     fetching legion/realm...")
      #  FetchContent_Populate(legion)
      #endif()
      #
      #add_subdirectory(${legion_SOURCE_DIR})
    else()
      # TODO: since our code generation occurs in the mid-stage of the compiler in kitsune+tapir 
      # we really don't need to worry about include files... 
      set(_realm_cfg_filename ${LLVM_BINARY_DIR}/share/kitsune/realm.cfg)
      create_cfg_file(${_realm_cfg_filename}
        "# NOTE: given realm is an ABI target, we only need link-time flags.\n"
        "-L${Realm_LIBRARY_DIR} ${Realm_LINK_LIBS}\n"
      )
 
      install(FILES
        ${_realm_cfg_filename}
        DESTINATION ${CMAKE_INSTALL_PREFIX}/share/kitsune/
      )
      unset(_realm_cfg_filename)
    endif()
  endif()
endif()

#
# Qthreads runtime support. 
#
option(KITSUNE_ENABLE_QTHREADS  "Enable the Qthreads runtime code generation target library." OFF)
if (KITSUNE_ENABLE_QTHREADS)
  if (DEFINED ENV{QTHREADS_PATH})
    message("kitsune: Qthreads runtime variable(s) found in user's environment.")
    set(Qthreads_INCLUDE_DIR $ENV{QTHREADS_PATH}/include CACHE STRING "Realm runtime include path.")
    set(Qthreads_LIBRARY_DIR $ENV{QTHREADS_PATH}/lib CACHE STRING "Realm runtime library path.")
    set(Qthreads_LINK_LIBS "-lqthread -lpthread" CACHE STRING "Link libraries for Realm runtime and supporting libraries.")
  else() 
    message(STATUS "kitsune: Looking for a Qthreads installation...")
    find_package(Qthreads)
    if (NOT Qthreads_FOUND)
      message(SEND_ERROR "kitsune: Qthreads was not found -- but KITSUNE_ENABLE_QTHREADS is set.")
      # TODO: qthreads doesn't neatly fit into an integrated Cmake build for now.  So, we'll punt 
      # here and let the user hand-build the install... 
    else()
      set(_qthreads_cfg_filename ${LLVM_BINARY_DIR}/share/kitsune/qthreads.cfg)
      create_cfg_file(${_qthreads_cfg_filename}
        "# NOTE: given qthreads is an ABI target, we only need link-time flags.\n"
        "-L${Qthreads_LIBRARY_DIR} ${Qthreads_LINK_LIBS}\n"
      )
      install(FILES
        ${_qthreads_cfg_filename}
        DESTINATION ${CMAKE_INSTALL_PREFIX}/share/kitsune/
      )
      unset(_qthreads_cfg_filename)
    endif()
  endif()
endif()

if (KITSUNE_INCLUDE_TESTS)
  add_subdirectory(tests)
endif()

if (KITSUNE_BUILD_EXAMPLES)
  add_kitsune_subdirectory(examples)
endif()
