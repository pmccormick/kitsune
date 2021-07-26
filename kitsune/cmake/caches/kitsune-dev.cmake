#
# This file sets up a basic configuration for building a Kitsune toolchain for
# active testing and development.  We typically use it as a ninja-based build
# with a command line of the form:
#
#      $ cmake -G Ninja [...] -C this-cache-file.cmake ../llvm
#

set(LLVM_PARALLEL_COMPILE_JOBS 14 CACHE STRING "")
set(LLVM_PARALLEL_LINK_JOBS 5 CACHE STRING "")
set(KITSUNE_ENABLED ON CACHE BOOL "")
set(LLVM_ENABLE_PROJECTS
  clang;clang-tools-extra;openmp;lld;lldb
  CACHE STRING "")

# Build a minimal set of targets under the assumption the
# build host is the appropriate platform.
set(LLVM_TARGETS_TO_BUILD host;NVPTX;AMDGPU CACHE STRING "")
set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "")

# Various settings for helpful development use.
set(LLVM_ENABLE_BACKTRACES ON CACHE BOOL "")
set(LLVM_ENABLE_DUMP ON CACHE BOOL "")
set(LLVM_ENABLE_TERMINFO ON CACHE BOOL "")
set(LLVM_ENABLE_THREADS ON CACHE BOOL "")
set(LLVM_INCLUDE_TESTS ON CACHE BOOL "")
set(LLVM_INCLUDE_UTILS ON CACHE BOOL "")
set(LLVM_INSTALL_UTILS ON CACHE BOOL "")
set(LLVM_INSTALL_BINUTILS_SYMLINKS ON CACHE BOOL "")

set(CLANG_BUILD_TOOLS ON CACHE BOOL "")
set(CLANG_INCLUDE_TESTS ON CACHE BOOL "")
set(CLANG_PLUGIN_SUPPORT ON CACHE BOOL "")

# Clean up some of the default NVIDIA targets so we're at least
# somewhat up-to-date...
set(CLANG_OPENMP_NVPTX_DEFAULT_ARCH sm_70 CACHE STRING "")
set(LIBOMPTARGET_NVPTX_COMPUTE_CAPABILITIES 70 CACHE STRING "")
set(OPENMP_ENABLE_LIBOMPTARGET ON CACHE BOOL "")  # do we really want to do this?
set(LLVM_ENABLE_RUNTIMES cheetah;kitsune CACHE STRING "")

set(KITSUNE_ENABLE_KOKKOS   ON  CACHE BOOL "")
set(KITSUNE_ENABLE_REALM    OFF CACHE BOOL "")
set(KITSUNE_ENABLE_CUDA     OFF CACHE BOOL "")
set(KITSUNE_ENABLE_QTHREADS OFF CACHE BOOL "")
set(CLANG_CONFIG_FILE_USER_DIR
    "~/.llvm/" CACHE STRING "")
set(KITSUNE_CONFIG_FILE_DIR 
    "${CMAKE_INSTALL_PREFIX}/share/kitsune/" 
    CACHE STRING "")

# Figure out some GCC stuff to avoid some hiccups...
execute_process(
  COMMAND /bin/bash --norc --noprofile -c "dirname \$(dirname `which gcc`)"
  OUTPUT_VARIABLE _gcc_prefix
  ECHO_OUTPUT_VARIABLE
  )

string(LENGTH ${_gcc_prefix} _gcc_prefix_len)
if (_gcc_prefix_len GREATER 0)
  string(REGEX REPLACE "\n$" "" _gcc_prefix "${_gcc_prefix}")
  set(GCC_INSTALL_PREFIX ${_gcc_prefix} CACHE STRING "")
endif()
