#
# This file provides a basic configuration for developer's working
# with the Kitsune+Tapir toolchain -- it is more designed for
# active testing and development vs. general use.
#
# We often use a ninja-based build with a command line of the
# form:
#
#      $ cmake -G Ninja [...] -C this-cache-file.cmake ../llvm
#
# For development the cmake log-level can be helpful.  For example,
#
#      $ cmake ...  --log-level=DEBUG ...
#
function(contains_project project project_list found)
  list(FIND project_list project loc)
  if (loc EQUAL -1)
    set(found FALSE)
  else()
    set(found TRUE)
  endif()
endfunction()


message(DEBUG "KITSUNE-DEV - loading example cache file...")
# Careful with a full debug build -- linking can be painfully slow
# and you'll want to tweak the parallel 'width' of the compile and
# link stages below to make sure all is within the capabilities of
# the system you are developing on (see compile+link thread count
# settings below).
set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "")

# You'll want to tweak this to reduce build times based on what
# you are working on.  By default we provide the full suite of
# clang+tools, openmp, lld, and a debugger via lldb.
set(LLVM_ENABLE_PROJECTS
  clang;clang-tools-extra;openmp;lld;lldb
  CACHE STRING "")

message(DEBUG
  "  --> KITSUNE-DEV - enabled LLVM projects: ${LLVM_ENABLE_PROJECTS}")

set(_runtimes_list "cheetah")
set(LLVM_ENABLE_RUNTIMES
  "cheetah" CACHE STRING "")

# Various helpful LLVM-level settings for development/debugging.
set(LLVM_ENABLE_BACKTRACES ON CACHE BOOL "")
set(LLVM_ENABLE_DUMP ON CACHE BOOL "")
set(LLVM_ENABLE_TERMINFO ON CACHE BOOL "")
set(LLVM_ENABLE_THREADS ON CACHE BOOL "")
set(LLVM_INCLUDE_TESTS ON CACHE BOOL "")
set(LLVM_INCLUDE_UTILS ON CACHE BOOL "")
set(LLVM_INSTALL_UTILS ON CACHE BOOL "")
set(LLVM_INSTALL_BINUTILS_SYMLINKS ON CACHE BOOL "")

# Various helpful Clang-level settings for development/debugging.
set(CLANG_BUILD_TOOLS ON CACHE BOOL "")
set(CLANG_INCLUDE_TESTS ON CACHE BOOL "")
set(CLANG_TOOL_CLANG_CHECK_BUILD ON CACHE BOOL "")
set(CLANG_PLUGIN_SUPPORT ON CACHE BOOL "")
set(CLANG_VENDOR "kitsune+tapir" CACHE STRING "")
set(CLANG_VENDOR_UTI "gov.lanl.kitsune" CACHE STRING "")

# Build a minimal set of targets under the assumption the
# build host is the appropriate platform.
set(LLVM_TARGETS_TO_BUILD host;NVPTX;AMDGPU CACHE STRING "")
message(DEBUG
  "  --> kitsune-dev: enabled LLVM targets: ${LLVM_TARGETS_TO_BUILD}")


# You should carefully look at the parallel workload parameters as
# LLVM builds can easily swamp systems if the amount of parallelism
# exceeds system resources.
set(LLVM_PARALLEL_COMPILE_JOBS 64 CACHE STRING "")
set(LLVM_PARALLEL_LINK_JOBS 16 CACHE STRING "")

# Enable Kitsune mode within the toolchain.
set(CLANG_ENABLE_KITSUNE ON CACHE BOOL
  "Enable Kitsune features in Clang.")

# Enable tailored Kokkos compilation.
set(KITSUNE_ENABLE_KOKKOS_SUPPORT ON CACHE BOOL
  "Enable custom recognition and compilation of Kokkos.")

# NOTE: The OpenCilk runtime is always enabled within our builds
# so it is no longer necessary to do separate work to make
# this happen...
set(_has_project FALSE)
contains_project("openmp" LLVM_ENABLE_PROJECTS _has_project)
if (_has_project)
  set(KITSUNE_ENABLE_OPENMP_TARGET ON CACHE BOOL "")
  # The nvidia architecture versions within the openmp project
  # are crufty and problematic -- set them to something sane
  # as the defaults can lead to a build failure.  Hopefully we
  # can deprecate this check soon...
  if (OPENMP_ENABLE_LIBOMPTARGET)
    set(CLANG_OPENMP_NVPTX_DEFAULT_ARCH sm_70 CACHE STRING "")
    set(LIBOMPTARGET_NVPTX_COMPUTE_CAPABILITIES 70 CACHE STRING "")
    message(WARNING
      "KITSUNE-DEV - openmp may need path to CUDA-supported GCC.")
  endif()
endif()

set(KITSUNE_ENABLE_QTHREADS_TARGET OFF CACHE BOOL "")
set(KITSUNE_ENABLE_REALM_TARGET OFF CACHE BOOL "")
set(KITSUNE_ENABLE_CUDATK_TARGET ON CACHE BOOL "")
set(KITSUNE_ENABLE_HIP_TARGET OFF CACHE BOOL "")
set(KITSUNE_ENABLE_OPENCL_TARGET OFF CACHE BOOL "")

if (KITSUNE_ENABLE_CUDATK_TARGET OR
    KITSUNE_ENABLE_HIP_TARGET)
  list(APPEND _runtimes_list "kitsune")
  message(DEBUG "--> KITSUNE-DEV - adding kitsune to runtime list.")
endif()

set(LLVM_ENABLE_RUNTIMES ${_runtimes_list} CACHE STRING "" FORCE)
message(DEBUG
  "  --> KITSUNE-DEV - enabled LLVM runtimes: ${LLVM_ENABLE_RUNTIMES}")

message(DEBUG "  --> KITSUNE-DEV: fixing gcc prefix path.")
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
message(DEBUG "  --> KITSUNE-DEV - gcc prefix: ${GCC_INSTALL_PREFIX}")

unset(_gcc_prefix)
unset(_gcc_prefix_len)
unset(_runtimes_list)

message(DEBUG "kitsune-dev: DONE.")
#
#####
