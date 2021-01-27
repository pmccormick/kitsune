#
# This file sets up a basic CMakeCache for building a Kitsune toolchain for
# active testing and development (not necessarily as the best possible choice
# for a full end-user install).
#
# Notes:
#
#  1. The use of ninja for building is recommended (vs. make).  This is
#     done with:
#
#      $ cmake -G Ninja [...] -C this-cache-file.cmake ../llvm
#
#     If you need to tweak ninja's parallel compilation and linking loads use
#     the LLVM_PARALLEL_COMPILE_JOBS and LLVM_PARALLEL_LINK_JOBS variables below
#     to tweak the settings for your system. By default they are not set which
#     will run the "widest" possible set of jobs during the build (you'll need
#     a nice hefty system to support this). .
#
#  2. This file uses a set of enviornment variables to configure runtime
#     targets for code generation in the mid-stage of the compiler (via Tapir
#     to llvm transformation).  Although there are several the runtime targets
#     will be enabled if the RT_BASE_DIR for each runtime is specified.  For
#     example:
#
#         $ export OPENCILKRT_BASE_DIR=/projects/kitsune/x86_64/
#         $ export QTHREADS_BASE_DIR=/projects/kitsune/x86_64/
#         $ export REALM_BASE_DIR=/projects/kitsune/x86_64/
#
#     in each of these cases the BASE_DIR is the base directory for the installed
#     location for each (i.e., includes header files, libraries, etc.).
#
#     Kokkos is handled in the same fashion:
#
#         $ export KOKKOS_BASE_DIR=/projects/kitsune/x86_64/
#
#####
#

set(LLVM_ENABLE_PACKAGES
  clang;clang-tools-extra;openmp
  CACHE STRING "")

set(LLVM_TARGETS_TO_BUILD
  host;NVPTX;AMDGPU
  CACHE STRING "")

set(CMAKE_BUILD_TYPE
  # Release
  RelWithDebInfo
  CACHE STRING "")

set(KITSUNE_ENABLED ON CACHE BOOL "") # not sure this is really needed these days...

if (DEFINED ENV{OPENCILKRT_BASE_DIR})
  set(KITSUNE_ENABLE_OPENCILK ON CACHE BOOL "")
  set(OpenCilk_INCLUDE_DIR $ENV{OPENCILKRT_BASE_DIR}/include CACHE STRING "")
  set(OpenCilk_LIBRARY_DIR $ENV{OPENCILKRT_BASE_DIR}/lib CACHE STRING "")
  if (DEFINED ENV{OPENCILK_LINK_LIBS})
    set(OpenCilk_LINK_LIBS $ENV{OPENCILK_LINK_LIBS} CACHE STRING "")
  else()
    set(OpenCilk_LINK_LIBS "-lopencilk -lopencilk-personality-cpp" CACHE STRING "")
  endif()
else()
  set(KITSUNE_ENABLE_OPENCILK OFF CACHE BOOL "")
endif()


if (DEFINED ENV{KOKKOS_BASE_DIR})
  set(KITSUNE_ENABLE_KOKKOS ON CACHE BOOL "")
  set(Kokkos_INCLUDE_DIR $ENV{KOKKOS_BASE_DIR}/include CACHE STRING "")
  set(Kokkos_LIBRARY_DIR $ENV{KOKKOS_BASE_DIR}/lib CACHE STRING "")
  if (DEFINED ENV{KOKKOS_LINK_LIBS})
    set(Kokkos_LINK_LIBS $ENV{KOKKOS_LINK_LIBS} CACHE STRING "")
  else()
    set(Kokkos_LINK_LIBS "-lkokkoscore -ldl -lrt" CACHE STRING "")
  endif()
else()
  set(KITSUNE_ENABLE_KOKKOS OFF CACHE BOOL "")
endif()


if (DEFINED ENV{QTHREADS_BASE_DIR})
  set(KITSUNE_ENABLE_QTHREADS ON CACHE BOOL "")
  set(Qthreads_INCLUDE_DIR $ENV{QTHREADS_BASE_DIR}/include CACHE STRING "")
  set(Qthreads_LIBRARY_DIR $ENV{QTHREADS_BASE_DIR}/lib CACHE STRING "")
  if (DEFINED ENV{QTHREADS_LINK_LIBS})
    set(Qthreads_LINK_LIBS $ENV{QTHREADS_LINK_LIBS} CACHE STRING "")
  else()
    set(Qthreads_LINK_LIBS "-lqthread -lhwloc" CACHE STRING "")
  endif()
else()
  set(KITSUNE_ENABLE_QTHREADS OFF CACHE BOOL "")
endif()


if (DEFINED ENV{REALM_BASE_DIR})
  set(KITSUNE_ENABLE_REALM ON CACHE BOOL "")
  set(Realm_INCLUDE_DIR $ENV{REALM_BASE_DIR}/include CACHE STRING "")
  set(Realm_LIBRARY_DIR $ENV{REALM_BASE_DIR}/lib CACHE STRING "")
  if (DEFINED ENV{REALM_LINK_LIBS})
    set(Realm_LINK_LIBS $ENV{REALM_LINK_LIBS} CACHE STRING "")
  else()
    set(Realm_LINK_LIBS "-lrealm -ldl -lpthread" CACHE STRING "")
  endif()
else()
  set(KITSUNE_ENABLE_REALM ON CACHE BOOL "")
endif()

# Defaults for NVIDIA GPU targets are old...  Let's bump it up
# a bit to make it more "modern"...
set(CLANG_OPENMP_NVPTX_DEFAULT_ARCH sm_70 CACHE STRING "")
set(LIBOMPTARGET_NVPTX_COMPUTE_CAPABILITIES 70 CACHE STRING "")
# After all that we're now going to shut off target offload...
# We're debugging build issues with this enabled on some systems.
set(OPENMP_ENABLE_LIBOMPTARGET OFF CACHE BOOL "")

# Get a little help w/ debugging issues.
set(LLVM_ENABLE_BACKTRACES ON CACHE BOOL "")
set(LLVM_ENABLE_DUMP ON CACHE BOOL "")
set(LLVM_ENABLE_TERMINFO ON CACHE BOOL "")
set(LLVM_ENABLE_THREADS ON CACHE BOOL "")
set(LLVM_INSTALL_TOOLCHAIN_ONLY OFF CACHE BOOL "")
set(LLVM_INCLUDE_TESTS ON CACHE BOOL "")
set(LLVM_INCLUDE_UTILS ON CACHE BOOL "")
set(LLVM_INSTALL_UTILS ON CACHE BOOL "")
set(LLVM_INSTALL_BINUTILS_SYMLINKS ON CACHE BOOL "")

set(CLANG_BUILD_TOOLS ON CACHE BOOL "")
set(CLANG_INCLUDE_TESTS ON CACHE BOOL "")
set(CLANG_PLUGIN_SUPPORT ON CACHE BOOL "")
set(CLANG_VENDOR Kitsune ON CACHE STRING "")
set(PACKAGE_VENDOR Kitsune CACHE STRING "")
