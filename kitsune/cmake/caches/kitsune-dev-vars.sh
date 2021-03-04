#!/bin/bash
#
#
# Set the location of the various runtimes base install directory.  From here the
# cache file will expand these variables inline to provide automatic search path
# and required libraries (see below for details on libraries).
#
[ ! -z "${KOKKOS_PATH}" ] && export KOKKOS_PATH=/projects/kitsune/x86_64
[ ! -z "${OPENCILKRTS_PATH}"] && export export OPENCILKRTS_PATH=/projects/kitsune/x86_64
[ ! -z "${QTHREADS_PATH}"] && export QTHREADS_PATH=/projects/kitsune/x86_64
[ ! -z "${REALM_PATH}"] && export REALM_PATH=/projects/kitsune/x86_64

#
# Depending upon how you build and configure each of these you can also provide some
# additional enviornment variables to control what libraries are automatically included
# in the mix when using Kitsune+Tapir.
#
# [ ! -z "${KOKKOS_LINK_LIBS}" ] && export KOKKOS_LINK_LIBS="-lkokkos-core -ldl -lrt"
# [ ! -z "${OPENCILKRTS_LINK_LIBS}" ] && export OPENCILKRTS_LINK_LIBS="-lopencilkrt -l..."
# [ ! -z "${QTHREADS_LINK_LIBS}" ] && export QTHREADS_LINK_LIBS="-lqhtread -l..."
# [ ! -z "${REALM_LINK_LIBS}" ] && export REALM_LINK_LIBS="-lrealm -ldl -l..."
#

# Base install of all libraries with minimal additions (e.g., hwloc, etc.)
[ ! -z "${KOKKOS_LINK_LIBS}" ] && export KOKKOS_LINK_LIBS="-lkokkos-core -ldl -lrt"
[ ! -z "${OPENCILKRTS_LINK_LIBS}" ] && export OPENCILKRTS_LINK_LIBS="-lopencilkrt -lopencilk-personality-cpp"
[ ! -z "${QTHREADS_LINK_LIBS}" ] && export QTHREADS_LINK_LIBS="-lqhtread"

######
# With these enviornment variables set you can then run cmake
# and point it at the kitsune-dev.cmake cache file.
#
#    $ mkdir -p build && cd build
#    $ cmake -G Ninja -C kitsune-dev.cmake -DCMAKE_INSTALL_PREFIX=inst-dir-path ../llvm
#    $ ninja
#
# Consider setting LLVM_PARALLEL_COMPILE_JOBS and LLVM_PARALLEL_LINK_JOBS when using
# Ninja to control the parallelism during the build.  The default will use all resources
# available and on systems with limited memory it is easy to exhuast memory when the
# llvm build starts linking (in this case you probably want LLVM_PARALLEL_COMPILE_JOBS
# to be greater than LLVM_PARALLEL_LINK_JOBS).
#
#    $ cmake -G Ninja -C kitsune-dev.cmake -DCMAKE_INSTALL_PREFIX=inst-dir-path \
#      -DLLVM_PARALLEL_COMPILE_JOBS=32 \
#      -DLLVM_PARALLEL_LINK_JOBS=12 \
#      ../llvm
#
#######
#
