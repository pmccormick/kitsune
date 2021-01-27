#!/bin/bash
#
export KOKKOS_BASE_DIR=/projects/kitsune/x86_64
export OPENCILKRT_BASE_DIR=/projects/kitsune/x86_64
export QTHREADS_BASE_DIR=/projects/kitsune/x86_64
export REALM_BASE_DIR=/projects/kitsune/x86_64

######
# With these enviornment variables set you can then run cmake
# and point it at the kitsune-dev.cmake cache file.
#
#    $ mkdir -p build && cd build
#    $ cmake -G Ninja -C kitsune-dev-var.sh -DCMAKE_INSTALL_PREFIX=inst-dir-path ../llvm
#    $ ninja
#
# Consider setting LLVM_PARALLEL_COMPILE_JOBS and LLVM_PARALLEL_LINK_JOBS when using
# Ninja to control the parallelism during the build.  The default will use all resources
# available and on systems with limited memory it is easy to exhuast memory when the
# llvm build starts linking (in this case you probably want LLVM_PARALLEL_COMPILE_JOBS
# to be greater than LLVM_PARALLEL_LINK_JOBS).
#
#    $ cmake -G Ninja -C kitsune-dev-var.sh -DCMAKE_INSTALL_PREFIX=inst-dir-path \
#      -DLLVM_PARALLEL_COMPILE_JOBS=32 \
#      -DLLVM_PARALLEL_LINK_JOBS=12 \
#      ../llvm
#######
#
