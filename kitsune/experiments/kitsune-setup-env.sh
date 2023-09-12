#!/bin/bash
# CUDA, ROCM, and HIP all should have (need?) environment 
# variables set to point to their install prefix. This 
# can be hand-coded here or set via environment variables 
# in dot-files, modules, etc. 
#
arch=`/usr/bin/uname -o -m | /usr/bin/awk '{print $1}'`
#export CUDA_PATH=/opt/cuda
#export ROCM_PATH=/opt/rocm
#export HIP_PATH=${ROCM_PATH}/hip
export KITSUNE_PREFIX=/projects/kitsune/$arch/16.x
export KITSUNE_BUILD=$KITSUNE_SRC/build
export LD_LIBRARY_PATH=$KITSUNE_PREFIX/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$KITSUNE_PREFIX/lib:$LIBRARY_PATH
export PATH=$KITSUNE_PREFIX/bin:$PATH

