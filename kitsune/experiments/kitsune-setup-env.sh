#!/bin/bash
#export CC=/usr/bin/gcc-11
#export CXX=/usr/bin/g++-11
export CUDA_HOME=/opt/cuda
export CUDAToolkit_ROOT=$CUDA_HOME
export ROCM_PATH=/opt/rocm
export HIP_PATH=$ROCM_PATH/hip
export KITSUNE_SRC=/home/pat/projects/kitsune-15.x
export KITSUNE_PREFIX=/projects/kitsune/15.x
export KITSUNE_BUILD=$KITSUNE_SRC/build

export LD_LIBRARY_PATH=$KITSUNE_PREFIX/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$KITSUNE_PREFIX/lib:$LIBRARY_PATH
#export PATH=$KITSUNE_PREFIX/bin:$PATH:$CUDA_HOME/bin
#cd $KITSUNE_SRC

