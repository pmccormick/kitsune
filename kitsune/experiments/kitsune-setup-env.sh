#!/bin/bash
#export ROCM_PATH=/opt/rocm
#export HIP_PATH=$ROCM_PATH/hip
export KITSUNE_SRC=$HOME/projects/kitsune-15.x
export KITSUNE_PREFIX=/projects/kitsune/15.x
export KITSUNE_BUILD=$KITSUNE_SRC/build
export LD_LIBRARY_PATH=$KITSUNE_PREFIX/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$KITSUNE_PREFIX/lib:$LIBRARY_PATH
export PATH=$KITSUNE_PREFIX/bin:$PATH
#cd $KITSUNE_SRC

