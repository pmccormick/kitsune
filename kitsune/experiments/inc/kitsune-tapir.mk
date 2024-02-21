#
# Kitsune+Tapir specific flags used by all the experiments.
#
# 
KITSUNE_PREFIX?=/projects/kitsune/${host_arch}/16.x
KITSUNE_OPTLEVEL?=3
KITSUNE_ABI_OPTLEVEL?=3
KITSUNE_OPTFLAGS?=-O$(KITSUNE_OPTLEVEL)
KITSUNE_FAST_MATH=-ffp-contract=fast

# For now we disable stripmining on GPUs.
GPU_STRIPMINE_FLAGS?=

##################################
TAPIR_CUDA_TARGET=-ftapir=cuda 
TAPIR_CUDA_TARGET_FLAGS?= -O$(KITSUNE_OPTLEVEL) \
 -mllvm -cuabi-opt-level=$(KITSUNE_ABI_OPTLEVEL) \
 -mllvm -cuabi-arch=$(CUDA_ARCH) \
 $(GPU_STRIPMINE_FLAGS) \
 $(TAPIR_CUDA_EXTRA_FLAGS)
TAPIR_CUDA_FLAGS=$(TAPIR_CUDA_TARGET) $(TAPIR_CUDA_TARGET_FLAGS)

 #-ffast-math -fno-vectorize \
 #-mllvm -cuabi-run-post-opts \
 # -mllvm -cuabi-streams=true \

TAPIR_CUDA_LTO_FLAGS?=-Wl,--tapir-target=cuda \
		      -Wl,--threads=1 \
		      -Wl,--lto-O${KITSUNE_OPTLEVEL} \
		      -Wl,-mllvm=-cuabi-opt-level=$(KITSUNE_ABI_OPTLEVEL) \
		      -Wl,-mllvm=-cuabi-arch=$(CUDA_ARCH)

ifneq ($(KITSUNE_VERBOSE),)
  TAPIR_CUDA_FLAGS+=-mllvm -debug-only=cuabi $(TAPIR_CUDA_DEBUG_FLAGS)
endif
##################################


##################################
TAPIR_HIP_TARGET=-ftapir=hip 
TAPIR_HIP_TARGET_FLAGS?= -O$(KITSUNE_OPTLEVEL) \
  -mllvm -hipabi-opt-level=$(KITSUNE_ABI_OPTLEVEL) \
  -mllvm -hipabi-host-opt-level=0 \
  -mllvm -hipabi-arch=$(AMDGPU_ARCH) \
  $(TAPIR_HIP_EXTRA_FLAGS) \
  $(GPU_STRIPMINE_FLAGS)
TAPIR_HIP_FLAGS= $(TAPIR_HIP_TARGET) $(TAPIR_HIP_TARGET_FLAGS)


TAPIR_HIP_LTO_FLAGS?=-Wl,--tapir-target=hip,--lto-O$(KITSUNE_OPTLEVEL),\
-mllvm,-hipabi-opt-level=$(KITSUNE_OPTLEVEL),-mllvm,-chipabi-arch=$(AMDGPU_ARCH),\
-mllvm,-stripmine-coarsen-factor=1,-mllvm,-stripmine-count=1

ifneq ($(KITSUNE_VERBOSE),)
  TAPIR_HIP_FLAGS+=-mllvm -debug-only=hipabi $(TAPIR_HIP_DEBUG_FLAGS)
endif
##################################

##################################
TAPIR_OPENCILK_TARGET=-ftapir=opencilk
TAPIR_OPENCILK_FLAGS?=-ftapir=opencilk -O$(KITSUNE_OPTLEVEL)
TAPIR_OPENCILK_BC_PATH=${KITSUNE_PREFIX}/lib/clang/16/lib/${host_arch}-unknown-linux-gnu
TAPIR_OPENCILK_BC_FILE=libopencilk-abi.bc
TAPIR_OPENCILK_LTO_FLAGS?=-ftapir=opencilk -Wl,--lto-O${KITSUNE_OPTLEVEL}
#TAPIR_OPENCILK_LTO_FLAGS?=-Wl,--tapir-target=opencilk,--lto-O${KITSUNE_OPTLEVEL} \
#                          -Wl,--opencilk-abi-bitcode=${TAPIR_OPENCILK_BC_PATH}/${TAPIR_OPENCILK_BC_FILE}

##################################

##################################
KITSUNE_KOKKOS_FLAGS?=-fkokkos -fkokkos-no-init 
##################################

KIT_CC=$(KITSUNE_PREFIX)/bin/clang $(C_FLAGS) -I$(KITSUNE_PREFIX)/include
ifneq ($(KITSUNE_VERBOSE),)
  KITCC+=-v 
endif

KIT_CXX=$(KITSUNE_PREFIX)/bin/clang++ $(CXX_FLAGS) -I$(KITSUNE_PREFIX)/include
ifneq ($(KITSUNE_VERBOSE),)
  KITCXX+=-v 
endif

CLANG=$(KITSUNE_PREFIX}/bin/clang
CLANGXX=$(KITSUNE_PREFIX}/bin/clang++

##################################
KITSUNE_MULTI_TARGET_FLAGS?= \
 -mllvm -cuabi-opt-level=$(KITSUNE_ABI_OPTLEVEL) \
 -mllvm -cuabi-prefetch=true \
 -mllvm -cuabi-arch=$(CUDA_ARCH) \
 $(GPU_STRIPMINE_FLAGS) \
 $(TAPIR_CUDA_EXTRA_FLAGS)

 KITSUNE_MULTI_TARGET_LINK_FLAGS?= \
  -L${CUDA_ROOT}/lib64 \
  -L$(KITSUNE_PREFIX)/lib \
  -L${KITSUNE_PREFIX}/lib/clang/16/lib/x86_64-unknown-linux-gnu \
  -lcudart -lkitrt -lcuda -lLLVM -lopencilk  -lnvToolsExt

