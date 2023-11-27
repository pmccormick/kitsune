#
# Kitsune+Tapir specific flags used by all the experiments.
#
# 
KITSUNE_PREFIX?=/projects/kitsune/${host_arch}/16.x
KITSUNE_OPTLEVEL?=3
KITSUNE_ABI_OPTLEVEL?=3
KITSUNE_OPTFLAGS?=-O$(KITSUNE_OPTLEVEL)

# For now we disable stripmining on GPUs.
GPU_STRIPMINE_FLAGS?=

##################################
TAPIR_CUDA_FLAGS?=-ftapir=cuda \
 -O$(KITSUNE_OPTLEVEL) \
 -mllvm -hipabi-opt-level=$(KITSUNE_ABI_OPTLEVEL) \
 -mllvm -cuabi-arch=$(CUDA_ARCH) \
 -ffp-contract=fast \
 -mllvm -cuabi-prefetch=true \
 -mllvm -cuabi-streams=false \
 $(GPU_STRIPMINE_FLAGS) \
 $(TAPIR_CUDA_EXTRA_FLAGS)
 #-mllvm -cuabi-run-post-opts \

TAPIR_CUDA_LTO_FLAGS?=-Wl,--tapir-target=cuda,--lto-O${KITSUNE_OPTLEVEL},\
-mllvm,-cuabi-opt-level=${KITSUNE_OPTLEVEL},-mllvm,-cuabi-arch=$(CUDA_ARCH),\
-mllvm,-cuabi-prefetch=true,-mllvm,-cuabi-streams=false,\
-mllvm,-stripmine-coarsen-factor=1,-mllvm,-stripmine-count=1

ifneq ($(KITSUNE_VERBOSE),)
  TAPIR_CUDA_FLAGS+=-mllvm -debug-only=cuabi $(TAPIR_CUDA_DEBUG_FLAGS)
endif
##################################


##################################
TAPIR_HIP_FLAGS?=-ftapir=hip \
  -O$(KITSUNE_OPTLEVEL) \
  -mllvm -hipabi-opt-level=$(KITSUNE_ABI_OPTLEVEL) \
  -ffp-contract=fast \
  -fno-vectorize \
  -mllvm -hipabi-arch=$(AMDGPU_ARCH) \
  -mllvm -hipabi-prefetch=true \
  -mllvm -hipabi-streams=true \
  $(TAPIR_HIP_EXTRA_FLAGS) \
  $(GPU_STRIPMINE_FLAGS)

#-mllvm -hipabi-xnack=true \


TAPIR_HIP_LTO_FLAGS?=-Wl,--tapir-target=hip,--lto-O$(KITSUNE_OPTLEVEL),\
-mllvm,-hipabi-opt-level=$(KITSUNE_OPTLEVEL),-mllvm,-chipabi-arch=$(AMDGPU_ARCH),\
-mllvm,-stripmine-coarsen-factor=1,-mllvm,-stripmine-count=1

ifneq ($(KITSUNE_VERBOSE),)
  TAPIR_HIP_FLAGS+=-mllvm -debug-only=hipabi $(TAPIR_HIP_DEBUG_FLAGS)
endif
##################################

##################################
TAPIR_OPENCILK_FLAGS?=-ftapir=opencilk -O$(KITSUNE_OPTLEVEL)
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
