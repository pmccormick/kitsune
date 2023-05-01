#
# Kitsune+Tapir specific flags used by all the experiments.
#
# 
KITSUNE_PREFIX?=/projects/kitsune/${host_arch}/15.x
KITSUNE_OPTLEVEL?=3
KITSUNE_ABI_OPTLEVEL?=$(KITSUNE_OPTLEVEL)
KITSUNE_OPTFLAGS?=-O$(KITSUNE_OPTLEVEL)

# For now we disable stripmining on GPUs.
GPU_STRIPMINE_FLAGS?=-mllvm -stripmine-count=1 -mllvm -stripmine-coarsen-factor=1

##################################
TAPIR_CUDA_FLAGS?=-ftapir=cuda \
 -O$(KITSUNE_OPTLEVEL) \
 -ffp-contract=fast \
 -mllvm -cuabi-opt-level=$(KITSUNE_ABI_OPTLEVEL) \
 -mllvm -cuabi-prefetch=true \
 -mllvm -cuabi-streams=false \
 -mllvm -cuabi-arch=$(CUDA_ARCH) \
 $(GPU_STRIPMINE_FLAGS)\
 $(TAPIR_CUDA_EXTRA_FLAGS)

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
  -ffp-contract=fast \
  -mllvm -hipabi-opt-level=$(KITSUNE_ABI_OPTLEVEL) \
  -mllvm -hipabi-arch=$(AMDGPU_ARCH) \
  -mllvm -amdgpu-internalize-symbols \
  -mllvm -amdgpu-function-calls=false \
  -mllvm -amdgpu-early-inline-all=true \
  -mllvm -vectorize-loops \
  -mllvm -vectorize-slp \
  $(GPU_STRIPMINE_FLAGS) \
  $(TAPIR_HIP_EXTRA_FLAGS)
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
