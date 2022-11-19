#
# Kitsune+Tapir specific flags used by all the experiments.
#
# 

# Install prefix for Kitsune+Tapir installation.
ifeq ($(KITSUNE_PREFIX),)
  kitsune_prefix = /projects/kitsune/${host_arch}
else
  kitsune_prefix = $(KITSUNE_PREFIX)
endif

##################################
# to disable stripming transformations.
ifeq ($(KITSUNE_STRIPMINE),)
  stripmine_flags = -mllvm -stripmine-count=1 \
   -mllvm -stripmine-coarsen-factor=1
else
  stripmine_flags = $(KITSUNE_STRIPMINE)
endif
#
##################################


##################################
# CUDA-centric target flags:
#  * cuabi-opt-level = [0...3] : kernel-centric optimization level.
#  * cuabi-prefetch=true|false : enable/disable data prefetching code gen.
#  * cuabi-streams=true|false : enable/disable pipelined stream generation.
#  * cuabi-run-post-opts=true|false : run an additional post-outline optimization
#                  pass on the host-side code.
#  * cuabi-arch=arch-name : nvidia gpu target architecture (e.g., sm_80). See the
#                  cuda.mk file for extra details. 
#  
tapir_cu_flags = -ftapir=cuda \
 -mllvm -cuabi-opt-level=${opt_level} \
 -mllvm -cuabi-prefetch=true \
 -mllvm -cuabi-streams=false \
 -mllvm -cuabi-run-post-opts=false \
 -mllvm -cuabi-arch=${NVARCH} \
 ${stripmine_flags}

tapir_cu_lto_flags = -Wl,--tapir-target=cuda,--lto-O${opt_level},-mllvm,-cuabi-opt-level=${opt_level},-mllvm,-cuabi-arch=${NVARCH},-mllvm,-cuabi-prefetch=true,-mllvm,-cuabi-streams=false,-mllvm,-cuabi-verbose=true,-mllvm,--debug-only=cuabi,-mllvm,-stripmine-coarsen-factor=1

ifneq ($(KITSUNE_VERBOSE),)
  tapir_cu_flags = ${tapir_cu_flags} -mllvm -cuabi-verbose=true 
endif

ifneq ($(KITSUNE_DEBUG),)
  tapir_cu_flags = ${tapir_cu_flags} -mllvm -debug-only=cuabi
endif
##################################


##################################
# HIP-centric target flags:
#  * hipabi-opt-level = [0...3] : kernel-centric optimization level.
#  * hipabi-disable-prefetch : disable data prefetching code gen.
#  * hipabi-streams=true|false : enable/disable pipelined stream generation.
#  * hipabi-run-post-opts=true|false : run an additional post-outline optimization
#                  pass on the host-side code.
#  * hipabiarch=arch-name : amd gpu target architecture and extra settings
#                  (e.g., gfx908). See the hip.mk file for extra details.
#  
tapir_hip_flags = -ftapir=hip \
${stripmine_flags} \
-mwavefrontsize64 \
-mllvm -hipabi-opt-level=${opt_level} \
-mllvm -hipabi-arch=${HIPARCH} \
-mllvm -hipabi-xnack=true
#-mllvm -amdgpu-internalize-symbols \
#-mllvm -amdgpu-function-calls=false \
#-mllvm --amdhsa-code-object-version=4

##################################
# OpenCilk target flags:
#
tapir_opencilk_flags = -ftapir=opencilk

##################################
# Verbose and debug mode flags.
# 
ifneq ($(KITSUNE_VERBOSE),)
  tapir_hip_flags = ${tapir_hip_flags} 
endif

ifneq ($(KITSUNE_DEBUG),)
  tapir_hip_flags = ${tapir_hip_flags} -mllvm -debug-only=hipabi
endif

##################################
# Kokkos support
kitsune_kokkos_flags = -fkokkos -fkokkos-no-init

##################################
# Experiments-wide flangs for Kitsune+Tapir 
kitflags = ${opt_flags} ${kitsune_flags} ${tapir_flags}
clang = ${kitsune_prefix}/bin/clang
clangxx = ${kitsune_prefix}/bin/clang++
opt = ${kitsune_prefix}/bin/opt 

$(info kitsune install prefix: ${kitsune_prefix})
$(info kitsune stripmine flags: ${stripmine_flags})
$(info kitsune compilation flags: ${kitflags})
$(info kitsune clang: ${clang})
##################################

