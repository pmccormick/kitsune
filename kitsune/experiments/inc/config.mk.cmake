# Kitsune support 
kitsune_install_prefix:=${CMAKE_INSTALL_PREFIX}
kitsune_build_prefix:=${CMAKE_LL_PREFIX}

# Cuda support 
kitsune_cuda_enable:="${KITSUNE_CUDA_ENABLE}"
ifeq ($(kitsune_cuda_enable),"ON")
  #$(info config: cuda target enabled.)
  KITSUNE_CUDA_ENABLE:=true
endif

# Hip support 
kitsune_hip_enable:="${KITSUNE_HIP_ENABLE}"
ifeq ($(kitsune_hip_enable),"ON")
  #$(info config: hip target enabled.)
  KITSUNE_HIP_ENABLE:=true
  ROCM_PATH:=${ROCM_PATH}
endif  

# Kokkos support 
kitsune_kokkos_enable:="${KITSUNE_KOKKOS_ENABLE}"
ifeq ($(kitsune_kokkos_enable),"ON")
  #$(info config: kokkos codegen enabled.)
  KITSUNE_KOKKOS_ENABLE:=true
endif

