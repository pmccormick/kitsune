# Kitsune support 
kitsune_install_prefix:=/projects/kitsune/x86_64/19.x


# Cuda support 
kitsune_cuda_enable:="ON"
ifeq ($(kitsune_cuda_enable),"ON")
  $(info config: cuda target enabled.)
  KITSUNE_CUDA_ENABLE:=true
endif

# Hip support 
kitsune_hip_enable:="ON"
ifeq ($(kitsune_hip_enable),"ON")
  $(info config: hip target enabled.)
  KITSUNE_HIP_ENABLE:=true
  ROCM_PATH:=/opt/rocm-6.3.2
endif  

# Kokkos support 
kitsune_kokkos_enable:=""
ifeq ($(kitsune_kokkos_enable),"ON")
  $(info config: kokkos codegen enabled.)
  KITSUNE_KOKKOS_ENABLE:=true
endif

