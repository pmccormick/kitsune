ifneq ($(ROCM_PATH),)
  HIPARCH?=gfx90a
  hip_prefix=$(ROCM_PATH)
  $(info hip/rocm install prefix: ${hip_prefix})
  $(info hip/rocm architecture: ${HIPARCH})

  hipcc=${ROCM_PATH}/bin/hipcc
  hipcc_cxx_flags = --offload-arch=${HIPARCH}
  clang_hip_flags = -x hip ${hipcc_cxx_flags}
  hip_ld_flags = -L${ROCM_PATH}/lib
  hip_libs = -lamdhip64 
endif


