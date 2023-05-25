ifneq ($(ROCM_PATH),)
  AMDGPU_ARCH?=gfx90a
  HIPCC=$(ROCM_PATH)/bin/hipcc
  HIPCC_CXX_FLAGS?=--offload-arch=$(AMDGPU_ARCH) \
    -fno-exceptions \
    -O$(KITSUNE_OPTLEVEL)

  KITSUNE_HIPCC=$(KITSUNE_PREFIX)/bin/clang++ -x hip 

  HIP_LIBS=-L$(ROCM_PATH)/lib -lamdhip64

  BUILD_HIP_EXPERIMENTS=true
  $(info note: hip experiments enabled)
else 
  BUILD_HIP_EXPERIMENTS=false
endif 


