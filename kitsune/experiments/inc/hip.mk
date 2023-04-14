ifneq ($(ROCM_PATH),)
  AMDGPU_ARCH?=gfx90a

  HIPCC=$(ROCM_PATH)/bin/hipcc
  HIPCC_CXX_FLAGS?=--offload-arch=$(AMDGPU_ARCH)\
    -fno-exceptions \
    -O$(KITSUNE_OPTLEVEL)

  CLANG_HIP=$(KITSUNE_PREFIX)/bin/clang 
  CLANG_HIP_FLAGS=-xhip --offload-arch=${AMDGPU_ARCH} \
    -O$(KITSUNE_OPTLEVEL)

  HIP_LIBS=-L$(ROCM_PATH)/lib -lamdhip64

  BUILD_HIP_EXPERIMENTS=true
  $(info note: hip experiments enabled)
else 
  BUILD_HIP_EXPERIMENTS=false
endif 


