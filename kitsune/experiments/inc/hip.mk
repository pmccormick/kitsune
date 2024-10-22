ifneq ($(KITSUNE_HIP_ENABLE),)
  AMDGPU_ARCH?=gfx90a
  $(info   hip: amdgpu arch: $(AMDGPU_ARCH))
  HIPCC=$(KITSUNE_PREFIX)/bin/clang++ 
  HIPCC_CXX_FLAGS?=--offload-arch=$(AMDGPU_ARCH) \
    -fno-exceptions \
    -O$(KITSUNE_OPTLEVEL)

  KITSUNE_HIPCC=$(kitsune_prefix)/bin/clang++ -x hip 
  HIP_LIBS=-L$(rocm_path)/lib -lamdhip64
  BUILD_HIP_EXPERIMENTS=true
endif 

