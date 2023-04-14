ifneq ($(CUDA_PATH),)
  CUDA_ARCH?=sm_80

  NVCC=$(CUDA_PATH)/bin/nvcc
  NVCC_C_FLAGS?=-arch=$(CUDA_ARCH)
  NVCC_CXX_FLAGS?=-arch=$(CUDA_ARCH) \
    --no-exceptions \
    --expt-extended-lambda \
    --expt-relaxed-constexpr \
    -O$(KITSUNE_OPTLEVEL)

  CLANG_CUDA=$(KITSUNE_PREFIX)/bin/clang 
  CLANG_CUDA_FLAGS=-xcuda --cuda-gpu-arch=$(CUDA_ARCH) \
    -O$(KITSUNE_OPTLEVEL)

  BUILD_CUDA_EXPERIMENTS=true
  $(info note: cuda experiments enabled)
else 
  BUILD_CUDA_EXPERIMENTS=false
endif 
