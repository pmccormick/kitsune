ifneq ($(CUDA_PATH),)
  NVARCH?=sm_80
  cuda_prefix=$(CUDA_HOME)
  $(info cuda install prefix: ${cuda_prefix})
  $(info cuda architecture: ${NVARCH})

  nvcc=${CUDA_HOME}/bin/nvcc
  nvcc_c_flags = -arch=${NVARCH}
  nvcc_cxx_flags = --std c++17 --no-exceptions --expt-extended-lambda \
		   --expt-relaxed-constexpr -arch=${NVARCH}
  clang_cu_flags=-xcuda --cuda-gpu-arch=${NVARCH} -O${opt_level} 
endif 

