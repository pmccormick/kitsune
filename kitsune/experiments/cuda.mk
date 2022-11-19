ifeq ($(CUDA_HOME),)
  $(error "required environment variable CUDA_HOME not set!")
  ${error "set it point to the install prefix of CUDA.")
  exit 1
endif

NVARCH?=sm_86
cuda_prefix=$(CUDA_HOME)

$(info cuda install prefix: ${cuda_prefix})
$(info cuda architecture: ${NVARCH})

nvcc=${CUDA_HOME}/bin/nvcc
nvcc_c_flags = -arch=${NVARCH}
nvcc_cxx_flags = --std c++17 \
 --no-exceptions \
 --expt-extended-lambda \
 -expt-relaxed-constexpr \
 -arch=${NVARCH}\
 -O${opt_level}

clang_cu_flags=-xcuda --cuda-gpu-arch=${NVARCH} -O${opt_level} 

