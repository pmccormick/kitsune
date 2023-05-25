
ifeq ($(BUILD_CUDA_EXPERIMENTS),true)
  KOKKOS_CUDA_PREFIX?=$(KITSUNE_PREFIX)/opt/kokkos/cuda
  KOKKOS_CUDA_LIBS=-L$(KOKKOS_CUDA_PREFIX)/lib -L$(KOKKOS_CUDA_PREFIX)/lib64 -L$(CUDA_PATH)/lib64 -lkokkoscore -lcudart -ldl
  KOKKOS_NVCC=$(KOKKOS_CUDA_PREFIX)/bin/nvcc_wrapper
  KOKKOS_NVCC_FLAGS?= $(NVCC_CXX_FLAGS) -std=c++17 -I$(KOKKOS_CUDA_PREFIX)/include/ 
  KOKKOS_CLANG=$(KITSUNE_PREFIX)/bin/clang
  KOKKOS_CLANG_CUDA_PREFIX?=$(KITSUNE_PREFIX)/opt/kokkos/clang-cuda
  KOKKOS_CLANG_CUDA_FLAGS?= $(CLANG_CUDA_FLAGS) -fPIC -std=c++17 -fno-exceptions -I$(KOKKOS_CUDA_PREFIX)/include/
  KOKKOS_CLANG_CUDA_LIBS=-L$(KOKKOS_CLANG_CUDA_PREFIX)/lib64 -L$(CUDA_PATH)/lib64 -lkokkoscore -lcudart -ldl -lstdc++
endif

ifeq ($(BUILD_HIP_EXPERIMENTS),true)
  KOKKOS_HIP_PREFIX?=$(KITSUNE_PREFIX)/opt/kokkos/hip
  KOKKOS_HIP_LIBS=-L$(KOKKOS_HIP_PREFIX)/lib -lkokkoscore -ldl
  KOKKOS_HIPCC=$(ROCM_PATH)/bin/hipcc 
  KOKKOS_HIP_FLAGS?=$(HIPCC_CXX_FLAGS) -std=c++17 -fno-exceptions -I$(KOKKOS_HIP_PREFIX)/include/ -ffp-contract=off
endif

