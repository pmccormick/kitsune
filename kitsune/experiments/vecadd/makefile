include ../experiments.mk

targets = vecadd-forall.opencilk.${host_arch}

ifeq ($(BUILD_CUDA_EXPERIMENTS),true)
  targets += vecadd-forall.cuda.${host_arch}
  targets += vecadd-kokkos.nvcc.${host_arch}
endif

ifeq ($(BUILD_HIP_EXPERIMENTS),true)
  targets += vecadd-forall.hip.${host_arch}
  targets += vecadd-kokkos.hipcc.${host_arch}
endif

all: ${targets}

# forall-based tests
vecadd-forall.opencilk.${host_arch}: vecadd-forall.cpp 
	/usr/bin/time $(KIT_CXX) $(TAPIR_OPENCILK_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
vecadd-forall.cuda.${host_arch}: vecadd-forall.cpp 
	/usr/bin/time $(KIT_CXX) $(TAPIR_CUDA_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
vecadd-forall.hip.${host_arch}: vecadd-forall.cpp 
	/usr/bin/time $(KIT_CXX) $(TAPIR_HIP_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib

# kokkos-based tests (w/out views)
vecadd-kokkos.cuda.kitsune.${host_arch}: vecadd-kokkos-no-views.cpp 
	/usr/bin/time $(KIT_CXX) $(TAPIR_CUDA_FLAGS) $(KITSUNE_KOKKOS_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
vecadd-kokkos.hip.kitsune.${host_arch}: vecadd-kokkos-no-views.cpp 
	/usr/bin/time $(KIT_CXX) $(TAPIR_HIP_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib

# kokkos-based tests (using views -- currently not compatible w/ kitsune)
vecadd-kokkos.nvcc.${host_arch}: vecadd-kokkos.cpp 
	/usr/bin/time $(KOKKOS_NVCC) $(KOKKOS_NVCC_FLAGS) -o $@ $< $(KOKKOS_CUDA_LIBS) -Xlinker -rpath=$(KOKKOS_CUDA_PREFIX)/lib64 
vecadd-kokkos.hipcc.${host_arch}: vecadd-kokkos.cpp 
	/usr/bin/time $(KOKKOS_HIPCC) $(KOKKOS_HIP_FLAGS) -o $@ $< $(KOKKOS_HIP_LIBS) -Xlinker -rpath=$(KOKKOS_HIP_PREFIX)/lib64 



clean:
	-rm -f *.${host_arch} *.ll *.o
	-rm -f *~ core *~

