include ../experiments.mk 

targets = saxpy-forall.opencilk.${host_arch}

ifeq ($(BUILD_CUDA_EXPERIMENTS),true)
  targets += saxpy-forall.cuda.${host_arch}
  targets += saxpy-kokkos.nvcc.${host_arch}
endif

ifeq ($(BUILD_HIP_EXPERIMENTS),true)
  targets += saxpy-forall.hip.${host_arch}
  targets += saxpy-kokkos.hipcc.${host_arch}
endif

all: ${targets}

# forall-based tests
saxpy-forall.opencilk.${host_arch}: saxpy-forall.cpp 
	/usr/bin/time $(KIT_CXX) $(TAPIR_OPENCILK_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
saxpy-forall.cuda.${host_arch}: saxpy-forall.cpp 
	/usr/bin/time $(KIT_CXX) $(TAPIR_CUDA_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
saxpy-forall.hip.${host_arch}: saxpy-forall.cpp 
	/usr/bin/time $(KIT_CXX) $(TAPIR_HIP_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib

# kokkos-based tests (w/out views)
#saxpy-kokkos.cuda.kitsune.${host_arch}: saxpy-kokkos-no-view.cpp 
#	/usr/bin/time $(KIT_CXX) $(TAPIR_CUDA_FLAGS) $(KITSUNE_KOKKOS_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
#saxpy-kokkos.hip.kitsune.${host_arch}: saxpy-kokkos-no-view.cpp 
#	/usr/bin/time $(KIT_CXX) $(TAPIR_HIP_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib

# kokkos-based tests (using views -- currently not compatible w/ kitsune)
saxpy-kokkos.nvcc.${host_arch}: saxpy-kokkos.cpp 
	/usr/bin/time $(KOKKOS_NVCC) $(KOKKOS_NVCC_FLAGS) -o $@ $< $(KOKKOS_CUDA_LIBS) -Xlinker -rpath=$(KOKKOS_CUDA_PREFIX)/lib64 
saxpy-kokkos.hipcc.${host_arch}: saxpy-kokkos.cpp 
	/usr/bin/time $(KOKKOS_HIPCC) $(KOKKOS_HIP_FLAGS) -o $@ $< $(KOKKOS_HIP_LIBS) -Xlinker -rpath=$(KOKKOS_HIP_PREFIX)/lib64 

clean:
	-rm -f *.${host_arch} *.ll *.o
	-rm -f *~ core 

