include ../experiments.mk 

targets = srad-forall.opencilk.${host_arch}

ifeq ($(BUILD_CUDA_EXPERIMENTS),true)
  targets += srad-forall.cuda.${host_arch}
  targets += srad-kokkos.nvcc.${host_arch}
  targets += srad.cuda.${host_arch}
endif

ifeq ($(BUILD_HIP_EXPERIMENTS),true)
  targets += srad-forall.hip.${host_arch}
  targets += srad-kokkos.hipcc.${host_arch}
endif

all: ${targets}

# forall-based tests
srad-forall.opencilk.${host_arch}: srad-forall.cpp 
	/usr/bin/time $(KIT_CXX) $(TAPIR_OPENCILK_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
srad-forall.cuda.${host_arch}: srad-forall.cpp 
	/usr/bin/time $(KIT_CXX) $(TAPIR_CUDA_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
srad-forall.hip.${host_arch}: srad-forall.cpp 
	/usr/bin/time $(KIT_CXX) $(TAPIR_HIP_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib

# kokkos-based tests (w/out views)
#srad-kokkos.cuda.kitsune.${host_arch}: srad-kokkos-no-view.cpp 
#	/usr/bin/time $(KIT_CXX) $(TAPIR_CUDA_FLAGS) $(KITSUNE_KOKKOS_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
#srad-kokkos.hip.kitsune.${host_arch}: srad-kokkos-no-view.cpp 
#	/usr/bin/time $(KIT_CXX) $(TAPIR_HIP_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib

# kokkos-based tests (using views -- currently not compatible w/ kitsune)
srad-kokkos.nvcc.${host_arch}: srad-kokkos.cpp 
	/usr/bin/time $(KOKKOS_NVCC) $(KOKKOS_NVCC_FLAGS) -o $@ $< $(KOKKOS_CUDA_LIBS) -Xlinker -rpath=$(KOKKOS_CUDA_PREFIX)/lib64 
srad-kokkos.hipcc.${host_arch}: srad-kokkos.cpp 
	/usr/bin/time $(KOKKOS_HIPCC) $(KOKKOS_HIP_FLAGS) -o $@ $< $(KOKKOS_HIP_LIBS) -Xlinker -rpath=$(KOKKOS_HIP_PREFIX)/lib64 

clean:
	-rm -f *.${host_arch} *.ll *.o
	-rm -f *~ core *~ *.ppm 

