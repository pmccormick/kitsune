include ../experiments.mk 

targets = euler3d-forall.opencilk.${host_arch}

ifeq ($(BUILD_CUDA_EXPERIMENTS),true)
  targets += euler3d-forall.cuda.${host_arch}
  targets += euler3d-kokkos.nvcc.${host_arch}
  targets += euler3d.cuda.${host_arch}
endif

ifeq ($(BUILD_HIP_EXPERIMENTS),true)
  targets += euler3d-forall.hip.${host_arch}
  targets += euler3d-kokkos.hipcc.${host_arch}
endif

all: ${targets}

# forall-based tests
euler3d-forall.opencilk.${host_arch}: euler3d-forall.cpp 
	/usr/bin/time $(KIT_CXX) $(TAPIR_OPENCILK_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
euler3d-forall.cuda.${host_arch}: euler3d-forall.cpp 
	/usr/bin/time $(KIT_CXX) $(TAPIR_CUDA_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
euler3d-forall.hip.${host_arch}: euler3d-forall.cpp 
	/usr/bin/time $(KIT_CXX) $(TAPIR_HIP_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib

# kokkos-based tests (w/out views)
euler3d-kokkos.cuda.kitsune.${host_arch}: euler3d-kokkos-no-view.cpp 
	/usr/bin/time $(KIT_CXX) $(TAPIR_CUDA_FLAGS) $(KITSUNE_KOKKOS_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
euler3d-kokkos.hip.kitsune.${host_arch}: euler3d-kokkos-no-view.cpp 
	/usr/bin/time $(KIT_CXX) $(TAPIR_HIP_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib

# kokkos-based tests (using views -- currently not compatible w/ kitsune)
euler3d-kokkos.nvcc.${host_arch}: euler3d-kokkos.cpp 
	/usr/bin/time $(KOKKOS_NVCC) $(KOKKOS_NVCC_FLAGS) -o $@ $< $(KOKKOS_CUDA_LIBS) -Xlinker -rpath=$(KOKKOS_CUDA_PREFIX)/lib64 
euler3d-kokkos.hipcc.${host_arch}: euler3d-kokkos.cpp 
	/usr/bin/time $(KOKKOS_HIPCC) $(KOKKOS_HIP_FLAGS) -o $@ $< $(KOKKOS_HIP_LIBS) -Xlinker -rpath=$(KOKKOS_HIP_PREFIX)/lib64 

clean:
	-rm -f *.${host_arch} *.ll *.o
	-rm -f *~ core *~ *.ppm 

