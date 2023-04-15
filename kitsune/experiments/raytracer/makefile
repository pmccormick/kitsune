include ../experiments.mk 

targets = raytracer-forall.opencilk.${host_arch}

ifeq ($(BUILD_CUDA_EXPERIMENTS),true)
  targets += raytracer-forall.cuda.${host_arch}
  targets += raytracer-kokkos.nvcc.${host_arch}
  targets += raytracer.cuda.${host_arch}
endif

ifeq ($(BUILD_HIP_EXPERIMENTS),true)
  targets += raytracer-forall.hip.${host_arch}
  targets += raytracer-kokkos.hipcc.${host_arch}
endif

all: ${targets}

# forall-based tests
raytracer-forall.opencilk.${host_arch}: raytracer-forall.cpp 
	/usr/bin/time $(KIT_CXX) $(TAPIR_OPENCILK_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
raytracer-forall.cuda.${host_arch}: raytracer-forall.cpp 
	/usr/bin/time $(KIT_CXX) $(TAPIR_CUDA_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
raytracer-forall.hip.${host_arch}: raytracer-forall.cpp 
	/usr/bin/time $(KIT_CXX) $(TAPIR_HIP_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib

# kokkos-based tests (w/out views)
raytracer-kokkos.cuda.kitsune.${host_arch}: raytracer-kokkos-no-view.cpp 
	/usr/bin/time $(KIT_CXX) $(TAPIR_CUDA_FLAGS) $(KITSUNE_KOKKOS_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
raytracer-kokkos.hip.kitsune.${host_arch}: raytracer-kokkos-no-view.cpp 
	/usr/bin/time $(KIT_CXX) $(TAPIR_HIP_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib

# kokkos-based tests (using views -- currently not compatible w/ kitsune)
raytracer-kokkos.nvcc.${host_arch}: raytracer-kokkos.cpp 
	/usr/bin/time $(KOKKOS_NVCC) $(KOKKOS_NVCC_FLAGS) -o $@ $< $(KOKKOS_CUDA_LIBS) -Xlinker -rpath=$(KOKKOS_CUDA_PREFIX)/lib64 
raytracer-kokkos.hipcc.${host_arch}: raytracer-kokkos.cpp 
	/usr/bin/time $(KOKKOS_HIPCC) $(KOKKOS_HIP_FLAGS) -o $@ $< $(KOKKOS_HIP_LIBS) -Xlinker -rpath=$(KOKKOS_HIP_PREFIX)/lib64 

# cuda version 
raytracer.cuda.${host_arch}: raytracer.cu
	/usr/bin/time $(NVCC) $(NVCC_CXX_FLAGS) -o $@ $<

# hip version 
raytracer.hip.${host_arch}: raytracer-hip.cpp
	/usr/bin/time $(HIPCC) $(HIPCC_CXX_FLAGS) -o $@ $<


clean:
	-rm -f *.${host_arch} *.ll *.o
	-rm -f *~ core *.ppm 

