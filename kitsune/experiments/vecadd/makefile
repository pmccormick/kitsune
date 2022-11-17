include ../experiments.mk

cxx_flags += -I${kitsune_prefix}/include 

targets = vecadd-forall.opencilk.${host_arch}

ifeq ($(KITSUNE_ENABLE_CUDA),TRUE)
  targets += vecadd-forall.cuda.${host_arch}
endif
ifeq ($(KITSUNE_ENABLE_HIP),TRUE)
  targets += vecadd-forall.hip.${host_arch}
endif

all: ${targets}

#############################
# Kitsune+Tapir versions.
vecadd-forall.opencilk.${host_arch}: vecadd-forall.cpp 
	/usr/bin/time ${clangxx} ${cxx_flags} ${kitflags} ${tapir_opencilk_flags} -o $@ $< \
	-Xlinker -rpath=${kitsune_prefix}/lib

vecadd-forall.cuda.${host_arch}: vecadd-forall.cpp
	/usr/bin/time ${clangxx} -v ${cxx_flags} ${kitflags} ${tapir_cu_flags} -o $@ $< -Xlinker -rpath=${kitsune_prefix}/lib

vecadd-forall.hip.${host_arch}: vecadd-forall.cpp
	/usr/bin/time ${clangxx} ${cxx_flags} ${kitflags} ${tapir_hip_flags} -o $@ $< -Xlinker -rpath=${kitsune_prefix}/lib

#############################
# Cuda version w/ both NVCC and Clang for comparison.
vecadd-cuda.nvcc.${host_arch}: vecadd.cu
	/usr/bin/time ${nvcc} -I${kitsune_prefix}/include ${nvcc_cxx_flags} -o $@ $<
vecadd-cuda.clang.${host_arch}: vecadd.cu
	/usr/bin/time ${clangxx} ${clang_cuda_flags} -o $@ $< -L${CUDA_HOME}/lib64 -lcudart_static -ldl -lrt -pthread

#############################
# Kokkos version. 
vecadd-kokkos.${host_arch}: vecadd-kokkos.cpp
	time ${nvcc_wrapper} ${kokkos_cu_flags} -o $@ $< ${kokkos_ld_flags} ${kokkos_ld_libs} -Xlinker -rpath=${kokkos_cu_prefix}/lib

clean:
	-rm -f *.${host_arch}
	-rm -f *.fatbin
	-rm -rf *-cfg-tmp
	-rm -f *.bc
	-rm -f *.ll *.ptx *.csv *.log *.fbin *.s
	
