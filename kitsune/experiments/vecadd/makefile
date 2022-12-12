include ../experiments.mk

cxx_flags += -I${kitsune_prefix}/include 

targets = vecadd-forall.opencilk.${host_arch}

ifeq ($(KITSUNE_ENABLE_CUDA),TRUE)
  targets += vecadd-forall.cuda.${host_arch}
  targets += vecadd-kokkos.nvcc.${host_arch}
  targets += vecadd-kokkos.clang.${host_arch}
endif
ifeq ($(KITSUNE_ENABLE_HIP),TRUE)
  targets += vecadd-forall.hip.${host_arch}
endif

all: ${targets}


#hip_junk=-idirafter /opt/rocm/include -include __clang_hip_runtime_wrapper.h -isystem /opt/rocm-5.3.0/llvm/lib/clang/15.0.0/include/.. -isystem /opt/rocm/hsa/include -isystem /opt/rocm/hip/include
hip_junk=

#############################
# Kitsune+Tapir versions.
vecadd-forall.opencilk.${host_arch}: vecadd-forall.cpp 
	/usr/bin/time ${clangxx} ${cxx_flags} ${kitflags} ${tapir_opencilk_flags} -o $@ $< \
	-Xlinker -rpath=${kitsune_prefix}/lib
vecadd-forall.cuda.${host_arch}: vecadd-forall.cpp
	/usr/bin/time ${clangxx} ${cxx_flags} ${kitflags} ${tapir_cu_flags} -o $@ $< -Xlinker -rpath=${kitsune_prefix}/lib
vecadd-forall.hip.${host_arch}: vecadd-forall.cpp
	/usr/bin/time ${clangxx} -v -mllvm -debug-only="hip-abi" ${cxx_flags} ${kitflags} ${tapir_hip_flags} ${hip_junk} -o $@ $< -Xlinker -rpath=${kitsune_prefix}/lib -lclang_rt.builtins-x86_64
# 
#############################


#############################
# Cuda version w/ both nvcc and clang for comparison.
vecadd-cuda.nvcc.${host_arch}: vecadd.cu
	/usr/bin/time ${nvcc} -I${kitsune_prefix}/include ${nvcc_cxx_flags} -o $@ $<
vecadd-cuda.clang.${host_arch}: vecadd.cu
	/usr/bin/time ${clangxx} ${clang_cuda_flags} -o $@ $< -L${CUDA_HOME}/lib64 -lcudart_static -ldl -lrt -pthread
# 
#############################


#############################
# Kokkos version w/ both nvcc and clang for comparison.
vecadd-kokkos.nvcc.${host_arch}: vecadd-kokkos.cpp
	/usr/bin/time ${nvcc_wrapper} ${kokkos_nvcc_flags} -I${kitsune_prefix}/include -o $@ $< ${kokkos_ld_flags} ${kokkos_ld_libs} -Xlinker -rpath=${kokkos_cu_prefix}/lib
vecadd-kokkos.clang.${host_arch}: vecadd-kokkos.cpp
	/usr/bin/time ${clangxx} ${cxx_flags} ${clang_cu_flags} ${kokkos_cu_flags} -I${kitsune_prefix}/include -o $@ $< ${kokkos_ld_flags} ${kokkos_ld_libs} -Xlinker -rpath=${kokkos_cu_prefix}/lib
# 
#############################

clean:
	-rm -f *.${host_arch}
	-rm -f *.fatbin
	-rm -rf *-cfg-tmp
	-rm -f *.bc
	-rm -f *.ll *.ptx *.csv *.log *.fbin *.s
	
