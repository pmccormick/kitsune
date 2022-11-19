include ../experiments.mk

cxx_flags += -I${kitsune_prefix}/include 

targets = copy-forall.opencilk.${host_arch}

ifeq ($(KITSUNE_ENABLE_CUDA),TRUE)
  targets += copy-forall.cuda.${host_arch}
  targets += copy-kokkos.nvcc.${host_arch}
  targets += copy-kokkos.clang.${host_arch}
endif
ifeq ($(KITSUNE_ENABLE_HIP),TRUE)
  targets += copy-forall.hip.${host_arch}
endif

all: ${targets}

#############################
# Kitsune+Tapir versions.
copy-forall.opencilk.${host_arch}: copy-forall.cpp 
	/usr/bin/time ${clangxx} ${cxx_flags} ${kitflags} ${tapir_opencilk_flags} -o $@ $< -Xlinker -rpath=${kitsune_prefix}/lib
copy-forall.cuda.${host_arch}: copy-forall.cpp
	/usr/bin/time ${clangxx} ${cxx_flags} ${kitflags} ${tapir_cu_flags} -o $@ $< -Xlinker -rpath=${kitsune_prefix}/lib
copy-forall.hip.${host_arch}: copy-forall.cpp
	/usr/bin/time ${clangxx} -mllvm -debug-only=hipabi ${cxx_flags} ${kitflags} ${tapir_hip_flags} -o $@ $< -Xlinker -rpath=${kitsune_prefix}/lib
#############################


#############################
# Kokkos version. 
copy-kokkos.nvcc.${host_arch}: copy-kokkos.cpp
	/usr/bin/time ${nvcc_wrapper} ${kokkos_nvcc_flags} -I${kitsune_prefix}/include -o $@ $< ${kokkos_ld_flags} ${kokkos_ld_libs} -Xlinker -rpath=${kokkos_cu_prefix}/lib
copy-kokkos.clang.${host_arch}: copy-kokkos.cpp
	/usr/bin/time ${clangxx} ${cxx_flags} ${clang_cu_flags} ${kokkos_cu_flags} -I${kitsune_prefix}/include -o $@ $< ${kokkos_ld_flags} ${kokkos_ld_libs} -Xlinker -rpath=${kokkos_cu_prefix}/lib
#
#############################


clean:
	-rm -f *.${host_arch}
	-rm -f *.fatbin
	-rm -rf *-cfg-tmp
	-rm -f *.bc
	-rm -f *.ll *.ptx *.csv *.log *.fbin *.s
	
