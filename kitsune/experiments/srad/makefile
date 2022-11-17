include ../experiments.mk

cxx_flags += -I${kitsune_prefix}/include 

targets = srad-forall.opencilk.${host_arch}

ifeq ($(KITSUNE_ENABLE_CUDA),TRUE)
  targets += srad-forall.cuda.${host_arch}
  targets += srad-kokkos.nvcc.${host_arch}
  targets += srad-kokkos.clang.${host_arch}
endif
ifeq ($(KITSUNE_ENABLE_HIP),TRUE)
  targets += srad-forall.hip.${host_arch}
endif

all: ${targets}

#############################
# Kitsune+Tapir versions.
srad-forall.opencilk.${host_arch}: srad-forall.cpp 
	/usr/bin/time ${clangxx} ${cxx_flags} ${kitflags} ${tapir_opencilk_flags} -o $@ $< \
	-Xlinker -rpath=${kitsune_prefix}/lib
srad-forall.cuda.${host_arch}: srad-forall.cpp
	/usr/bin/time ${clangxx} ${cxx_flags} ${kitflags} ${tapir_cu_flags} -o $@ $< -Xlinker -rpath=${kitsune_prefix}/lib
srad-forall.hip.${host_arch}: srad-forall.cpp
	/usr/bin/time ${clangxx} ${cxx_flags} ${kitflags} ${tapir_hip_flags} -o $@ $< -Xlinker -rpath=${kitsune_prefix}/lib
#############################


#############################
# Kokkos version w/ both nvcc and clang for comparison. 
srad-kokkos.nvcc.${host_arch}: srad-kokkos.cpp
	        /usr/bin/time ${nvcc_wrapper} ${kokkos_nvcc_flags} -I${kitsune_prefix}/include -o $@ $< ${kokkos_ld_flags} ${kokkos_ld_libs} -Xlinker -rpath=${kokkos_cu_prefix}/lib
srad-kokkos.clang.${host_arch}: srad-kokkos.cpp
	        /usr/bin/time ${clangxx} ${cxx_flags} ${clang_cu_flags} ${kokkos_cu_flags} -I${kitsune_prefix}/include -o $@ $< ${kokkos_ld_flags} ${kokkos_ld_libs} -Xlinker -rpath=${kokkos_cu_prefix}/lib


clean:
	-rm -f *.${host_arch}
	-rm -f *.fbin *.s *.ptx *.ll 
	-rm -rf *-cfg-tmp
	-rm -f *.bc
	-rm -f *.fatbin
	-rm -f *.csv *.log *.s 
