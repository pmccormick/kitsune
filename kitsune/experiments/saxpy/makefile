include ../experiments.mk

cxx_flags += -I${kitsune_prefix}/include 

targets = saxpy-forall.opencilk.${host_arch}

ifeq ($(KITSUNE_ENABLE_CUDA),TRUE)
  targets += saxpy-forall.cuda.${host_arch}
  targets += saxpy-kokkos.nvcc.${host_arch}
  targets += saxpy-kokkos.clang.${host_arch}
endif
ifeq ($(KITSUNE_ENABLE_HIP),TRUE)
  targets += saxpy-forall.hip.${host_arch}
endif

all: ${targets}


#############################
# Kitsune+Tapir versions.
saxpy-forall.opencilk.${host_arch}: saxpy-forall.cpp 
	/usr/bin/time ${clangxx} ${cxx_flags} ${kitflags} ${tapir_opencilk_flags} -o $@ $< \
	-Xlinker -rpath=${kitsune_prefix}/lib
saxpy-forall.cuda.${host_arch}: saxpy-forall.cpp
	/usr/bin/time ${clangxx} ${cxx_flags} ${kitflags} ${tapir_cu_flags} -o $@ $< -Xlinker -rpath=${kitsune_prefix}/lib
saxpy-forall.hip.${host_arch}: saxpy-forall.cpp
	/usr/bin/time ${clangxx} ${cxx_flags} ${kitflags} ${tapir_hip_flags} -o $@ $< -Xlinker -rpath=${kitsune_prefix}/lib
#
#############################


#############################
# Kokkos versions w/ both nvcc and clang for comparison. 
saxpy-kokkos.nvcc.${host_arch}: saxpy-kokkos.cpp
	/usr/bin/time ${nvcc_wrapper} ${kokkos_nvcc_flags} -I${kitsune_prefix}/include -o $@ $< ${kokkos_ld_flags} ${kokkos_ld_libs} -Xlinker -rpath=${kokkos_cu_prefix}/lib 
saxpy-kokkos.clang.${host_arch}: saxpy-kokkos.cpp
	/usr/bin/time ${clangxx} ${cxx_flags} ${clang_cu_flags} ${kokkos_cu_flags} -I${kitsune_prefix}/include -o $@ $< ${kokkos_ld_flags} ${kokkos_ld_libs} -Xlinker -rpath=${kokkos_cu_prefix}/lib 
#
#############################

clean:
	-rm -f *.${host_arch}
	-rm -f *.fbin *.s *.ptx *.ll 
	-rm -rf *-cfg-tmp
	-rm -f *.bc
	-rm -f *.fatbin
	-rm -f *.csv *.log *.s 
