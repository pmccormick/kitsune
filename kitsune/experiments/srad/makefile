include ../common.mk
include ../cuda.mk
include ../kokkos.mk

# General (shared) flags (see common.mk)
clang_flags=${cxx_flags} ${opt_flags} ${clang_info_flags}
tapir_cu_flags=-ftapir=cuda -ffp-contract=fast -mllvm -cuabi-arch=${NVARCH} -I${CUDA_HOME}/include 
tapir_stripmine_flags=-mllvm -stripmine-count=1 -mllvm -stripmine-coarsen-factor=1
####

kokkos_cu_flags=-I${kokkos_cu_prefix}/include ${nvcc_cxx_flags} -m64 -arch=${NVARCH} --resource-usage -O3 --no-exceptions -I${kitsune_prefix}/include

all: srad.${host_arch} srad-forall.cuda.${host_arch} srad-kokkos.${host_arch}

srad-forall.cuda.${host_arch}: srad-forall.cpp
	${clangxx} -ftime-report ${clang_flags} ${tapir_cu_flags} ${tapir_stripmine_flags} -o $@ $< -Xlinker -rpath=${kitsune_prefix}/lib

srad.${host_arch}: srad.cpp
	${clangxx} -ftime-report ${clang_flags} -o $@ $<

srad-kokkos.${host_arch}: srad-kokkos.cpp
	${nvcc_wrapper} ${kokkos_cu_flags} -o $@ $< ${kokkos_ld_flags} ${kokkos_ld_libs} -Xlinker -rpath=${kokkos_cu_p\
refix}/lib


clean:
	-rm -f *.${host_arch}
	-rm -f *.fbin *.s *.ptx *.ll 
	-rm -rf *-cfg-tmp
	-rm -f *.bc
	-rm -f *.fatbin
	-rm -f *.csv *.log *.s 
