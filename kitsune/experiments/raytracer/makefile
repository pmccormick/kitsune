include ../common.mk
include ../cuda.mk
include ../kokkos.mk
include ../gcc.mk 

# General (shared) flags (see common.mk)
clang_flags=${cxx_flags} ${opt_flags} ${clang_info_flags}
# kitsune kokkos mode flags.
kit_kokkos_flags=-fkokkos -fkokkos-no-init

#### Kitsune+Tapir options 
#  Notes:
#   -ftapir=cuda will target the CUDA ABI transform. 
#   -cuabi-verbose dumps kernel stats via ptxas output. 
#   -cuabi-keep-files saves intermediate files (.ll, .ptx, .fatbin) on disk for a closer look at codegen details. 
#
tapir_cu_launch_flags=-I${CUDA_HOME}/include
tapir_cu_flags=-ftapir=cuda -ffp-contract=fast -mllvm -cuabi-verbose -mllvm -cuabi-arch=${NVARCH} ${tapir_cu_launch_flags}
tapir_stripmine_flags=-mllvm -stripmine-count=1 -mllvm -stripmine-coarsen-factor=1 -mllvm -cuabi-default-grainsize=1
####

#### CUDA flags
nvcc_flags=--std c++17 -m64 -arch=${NVARCH} --resource-usage -O3 --no-exceptions -I${kitsune_prefix}/include
clang_cuda_flags=-xcuda --cuda-gpu-arch=${NVARCH} ${cxx_flags} ${opt_flags} -I${kitsune_prefix}/include
####

#### Kokkos stuff 
kokkos_cu_flags=-I${kokkos_cu_prefix}/include ${nvcc_cxx_flags} -m64 -arch=${NVARCH} --resource-usage -O3 --no-exceptions -I${kitsune_prefix}/include


all: raytrace-forall.cuda.${host_arch} \
     raytrace.nvcc.${host_arch} \
     raytrace.clang.${host_arch} \
     raytrace.kokkos.${host_arch} \
     raytrace.kokkos.noview.${host_arch} \
     raytrace.openmp.clang.${host_arch} 
#     raytrace.openmp.gcc.${host_arch} 


# Kitsune-Tapir versions.
raytrace-forall.cuda.${host_arch}: raytrace-forall.cpp
	${clangxx} ${clang_flags} ${tapir_cu_flags} ${tapir_stripmine_flags} -o $@ $< -Xlinker -rpath=${kitsune_prefix}/lib
raytrace.kokkos.noview.${host_arch}: raytrace-kokkos-noview.cpp
	${clangxx} ${clang_flags} ${tapir_cu_flags} ${kit_kokkos_flags} ${tapir_stripmine_flags} -o $@ $< -Xlinker -rpath=${kitsune_prefix}/lib


# Cuda version.
raytrace.nvcc.${host_arch}: raytrace.cu
	${nvcc} ${nvcc_flags} -o $@ $<
raytrace.clang.${host_arch}: raytrace.cu
	${clangxx} ${clang_cuda_flags} -o $@ $< -L${CUDA_HOME}/lib64 -lcudart_static -ldl -lrt -pthread
raytrace.kokkos.${host_arch}: raytrace-kokkos.cpp
	${nvcc_wrapper} ${kokkos_cu_flags} -o $@ $< ${kokkos_ld_flags} ${kokkos_ld_libs} -Xlinker -rpath=${kokkos_cu_prefix}/lib

# OpenMP version.
raytrace.openmp.clang.${host_arch}: raytrace-openmp.cpp
	${clangxx} ${clang_flags} -fopenmp -fopenmp-targets=nvptx64-cuda -Xopenmp-target -march=${NVARCH} -I${kitsune_prefix}/include -fopenmp -o $@ $<
#raytrace.openmp.gcc.${host_arch}: raytrace-openmp.cpp
#	${gccxx} ${gcc_flags} -fopenmp -foffload=nvptx-none="-O3 -lm" -o $@ $<

clean:
	-rm -f *.${host_arch}
	-rm -f *.fatbin
	-rm -rf *-cfg-tmp
	-rm -f *.bc
	-rm -f *.fatbin
	-rm -f *.ppm *.jpg 
	-rm -f *.ll *.ptx *.csv *.log *.s *.fbin
