include ../common.mk
include ../cuda.mk
include ../kokkos.mk

# General (shared) flags (see common.mk)
clang_flags=${cxx_flags} ${opt_flags} ${clang_info_flags}
# kitsune kokkos mode flags.
kit_kokkos_flags=-fkokkos -fkokkos-no-init


#### Tapir parameters 
COARSEN_FACTOR?=1
GRAINSIZE?=1
THREADS_PER_BLOCK?=256
BLOCKS_PER_GRID?=0

$(info coarsen factor    : ${COARSEN_FACTOR})
$(info grainsize         : ${GRAINSIZE})
$(info blocks per grid   : ${BLOCKS_PER_GRID})
$(info threads per block : ${THREADS_PER_BLOCK})

#### Kitsune+Tapir options 
#  Notes:
#   -ftapir=cuda will target the CUDA ABI transform. 
#   -cuabi-verbose dumps kernel stats via ptxas output. 
#   -cuabi-keep-files saves intermediate files (.ll, .ptx, .fatbin) on disk for a closer look at codegen details. 
#
tapir_cu_launch_flags=-I${CUDA_HOME}/include
#tapir_cu_launch_flags=-mllvm -cuabi-threads-per-block=${THREADS_PER_BLOCK} -mllvm -cuabi-blocks-per-grid=${BLOCKS_PER_GRID}
tapir_cu_flags=-ffast-math -ftapir=cuda -mllvm -cuabi-verbose -mllvm -cuabi-arch=${NVARCH} ${tapir_cu_launch_flags}
tapir_stripmine_flags=-mllvm -stripmine-count=${GRAINSIZE} -mllvm -stripmine-coarsen-factor=${COARSEN_FACTOR}  -mllvm -cuabi-default-grainsize=${GRAINSIZE}
####

#### CUDA flags
nvcc_flags = --std c++17 -m64 -arch ${NVARCH} --resource-usage --optimize 3 --no-exceptions -I${kitsune_prefix}/include
nvcc_kokkos_flags=${nvcc_flags} -I${kokkos_cuda_prefix}/include -I${kitsune_prefix}/include
clang_cuda_flags = -xcuda --cuda-gpu-arch=${NVARCH} ${cxx_flags} ${opt_flags} -I${kitsune_prefix}/include
####

### Kokkos stuff 
kokkos_cu_flags=-I${kokkos_cu_prefix}/include ${nvcc_cxx_flags} -m64 -arch=${NVARCH} --resource-usage -O3 --no-exceptions -I${kitsune_prefix}/include

all: vecadd-forall.cuda.${host_arch} \
     vecadd-forall.cuda.${GRAINSIZE}.${host_arch} \
     vecadd.nvcc.${host_arch} \
     vecadd.clang.${host_arch} \
		 vecadd-kokkos.${host_arch}

# Kitsune-Tapir versions.
vecadd-forall.cuda.${host_arch}: vecadd-forall.cpp
	${clangxx} ${clang_flags} ${tapir_cu_flags} -o $@ $< -Xlinker -rpath=${kitsune_prefix}/lib

vecadd-forall.cuda.${GRAINSIZE}.${host_arch}: vecadd-forall.cpp
	${clangxx} ${clang_flags} ${tapir_cu_flags} ${tapir_stripmine_flags} -o $@ $< -Xlinker -rpath=${kitsune_prefix}/lib

# Cuda version.
vecadd.nvcc.${host_arch}: vecadd.cu
	${nvcc} ${nvcc_flags} -o $@ $<
vecadd.clang.${host_arch}: vecadd.cu
	${clangxx} ${clang_cuda_flags} -o $@ $< -L${CUDA_HOME}/lib64 -lcudart_static -ldl -lrt -pthread

# Kokkos version. 
vecadd-kokkos.${host_arch}: vecadd-kokkos.cpp
	${nvcc_wrapper} ${kokkos_cu_flags} -o $@ $< ${kokkos_ld_flags} ${kokkos_ld_libs} -Xlinker -rpath=${kokkos_cu_prefix}/lib

clean:
	-rm -f *.${host_arch}
	-rm -f *.fatbin
	-rm -rf *-cfg-tmp
	-rm -f *.bc

realclean: clean
	-rm -f *.ll *.ptx *.csv *.log *.fatbin
	
