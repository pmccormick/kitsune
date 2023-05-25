include ../experiments.mk


# General (shared) flags (see common.mk)
clang_flags=${cxx_flags} ${opt_flags} ${clang_info_flags}


#### Tapir parameters 
COARSEN_FACTOR?=1
GRAINSIZE?=1
THREADS_PER_BLOCK?=256
BLOCKS_PER_GRID?=0

# $(info coarsen factor    : ${COARSEN_FACTOR})
# $(info grainsize         : ${GRAINSIZE})
# $(info blocks per grid   : ${BLOCKS_PER_GRID})
# $(info threads per block : ${THREADS_PER_BLOCK})

#### Kitsune+Tapir options 
#  Notes:
#   -ftapir=cuda will target the CUDA ABI transform. 
#   -cuabi-verbose dumps kernel stats via ptxas output. 
#   -cuabi-keep-files saves intermediate files (.ll, .ptx, .fatbin) on disk for a closer look at codegen details. 
# a dirty way to let the preprocessor know we are using Kitsune (-DKITSUNE)
tapir_cu_compile_flags=-D_KITSUNE_ -I${CUDA_HOME}/include -mllvm -cuabi-arch=${NVARCH} 
#  -mllvm -cuabi-verbose
tapir_cu_link_flags= -L${CUDA_HOME}/targets/x86_64-linux/lib -lnvToolsExt -Xlinker -rpath=${kitsune_prefix}/lib
#tapir_cu_launch_flags=-mllvm -cuabi-threads-per-block=${THREADS_PER_BLOCK} -mllvm -cuabi-blocks-per-grid=${BLOCKS_PER_GRID}
#tapir_cu_flags=-ffp-contract=fast -ftapir=cuda -mllvm -cuabi-verbose -mllvm -cuabi-arch=${NVARCH} ${tapir_cu_launch_flags}
tapir_cu_flags=-ffast-math -ftapir=opencilk 
tapir_stripmine_flags=-mllvm -stripmine-count=${GRAINSIZE} -mllvm -stripmine-coarsen-factor=${COARSEN_FACTOR}  -mllvm -cuabi-default-grainsize=${GRAINSIZE}
#
####

#### CUDA flags
# nvcc_flags =--std c++17 -m64 -arch ${NVARCH} --resource-usage  --optimize 0 -g --no-exceptions -I${kitsune_prefix}/include -I${CUDA_ROOT}/include -L${CUDA_ROOT}/lib -lnvToolsExt
nvcc_flags =--std c++17 -m64 -arch ${NVARCH} --resource-usage  --optimize 3 --no-exceptions -I${kitsune_prefix}/include -I${CUDA_ROOT}/include -L${CUDA_ROOT}/lib -lnvToolsExt
clang_cuda_flags = -xcuda -- cuda-gpu-arch=${NVARCH} ${cxx_flags} ${opt_flags} -I${kitsune_prefix}/include
####

all: intersect.kitsune \
	 intersect.nvcc

intersect: 
	rm -f intersect.{nvcc,kitsune}*  common.{nvcc.kitsune}* kernels.nvcc*
	make intersect.nvcc intersect.kitsune

# Kitsune-Tapir versions.
intersect.kitsune: intersect.kitsune.o common.kitsune.o 
	${clangxx} ${clang_flags} ${tapir_cu_flags} ${tapir_cu_link_flags} -o $@ $^ 

intersect.kitsune.o: intersect.cpp
	${clangxx} -c ${clang_flags} ${tapir_cu_flags} ${tapir_cu_compile_flags} -o $@ $< 

common.kitsune.o: common.cpp
	${clangxx} -c ${clang_flags} ${tapir_cu_flags} ${tapir_cu_compile_flags} -o $@ $< 

# Cuda versions
intersect.nvcc: intersect.nvcc.o common.nvcc.o kernels.nvcc.o
	${nvcc} ${nvcc_flags} -o $@ $^

intersect.nvcc.o: intersect.cu 
	${nvcc} -c ${nvcc_flags} -o $@ $<

# -x cu allows .cpp files to compile cuda
common.nvcc.o: common.cpp
	${nvcc} -c -x cu ${nvcc_flags} -o $@ $<

kernels.nvcc.o: kernels.cu
	${nvcc} -c ${nvcc_flags} -o $@ $<

# serial_algorithms.nvcc.o: serial_algorithms.cpp
# 	${nvcc} -c ${nvcc_flags} -o $@ $<

clean:
	-rm -f *.${host_arch} *.fatbin *.bc *.o *.{kitsune,nvcc}* core*
	-rm -rf *-cfg-tmp

realclean: clean
	make clean
	rm -f *.ll *.ptx *.csv *.log *report* 

test: all
	./intersect.nvcc
	./intersect.kitsune
# ### Kokkos stuff (not used yet)
# include ../kokkos.mk
# # kitsune kokkos mode flags.
# kit_kokkos_flags=-fkokkos -fkokkos-no-init
# nvcc_kokkos_flags=${nvcc_flags} -I${kokkos_cuda_prefix}/include -I${kitsune_prefix}/include
# kokkos_cu_flags=-I${kokkos_cu_prefix}/include ${nvcc_cxx_flags} -m64 -arch=${NVARCH} --resource-usage -O3 --no-exceptions -I${kitsune_prefix}/include

