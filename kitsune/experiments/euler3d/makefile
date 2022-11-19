include ../experiments.mk 

cxx_flags += -I${kitsune_prefix}/include -fPIC -fno-vectorize 

targets = euler3d-forall.opencilk.${host_arch}

ifeq ($(KITSUNE_ENABLE_CUDA), TRUE)
  targets += euler3d-forall.cuda.${host_arch}
  targets += euler3d-kokkos.nvcc.${host_arch}
  targets += euler3d-kokkos.clang.${host_arch}
endif 

ifeq ($(KITSUNE_ENABLE_HIP), TRUE)
  targets += euler3d-forall.hip.${host_arch}
  targets += euler3d-kokkos.hip.${host_arch}
endif 

all: ${targets} 


#############################
# Kitsune+Tapir versions.
#
euler3d-forall.opencilk.${host_arch}: euler3d-forall.cpp
	/usr/bin/time ${clangxx} ${cxx_flags} ${kitflags} ${tapir_opencilk_flags} -o $@ $< -Xlinker -rpath=${kitsune_prefix}/lib
euler3d-forall.cuda.${host_arch}: euler3d-forall.cpp
	/usr/bin/time ${clangxx} ${cxx_flags} ${kitflags} ${tapir_cu_flags} -o $@ $< -Xlinker -rpath=${kitsune_prefix}/lib
euler3d-forall.hip.${host_arch}: euler3d-forall.cpp
	/usr/bin/time ${clangxx} ${cxx_flags} ${kitflags} ${tapir_hip_flags} -o $@ $< -Xlinker -rpath=${kitsune_prefix}/lib
#
#############################


#############################
# Kokkos versions w/ both nvcc and clang for comparison.
euler3d-kokkos.nvcc.${host_arch}: euler3d-kokkos.cpp
	/usr/bin/time ${nvcc_wrapper} ${kokkos_nvcc_flags} -I${kitsune_prefix}/include -o $@ $< ${kokkos_ld_flags} ${kokkos_ld_libs} -Xlinker -rpath=${kokkos_cu_prefix}/lib
euler3d-kokkos.clang.${host_arch}: euler3d-kokkos.cpp
	/usr/bin/time ${clangxx} ${cxx_flags} ${clang_cu_flags} ${kokkos_cu_flags} -I${kitsune_prefix}/include -o $@ $< ${kokkos_ld_flags} ${kokkos_ld_libs} -Xlinker -rpath=${kokkos_cu_prefix}/lib
euler3d-kokkos.clang.${host_arch}: euler3d-kokkos.cpp
	/usr/bin/time ${clangxx} ${cxx_flags} ${clang_hip_flags} ${kokkos_hip_flags} -I${kitsune_prefix}/include -o $@ $< ${kokkos_ld_flags} ${kokkos_ld_libs} -Xlinker -rpath=${kokkos_cu_prefix}/lib
# 
#############################

clean:
	-rm -f *.${host_arch}
	-rm -f *.fatbin
	-rm -rf *-cfg-tmp
	-rm -f *.bc
	-rm -f *.fatbin
	-rm -f *.ppm *.jpg
	-rm -f *.ll *.ptx *.csv *.log *.s *.fbin *.tapir *.o 
	-rm -f *.dat

#### CUDA flags
#nvcc_flags=--std c++17 -m64 -arch=${NVARCH} --resource-usage ${opt_flags} --no-exceptions -I${kitsune_prefix}/include
#nvcc_kokkos_flags=${nvcc_flags} -I${kokkos_cuda_prefix}/include -I${kitsune_prefix}/include
#clang_cuda_flags=-xcuda --cuda-gpu-arch=${NVARCH} ${cxx_flags} ${opt_flags} -I${kitsune_prefix}/include
####

#### Kokkos stuff
#kokkos_cu_flags=-I${kokkos_cu_prefix}/include
#kokkos_clang_flags=${clang_cuda_flags} -I${kokkos_cu_prefix}/include
#kokkos_nvcc_flags=-x cu -I${kokkos_cu_prefix}/include ${nvcc_cxx_flags} ${opt_flags} -m64 -arch=${NVARCH} --resource-usage --no-exceptions -I${kitsune_prefix}/include
#all: euler3d.${host_arch} \
#     euler3d-forall.cuda.${host_arch} \
#     euler3d-kokkos.nvcc.${host_arch} \
#     euler3d-kokkos.clang.${host_arch} \
#     euler3d-kokkos-noview.kitsune.${host_arch} \
#     euler3d-kokkos-noview.clang.${host_arch}


#euler3d-kokkos.nvcc.${host_arch}: euler3d-kokkos.cpp
#	 /usr/bin/time ${nvcc} ${kokkos_nvcc_flags} -o $@ $< ${kokkos_ld_flags} ${kokkos_ld_libs} -Xlinker -rpath=${kokkos_cu_prefix}/lib

#euler3d-kokkos.clang.${host_arch}: euler3d-kokkos.cpp
#	 /usr/bin/time ${clangxx} ${clang_cuda_flags} ${kokkos_cu_flags} -o $@ $< ${kokkos_ld_flags} ${kokkos_ld_libs} -Xlinker -rpath=${kokkos_cu_prefix}/lib

#euler3d-kokkos-noview.clang.${host_arch}: euler3d-kokkos-noview.cpp
#	 /usr/bin/time ${clangxx} ${clang_cuda_flags} ${kokkos_cu_flags} -o $@ $< ${kokkos_ld_flags} ${kokkos_ld_libs} -lkitrt -lLLVM -Xlinker -rpath=${kokkos_cu_prefix}/lib
