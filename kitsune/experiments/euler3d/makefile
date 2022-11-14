include ../experiments.mk 

clangxx_flags=${clangxx_flags} ${opt_flags}

all: euler3d.${host_arch} \
     euler3d-forall.cuda.${host_arch} \
     euler3d-forall.hip.${host_arch} 

# Serial version.
euler3d.${host_arch}: euler3d.cpp
	/usr/bin/time ${clangxx} ${clangxx_flags} -o $@ $<

# Kitsune forall versions. 
euler3d-forall.cuda.${host_arch}: euler-forall.cpp
	/usr/bin/time ${clangxx} ${clangxx_flags} ${tapir_cu_flags} -o $@ $< -Xlinker -rpath=${kitsune_prefix}/lib
euler3d-forall.hip.${host_arch}: euler-forall.cpp
	/usr/bin/time ${clangxx} ${clangxx_flags} ${tapir_hip_flags} -o $@ $< -Xlinker -rpath=${kitsune_prefix}/lib
euler3d-forall.cheetah.${host_arch}: euler-forall.cpp
	/usr/bin/time ${clangxx} ${clangxx_flags} ${tapir_cheetah_flags} -o $@ $< -Xlinker -rpath=${kitsune_prefix}/lib

clean:
	-rm -f *.${host_arch}
	-rm -f *.fatbin
	-rm -rf *-cfg-tmp
	-rm -f *.bc
	-rm -f *.fatbin
	-rm -f *.ppm *.jpg
	-rm -f *.ll *.ptx *.csv *.log *.s *.fbin *.tapir
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
