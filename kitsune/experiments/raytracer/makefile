include ../experiments.mk 

cxx_flags += -Xclang -no-opaque-pointers -I${kitsune_prefix}/include
opt_flags += -ffp-contract=fast

targets = raytracer-forall.opencilk.${host_arch}

ifeq ($(KITSUNE_ENABLE_CUDA), TRUE)
  targets += raytracer-forall.cuda.${host_arch}
  targets += raytracer-kokkos.nvcc.${host_arch}
  targets += raytracer-kokkos.clang.${host_arch}
endif 

ifeq ($(KITSUNE_ENABLE_HIP), TRUE)
  targets += raytracer-forall.hip.${host_arch}
  targets += raytracer-kokkos.hipcc.${host_arch}
  targets += raytracer-kokkos.clang.${host_arch}
  targets += raytracer-hip.hipcc.${host_arch}
  targets += raytracer-hip.clang.${host_arch}
endif 

all: ${targets} 

#############################
# Kitsune+Tapir versions.
#
raytracer-forall.opencilk.${host_arch}: raytracer-forall.cpp
	/usr/bin/time ${clangxx} ${cxx_flags} ${kitflags} ${tapir_opencilk_flags} -o $@ $< -Xlinker -rpath=${kitsune_prefix}/lib
raytracer-forall.cuda.${host_arch}: raytracer-forall.cpp
	/usr/bin/time ${clangxx} ${cxx_flags} ${kitflags} ${tapir_cu_flags} -o $@ $< -Xlinker -rpath=${kitsune_prefix}/lib
raytracer-forall.hip.${host_arch}: raytracer-forall.cpp
	/usr/bin/time ${clangxx} ${cxx_flags} ${kitflags} ${tapir_hip_flags} ${opt_flags} -o $@ $< -Xlinker -rpath=${kitsune_prefix}/lib 
#
#############################


#############################
# Kokkos versions 
raytracer-kokkos.nvcc.${host_arch}: raytracer-kokkos.cpp
	/usr/bin/time ${nvcc_wrapper} ${kokkos_nvcc_flags} -O${opt_level} -I${kitsune_prefix}/include -o $@ $< ${kokkos_ld_flags} ${kokkos_ld_libs} -Xlinker -rpath=${kokkos_cu_prefix}/lib
raytracer-kokkos.clang.${host_arch}: raytracer-kokkos.cpp
	/usr/bin/time ${clangxx} ${cxx_flags} ${clang_cu_flags} ${kokkos_cu_flags} -I${kitsune_prefix}/include -o $@ $< ${kokkos_ld_flags} ${kokkos_ld_libs} -Xlinker -rpath=${kokkos_cu_prefix}/lib

raytracer-kokkos.hipcc.${host_arch}: raytracer-kokkos.cpp
	/usr/bin/time ${hipcc} -v ${cxx_flags} ${opt_flags} ${kokkos_hip_flags} -I${kitsune_prefix}/include -o $@ $< ${kokkos_hip_ld_flags} ${kokkos_libs} -Xlinker -rpath=${kokkos_hip_prefix}/lib64
raytracer-kokkos.clang.${host_arch}: raytracer-kokkos.cpp
	/usr/bin/time ${clangxx} ${clang_hip_flags} ${cxx_flags} ${opt_flags} ${kokkos_hip_flags} -I${kitsune_prefix}/include -o $@ $< ${kokkos_hip_ld_flags} ${kokkos_libs} ${hip_ld_flags} ${hip_libs} -Xlinker -rpath=${kokkos_hip_prefix}/lib64

# 
#############################

raytracer.cuda.${host_arch}: raytracer.cu 
	/usr/bin/time ${nvcc} ${nvcc_cxx_flags} -O${opt_level} -I${kitsune_prefix}/include -o $@ $< ${cuda_ld_flags} ${cuda_libs} 

raytracer-hip.hipcc.${host_arch}: raytracer-hip.cpp
	/usr/bin/time ${hipcc} ${cxx_flags} ${opt_flags} -I${kitsune_prefix}/include -o $@ $<

raytracer-hip.clang.${host_arch}: raytracer-hip.cpp
	/usr/bin/time ${clangxx} ${clang_hip_flags} ${cxx_flags} ${opt_flags} -I${kitsune_prefix}/include -o $@ $< ${hip_ld_flags} ${hip_libs} 

clean:
	-rm -f *.${host_arch}
	-rm -f *.fatbin
	-rm -rf *-cfg-tmp
	-rm -f *.bc
	-rm -f *.o *.so *.a *.hipfb
	-rm -f *.ppm *.jpg
	-rm -f *.ll *.ptx *.csv *.log *.s *.fbin *.tapir
	-rm -f *.dat

#### CUDA flags
#nvcc_flags=--std c++17 -m64 -arch=${NVARCH} --resource-usage ${opt_flags} --no-exceptions -I${kitsune_prefix}/include
#nvcc_kokkos_flags=${nvcc_flags} -I${kokkos_cuda_prefix}/include -I${kitsune_prefix}/include
#clang_cuda_flags=-xcuda --cuda-gpu-arch=${NVARCH} ${cxx_flags} ${opt_flags} -I${kitsune_prefix}/include
####

