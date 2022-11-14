include ../../common.mk
include ../../cuda.mk
include ../../kokkos.mk

kokkos_clang_flags=${clang_cuda_flags} -I${kokkos_cu_prefix}/include
clang_flags=${cxx_flags} ${opt_flags} ${clang_info_flags} ${kokkos_clang_flags}
clang_cuda_flags=-xcuda --cuda-gpu-arch=${NVARCH}

link_flags = 
objs = compute_flux.o compute_step_factor.o euler3d-kokkos.o time_step.o

euler3d-kokkos.clang.${host_arch}: ${objs}
	/usr/bin/time ${clangxx} ${link_flags} -o $@ ${objs} ${kokkos_ld_flags} ${kokkos_ld_libs} -Xlinker -rpath=${kokkos_cu_prefix}/lib 

%.o: %.cpp
	/usr/bin/time $(clangxx) ${clang_flags} -c $< -o $@

clean:
	-rm -f *.${host_arch} *.o
	-rm -f *.fatbin
	-rm -rf *-cfg-tmp
	-rm -f *.bc
	-rm -f *.fatbin
	-rm -f *.ppm *.jpg
	-rm -f *.ll *.ptx *.csv *.log *.s *.fbin *.tapir
	-rm -f *.dat
