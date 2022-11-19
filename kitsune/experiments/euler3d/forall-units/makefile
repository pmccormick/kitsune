include ../../common.mk
include ../../cuda.mk

tapir_flags = -ffp-contract=fast -ftapir=cuda -mllvm -stripmine-count=1 -mllvm -stripmine-coarsen-factor=1 -mllvm -cuabi-default-grainsize=1 -mllvm -cuabi-streams=false -mllvm -cuabi-arch=${NVARCH} -mllvm -cuabi-opt-level=3 -mllvm -cuabi-run-post-opts=false 
# -mllvm -debug-only=cudaabi
clang_flags = ${cxx_flags} ${opt_flags} ${clang_info_flags} -I${CUDA_HOME}/include ${tapir_flags}

link_flags=-ftapir=cuda 

objs = compute_flux.o compute_step_factor.o euler3d-forall.o time_step.o

euler3d-forall.cuda.${host_arch}: ${objs}
	/usr/bin/time ${clangxx} ${link_flags} ${opt_flags} -o $@ ${objs} -Xlinker -rpath=${kitsune_prefix}/lib

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
