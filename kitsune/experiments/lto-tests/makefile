include ../experiments.mk

targets += vecadd.cuda.${host_arch}

VPATH := ./

forall_lto_src = vecadd-init.cpp vecadd-add.cpp vecadd-forall.cpp

all: ${targets}

#############################
# cuda target
# 
forall_cuda_lto_objs := $(patsubst %.cpp, %.cuda.lto.o, $(notdir $(forall_lto_src)))

%.cuda.lto.o: %.cpp 
	${KIT_CXX} -ftapir=cuda -c -flto -O1 -o $@ $<
#	${KITSUNE_PREFIX}/bin/llvm-dis -o $@.ll $@ 

vecadd.cuda.${host_arch}: ${forall_cuda_lto_objs}
	${KIT_CXX} -flto -fuse-ld=lld -ftapir=cuda ${TAPIR_CUDA_LTO_FLAGS} -o $@ ${forall_cuda_lto_objs} \
	-Xlinker -rpath=${KITSUNE_PREFIX}/lib
	cuobjdump -symbols $@

clean:
	-rm -f *.${host_arch} *.o *.lto.o
	-rm -f *.fatbin
	-rm -rf *-cfg-tmp
	-rm -f *.bc
	-rm -f *.fatbin
	-rm -f *.ppm *.jpg
	-rm -f *.ll *.ptx *.csv *.log *.s *.fbin *.tapir
	-rm -f *.dat
