include ../experiments.mk

cxx_flags += -I${kitsune_prefix}/include 
targets = euler3d-forall.opencilk.${host_arch}
ifeq ($(KITSUNE_ENABLE_CUDA),TRUE)
  targets += euler3d-forall.cuda.${host_arch}
endif 

VPATH := ./forall/


forall_lto_src = forall/compute_flux.cpp \
   forall/compute_step_factor.cpp \
   forall/time_step.cpp \
   forall/euler3d-forall.cpp 

forall_src = forall/euler3d-forall.cpp 

all: ${targets}


#############################
# opencilk target
#
forall_opencilk_lto_objs := $(patsubst %.cpp, forall/%.opencilk.lto.o, $(notdir $(forall_lto_src)))
forall_opencilk_objs := $(patsubst %.cpp, forall/%.opencilk.o, $(notdir $(forall_src)))

forall/%.opencilk.lto.o: forall/%.cpp 
	${KIT_CXX} -c -flto ${TAPIR_OPENCILK_FLAGS} -o $@ $<
forall/%.opencilk.o: forall/%.cpp
	${KIT_CXX} -c ${TAPIR_OPENCILK_FLAGS} -o $@ $<
euler3d-forall.opencilk.${host_arch}: ${forall_opencilk_lto_objs} ${forall_opencilk_objs}
	${KIT_CXX} -flto -fuse-ld=lld -ftapir=opencilk \
	${TAPIR_OPENCILK_LTO_FLAGS} -o $@ ${forall_opencilk_lto_objs} ${forall_opencilk_objs} ${link_flags} \
	-Xlinker -rpath=${KITSUNE_PREFIX}/lib

#############################
# cuda target
# 
forall_cuda_lto_objs := $(patsubst %.cpp, forall/%.cuda.lto.o, $(notdir $(forall_lto_src)))
forall_cuda_objs := $(patsubst %.cpp, forall/%.cuda.o, $(notdir $(forall_src)))

forall/%.cuda.lto.o: forall/%.cpp 
	${KIT_CXX} -c -flto -ftapir=cuda -O${KITSUNE_OPTLEVEL} -o $@ $<
euler3d-forall.cuda.${host_arch}: ${forall_cuda_lto_objs}
	${KIT_CXX} -flto -fuse-ld=lld -ftapir=cuda ${TAPIR_CUDA_LTO_FLAGS} -o $@ ${forall_cuda_lto_objs} \
	-Xlinker -rpath=${KITSUNE_PREFIX}/lib

clean:
	-rm -f *.${host_arch} *.o forall/*.o  kokkos/*.o 
	-rm -f *.fatbin
	-rm -rf *-cfg-tmp
	-rm -f *.bc
	-rm -f *.fatbin
	-rm -f *.ppm *.jpg
	-rm -f *.ll *.ptx *.csv *.log *.s *.fbin *.tapir
	-rm -f *.dat
