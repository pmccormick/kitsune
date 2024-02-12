include ../experiments.mk

cxx_flags += -I${kitsune_prefix}/include 

ifeq ($(KITSUNE_ENABLE_CUDA),TRUE)
  targets += euler3d-forall.cuda.${host_arch}
endif 

ifeq ($(KITSUNE_ENABLE_HIP), TRUE)
  targets += euler3d-forall.hip.${host_arch}
endif 

VPATH := ./forall/

all: ${targets} 

forall_lto_src = forall/compute_flux.cpp \
   forall/compute_step_factor.cpp \
   forall/time_step.cpp 

forall_src = forall/euler3d-forall.cpp 

forall_lto_objs := $(patsubst %.cpp, forall/%.lto.o, $(notdir $(forall_lto_src)))
forall_objs := $(patsubst %.cpp, forall/%.o, $(notdir $(forall_src)))
forall/%.lto.o: forall/%.cpp 
	${KIT_CXX} -c -flto ${TAPIR_CUDA_FLAGS} -o $@ $<
forall/%.o: forall/%.cpp
	${KIT_CXX} -c ${TAPIR_CUDA_FLAGS} -o $@ $<

euler3d-forall.cuda.${host_arch}:  ${forall_lto_objs} ${forall_objs}
	${KIT_CXX} -flto -fuse-ld=lld -ftapir=cuda \
	${tapir_cu_lto_flags} -o $@ ${forall_lto_objs} ${forall_objs} ${link_flags} \
	-Xlinker -rpath=${kitsune_prefix}/lib

clean:
	-rm -f *.${host_arch} *.o forall/*.o  kokkos/*.o 
	-rm -f *.fatbin
	-rm -rf *-cfg-tmp
	-rm -f *.bc
	-rm -f *.fatbin
	-rm -f *.ppm *.jpg
	-rm -f *.ll *.ptx *.csv *.log *.s *.fbin *.tapir
	-rm -f *.dat
