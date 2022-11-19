include ../experiments.mk

cxx_flags += -I${kitsune_prefix}/include 

ifeq ($(KITSUNE_ENABLE_CUDA),TRUE)
  targets += euler3d-forall.cuda.${host_arch}
  targets += euler3d-kokkos.cuda.${host_arch}
endif 

ifeq ($(KITSUNE_ENABLE_HIP), TRUE)
  targets += euler3d-forall.hip.${host_arch}
endif 

VPATH := ./forall/:./kokkos/

all: ${targets} 

forall_lto_src = forall/compute_flux.cpp \
   forall/compute_step_factor.cpp \
   forall/time_step.cpp 

forall_src = forall/euler3d-forall.cpp 

forall_lto_objs := $(patsubst %.cpp, forall/%.lto.o, $(notdir $(forall_lto_src)))
forall_objs := $(patsubst %.cpp, forall/%.o, $(notdir $(forall_src)))
forall/%.lto.o: forall/%.cpp 
	/usr/bin/time ${clangxx} -c -flto ${cxx_flags} ${kitflags} ${tapir_cu_flags} -o $@ $<
forall/%.o: forall/%.cpp
	/usr/bin/time ${clangxx} -c ${cxx_flags} ${kitflags} ${tapir_cu_flags} -o $@ $<

euler3d-forall.cuda.${host_arch}:  ${forall_lto_objs} ${forall_objs}
	/usr/bin/time ${clangxx} -flto -fuse-ld=lld -ftapir=cuda \
	${tapir_cu_lto_flags} -o $@ ${forall_lto_objs} ${forall_objs} ${link_flags} \
	-Xlinker -rpath=${kitsune_prefix}/lib

kokkos_src = kokkos/compute_flux.cpp \
   kokkos/compute_step_factor.cpp \
   kokkos/time_step.cpp \
   kokkos/euler3d-kokkos.cpp
kokkos_objs := $(patsubst %.cpp, kokkos/%.o, $(notdir $(kokkos_src)))
kokkos/%.o: kokkos/%.cpp 
	/usr/bin/time ${clangxx} -c ${cxx_flags} ${clang_cu_flags} ${kokkos_cu_flags} \
		-I${kitsune_prefix}/include -o $@ $<

euler3d-kokkos.cuda.${host_arch}: ${kokkos_objs}
	/usr/bin/time ${clangxx} ${link_flags} -o $@ ${kokkos_objs} ${kokkos_ld_flags} ${kokkos_ld_libs} \
		-Xlinker -rpath=${kokkos_cu_prefix}/lib

clean:
	-rm -f *.${host_arch} *.o forall/*.o  kokkos/*.o 
	-rm -f *.fatbin
	-rm -rf *-cfg-tmp
	-rm -f *.bc
	-rm -f *.fatbin
	-rm -f *.ppm *.jpg
	-rm -f *.ll *.ptx *.csv *.log *.s *.fbin *.tapir
	-rm -f *.dat
