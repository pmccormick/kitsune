include ../experiments.mk 

cxx_flags += -I${kitsune_prefix}/include  -fPIC

targets = struct-args.opencilk.${host_arch}

ifeq ($(KITSUNE_ENABLE_CUDA), TRUE)
  targets += struct-args.cuda.${host_arch}
  targets += struct-args-kokkos.nvcc.${host_arch}
  targets += struct-args-kokkos.clang.${host_arch}
endif 

ifeq ($(KITSUNE_ENABLE_HIP), TRUE)
  targets += struct-args.hip.${host_arch}
endif 

all: ${targets} 

#############################
# Kitsune+Tapir versions.
#
struct-args.opencilk.${host_arch}: struct-args.cpp
	/usr/bin/time ${clangxx} ${cxx_flags} ${kitflags} ${tapir_opencilk_flags} -o $@ $< -Xlinker -rpath=${kitsune_prefix}/lib
struct-args.cuda.${host_arch}: struct-args.cpp
	/usr/bin/time ${clangxx} ${cxx_flags} ${kitflags} ${tapir_cu_flags} -o $@ $< -Xlinker -rpath=${kitsune_prefix}/lib
struct-args.hip.${host_arch}: struct-args.cpp
	/usr/bin/time ${clangxx} -mllvm -debug-only=hipabi ${cxx_flags} ${kitflags} ${tapir_hip_flags} -o $@ $< -Xlinker -rpath=${kitsune_prefix}/lib
#
#############################

clean:
	-rm -f *.${host_arch}
	-rm -f *.fatbin *.hipfb
	-rm -rf *-cfg-tmp
	-rm -f *.bc
	-rm -f *.o *.a *.so
	-rm -f *.ppm *.jpg
	-rm -f *.ll *.ptx *.csv *.log *.s *.fbin *.tapir

