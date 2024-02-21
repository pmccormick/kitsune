include ../experiments.mk

targets = vecadd.opencilk.${host_arch}

ifeq ($(BUILD_CUDA_EXPERIMENTS),true)
  targets += vecadd.cuda.${host_arch}
endif

forall_lto_src = vecadd-init.cpp vecadd-add.cpp vecadd-forall.cpp

all: ${targets}

#############################
# opencilk target
# 
forall_opencilk_lto_objs := $(patsubst %.cpp, %.opencilk.lto.o, $(notdir $(forall_lto_src)))

%.opencilk.lto.o: %.cpp 
	@echo opencilk lto target: $@
	@${TIME_CMD} ${KIT_CXX} -ftapir=opencilk -c -flto ${KITSUNE_OPT_LEVEL} -o $@ $<

vecadd.opencilk.${host_arch}: ${forall_opencilk_lto_objs}
	@echo opencilk lto executable: $@
	@${TIME_CMD} ${KIT_CXX} -flto -fuse-ld=lld ${TAPIR_OPENCILK_LTO_FLAGS} \
                   -o $@ ${forall_opencilk_lto_objs} -Xlinker -rpath=${KITSUNE_PREFIX}/lib
	@$(FILE_SIZE)
	@echo 


#############################
# cuda target
# 
forall_cuda_lto_objs := $(patsubst %.cpp, %.cuda.lto.o, $(notdir $(forall_lto_src)))

%.cuda.lto.o: %.cpp 
	@echo cuda lto target: $@
	@${TIME_CMD} ${KIT_CXX} -ftapir=cuda -c -flto ${KITSUNE_OPT_LEVEL} -o $@ $<

vecadd.cuda.${host_arch}: ${forall_cuda_lto_objs}
	@echo cuda lto executable: $@
	@${TIME_CMD} ${KIT_CXX} -flto -fuse-ld=lld -ftapir=cuda ${TAPIR_CUDA_LTO_FLAGS} \
            -o $@ ${forall_cuda_lto_objs} -Xlinker -rpath=${KITSUNE_PREFIX}/lib
	@$(FILE_SIZE)
	@echo

#####
define newline 


endef

run: $(targets) 
	@echo "running generated executables..."
	@$(foreach prog,$(targets),$(call RUN_test,$(prog)))

launch_bench: copy-forall.cuda.${host_arch}
	../launch-bench.sh $<

#####

clean:
	-rm -f *.${host_arch} *.o *.lto.o
	-rm -f *.fatbin
	-rm -rf *-cfg-tmp
	-rm -f *.bc
	-rm -f *.fatbin
	-rm -f *.ppm *.jpg
	-rm -f *.ll *.ptx *.csv *.log *.s *.fbin *.tapir
	-rm -f *.dat

