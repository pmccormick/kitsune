include ../experiments.mk 

ifeq ($(BUILD_CUDA_EXPERIMENTS),true)
  targets += srad-forall.cuda.${host_arch}
  targets += srad-forall-exprs.cuda.${host_arch}
endif

THREADS_PER_BLOCK?=16

all: ${targets}

# forall-based tests
srad-forall.cuda.${host_arch}: srad-forall.cpp 
	@echo $@
	@$(TIME_CMD) $(KIT_CXX) $(TAPIR_CUDA_FLAGS)  -DTHREADS_PER_BLOCK=$(THREADS_PER_BLOCK) \
		-o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
	@$(FILE_SIZE)

srad-forall-exprs.cuda.${host_arch}: srad-forall-exprs.cpp 
	@echo $@
	@$(TIME_CMD) $(KIT_CXX) $(TAPIR_CUDA_FLAGS)  -DTHREADS_PER_BLOCK=$(THREADS_PER_BLOCK) \
		-mllvm --cuabi-occupancy-launches=false -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib

run: $(targets) 
	@echo "running generated executables..."
	$(foreach prog,$(targets),$(call RUN_test,$(prog)))

launch_bench: srad-forall.cuda.${host_arch}
	../launch-bench.sh $<
#####

.PHONY: all clean run launch_bench

clean:
	-rm -f *.${host_arch} *.ll *.o
	-rm -f *~ core *.log 
	-rm -f *.dat

