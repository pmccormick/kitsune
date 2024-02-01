include ../experiments.mk

targets = noforall.opencilk.${host_arch}

ifeq ($(BUILD_CUDA_EXPERIMENTS),true)
  targets += noforall.cuda.${host_arch}
endif

ifeq ($(BUILD_HIP_EXPERIMENTS),true)
  targets += noforall.hip.${host_arch}
endif

all: ${targets}

noforall.opencilk.${host_arch}: noforall.cpp 
	@echo $@ 
	@${TIME_CMD} $(KIT_CXX) $(TAPIR_OPENCILK_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
	@${FILE_SIZE}
noforall.cuda.${host_arch}: noforall.cpp 
	@echo $@
	@${TIME_CMD} $(KIT_CXX) $(TAPIR_CUDA_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
	@${FILE_SIZE}
noforall.hip.${host_arch}: noforall.cpp 
	@echo $@
	@${TIME_CMD} $(KIT_CXX) $(TAPIR_HIP_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
	@${FILE_SIZE}

run: $(targets) 
	@echo "running generated exeutables..."
	@$(foreach prog,$(targets),$(call RUN_test,$(prog)))

.PHONY: all clean run

launch_bench: 
	@echo "launch benchmark invalid... skipping..."

clean:
	-rm -f *.${host_arch} *.ll *.o
	-rm -f *~ core *.log 

