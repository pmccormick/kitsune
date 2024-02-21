include ../experiments.mk 

targets = saxpy-forall.opencilk.${host_arch}

ifeq ($(BUILD_CUDA_EXPERIMENTS),true)
  targets += saxpy-forall.cuda.${host_arch}
endif

ifeq ($(BUILD_HIP_EXPERIMENTS),true)
  targets += saxpy-forall.hip.${host_arch}
endif

all: ${targets}

# forall-based tests
saxpy-forall.opencilk.${host_arch}: saxpy-forall.cpp 
	@echo $@
	@$(TIME_CMD) $(KIT_CXX) $(TAPIR_OPENCILK_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
	@$(FILE_SIZE)
saxpy-forall.cuda.${host_arch}: saxpy-forall.cpp 
	@echo $@
	@$(TIME_CMD)  $(KIT_CXX) -ftime-trace=. $(TAPIR_CUDA_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
	@$(FILE_SIZE)
saxpy-forall.hip.${host_arch}: saxpy-forall.cpp 
	@echo $@
	@$(TIME_CMD) $(KIT_CXX) $(TAPIR_HIP_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
	@$(FILE_SIZE)
# kokkos-based tests (w/out views)
#saxpy-kokkos.cuda.kitsune.${host_arch}: saxpy-kokkos-no-view.cpp 
#	/usr/bin/time $(KIT_CXX) $(TAPIR_CUDA_FLAGS) $(KITSUNE_KOKKOS_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
#saxpy-kokkos.hip.kitsune.${host_arch}: saxpy-kokkos-no-view.cpp 
#	/usr/bin/time $(KIT_CXX) $(TAPIR_HIP_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib

# kokkos-based tests (using views -- currently not compatible w/ kitsune)
saxpy-kokkos.nvcc.${host_arch}: saxpy-kokkos.cpp 
	@echo $@
	@$(TIME_CMD) $(KOKKOS_NVCC) $(KOKKOS_NVCC_FLAGS) -o $@ $< $(KOKKOS_CUDA_LIBS) -Xlinker -rpath=$(KOKKOS_CUDA_PREFIX)/lib64
	@$(FILE_SIZE)

saxpy-kokkos.clang.${host_arch}: saxpy-kokkos.cpp 
	@echo $@
	@$(TIME_CMD) $(CLANG_CUDA) $(KOKKOS_CLANG_CUDA_FLAGS) -ftime-trace=. -fno-exceptions -o $@ $< $(KOKKOS_CLANG_CUDA_LIBS) -Xlinker -rpath=$(KOKKOS_CUDA_PREFIX)/lib64
	@$(FILE_SIZE)

saxpy-kokkos.hipcc.${host_arch}: saxpy-kokkos.cpp 
	@echo $@
	@$(TIME_CMD) $(KOKKOS_HIPCC) $(KOKKOS_HIP_FLAGS) -o $@ $< $(KOKKOS_HIP_LIBS) -Xlinker -rpath=$(KOKKOS_HIP_PREFIX)/lib64 
	@$(FILE_SIZE)
#####
define newline 


endef 

run: $(targets) 
	@echo "running generated executables..."
	$(foreach prog,$(targets),$(call RUN_test,$(prog)))

launch_bench: saxpy-forall.cuda.${host_arch}
	../launch-bench.sh $<
##### 

.PHONY: all clean run

clean:
	-rm -f *.${host_arch} *.ll *.o *.json
	-rm -f *~ core *.log  

