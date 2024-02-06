include ../experiments.mk 

targets = srad-forall.opencilk.${host_arch}

ifeq ($(BUILD_CUDA_EXPERIMENTS),true)
  targets += srad-forall.cuda.${host_arch}
  targets += srad-kokkos.nvcc.${host_arch}
  targets += srad-kokkos.clang.cuda.${host_arch}
endif

ifeq ($(BUILD_HIP_EXPERIMENTS),true)
  targets += srad-forall.hip.${host_arch}
endif

all: ${targets}

# forall-based tests
srad-forall.opencilk.${host_arch}: srad-forall.cpp 
	@echo $@
	@$(TIME_CMD) $(KIT_CXX) $(TAPIR_OPENCILK_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
	@$(FILE_SIZE)
srad-forall.cuda.${host_arch}: srad-forall.cpp 
	@echo $@
	@$(TIME_CMD) $(KIT_CXX) $(TAPIR_CUDA_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
	@$(FILE_SIZE)
srad-forall.hip.${host_arch}: srad-forall.cpp 
	@echo $@
	@$(TIME_CMD) $(KIT_CXX) $(TAPIR_HIP_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
	@$(FILE_SIZE)

# kokkos-based tests (w/out views)
srad-kokkos.cuda.kitsune.${host_arch}: srad-kokkos-no-view.cpp 
	/usr/bin/time $(KIT_CXX) $(TAPIR_CUDA_FLAGS) $(KITSUNE_KOKKOS_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
#srad-kokkos.hip.kitsune.${host_arch}: srad-kokkos-no-view.cpp 
#	/usr/bin/time $(KIT_CXX) $(TAPIR_HIP_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib

# kokkos-based tests (using views -- currently not compatible w/ kitsune)
srad-kokkos.nvcc.${host_arch}: srad-kokkos.cpp 
	@echo $@
	@$(TIME_CMD) $(KOKKOS_NVCC) $(KITSUNE_FAST_MATH) $(KOKKOS_NVCC_FLAGS) -o $@ $< $(KOKKOS_CUDA_LIBS) -Xlinker -rpath=$(KOKKOS_CUDA_PREFIX)/lib64 
	@$(FILE_SIZE)
srad-kokkos.clang.cuda.${host_arch}: srad-kokkos.cpp 
	@echo $@
	$(TIME_CMD) $(kitclangxx) $(KOKKOS_CLANG_CUDA_FLAGS) -o $@ $< $(KOKKOS_CUDA_LIBS) -Xlinker -rpath=$(KOKKOS_CUDA_PREFIX)/lib64 
	@$(FILE_SIZE)

srad-kokkos.hipcc.${host_arch}: srad-kokkos.cpp 
	@echo $@
	@$(TIME_CMD) $(KOKKOS_HIPCC) $(KOKKOS_HIP_FLAGS) -o $@ $< $(KOKKOS_HIP_LIBS) -Xlinker -rpath=$(KOKKOS_HIP_PREFIX)/lib64 
	@$(FILE_SIZE)
srad-kokkos.clang.hip.${host_arch}: srad-kokkos.cpp 
	@echo $@
	@$(TIME_CMD) $(kitclangxx) -fPIC --language=hip $(KOKKOS_HIP_FLAGS) -o $@ $< $(KOKKOS_HIP_LIBS) $(HIP_LIBS) -Xlinker -rpath=$(KOKKOS_HIP_PREFIX)/lib64
	@$(FILE_SIZE)
#####

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

