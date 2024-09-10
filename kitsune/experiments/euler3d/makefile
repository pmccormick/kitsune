include ../experiments.mk 

targets = euler3d-forall.opencilk.${host_arch}

ifeq ($(BUILD_CUDA_EXPERIMENTS),true)
  targets += euler3d-forall.cuda.${host_arch}
  targets += euler3d-kokkos.kitsune.cuda.${host_arch}
endif

ifeq ($(BUILD_HIP_EXPERIMENTS),true)
  targets += euler3d-forall.hip.${host_arch}
  targets += euler3d-kokkos.kitsune.hip.${host_arch}
endif

all: ${targets}

# forall-based tests
euler3d-forall.opencilk.${host_arch}: euler3d-forall.cpp 
	@echo $@
	@$(TIME_CMD) $(KIT_CXX) $(TAPIR_OPENCILK_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
	@$(FILE_SIZE)
euler3d-forall.cuda.${host_arch}: euler3d-forall.cpp 
	@echo $@
	@$(TIME_CMD) $(KIT_CXX) $(TAPIR_CUDA_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
	@$(FILE_SIZE)
euler3d-forall.hip.${host_arch}: euler3d-forall.cpp 
	@echo $@
	@$(TIME_CMD) $(KIT_CXX) $(TAPIR_HIP_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
	@$(FILE_SIZE)

# kokkos-based tests (w/out views)
euler3d-kokkos.kitsune.cuda.${host_arch}: euler3d-kokkos-no-view.cpp 
	@echo $@
	@$(TIME_CMD) $(KIT_CXX) $(TAPIR_CUDA_FLAGS) $(KITSUNE_KOKKOS_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
	@$(FILE_SIZE)
euler3d-kokkos.kitsune.hip.${host_arch}: euler3d-kokkos-no-view.cpp 
	@echo $@
	@$(TIME_CMD) $(KIT_CXX) $(TAPIR_HIP_FLAGS) $(KITSUNE_KOKKOS_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
	@$(FILE_SIZE)

# kokkos-based tests (using views -- currently not compatible w/ kitsune)
euler3d-kokkos.nvcc.${host_arch}: euler3d-kokkos.cpp 
	@echo $@
	@$(TIME_CMD) $(KOKKOS_NVCC) $(KOKKOS_NVCC_FLAGS) -o $@ $< $(KOKKOS_CUDA_LIBS) -Xlinker -rpath=$(KOKKOS_CUDA_PREFIX)/lib64 
	@$(FILE_SIZE)
euler3d-kokkos.hipcc.${host_arch}: euler3d-kokkos.cpp 
	@echo $@
	@echo $(KOKKOS_HIPCC) $(KOKKOS_HIP_FLAGS) -o $@ $< $(KOKKOS_HIP_LIBS) -Xlinker -rpath=$(KOKKOS_HIP_PREFIX)/lib64
	@$(TIME_CMD) $(KOKKOS_HIPCC) $(KOKKOS_HIP_FLAGS) -o $@ $< $(KOKKOS_HIP_LIBS) -Xlinker -rpath=$(KOKKOS_HIP_PREFIX)/lib64 
	@$(FILE_SIZE)
euler3d-kokkos.hip.${host_arch}: euler3d-kokkos.cpp 
	@echo $@
	@$(TIME_CMD) $(KITSUNE_HIPCC) $(KOKKOS_HIP_FLAGS) -I$(ROCM_PATH)/include -o $@ $< -L$(ROCM_PATH)/lib $(KOKKOS_HIP_LIBS) -lamdhip64 -Xlinker -rpath=$(KOKKOS_HIP_PREFIX)/lib64 
	@$(FILE_SIZE)

#####

define RUN_euler_test
	@echo "tapir cuda flags: $(TAPIR_CUDA_FLAGS)" > $(1).log 
	@echo "tapir hip flags: $(TAPIR_HIP_FLAGS)" >> $(1).log
	$$(./$(1) fvcorr.domn.193K >> $(1).log) $(newline) 
endef

run: $(targets) 
	@echo "running generated executables..."
	$(foreach prog,$(targets),$(call RUN_euler_test,$(prog)))

launch_bench: euler3d-forall.cuda.${host_arch}
	../launch-bench.sh $< fvcorr.domn.193K

#####

.PHONY: all clean run

clean:
	-rm -f *.${host_arch} *.ll *.o
	-rm -f *~ core *.log *.dat
