include ../experiments.mk 

targets = euler3d-forall.opencilk.${host_arch}

ifeq ($(BUILD_CUDA_EXPERIMENTS),true)
  targets += euler3d-forall.cuda.${host_arch}
  targets += euler3d-kokkos.nvcc.${host_arch}
endif

ifeq ($(BUILD_HIP_EXPERIMENTS),true)
  targets += euler3d-forall.hip.${host_arch}
  targets += euler3d-kokkos.hipcc.${host_arch}
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
euler3d-kokkos.cuda.kitsune.${host_arch}: euler3d-kokkos-no-view.cpp 
	@echo $@
	@$(TIME_CMD) $(KIT_CXX) $(TAPIR_CUDA_FLAGS) $(KITSUNE_KOKKOS_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
	@$(FILE_SIZE)
euler3d-kokkos.hip.kitsune.${host_arch}: euler3d-kokkos-no-view.cpp 
	@echo $@
	@$(TIME_CMD) $(KIT_CXX) $(TAPIR_HIP_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
	@$(FILE_SIZE)

# kokkos-based tests (using views -- currently not compatible w/ kitsune)
euler3d-kokkos.nvcc.${host_arch}: euler3d-kokkos.cpp 
	@echo $@
	@$(TIME_CMD) $(KOKKOS_NVCC) $(KOKKOS_NVCC_FLAGS) -o $@ $< $(KOKKOS_CUDA_LIBS) -Xlinker -rpath=$(KOKKOS_CUDA_PREFIX)/lib64 
	@$(FILE_SIZE)
euler3d-kokkos.hipcc.${host_arch}: euler3d-kokkos.cpp 
	@echo $@
	@$(TIME_CMD) $(KOKKOS_HIPCC) $(KOKKOS_HIP_FLAGS) -o $@ $< $(KOKKOS_HIP_LIBS) -Xlinker -rpath=$(KOKKOS_HIP_PREFIX)/lib64 
	@$(FILE_SIZE)

#####
define newline 


endef 

define RUN_test
	$$(./$(1) fvcorr.domn.193K > $(1).log) $(newline) 
endef

run: $(targets) 
	@echo "running generated exeutables..."
	$(foreach prog,$(targets),$(call RUN_test,$(prog)))
#####

.PHONY: all clean run

clean:
	-rm -f *.${host_arch} *.ll *.o
	-rm -f *~ core *.log *.dat
