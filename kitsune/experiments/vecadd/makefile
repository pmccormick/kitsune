include ../experiments.mk

targets = vecadd-forall.opencilk.${host_arch}

ifeq ($(BUILD_CUDA_EXPERIMENTS),true)
  targets += vecadd-forall.cuda.${host_arch}
endif

ifeq ($(BUILD_HIP_EXPERIMENTS),true)
  targets += vecadd-forall.hip.${host_arch}
endif

all: ${targets}

# forall-based tests
vecadd-forall.opencilk.${host_arch}: vecadd-forall.cpp
	@echo $@
	@$(TIME_CMD) $(KIT_CXX) $(TAPIR_OPENCILK_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
	@$(FILE_SIZE)
vecadd-forall.cuda.${host_arch}: vecadd-forall.cpp
	@echo $@
	@$(TIME_CMD) $(KIT_CXX) $(TAPIR_CUDA_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
	@$(FILE_SIZE)
vecadd-forall.hip.${host_arch}: vecadd-forall.cpp 
	@echo $@
	@$(TIME_CMD) $(KIT_CXX) $(TAPIR_HIP_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
	@$(FILE_SIZE)

# kokkos-based tests (w/out views)
vecadd-kokkos.opencilk.kitsune.${host_arch}: vecadd-kokkos-no-views.cpp 
	@echo $@
	@$(TIME_CMD) $(KIT_CXX) $(TAPIR_OPENCILK_FLAGS) $(KITSUNE_KOKKOS_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
	@$(FILE_SIZE)
vecadd-kokkos.cuda.kitsune.${host_arch}: vecadd-kokkos-no-views.cpp 
	@echo $@
	@$(TIME_CMD) $(KIT_CXX) $(TAPIR_CUDA_FLAGS) $(KITSUNE_KOKKOS_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
	@$(FILE_SIZE)
vecadd-kokkos.hip.kitsune.${host_arch}: vecadd-kokkos-no-views.cpp 
	@echo $@
	@$(TIME_CMD) $(KIT_CXX) $(TAPIR_HIP_FLAGS) $(KITSUNE_KOKKOS_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
	@$(FILE_SIZE)

# kokkos-based tests (using views -- currently not compatible w/ kitsune)
vecadd-kokkos.nvcc.${host_arch}: vecadd-kokkos.cpp 
	@echo $@
	@$(TIME_CMD) $(KOKKOS_NVCC) $(KOKKOS_NVCC_FLAGS) -o $@ $< $(KOKKOS_CUDA_LIBS) -Xlinker -rpath=$(KOKKOS_CUDA_PREFIX)/lib64 
	@$(FILE_SIZE)
vecadd-kokkos.hipcc.${host_arch}: vecadd-kokkos.cpp 
	@echo $@
	@$(TIME_CMD) $(KOKKOS_HIPCC) $(KOKKOS_HIP_FLAGS) -o $@ $< $(KOKKOS_HIP_LIBS) -Xlinker -rpath=$(KOKKOS_HIP_PREFIX)/lib64 
	@$(FILE_SIZE)

#####
define newline 


endef 

define RUN_test
	$$(./$(1) > $(1).log) $(newline) 
endef

run: $(targets) 
	@echo "running generated executables..."
	$(foreach prog,$(targets),$(call RUN_test,$(prog)))

launch_bench: vecadd-forall.cuda.${host_arch}
	../launch-bench.sh $<
#####

.PHONY: all clean run

clean:
	-rm -f *.${host_arch} *.ll *.o
	-rm -f *~ core *~ *.log 

