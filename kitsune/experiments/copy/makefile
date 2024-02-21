include ../experiments.mk

targets = copy-forall.opencilk.${host_arch}

ifeq ($(BUILD_CUDA_EXPERIMENTS),true)
  targets += copy-forall.cuda.${host_arch}
endif

ifeq ($(BUILD_HIP_EXPERIMENTS),true)
  targets += copy-forall.hip.${host_arch}
  targets += copy-hip.${host_arch}
endif

all: ${targets}

# forall-based tests
copy-forall.opencilk.${host_arch}: copy-forall.cpp 
	@echo $@ 
	@${TIME_CMD} $(KIT_CXX) $(TAPIR_OPENCILK_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
	@${FILE_SIZE}
copy-forall.cuda.${host_arch}: copy-forall.cpp 
	@echo $@
	@${TIME_CMD} $(KIT_CXX) $(TAPIR_CUDA_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
	@${FILE_SIZE}
copy-forall.hip.${host_arch}: copy-forall.cpp 
	@echo $@
	@${TIME_CMD} $(KIT_CXX) $(TAPIR_HIP_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
	@${FILE_SIZE}

# kokkos-based tests (w/out views)
copy-kokkos.cuda.kitsune.${host_arch}: copy-kokkos-no-views.cpp 
	@echo $@
	@${TIME_CMD} $(KIT_CXX) $(TAPIR_CUDA_FLAGS) $(KITSUNE_KOKKOS_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
	@${FILE_SIZE}
copy-kokkos.hip.kitsune.${host_arch}: copy-kokkos-no-views.cpp 
	@echo $@
	@${TIME_CMD} $(KIT_CXX) $(TAPIR_HIP_FLAGS) $(KITSUNE_KOKKOS_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
	@${FILE_SIZE}

# kokkos-based tests (using views -- currently not compatible w/ kitsune)
copy-kokkos.nvcc.${host_arch}: copy-kokkos.cpp 
	@echo $@
	@${TIME_CMD} $(KOKKOS_NVCC) $(KOKKOS_NVCC_FLAGS) -o $@ $< $(KOKKOS_CUDA_LIBS) -Xlinker -rpath=$(KOKKOS_CUDA_PREFIX)/lib64 
	@${FILE_SIZE}
copy-kokkos.hipcc.${host_arch}: copy-kokkos.cpp 
	@echo $@
	@${TIME_CMD} $(KOKKOS_HIPCC) $(KOKKOS_HIP_FLAGS) -o $@ $< $(KOKKOS_HIP_LIBS) -Xlinker -rpath=$(KOKKOS_HIP_PREFIX)/lib64 
	@${FILE_SIZE}

# hip version 
copy-hip.${host_arch}: copy-hip.cpp
	@echo $@
	@$(TIME_CMD) $(HIPCC) $(HIPCC_CXX_FLAGS) -o $@ $<
	@$(FILE_SIZE)
#####
define newline 


endef 

run: $(targets) 
	@echo "running generated executables..."
	@$(foreach prog,$(targets),$(call RUN_test,$(prog)))

launch_bench: copy-forall.cuda.${host_arch}
	../launch-bench.sh $<
##### 

.PHONY: all clean run

clean:
	-rm -f *.${host_arch} *.ll *.o
	-rm -f *~ core *.log 

