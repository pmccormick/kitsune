include ../experiments.mk 

targets = raytracer-forall.opencilk.${host_arch}

ifeq ($(BUILD_CUDA_EXPERIMENTS),true)
  targets += raytracer-forall.cuda.${host_arch}
  targets += raytracer-cuda.${host_arch}
endif

ifeq ($(BUILD_HIP_EXPERIMENTS),true)
  targets += raytracer-forall.hip.${host_arch}
  targets += raytracer-hip.${host_arch}
endif

all: ${targets}

# forall-based tests
raytracer-forall.opencilk.${host_arch}: raytracer-forall.cpp 
	@echo $@
	@$(TIME_CMD) $(KIT_CXX) $(TAPIR_OPENCILK_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
	@$(FILE_SIZE)
raytracer-forall.cuda.${host_arch}: raytracer-forall.cpp 
	@echo $@
	@$(TIME_CMD) $(KIT_CXX) $(KITSUNE_FAST_MATH) $(TAPIR_CUDA_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
	@$(FILE_SIZE)

raytracer-forall.hip.${host_arch}: raytracer-forall.cpp 
	@echo $@
	@$(TIME_CMD) $(KIT_CXX) -v $(TAPIR_HIP_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
	@$(FILE_SIZE)

# kokkos-based tests (w/out views)
raytracer-kokkos.cuda.kitsune.${host_arch}: raytracer-kokkos-no-view.cpp 
	@echo $@
	@$(TIME_CMD) $(KIT_CXX) $(TAPIR_CUDA_FLAGS) $(KITSUNE_KOKKOS_FLAGS) -I$(KOKKOS_CUDA_PREFIX)/include/ -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
	@$(FILE_SIZE)
raytracer-kokkos.hip.kitsune.${host_arch}: raytracer-kokkos-no-view.cpp 
	@echo $@
	@$(TIME_CMD) $(KIT_CXX) $(TAPIR_HIP_FLAGS) $(KITSUNE_KOKKOS_FLAGS) -I$(KOKKOS_HIP_PREFIX)/include/ -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
	@$(FILE_SIZE)

# kokkos-based tests (using views -- currently not compatible w/ kitsune)
raytracer-kokkos.nvcc.${host_arch}: raytracer-kokkos.cpp 
	@echo $@
	@$(TIME_CMD) $(KOKKOS_NVCC) $(KOKKOS_NVCC_FLAGS) -o $@ $< $(KOKKOS_CUDA_LIBS) -Xlinker -rpath=$(KOKKOS_CUDA_PREFIX)/lib64 
	@$(FILE_SIZE)
raytracer-kokkos.hipcc.${host_arch}: raytracer-kokkos.cpp 
	@echo $@
	@$(TIME_CMD) $(KOKKOS_HIPCC) -v $(KOKKOS_HIP_FLAGS) -o $@ $< $(KOKKOS_HIP_LIBS) -Xlinker -rpath=$(KOKKOS_HIP_PREFIX)/lib64 
	@$(FILE_SIZE)
raytracer-kokkos.hip.${host_arch}: raytracer-kokkos.cpp 
	@echo $@
	@$(TIME_CMD) $(KIT_CXX) -x hip -v $(KOKKOS_HIP_FLAGS) -I$(ROCM_PATH)/include -o $@ $< -L$(ROCM_PATH)/lib $(KOKKOS_HIP_LIBS) -lamdhip64 -Xlinker -rpath=$(KOKKOS_HIP_PREFIX)/lib64 
	@$(FILE_SIZE)

# cuda version 
raytracer-cuda.${host_arch}: raytracer.cu
	@echo $@
	@$(TIME_CMD) $(NVCC) $(NVCC_CXX_FLAGS) -o $@ $<
	@$(FILE_SIZE)

# hip version 
raytracer-hip.${host_arch}: raytracer-hip.cpp
	@echo $@
	@$(TIME_CMD) $(HIPCC) $(HIPCC_CXX_FLAGS) -o $@ $<
	@$(FILE_SIZE)

raytracer-hip.clang.${host_arch}: raytracer-hip.cpp
	@echo $@
	@$(KIT_CXX) -x hip -v $(HIPCC_CXX_FLAGS) -o $@ $< -L$(ROCM_PATH)/lib -lamdhip64
	@$(FILE_SIZE)
#####
define newline 


endef 

run: $(targets) 
	@echo "running generated executables..."
	$(foreach prog,$(targets),$(call RUN_test,$(prog)))

launch_bench: raytracer-forall.cuda.${host_arch}
	../launch-bench.sh $<
#####

.PHONY: all clean run

clean:
	-rm -f *.${host_arch} *.ll *.o
	-rm -f *~ core *.ppm *.log 

