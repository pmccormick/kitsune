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
	@$(TIME_CMD) $(KIT_CXX) $(TAPIR_CUDA_FLAGS) -mllvm -debug-only=cuabi -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
	@$(FILE_SIZE)
euler3d-attr-forall.cuda.${host_arch}: euler3d-attr-forall.cpp 
	$(KIT_CXX) $(TAPIR_CUDA_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib

euler3d-forall.hip.${host_arch}: euler3d-forall.cpp 
	@echo $@
	@$(TIME_CMD) $(KIT_CXX) $(TAPIR_HIP_FLAGS) -fuse-ld=lld -Xclang -vectorize-loops -Xclang -vectorize-slp -mllvm --hipabi-keep-files=true -mllvm -debug-only=hipabi -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
	@$(FILE_SIZE)
euler3d-attr-forall.hip.${host_arch}: euler3d-attr-forall.cpp 
	$(KIT_CXX) $(TAPIR_HIP_FLAGS) -fuse-ld=lld -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib

# kokkos-based tests (w/out views)
euler3d-kokkos.kitsune.cuda.${host_arch}: euler3d-kokkos-no-view.cpp 
	@echo $@
	$(TIME_CMD) $(KIT_CXX) $(TAPIR_CUDA_FLAGS) $(KITSUNE_KOKKOS_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
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
euler3d-kokkos.clang.${host_arch}: euler3d-kokkos.cpp 
	$(KOKKOS_CLANG) $(KOKKOS_CLANG_FLAGS) -x cuda -o $@ $< -O3 -ffp-contract=on --cuda-gpu-arch=sm_90 $(KOKKOS_CUDA_LIBS) -Xlinker -rpath=$(KOKKOS_CUDA_PREFIX)/lib64 

euler3d-kokkos.hipcc.${host_arch}: euler3d-kokkos.cpp 
	$(KOKKOS_HIPCC) $(KOKKOS_HIP_FLAGS) -save-temps -o $@ $< $(KOKKOS_HIP_LIBS) -Xlinker -rpath=$(KOKKOS_HIP_PREFIX)/lib64 
euler3d-kokkos.hip.${host_arch}: euler3d-kokkos.cpp 
	$(KITSUNE_HIPCC) $(KOKKOS_HIP_FLAGS) -I$(ROCM_PATH)/include -o $@ $< -L$(ROCM_PATH)/lib $(KOKKOS_HIP_LIBS) -lamdhip64 -Xlinker -rpath=$(KOKKOS_HIP_PREFIX)/lib64 



euler3d-cuda.${host_arch}: euler3d.cu
	@echo $@
	$(TIME_CMD) $(NVCC) $(NVCC_CXX_FLAGS) -o $@ $<
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
	-rm -f *.${host_arch} *.ll *.o *.bc *.s *.hipi *.ptx  *.resolution.txt *.out  *.hipfb 
	-rm -f *~ core *.log *.dat
