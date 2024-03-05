include ../experiments.mk

ifeq ($(BUILD_CUDA_EXPERIMENTS),true)
  targets += vecadd.cuda.${host_arch}
  targets += nested-launch.cuda.${host_arch}
  targets += vecadd-kokkos.cuda.${host_arch}
endif

ifeq ($(BUILD_HIP_EXPERIMENTS),true)
  targets += vecadd.hip.${host_arch}
  targets += nested-launch.hip.${host_arch}
endif

all: ${targets}

# forall-based tests
multi-target.cuda.${host_arch}: multi-target.cpp
	@echo $@
	@$(TIME_CMD) $(KIT_CXX) $(TAPIR_OPENCILK_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
	@$(FILE_SIZE)

kokkos-multi-target.cuda.${host_arch}: kokkos-multi-target.cpp 
	@echo $@
	@$(TIME_CMD) $(KIT_CXX) $(TAPIR_CUDA_FLAGS) $(KITSUNE_KOKKOS_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
	@$(FILE_SIZE)

# build targets for custom config files -- assumes -ftapir=opencilk
# with cuda/hip attributed targets...

opencilk-cuda.cfg:
	@cat $(KITSUNE_PREFIX)/share/kitsune/cuda.cfg > $@
	@echo "$(TAPIR_CUDA_FLAGS)" >> $@
	@echo "-L/$(CUDA_PATH)/lib64/ -lcuda" >> $@

opencilk-hip.cfg:
	@cat $(KITSUNE_PREFIX)/share/kitsune/hip.cfg > $@
	@echo "$(TAPIR_HIP_FLAGS)" >> $@
	@echo "-L/$(CUDA_PATH)/lib64 -lcudart -lcuda" >> $@

# forall-based tests

# cuda 
vecadd.cuda.${host_arch}: vecadd.cpp opencilk-cuda.cfg 
	@${TIME_CMD} $(KIT_CXX) $(TAPIR_OPENCILK_FLAGS) --config=./opencilk-cuda.cfg -o $@ $< 

nested-launch.cuda.${host_arch}: nested-launch.cpp opencilk-cuda.cfg
	@echo $@
	@${TIME_CMD} $(KIT_CXX) $(TAPIR_OPENCILK_FLAGS) --config=./opencilk-cuda.cfg -o $@ $<
	@${FILE_SIZE}

# hip 
nested-launch.hip.${host_arch}: nested-launch.cpp opencilk-hip.cfg
	@echo $@
	@${TIME_CMD} $(KIT_CXX) $(TAPIR_OPENCILK_FLAGS) --config=./opencilk-hip.cfg -o $@ $<
	@${FILE_SIZE}

# kokkos-based tests
vecadd-kokkos.cfg:
	@echo "building custom multi-target config file..."
	@echo "-D_tapir_cuda_target" > vecadd-kokkos.cfg
	@echo "$(TAPIR_CUDA_FLAGS)" >> vecadd-kokkos.cfg
	@cat $(KITSUNE_PREFIX)/share/kitsune/kokkos.cfg >> vecadd-kokkos.cfg
	@cat $(KITSUNE_PREFIX)/share/kitsune/cuda.cfg >> vecadd-kokkos.cfg

vecadd-kokkos.cuda.${host_arch}: vecadd-kokkos.cpp vecadd-kokkos.cfg
	@echo $@
	@${TIME_CMD} $(KIT_CXX) $(TAPIR_OPENCILK_FLAGS) $(KITSUNE_KOKKOS_FLAGS) --config=./vecadd-kokkos.cfg -o $@ $<
	@${FILE_SIZE}

# multi file multi target mesh tests

# VPATH := ./forall/:./kokkos/

intersect_lto_src = common.cpp intersect.cpp

# intersect_src = common.cpp intersect.cpp

# notdir below is unnecessary if no source subfolder
intersect_lto_objs := $(patsubst %.cpp, %.lto.o, $(notdir $(intersect_lto_src)))
intersect_objs := $(patsubst %.cpp, %.o, $(notdir $(intersect_src)))

%.lto.o: %.cpp opencilk-cuda.cfg
	/usr/bin/time ${KIT_CXX} $(TAPIR_OPENCILK_FLAGS) --config=./opencilk-cuda.cfg -I ${CUDA_PATH}/targets/x86_64-linux/include -c -flto -o $@ $<
%.o: %.cpp opencilk-cuda.cfg
	/usr/bin/time ${KIT_CXX} $(TAPIR_OPENCILK_FLAGS) --config=./opencilk-cuda.cfg -I ${CUDA_PATH}/targets/x86_64-linux/include -c -o $@ $<

# intersect.${host_arch}: ${intersect_objs} opencilk-cuda.cfg
# 	/usr/bin/time ${KIT_CXX} $(TAPIR_OPENCILK_FLAGS) --config=./opencilk-cuda.cfg -fuse-ld=lld -o $@ ${intersect_objs} \
# 	-L${CUDA_ROOT}/lib64 -lnvToolsExt
intersect.${host_arch}: ${intersect_lto_objs} ${intersect_objs} opencilk-cuda.cfg
	/usr/bin/time ${KIT_CXX} $(TAPIR_OPENCILK_FLAGS) --config=./opencilk-cuda.cfg -flto -fuse-ld=lld -o $@ ${intersect_lto_objs} ${intersect_objs} \
	-L${CUDA_ROOT}/lib64 -lnvToolsExt
# /usr/bin/time ${KIT_CXX} $(TAPIR_OPENCILK_FLAGS) --config=./opencilk-cuda.cfg -flto -fuse-ld=lld -o $@ ${intersect_lto_objs} ${intersect_objs} 














#####
define newline 


endef 

define RUN_test
	$$(./$(1) > $(1).log) $(newline) 
endef

run: $(targets) 
	@echo "running generated exeutables..."
	$(foreach prog,$(targets),$(call RUN_test,$(prog)))
#####

.PHONY: all clean run

clean:
	-rm -f *.${host_arch} *.ll *.o *.cfg
	-rm -f *~ core *~ *.log
	-rm -f *.sqlite *.nsys-rep

