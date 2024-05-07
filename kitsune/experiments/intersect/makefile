include ../experiments.mk

ifeq ($(BUILD_CUDA_EXPERIMENTS),true)
  targets += intersect.cuda.${host_arch}
endif

ifeq ($(BUILD_HIP_EXPERIMENTS),true)
  # no-op for now... 
endif

all: ${targets}

opencilk-cuda.cfg:
	@cat $(KITSUNE_PREFIX)/share/kitsune/cuda.cfg > $@
	@echo "$(TAPIR_CUDA_FLAGS)" >> $@
	@echo "-L/$(CUDA_PATH)/lib64/ -lcuda" >> $@

opencilk-hip.cfg:
	@cat $(KITSUNE_PREFIX)/share/kitsune/hip.cfg > $@
	@echo "$(TAPIR_HIP_FLAGS)" >> $@
	@echo "-L/$(CUDA_PATH)/lib64 -lcudart -lcuda" >> $@


intersect_lto_src = common.cpp intersect.cpp
# notdir below is unnecessary if no source subfolder
intersect_lto_objs := $(patsubst %.cpp, %.lto.o, $(notdir $(intersect_lto_src)))
intersect_objs := $(patsubst %.cpp, %.o, $(notdir $(intersect_src)))

%.lto.o: %.cpp opencilk-cuda.cfg
	/usr/bin/time ${KIT_CXX} $(TAPIR_OPENCILK_FLAGS) --config=./opencilk-cuda.cfg -I ${CUDA_PATH}/targets/x86_64-linux/include -c -flto -o $@ $<
%.o: %.cpp opencilk-cuda.cfg
	/usr/bin/time ${KIT_CXX} $(TAPIR_OPENCILK_FLAGS) --config=./opencilk-cuda.cfg -I ${CUDA_PATH}/targets/x86_64-linux/include -c -o $@ $<

intersect.cuda.${host_arch}: ${intersect_lto_objs} ${intersect_objs} opencilk-cuda.cfg
	/usr/bin/time ${KIT_CXX} $(TAPIR_OPENCILK_FLAGS) --config=./opencilk-cuda.cfg -Wl,--mllvm=-cuabi-arch=$(CUDA_ARCH) -flto -fuse-ld=lld -o $@ ${intersect_lto_objs} ${intersect_objs} \
	-L${CUDA_ROOT}/lib64 -lnvToolsExt
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

