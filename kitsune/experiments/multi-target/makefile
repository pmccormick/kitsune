include ../experiments.mk

ifeq ($(BUILD_CUDA_EXPERIMENTS),true)
  targets += multi-target.cuda.${host_arch}
  targets += kokkos-multi-target.cuda.${host_arch}
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
	-rm -f *.${host_arch} *.ll *.o
	-rm -f *~ core *~ *.log 

