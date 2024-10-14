include ../experiments.mk

targets = exp.${host_arch}

all: ${targets}

# forall-based tests
exp.cuda.${host_arch}: exp.cpp
	@echo $@
	@$(TIME_CMD) $(KIT_CXX) $(TAPIR_CUDA_FLAGS) -o $@ $< 
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

