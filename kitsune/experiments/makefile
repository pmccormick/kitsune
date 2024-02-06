

subdirs := \
  noforall \
  copy \
  vecadd \
  saxpy \
  srad \
  raytracer \
  euler3d \
  multi-target

.PHONY: all clean run $(subdirs)

all: $(subdirs) 
clean: $(subdirs) 

$(subdirs):
	$(MAKE) -C $@ $(MAKECMDGOALS)

run: $(subdirs)
	$(MAKE) -C $< $(MAKECMDGOALS)
	
launch_bench: $(subdirs)
	$(MAKE) -C $< $(MAKECMDGOALS)

