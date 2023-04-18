

subdirs := \
  copy \
  vecadd \
  saxpy \
  srad \
  raytracer \
  euler3d 

.PHONY: all clean run $(subdirs)

all: $(subdirs) 
clean: $(subdirs) 

$(subdirs):
	$(MAKE) -C $@ $(MAKECMDGOALS)

run: $(subdirs)
	$(MAKE) -C $< $(MAKECMDGOALS)
	


