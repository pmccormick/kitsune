

subdirs := \
  copy \
  vecadd \
  saxpy \
  srad \
  raytracer \
  euler3d 

all: $(subdirs) 
clean: $(subdirs) 

$(subdirs):
	$(MAKE) -C $@ $(MAKECMDGOALS)


.PHONY: all clean $(subdirs)

