SHELL=/bin/bash
host_arch := $(shell uname -o -m | awk '{print $$1}')
$(info kitsune experiments host architecture: ${host_arch})

ifneq ($(KITSUNE_OPT_LEVEL),)
  opt_level = $(KITSUNE_OPT_LEVEL)
else
  opt_level = 3
endif

ifneq ($(KITSUNE_OPT_FLAGS),)
  opt_flags = $(KITSUNE_OPT_FLAGS)
else
  opt_flags = -O${opt_level}
endif 

##################################
# EXPERIMENTS-WIDE C and CXX FLAGS 
#c_flags = -march=native $(CFLAGS) 
c_flags = $(CFLAGS) 
cxx_flags = -std=c++17 -march=native -fno-exceptions $(CXX_FLAGS)

ifneq ($(KITSUNE_VERBOSE),)
  clang_flags = ${clang_flags} -v
  clangxx_flags = ${clangxx_flags} -v
endif

$(info common c flags: ${c_flags})
$(info common c++ flags: ${cxx_flags})
