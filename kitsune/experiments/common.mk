# 
#
# 
SHELL=/bin/bash
host_arch:=$(shell uname -o -m | awk '{print $$1}')
$(info host architecture: ${host_arch})

CXX_FLAGS+=-std=c++17 -fno-exceptions
C_FLAGS?=

