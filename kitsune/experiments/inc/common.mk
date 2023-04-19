# 
#
# 
SHELL=/bin/bash
host_arch:=$(shell uname -o -m | awk '{print $$1}')
$(info host architecture: ${host_arch})

CXX_FLAGS+=-std=c++17 -fno-exceptions
C_FLAGS?=

TIME_CMD=/usr/bin/time -f "  compile time: %E sec., high-water memory usage: %M KB"
FILE_SIZE=/usr/bin/ls -lh '$@' | /usr/bin/awk '{ print "  executable size:",$$5 }'

