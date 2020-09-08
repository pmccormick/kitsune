#ifndef __KITSUNE__H__
#define __KITSUNE__H__
#include <stdint.h>

#define ocl_mmap(a, n) __kitsune_opencl_mmap_marker((void*)a, n)
void* __kitsune_opencl_mmap_marker(void* ptr, uint64_t n);
#define spawn _kitsune_spawn
#define sync _kitsune_sync
#define forall _kitsune_forall

#endif
