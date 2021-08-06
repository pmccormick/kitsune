/*
 * Copyright (c) 2020 Triad National Security, LLC
 *                         All rights reserved.
 *
 * This file is part of the kitsune/llvm project.  It is released under
 * the LLVM license.
 */
#ifndef __CLANG_KITSUNE_RT_H__
#define __CLANG_KITSUNE_RT_H__

/* expose some runtime calls so we can tinker with things
   at the code level without too much effort.
*/

#if defined(KITSUNE_ENABLE_OPENCL_TARGET)
#define ocl_mmap(a, n) __kitsune_opencl_mmap_marker((void*)a, n)
extern "C" void __kitsune_opencl_mmap_marker(void* ptr, uint64_t n);
#endif

#endif
