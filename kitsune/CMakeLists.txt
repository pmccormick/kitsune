project(kitsune)

set(KITSUNE_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(KITSUNE_LIBRARY_DIR ${CMAKE_SOURCE_DIR}/lib)


option(KITSUNE_ENABLE_CUDA_RUNTIME "Build and enable targeting Kitsune's CUDA runtime." OFF)
option(KITSUNE_ENABLE_OPENCL_RUNTIME "Build and enable targeting Kitsune's OpenCL runtime." OFF)
option(KITSUNE_ENABLE_QTHREADS_RUNTIME "Enable targeting the Qthreads runtime." OFF)
option(KITSUNE_ENABLE_CILK_RUNTIME "Enable targeting the Cilk runtime." OFF)
option(KITSUNE_ENABLE_LIBOMP_RUNTIME "Enable targeting the OpenMP runtime." OFF)

option(KITSUNE_ENABLE_KOKKOS_MODE "Enable Kokkos-aware features in the compiler." OFF)


if (KITSUNE_ENABLE_CUDA_RUNTIME)
   add_subdirectory(lib/CudaRT)
endif()

