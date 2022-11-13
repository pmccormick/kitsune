
kokkos_prefix=${kitsune_prefix} # default kokkos (serial) install w/ kitsune 
kokkos_cuda_prefix=${kitsune_prefix}/opt/kokkos_cuda
kokkos_hip_prefix=${kitsune_prefix}/opt/kokkos_hip

nvcc_wrapper=${kokkos_cu_prefix}/bin/nvcc_wrapper

kokkos_cuda_flags=-I${kokkos_cu_prefix}/include
kokkos_libs=-lkokkoscore -lcuda -lcudart -ldl -lrt
kokkos_ld_flags=-L${kokkos_cu_prefix}/lib -L${CUDA_HOME}/lib64 ${kokkos_libs} 



