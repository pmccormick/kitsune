# serial kokkos (part of the default kitsune install)
kokkos_prefix=${kitsune_prefix} 
kokkos_libs=-lkokkoscore -ldl 

# kokkos w/ cuda.
kokkos_cu_prefix=${kitsune_prefix}/opt/kokkos/
kokkos_cu_flags=-I${kokkos_cu_prefix}/include
nvcc_wrapper=${kokkos_cu_prefix}/bin/nvcc_wrapper
kokkos_nvcc_flags=${kokkos_cu_flags} ${nvcc_cxx_flags}
kokkos_ld_flags=-L${kokkos_cu_prefix}/lib -L${CUDA_HOME}/lib64 ${kokkos_libs}

# kokkos w/ hip
kokkos_hip_prefix=${kitsune_prefix}/opt/kokkos_hip
kokkos_hip_flags=-I${kokkos_hip_prefix}/include 
kokkos_hip_ld_flags=-L${kokkos_hip_prefix}/lib64 



