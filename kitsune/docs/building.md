
## Building Kitsune

In general Kitsune follows the overall LLVM build process.  You can learn more about 
LLVM [here](https://releases.llvm.org/10.0.0/docs/index.html) and it is recommended
that you first  you first become familiar with building LLVM's before diving into the
specifics of Kitsune.  This page is a good starting point: 

   https://releases.llvm.org/10.0.0/docs/GettingStarted.html#getting-the-source-code-and-building-llvm

### Build Notes & Suggestions  

* __Ninja__: In general we recommend using ``ninja`` for building as it tends to be
  faster than ``make.`` However, given the size and resource requirements for a
  parallel build of the LLVM infrastructure it is recommended that you set the
  CMake parameters ``LLVM_PARALLEL_COMPILE_JOBS`` and
  ``LLVM_PARALLEL_LINK_JOBS`` to a number that does overwhelm the system 
  you are building on; note that *ninja-based* builds will often (somewhat 
  silently) fail during linking when system memory is exhausted.

* __CUDA__ support with new C++ compilers.  ``nvcc`` often fails if you build with
  recent releases of the GCC toolchain (e.g., 10.2).  You can still use this version
  of GCC to build LLVM but need to use an older version of GCC as the CUDA
  host compiler. To specify this, set the CMake variable ``CUDA_HOST_COMPILER``
  to a (CUDA) supported version of GCC.  
  
  For example,

    ```bash
      $ cmake ... -DCUDA_HOST_COMPILER=gcc-8 ... ../llvm
    ```

  *Note: This is typically only an issue when building OpenMP support in Clang
  and including the OpenMP runtime library as part of the build.*

* __CUDA compute capabilities__: By default LLVM/Clang/OpenMP has some old
  architectures as their default settings (in fact, some are soon to be
  deprecated). We suggest using the following parameters to instead target 
  something more modern:

    ```bash
       $ cmake ... -DLIBOMPTARGET_NVPTX_COMPUTE_CAPABILITIES=70 \
         -DCLANG_OPENMP_NVPTX_DEFAULT_ARCH=sm_70 ... ../llvm
    ```

  *Note: This is typically only an issue when building OpenMP support (in Clang)
  and including the OpenMP runtime library as part of the build.* Using this
  approach will typically reduce several warnings that are produced during the
  build but you should make sure the details match the hardware configuration
  you are targeting (e.g., replace ``70`` and ``sm_70`` with values appropriate for
  your system).

* __Reduce build times__ by simplifying/Reducing the number of architecture 
  targets supported by the build. For example,

     ```bash
       $ cmake ... -DLLVM_TARGETS_TO_BUILD="X86;AMDGPU;NPTX" ... ../llvm
     ``` 

* __Debugging__: Debug symbols can be problematic in terms of the resources
  needed to build the full LLVM infrastructure (debug symbols require more
  memory). In addition, the resulting compiler can be painfully slow and eat
  into development time significantly.  In general, we recommend that Kitsune
  be built with ``CMAKE_BUILD_TYPE=RelWithDebInfo``.  For developers we would
  then suggest (re)enabling certain debugging mechanisms that are disabled in this
  build mode.  For example, ``LLVM_ENABLE_DUMP=ON`` will enable the ``dump()``
  method in non-debug builds.

* __Example CMake configuration:__

    ```bash
      $ cmake -GNinja -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;libcxx;libcxxabi;lld" \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLLVM_TARGETS_TO_BUILD="X86;AMDGPU;NVPTX" \
        -DCMAKE_INSTALL_PREFIX=/path/to/install/dir -DLLVM_ENABLE_DUMP=ON \
        -DLIBOMPTARGET_NVPTX_COMPUTE_CAPABILITIES=70 -DCLANG_OPENMP_NVPTX_DEFAULT_ARCH=sm70 \
        -DLLVM_PARALLEL_COMPILE_JOBS=14 -DLLVM_PARALLEL_LINK_JOBS=4 \
        ../llvm
    ```
