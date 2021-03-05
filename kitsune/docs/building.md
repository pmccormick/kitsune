
## Building Kitsune+Tapir

In general Kitsune follows the overall LLVM build process.  You can learn more about the release of LLVM that underlies Kitsune+Tapir [here](https://releases.llvm.org/10.0.0/docs/index.html).  Becoming familiar with building LLVM is helpful, if you plan to do development work on Kitsune+Tapir and this is a good starting point. 

   https://releases.llvm.org/10.0.0/docs/GettingStarted.html#getting-the-source-code-and-building-llvm

However, we have some suggestions and time saving steps below that can be helpful for getting up and running.

### Build Notes & Suggestions  

* __Ninja__: In general we recommend using ``ninja`` for building as it tends to be faster than ``make``. However, given the size and resource requirements for a  parallel build of the LLVM infrastructure it is recommended that you set the   CMake parameters ``LLVM_PARALLEL_COMPILE_JOBS`` and ``LLVM_PARALLEL_LINK_JOBS`` to values that do not overwhelm the system you are building on.  Building a ``Release`` or ``RelWithDebInfo`` version will save on disk space and RAM usage during building, linking and installation.  

* __CUDA__ support with new C++ compilers.  ``nvcc`` often fails if you build with recent releases of the GCC toolchain (e.g., 10.x).  You can still use this version of GCC to build LLVM but need to use an older version of GCC as the CUDA host compiler. To specify this, set the CMake variable ``CUDA_HOST_COMPILER`` to a (CUDA) supported version of GCC:

    ```bash
      $ cmake ... -DCUDA_HOST_COMPILER=gcc-8 ... ../llvm
    ```

  **Note**: This is typically encountered as an when building with OpenMP support. 

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


## Using a CMake Cache 

The Kitsune+Tapir release comes with an example CMake cache file that contains 
some configuration examples for a basic installation.  This cache file is located 
in ``kitsune/cmake/caches/kitsune-dev.cmake``.   Here is a simple example of using 
this cache file: 

  ```bash
    $ mkdir build; cd build 
    $ cmake -G Ninja -C ../kitsune/cmake/caches/kitsune-dev.cmake \
      -DCMAKE_INSTALL_PREFIX=/home/kitsune/local \
      ../llvm 
  ``` 
As the cache file can be tailored and configured for specific needs it is often 
easier than other approaches.  In addition, this cache file can be used in 
concert with the module files located in ``kitsune/modules/`` as they provide a 
set of enviornment variables that can help configure the build parameters.  For
example, using the module files we can load the specific runtime systems we 
would like to enable: 

  ```bash
    $ export MODULEPATH=../kitsune/modules:$MODULEPATH 
    $ module load kokkos 
    $ module load opencilkrt 
    $ module load legion   # for realm runtime target.
    $ mkdir build; cd build 
    $ cmake -G Ninja -C ../kitsune/cmake/caches/kitsune-dev.cmake \
      -DCMAKE_INSTALL_PREFIX=/home/kitsune/local \
      ../llvm 
  ``` 

**NOTE**: You will need to tailor the details of the provided module 
files to match the details of your target system. 





