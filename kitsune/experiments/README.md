
## Experiments Notes

The experiments all use a pure makefile-based build process that 
relies on a set of environment variables.  We have explicitly done 
this to test the toolchain outside of the confines of LLVM's CMake 
mechanisms. 

The most basic environment variable is the install prefix for 
the kitsune build.  It should be set via the `KITSUNE_PREFIX`
variable: 

```console
  $ export KITSUNE_PREFIX=/projects/kitsune/x86_64/15.x
```  

With this set, each example/experiment will invoke the kitsune
version of clang via the full install prefix.  For a basic 
spin around the block with a CPU-only target (`-ftapir=opencilk`)
this is often all that you will need but more steps are needed for 
enabling GPU support and testing.  First, make sure you have built 
kitsune w/ GPU support enabled (see the README file in the toplevel
kitsune subfolder of our LLVM-based distribution). 

Builds of *CUDA* and *HIP* versions of the experiments are 
triggered via (*somewhat traditional?*) environment variables
that point to the install prefix for each installation.  These
two variables are `CUDA_PATH` and `ROCM_PATH`.  For example, 
```console
  $ export CUDA_PATH=/opt/cuda
  $ export ROCM_PATH=/opt/rocm-5.4.3
```
With these two variables set, the build with use the default 
CPU-based target using `-ftapir=cheetah` (OpenCilk) target plus
both *CUDA* (`-ftapir=cuda`) and *HIP* (`-ftapir=hip`) targets.
If you don't want one of the two GPU targets, skip setting
the corresponding environment variable. 

Once these variables are set you can compile each experiment by running, 
```console
$ make
``` 
from the top-level *experiments* directory.  If you would like to
compile and run the different executables you can use:
```console
$ make run
``` 
The default targets will provide feedback on compile times, 
*high-water* memory usage statistics, and the final executable 
file sizes.  The `run` target will include these details as well
and run each produced executable with output gathered in a log 
file within each experiment's subdirectory.

### Kokkos Tests

Note that additional Kokkos versions of the experiments will be 
built (and optionally run) under the assumption that the 
appropriate GPU-enabled versions of Kokkos are installed in the 
kitsune installation under:
```$KITSUNE_PREFIX/opt/kokkos/[cuda|hip]/```.
You can override this by setting the `KOKKOS_PREFIX`
environment variable but the subdirectory path is still assumed
to contain the cuda and/or hip entries (corresponding to the 
setting of `CUDA_PATH` and/or `ROCM_PATH`).

## For Developers/Testers

There are various other environment variables exposed
via the makefile infrastructure (see the inc/ directory).
These can be used to tailor various aspects of the
build and can be helpful when debugging or evaluating 
other aspects of the code base (e.g., runtime components).
A quick summary of some of the additional variables is 
provided below:

* `KITSUNE_OPTLEVEL`: This is a single numerical value
that corresponds to the value added to the common compiler 
optimization flag (e.g., -O**3**).  By default this will be 
applied to **both host and device** optimization stages.

* `KITSUNE_ABI_OPTLEVEL`: This is a single numerical 
value that corresponds to the value added to the common 
compiler optimization flag (e.g., -O**3**) but will 
**only be applied to the device** optimization stage.

* `TAPIR_CUDA_FLAGS` | `TAPIR_HIP_FLAGS`:  These variables
may be used to ***completely*** override (replace) the 
defaults provided in the default makefile settings.

* `TAPIR_CUDA_EXTRA_FLAGS` | `TAPIR_HIP_EXTRA_FLAGS`: Can be 
used to add additional flags to the default arguments provided
in the makefile settings. These are useful for passing flags to enable verbose debug output from the Tapir target transformation
stages.  For example, 
  ```console
  $ export TAPIR_CUDA_EXTRA_FLAGS='-mllvm -debug-only=cuabi'
  ```

* `CUDA_ARCH`: The CUDA architecture to target (e.g., `sm_80`).

* `AMDGPU_ARCH`: The AMDGPU architecture to target (e.g., 
`gfx90a`).

* `KITSUNE_VERBOSE`: Enable the verbose debugging output for the 
Tapir-target transformations.  This will essentially add the 
appropriate flag to both `$TAPIR_CUDA_FLAGS` and `$TAPIR_HIP_FLAGS`
to put the target transformations into debug mode.  Note that 
this mode will also dump intermediate stage files such as LLVM
IR files that are helpful when debugging the compiler's code 
generation phases.  This environment variable simply needs to be
set to trigger the extra flags; the value it is set to is ignored 
and a check is simply made to see if it is set or not.


