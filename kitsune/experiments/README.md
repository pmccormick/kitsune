
The experiments all use a build process that relies on
a few environment variables.  Specifically, the install
prefix for kitsune should be set via the KITSUNE_PREFIX
variable:

  $ export KITSUNE_PREFIX=/projects/kitsune/x86_64/15.x

With this set the makefiles will invoke the kitsune
version of clang via the full path.

Builds of cuda- and hip-based versions of executables
are enabled via (somewhat traditional) enviornment
variables that point to the install prefix of cuda
and rocm/hip.  These two variabes are (for example):

  $ export CUDA_PATH=/opt/cuda
  $ export ROCM_PATH=/opt/rocm-5.4.3

There are various other environment variables exposed
via the makefile infrastructure (see the inc/ directory).
These can be used to tailor various aspects of the
build.

Kokkos Code

Most of the experiments include some Kokkkos-specific
versions.  By default it is assumed that the Kokkos
install prefix is $(KITSUNE_PREFIX)/opt/kokkos/[cuda|hip].
You can override this by setting the KOKKOS_PREFIX
environment variable.
