=============================
User Guide for CudaABI Target Transform
=============================

.. contents::
   :local:
   :depth: 3


Introduction
============

The CudaABI target transformation is responsible for converting Tapir
IR form into a form suitable for running as a CUDA kernel.  The
transformation is dependent upon the Kitsune runtime that loosely follows 
the CUDA driver API semantics but with an extended feature set to simplify 
code generation and allow for a larger feature set.

The transformation is also dependent upon the NVPTX back-end as well
as the ``ptxas`` and ``fatbinary`` components from the CUDA Toolkit
installation.

// note::
More information about the CUDA Driver API and the PTX assembly
language can be found `here <http://docs.nvidia.com/cuda/index.html>`_.

.. _options:
CudaABI Options
===========

The transformation has a number of special command line arguments that
can help to tailor the code generation in terms of optimizations,
special runtime modes, and feedback on the generated code.  This
section quickly reviews these arguments.

These options are typically specified by preceding them with the
``--mllvm`` flag via Clang.

.. _target_options:
Architecture Targets / Code Generation 
---------------------------------------

``-cuabi-arch``: The target NVIDIA GPU architecture to generated code
for.  This goes hand-in-hand with the target name used by NVPTX and
must also match with the targets supported by the installed version of
CUDA.  For example,

.. code-block:: text

   --mllvm -cuabi-arch=sm_80

The default architecture can be specified at build time by defining
``CUDAABI_DEFAULT_ARCH``.  In addition, the environment variable
``CUDAABI_ARCH`` can be set to **override** the default but it will
not override an explicit use of the ``-cuabi-arch`` argument.

``-cuabi-march=64|32``: **deprecated** Specify if the host
architecture is a 32- or 64-bit processor.  This will currently
default to 64-bit and will soon be deprecated to match its deprecation
in the latest CUDA releases.

``-cuabi-maxregcount=N``: Specify the maximum amount of registers that
GPU functions can use -- higher values will (in general) increase the
performance of threads; however, because there is a global pool of
registers on a GPU, higher values will reduce the thread block size
and reduce the amount of thread parallelism.  Also note that not all
device registers may be available for use as some could be reserved
for use by the compiler.

``-cuabi-opt-level=N``: Specify the optimization level to use.  This
level will apply to both a post-transformation set of LLVM passes as
well as the ``ptxas`` assembler.

``--cuabi-reg-usage-level=[0..10]``: **BETA** Control how aggressive
the compiler (``ptxas``) will be in terms of applying optimizations
that impact register usage.  Higher levels optimize by trading off
using additional registers to improve performance, while lower values
will inhibit register use.  In general, this flag should be used in
concert with ``-cuab-maxregcount`` and the kernel launch parameters.
See the ``ptxas`` documentation for further details.  Default value
is ``5``. 

``-cuabi-pic``: Generate position-independent code. 

.. _feedback_options:
Feedback
--------

``-cuabi-verbose``:  Provide feedback about the generated GPU code
(e.g., register usage).  This output is primary provided by the
output produced by ``ptxas`` and other CUDA-specific toolchain
programs used by the transformation.

``-cuabi-warn-on-spills``: Issue a warning when registers are 
spilled to local memory. 

``-cuabi-warn-as-error``: All warnings result in fatal errors.

.. _debug_options:
Debug Code Generation 
---------------------

``-cuabi-debug``: Enable debug mode.  This mode is independent of more
common debugging flags (e.g., ``-g``).  This is primarily due to the
fact that Tapir target transformations occur later in the pipeline and
in general are disabled when the full pipeline is in debug mode.  At
present, the CUDA toolchain does not support optimized debug modes.

``-cuabi-generate-line-info``: Generate line information for the
generated GPU code. 

``-mllvm -debug-only="cuabi"``: Place the transformation into debug
output mode.  This often results in long and detailed output that
is not generally helpful but can be extremely valuable when bug
chasing!


