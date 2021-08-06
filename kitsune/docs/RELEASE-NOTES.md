## Kitsune+Tapir Release Notes

Kitsune follows the LLVM version numbering scheme and thus adopts the base
functionality of the corresponding LLVM release.  In the sections below each
release includes this version number for reference and a pointer to the corresponding 
set of LLVM release notes.

### Release 10.0.1 (08-XX-2021 update)

__Date__: August XX, 2021 
__Internal/ECP Release Milestone: ST_NS-XX-YYYY (__tagged__)

**TODO**: Can we catch up (merge) with LLVM 12.x/13.x???

[LLVM 10 Release Notes](https://releases.llvm.org/10.0.0/docs/ReleaseNotes.html)

### What's New 

  * Overhaul of the build configuration to account for new bitcode interfaces to runtime targets, better incorporation of using examples for testing across enabled architectures and runtime targets. 
  * Support for some Kokkos MDRange constructs (nested loops).
  * Updated progress on AMD GPU (HIP) code generation. 
  * Significant update and overhaul to the libomp runtime ABI target. 
  * Code refactoring to unify Kokoks and ``forall`` code generation infrastructure. 
  * Exploring possibliities and restrictions of using NVIDIA's NVVM library with LLVM 10.x (and beyond).
  * Improved internal error handling for bitcode loading cases that would previously crash the compiler. 
  * Additional bug fixes and compiler introduced race conditions as part of the ABI rutnime target code generation.
  * Improved flexbility by leveraging Clang's ``--config`` capabilities (should benefit advanced end users and developers by avoiding the need to rebuild the toolchain for certain use cases).
  * Some updates to the documentation for building and using the toolchain. 

### Release 10.0.1 (03-2021 update)

__Date__: March 9, 2021 
__Internal/ECP Release Milestone__: ST_NS-XX-YYYY (__tagged__)
[__LLVM 10 Release Notes__](https://releases.llvm.org/10.0.0/docs/ReleaseNotes.html)


*Note: Due to late bugs in LLVM 11.x and a delay of 12.x the default branch remains at ``release/10.x``*

### What's New

  * Build and configuration bug fixes for various search paths libaries. 

  * New CMake cache file for simplifying building for end-users and developers (helping to establish a fixed configuration set for release, testing, and reporting). 

  * New module files support for end-users and developers not leveraging Spack. 

  * Initial Realm ABI lowering is present but should be considered unstable for this release.  Note the Realm ABI will require some updates to Legion/Realm to squash some bugs that were discovered during development and testing. 

  * NVIDIA/CUDA/PTX, AMD/HIP, Intel/LevelZero support is awaiting attribute support for data movement operations.  


### Release 10.0.1 

__Date__: October 28, 2020
__Internal/ECP Release Milestone__: ST-NS-01-1330 (__tagged__)
[__LLVM 10 Release Notes__](https://releases.llvm.org/10.0.0/docs/ReleaseNotes.html)

*Note: Updated default branch to ``release/10.x``*

#### What's New

* Rebased to LLVM 10.0.1.

* Squashed a nasty bug related to avoiding race conditions and type-based alias analysis 
  (TBAA). Added some code to support early TBAA verification to make similar errors appear 
  earlier in the compilation stages (easier bug tracing).

* Updates to some configuration details for different (new) runtime targets (e.g., OpenCL 
  and a wrapper around CUDA for simplifying code generation implementation).

* Improved __``forall``__ support and bug fixes (traditional and range-based):

  * Fixed a couple of race condition in generated code -- specifically moved
    location of ``alloca`` statements such that they become task-/thread-local.

  * Found another race condition that appears to be specifically related to a
    bug in the ``qthreads`` target (i.e., ``-ftapir=qthreads``). Will report issue
    Qthreads developers.  *__Note__: We currently advise the careful use of
    the Qthreads target until this bug can be squashed.*

  * Fixed an issue related to the location of the ``detach`` instruction in ``forall``
    code generation.

* __Early CUDA ABI support__: New CUDA ABI (code generation) transformation
  support.  The current implementation has been tested and works but only under
  very specific conditions:
  * Should be considered extremely fragile...
  * Only targets CUDA 10.1, 64-bit code and the ``sm_70`` (Volta) architecture.
  * The compiler and ABI do not support the generation of data movement between
    CPU and GPU memories.
  * We have only tested on simple loops at specific optimization levels ``-O[1-3]``.
    Although simple, there are early signs of outperforming ``nvcc`` in some cases.

* __New CUDA runtime target__: A new runtime target library based on the NVIDIA
  driver API is provided in this release.  It is not yet supported by our code
  generation but aims to simplify code generation by hiding complex data structures
  and CUDA-centric data types that make for cumbersome code generation. Source
  and example program in kitsune/lib/CudaRT. 

* __Parallel IR (Tapir) updates__:  
  * __Verifier__: Add simple checks to verify that Tapir tasks have a valid
    structure and that ``sync`` regions are properly used.
  * __LoopSpawningTI__: Enable Tapir targets to maintain the same
    *LoopOutlineProcessor* structure beyond a Tapir single loop.
  * __LoopStripMine__: Prototype change to support loop strip mining to generate
    two nested parallel loops from a single parallel loop.  So far only tested on
    a simple loop with no exception-handling code.

* Kitsune has been merged/rebased with OpenCilk (beta 2) that includes both
  Tapir bug fixes and new features.

* Tweaks and changes to the CMake configuration to support building examples and
  tests for each enabled backend runtime target.  An initial set of [notes](building.md) about
  building Kitsune is now provided as part of this release.

* Starting to add a larger set of examples and tests.  A minimal set of items for
  this release but we will have more in the future as we squash some more bugs and
  get the basic CUDA and OpenCL targets behaving a bit more reliably.

### Bugs/Issues & "To Do" Items

* There can be significant performance issues with complex application codes and
  especially so when complicated iterator types are used for loops.  Some divergence
  bewteen ``forall`` and Kokkos ``parallel_for`` forms (which should lower to
  essentially the same parallel IR) show distinctly different behaviors.  This is
  likely a combination of issues that need to be separated and explored.

* An OpenCL backend target (parallel IR --> OpenCL) is nearing completion.  This
  should provide support for paths to different architectures such as GPUs from
  Intel and AMD.  We were unable to work a couple of nasty bugs out in time for
  this release.  We hope to push push out an update in the near future that will
  provide this support.

### Release 10.0.0

__Date__: June 18, 2020  
__Internal/ECP Release Milestone__: STNS01-11
[__LLVM 10 Release Notes__](https://releases.llvm.org/10.0.0/docs/ReleaseNotes.html)

* Rebased for LLVM 10.0.0. 

* __``forall``__ keyword support: Provides a parallel extension to standard C/C++
  ``for`` statements that enable parallel loop code generation.  The semantics of
  ``forall`` match a common data-parallel execution that requires that each loop
  iteration be independent of all others and no specific order of execution among
  the iterations should be assumed by the programmer.  It is the programmer's  responsibility to guarantee code follows these guidelines.

  * Both *"standard"* C-style ``for`` statements as well as C++'s more recent
    range-based loops styles are supported.  Assuming there are thread safety
    guarantees, these features can be enabled using various types of iterators
    that include both integral and more complex C++-style iterators from the
    STL.  

* __C++ Attributes for Tapir Code Generation Hints__:

  * New mechanisms for specifying per-statement code generation details have been
    added but are not yet fully supported in this release.  A runtime target may be
    specified via the ``tapir::rt_target`` attribute.  In addition, the statement
    *execution strategy* (e.g., divide-and-conquer) can be specified via the
    ``tapir::strategy`` attribute.  At present these may only be applied to ``forall``
    statements (both traditional and range-based).

    * Future functionality will allow for GPU and CPU targets to be mixed in
      application code -- this overcomes the limitations of a single target support
      when using the ``-ftapir=runtime-target`` argument on the command line.  

* __``spawn`` and ``sync`` Statements__:
  Although primarily used for development testing two new statements are also
  enabled in the ``-fkitsune`` mode of Clang.  The ``spawn`` and ``sync``
  constructs support non-nested concurrency -- thus allowing interleaved tasks.  
  Conceptually a construct similar to ``forall`` can be captured via this
  construct:

  ```c++
    #include <kitsune.h> // required to avoid naming collisons...
    #include <stdio.h>

    int main() {
      for(int i = 0; i < 10; i++) spawn pl {
        printf("Hello %d\n", i);
      }
      printf("Done with the loop.\n");
      sync pl;
    }
    ```

* __Kokkos Support__: This release has support for recognizing Kokkos ``parallel_for``
  statements and lowering them to the parallel IR (Tapir). This path will actually
  disable all template-based mechanisms (disabling all Kokkos library functionality)
  and replace the lambda form with an intermediate representation that matches a
  more traditional loop construct.  This is essentially the same path taken by
  the ``forall`` constructs mentioned above.

  * The supported Kokkos constructs are a minimal set of all possible approaches.
    At present only lambda-based (within a single compilation unit) are supported
    and match this basic form:

    ```c++
      Kokkos::parallel_for(N, KOKKOS_LAMBDA(const int i) {
        C[i] = A[i] + B[i];
      });
    ```
