## Kitsune 10.x Release Notes

Kitsune follows the LLVM version numbering scheme and thus adopts the base 
functionality of the toolchain corresponding to that release.

*Date*: June 18, 2020  
*Internal/ECP Release Reference*: STNS01-11

  * Corresponding LLVM 10.x [Release Notes](https://releases.llvm.org/10.0.0/docs/ReleaseNotes.html)
  

<!--- Clang -- encourage markdown to add space... --->
* ___Clang Code Generation Details:___
    * Corresponding Clang 10.x [Release Notes](https://releases.llvm.org/10.0.0/tools/clang/docs/ReleaseNotes.html)

    * __``forall`` statements__:
        * The ``forall`` keyword is an extension to standard C/C++ ``for`` statements that enable parallel loop code generation.  The sematnics of the construct match common data-parallel execution and the programmer is responsible for assuming that each loop iteration is independent of all others and no explicit ordering of iterations should be assumed.
        * This release squashes a nasty (intermittent) race condition (typically encountered when debugging was enabled via ``-g`` as the optimizer helpfully *fixed* the issue... sometimes...).  Specifically, this bug was caused by issues related to the code generation approach for the loop induction variable. All references to the induction variables in the loop body are now handled as a thread local copy. 

        * ``forall`` support is enabled by using the ``-fkitsune`` command line argument. 

    * __``forall`` range statements__: 

        * Like traditional C/C++ ``for`` statements this release extends  ``forall`` to support C++ ``for`` range statements.  The same rules of execution apply to range statements (e.g., every loop iteration must be safe to execute independently of all others, no ordering of iterations should be expected).  In general, C++ iterators should be safe in this use case provided they are thread safe (i.e. don't update global state, etc.).  

        * This release addresses the same loop induction variable race issue discused above in the traditional *for-style* loops.

        * Testing coverage was added for arrays, STL vectors, and STL maps.

        * ``forall`` range statements are enabled by using the ``-fkitsune`` command line argument.

    * __New C++ Attributes for Tapir Code Generation Hints__:

        * New mechanisms for specifying per-statement code generation details have been added but are not yet fully supported in this
        release.  At a high-level, a new Tapir runtime target may be specified via the ``tapir::rt_target`` attribute.  In addition, the statement ``execution strategy`` (e.g., divide-and-conquer) can be specified via the ``tapir::strategy`` attribute.  At present these may only be applied to ``forall`` statements (both traditional and range-based).

        * Future functionality will allow for GPU and CPU targets to be mixed in application code -- this overcomes the limitations of a single target support currently encountered when using the ``-ftapir=runtime-target`` command line argument. 
        
    * __``spawn`` and ``sync`` Statements__:
        * Although primarily used for development testing two new statements are also enabled in the ``-fkitsune`` mode of Clang.  The ``spawn`` and ``sync`` constructs support non-nested concurrency -- thus allowing interleaved tasks.  Conceptually a construct similar to ``forall`` can be captured via this construct:
        ```
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
        * Notes about ``spawn`` and ``sync`` syntax. 

    * __Kokkos Support__
        * ``parallel_for`` --- partial support 
        * ``parallel_reduce`` --- unsupported 

<!--- skip -->
* __LLVM Feature Set__
    * __Target runtimes__: 
        * Qthreads --- 
        * Realm --- 
        * CilkRTS --- 
        * CUDA+NVPTX --- 
        * OpenMP -- 

