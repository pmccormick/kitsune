Kitsune Runtime CUDA Interface
------------------------------
The details of the CUDA portion of the Kitsune runtime are closely aligned with the code generation details within the Tapir CUDA ABI transform.  The most prominent goal of the design is to reduce the complexity of the code generation.  In general, this is currently done by:

  - Hiding CUDA-centric data types in return types or parameters. In most cases a void pointer is used to obscure these types. 
  - Using C calling conventions to avoid hassles with C++ naming. 
  - Finally, by reducing the number of direct CUDA calls into higher level building blocks.

High-Level Design Details
-------------------------
There are several current internal implementation design choices that impact performance, feature set, and ease of code generation.  This section quickly discusses these specific choices and any associated details that can impact correctness, performance, and overall behavior. 

  - **Managed Memory**: The runtime currently only support the use of managed memory allocations.  This is also true of the coding style and code generation mechanisms in the compiler.  This is primarily done for correctness, simplification, and avoiding explicit memory movement code in user application code.  The compiler will insert asynchronous prefetch calls on managed memory pointers to attempt overlapping data movement and compute operations.  The runtime tracks each managed memory allocation to issue prefetch calls but does not attempt to track its location and status in device or host memories.
  - **Execution Streams**: The runtime constructs a stream per calling thread and each thread uses that stream for prefetching data, launching kernels, and synchronizing execution.  Care must be taken to avoid overwhelming GPU resources.  This is particularly important when sharing managed memory allocations as this can lead to excessive page faults that can have an extremely negative impact on performance.  These streams can be used in tandem with CPU-based computations using the `-ftapir=opencilk` target.  At present this feature set should be considered experimental as the two runtime systems are not aware of each other and the potential for unfriendly interactions are possible.
  - **Launch Parameters**: The current kernel launching support is very basic and only uses one-dimensional launches.  There are entry points for customizing these parameters but they are largely unused by the compiler in the current implementation. Some parameters may be overridden via environment variables and can help during experimentation and debugging alongside changes in the compiler's code generation details. 
  - **Profiling Hooks**: The primary code generation entry points in the runtime are decorated with calls into NVIDIA's profiling tool API (NVTX). These details can be helpful when used with the NSight tool suite.  There are cases where the Kitsune overheads are higher than CUDA's (due to the underlying bookkeeping details) and these tools can help address spots for performance improvements within the runtime. 
  - **Error Conditions**: All CUDA calls in the runtime are guarded and checked for errors.  When a CUDA call returns an error condition, the runtime will report the failure, CUDA's error message, and then abort program execution. The runtime also makes liberal use of assertions to catch unexpected error states (e.g., an unexpected null pointer).  The vast majority of these errors are most often encountered when hitting compiler code generation errors and a hard error is often the easiest path for debugging. 

Additional Information 
----------------------
More information for the implementation details can be found directly in the runtime's source code.  Instead of higher-level CUDA API, the runtime uses the [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/index.html).

The compiler code generation for CUDA is enabled using the `-ftapir=cuda` flag.  For a complete picture of the connection between the runtime and the compiler referring to both implementations is often important. 

