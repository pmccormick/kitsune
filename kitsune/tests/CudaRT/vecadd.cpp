//
// A simple vector addition example -- this is essentially a version
// of the CUDA example in cuda/samples/0_Simple/vectorAddDrv modified
// to use the kitsune cuda runtime interface. 
//
#include <stdio.h>
#include <math.h>
#include "../cudart.h"

using namespace std;

// NOTE: PTX can be finicky in terms of whitespace characters and
// such. The tell-tale sign of an issue is often a complete hang in
// the CUDA driver api when loading the module.
const char *PTX_CODE =
  ".version 7.0\n\
   .target sm_35\n\
   .address_size 64\n\
   // .globl VecAdd_kernel\n\
   .visible .entry VecAdd_kernel(.param .u64 VecAdd_kernel_param_0,\n\
                                 .param .u64 VecAdd_kernel_param_1,\n\
                                 .param .u64 VecAdd_kernel_param_2,\n\
                                 .param .u32 VecAdd_kernel_param_3) {\n\
     .reg .pred      %p<2>;\n\
     .reg .f32       %f<4>;\n\
     .reg .b32       %r<6>;\n\
     .reg .b64       %rd<11>;\n\
     \n\
     \n\
     ld.param.u64    %rd1, [VecAdd_kernel_param_0];\n\
     ld.param.u64    %rd2, [VecAdd_kernel_param_1];\n\
     ld.param.u64    %rd3, [VecAdd_kernel_param_2];\n\
     ld.param.u32    %r2, [VecAdd_kernel_param_3];\n\
     mov.u32         %r3, %ntid.x;\n\
     mov.u32         %r4, %ctaid.x;\n\
     mov.u32         %r5, %tid.x;\n\
     mad.lo.s32      %r1, %r4, %r3, %r5;\n\
     setp.ge.s32     %p1, %r1, %r2;\n\
     @%p1 bra        BB0_2;\n\
     \n\
     cvta.to.global.u64      %rd4, %rd1;\n\
     mul.wide.s32    %rd5, %r1, 4;\n\
     add.s64         %rd6, %rd4, %rd5;\n\
     cvta.to.global.u64      %rd7, %rd2;\n\
     add.s64         %rd8, %rd7, %rd5;\n\
     ld.global.f32   %f1, [%rd8];\n\
     ld.global.f32   %f2, [%rd6];\n\
     add.f32         %f3, %f2, %f1;\n\
     cvta.to.global.u64      %rd9, %rd3;\n\
     add.s64         %rd10, %rd9, %rd5;\n\
     st.global.f32   [%rd10], %f3;\n\
     BB0_2:\n\
     ret;\n\
     }\n";

void random_fill(float *data, size_t N) {
  for(unsigned int i = 0; i < N; ++i) 
    data[i] = rand() / (float)RAND_MAX;
}


int main(int argc, char *argv[]) {
  
  // initialize the runtime. 
  __kitsune_cudart_initialize();

  // Make sure we have at least one GPU to work with...
  // 
  // Each GPU target can be identified by a integer value from
  // [0, 1, ... N-1], where N is the number of GPUs/devices
  // reported by the runtime. 
  if (__kitsune_cudart_ndevices() > 0) {
    int N = 50000;
    size_t Nbytes = N * sizeof(float);
    
    float *hostA = (float *)malloc(Nbytes);
    random_fill(hostA, N);
    
    float *hostB = (float *)malloc(Nbytes);
    random_fill(hostB, N);
    
    float *hostC = (float *)malloc(Nbytes);
    random_fill(hostC, N);    
    
    // Create a kernel on the given device (device = 0), from the given
    // PTX source buffer, and the name of the function/kernel call in
    // the PTX source. 
    int kID = __kitsune_cudart_create_kernel(0, PTX_CODE, "VecAdd_kernel");

    // Now, add a series of arguments to the kernel.  We must
    // associate each argument with the kernel ID (returned by the
    // create_kernel() call), a pointer to the argument on the host,
    // the "kind" of argument, and the access mode for the argument.
    //
    // The argument kind idenitifies scalars and data blocks
    // (arrays/buffers).
    //
    // The access mode determines if the data needs to copied to the
    // GPU for read only operations, copied to and from the GPU for
    // read and write, or copied back from the GPU after kernel
    // execution for write only.
    //
    // For this example we're executing
    //
    //    C = A + B
    //
    // So, A and B are specified as read-only data blocks and C is a
    // write-only data block.
    //
    // The kernel also needs to know the number of elements in the
    // data blocks (N) -- thus, we also provide N as a scalar argument
    // to the kernel.  This corresponds to the PTX function signature
    // above:
    //
    //  .visible .entry VecAdd_kernel(.param .u64 VecAdd_kernel_param_0,
    //                                .param .u64 VecAdd_kernel_param_1,
    //                                .param .u64 VecAdd_kernel_param_2,
    //                                .param .u32 VecAdd_kernel_param_3)
    //
    // Binding to parameters is done in call order (parameter #1
    // first, #2 next, etc.
    // 
    __kitsune_cudart_add_arg(kID, hostA, Nbytes,
			     CUDART_ARG_TYPE_DATA_BLK,
			     CUDART_ARG_READ_ONLY);
  
    __kitsune_cudart_add_arg(kID, hostB, Nbytes,
			     CUDART_ARG_TYPE_DATA_BLK,
			     CUDART_ARG_READ_ONLY);
  
    __kitsune_cudart_add_arg(kID, hostC, Nbytes,
			     CUDART_ARG_TYPE_DATA_BLK,
			     CUDART_ARG_WRITE_ONLY);

    __kitsune_cudart_add_arg(kID, &N, sizeof(int),
			     CUDART_ARG_TYPE_SCALAR,
			     CUDART_ARG_READ_ONLY);

    // Finally, set some specific parameters and launch the
    // kernel.
    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;  
    __kitsune_cudart_set_grid_dims(kID, blocks_per_grid, 1, 1);
    __kitsune_cudart_set_block_dims(kID, threads_per_block, 1, 1);
    __kitsune_cudart_launch_kernel(kID);
  
    // Verify correct result (taking some floating point nuances into
    // play)...
    size_t i = 0;
    for(i = 0; i < N; ++i) {
      float sum = hostA[i] + hostB[i];
      if (fabs(hostC[i] - sum) > 1e-7f)
	break; // whoops, that's not good... 
    }

    fprintf(stdout, "Result = %s\n", (i == N) ? "PASS" : "FAIL");
    free((void*)hostA);
    free((void*)hostB);
    free((void*)hostC);
    __kitsune_cudart_finalize();
    exit((i == N) ? EXIT_SUCCESS : EXIT_FAILURE);
  }
  fprintf(stdout, "no gpu devices found!\n");
  exit(EXIT_FAILURE);
}
  
