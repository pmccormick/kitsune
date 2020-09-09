#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void kitsune_spirv_kernel(  __global double *a,                       
                       __global double *b,                       
                       __global double *c)
{                                                               
    //Get our global thread ID                                  
    int id = get_global_id(0);                                  
    c[id] = a[id] + b[id];                                  
}                                                               
