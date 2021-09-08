extern "C"
__global__ void f(float *x, float *y, float *out)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    out[tid] = x[tid] + y[tid];
}
