#include <cuda_runtime.h>
#include "stable_topk6.cuh"
#ifndef P3_SOURCE_SHA
#define P3_SOURCE_SHA "UNSET"
#endif
__global__ void p3_topk_kernel(const float* q,int32_t* ids,int T,p3::P3StatusRecord* status,
                               unsigned long long invocation) {
    int row=blockIdx.x*blockDim.x+threadIdx.x;
    if (row<T) p3::stable_topk6_row(q+row*256,ids+row*6,status);
    if (blockIdx.x==0 && threadIdx.x==0) {
        __threadfence(); atomicExch(&status->invocation_echo,invocation);
    }
}
__global__ void p3_math_kernel(const float* x,float* out,int N,int op) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=N) return;
    switch(op) {
        case 0: out[i]=p3::exp(x[i]); break;
        case 1: out[i]=p3::log1p(x[i]); break;
        case 2: out[i]=p3::softplus(x[i]); break;
        case 3: out[i]=p3::sigmoid(x[i]); break;
        case 4: out[i]=p3::root(x[i]); break;
        case 5: out[i]=p3::bf16(x[i]); break;
    }
}
extern "C" int p3_launch_topk(const float* q,int32_t* ids,int T,p3::P3StatusRecord* status,
                              unsigned long long invocation,void* stream,int block) {
    p3_topk_kernel<<<(T+block-1)/block,block,0,(cudaStream_t)stream>>>(q,ids,T,status,invocation);
    return (int)cudaGetLastError();
}
extern "C" int p3_launch_math(const float* x,float* out,int N,int op,void* stream) {
    if (!N) return 0;
    p3_math_kernel<<<(N+127)/128,128,0,(cudaStream_t)stream>>>(x,out,N,op);
    return (int)cudaGetLastError();
}
extern "C" const char* p3_source_sha() { return P3_SOURCE_SHA; }
extern "C" const char* p3_build_flags() {
    return "sm_90;ftz=false;fmad=false;prec-div=true;prec-sqrt=true;fast-math=false";
}
extern "C" int p3_binary_arch() {
    cudaFuncAttributes a; if (cudaFuncGetAttributes(&a,p3_topk_kernel)!=cudaSuccess) return -1;
    return a.binaryVersion;
}
