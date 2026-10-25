// SPDX-License-Identifier: Apache-2.0
// Fixed-tile FP32 primitives. No atomic reductions or shape-dependent dispatch.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace {
struct Fp32Pair { float hi, lo; };
__device__ __forceinline__ Fp32Pair pair_add(Fp32Pair a, Fp32Pair b) {
    float s=a.hi+b.hi;
    float v=s-a.hi;
    float error=(a.hi-(s-v))+(b.hi-v);
    float t=(a.lo+b.lo)+error;
    float hi=s+t;
    return {hi,t-(hi-s)};
}
// Each output keeps one reduction order regardless of the number of rows.
__global__ void timestep_mm_short(const float* a, const float* b, float* c,
                                  int m, int n, int k, int64_t as0, int64_t as1,
                                  int64_t bs0, int64_t bs1) {
    int out = blockIdx.x * blockDim.x + threadIdx.x;
    if (out >= m*n) return;
    int row=out/n, col=out%n;
    float acc=0.f;
    for (int i=0; i<k; ++i) acc=__fmaf_rn(a[row*as0+i*as1],b[i*bs0+col*bs1],acc);
    c[out]=acc;
}
__global__ void timestep_mm_warp(const float* a, const float* b, float* c,
                                 int m, int n, int k, int64_t as0, int64_t as1,
                                 int64_t bs0, int64_t bs1) {
    int lane=threadIdx.x%32;
    int out=blockIdx.x*(blockDim.x/32)+threadIdx.x/32;
    if (out >= m*n) return;
    int row=out/n, col=out%n;
    Fp32Pair acc{0.f,0.f};
    for (int i=lane; i<k; i+=32) {
        float product=a[row*as0+i*as1]*b[i*bs0+col*bs1];
        float residual=__fmaf_rn(a[row*as0+i*as1],b[i*bs0+col*bs1],-product);
        acc=pair_add(acc,{product,residual});
    }
    for (int delta=16; delta>0; delta/=2) {
        Fp32Pair other{__shfl_down_sync(0xffffffff,acc.hi,delta),
                       __shfl_down_sync(0xffffffff,acc.lo,delta)};
        acc=pair_add(acc,other);
    }
    if (lane==0) c[out]=acc.hi+acc.lo;
}
__global__ void timestep_embedding(const float* t, const float* f, float* e, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * 128) return;
    int row=i/128, col=i%128;
    float p = (t[row]*f[col])*1000.f;
    e[row*256+col] = cosf(p);
    e[row*256+128+col] = sinf(p);
}
__global__ void timestep_silu(const float* x, float* y, int n, bool derivative) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v=x[i], s=1.f/(1.f+expf(-v));
    y[i] = derivative ? s+(v*s)*(1.f-s) : v*s;
}
__global__ void timestep_dt(const float* de, const float* e, const float* f,
                            float* dt, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    Fp32Pair result{0.f,0.f};
    for (int i=0; i<128; ++i) {
        float term=(-de[row*256+i]*e[row*256+128+i])
                  + de[row*256+128+i]*e[row*256+i];
        result=pair_add(result,{(term*1000.f)*f[i],0.f});
    }
    dt[row]=result.hi+result.lo;
}
void check(const torch::Tensor& x) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type()==at::kFloat, "expected CUDA FP32");
}
}

torch::Tensor mm(torch::Tensor a, torch::Tensor b) {
    check(a); check(b);
    TORCH_CHECK(a.dim()==2 && b.dim()==2 && a.size(1)==b.size(0), "bad mm shape");
    TORCH_CHECK(a.device()==b.device(), "device mismatch");
    c10::cuda::CUDAGuard guard(a.device());
    auto c=torch::empty({a.size(0),b.size(1)},a.options());
    if (c.numel()) {
        if (a.size(1)<=32) {
            timestep_mm_short<<<(c.numel()+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(
                a.data_ptr<float>(),b.data_ptr<float>(),c.data_ptr<float>(),
                a.size(0),b.size(1),a.size(1),a.stride(0),a.stride(1),b.stride(0),b.stride(1));
        } else {
            // Make the reduction dimension contiguous for coalesced warp loads.
            if (b.stride(0)!=1) b=b.transpose(0,1).contiguous().transpose(0,1);
            timestep_mm_warp<<<(c.numel()+7)/8,256,0,at::cuda::getCurrentCUDAStream()>>>(
                a.data_ptr<float>(),b.data_ptr<float>(),c.data_ptr<float>(),
                a.size(0),b.size(1),a.size(1),a.stride(0),a.stride(1),b.stride(0),b.stride(1));
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return c;
}
torch::Tensor embedding(torch::Tensor t, torch::Tensor f) {
    check(t); check(f);
    TORCH_CHECK(t.dim()==1 && t.is_contiguous() && f.numel()==128 && f.is_contiguous(), "bad embedding shape");
    TORCH_CHECK(t.device()==f.device(), "device mismatch");
    c10::cuda::CUDAGuard guard(t.device());
    auto e=torch::empty({t.numel(),256},t.options());
    if (t.numel()) {
        timestep_embedding<<<(t.numel()*128+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(t.data_ptr<float>(),f.data_ptr<float>(),e.data_ptr<float>(),t.numel());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return e;
}
torch::Tensor unary(torch::Tensor x, bool derivative) {
    check(x); TORCH_CHECK(x.is_contiguous(), "expected contiguous unary input");
    c10::cuda::CUDAGuard guard(x.device());
    auto y=torch::empty_like(x);
    if (x.numel()) {
        timestep_silu<<<(x.numel()+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(),y.data_ptr<float>(),x.numel(),derivative);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return y;
}
torch::Tensor timestep_grad(torch::Tensor de, torch::Tensor e, torch::Tensor f) {
    check(de); check(e); check(f);
    TORCH_CHECK(de.dim()==2 && de.size(1)==256 && e.sizes()==de.sizes() && f.numel()==128, "bad dt shape");
    TORCH_CHECK(de.is_contiguous() && e.is_contiguous() && f.is_contiguous(), "expected contiguous dt inputs");
    TORCH_CHECK(de.device()==e.device() && de.device()==f.device(), "device mismatch");
    c10::cuda::CUDAGuard guard(de.device());
    auto dt=torch::empty({de.size(0)},de.options());
    if (dt.numel()) {
        timestep_dt<<<(dt.numel()+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(de.data_ptr<float>(),e.data_ptr<float>(),f.data_ptr<float>(),dt.data_ptr<float>(),dt.numel());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return dt;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("mm",&mm); m.def("embedding",&embedding);
    m.def("unary",&unary); m.def("timestep_grad",&timestep_grad);
}
