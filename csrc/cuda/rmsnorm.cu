#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

template <typename scalar_t>
__device__ __forceinline__ float load_as_float(const scalar_t* ptr) {
    return static_cast<float>(*ptr);
}

template <>
__device__ __forceinline__ float load_as_float<at::Half>(const at::Half* ptr) {
    const __half* p = reinterpret_cast<const __half*>(ptr);
    return __half2float(*p);
}

template <>
__device__ __forceinline__ float load_as_float<at::BFloat16>(const at::BFloat16* ptr) {
    const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(ptr);
    return __bfloat162float(*p);
}


template <typename scalar_t>
__device__ __forceinline__ void store_from_float(scalar_t* ptr, float v) {
    *ptr = static_cast<scalar_t>(v);
}

template <>
__device__ __forceinline__ void store_from_float<at::Half>(at::Half* ptr, float v) {
    __half* p = reinterpret_cast<__half*>(ptr);
    *p = __float2half(v);
}

template <>
__device__ __forceinline__ void store_from_float<at::BFloat16>(at::BFloat16* ptr, float v) {
    __nv_bfloat16* p = reinterpret_cast<__nv_bfloat16*>(ptr);
    *p = __float2bfloat16(v);
}


__device__ __forceinline__ float block_reduce_sum(float v) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;

    smem[tid] = v;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    return smem[0];
}


static int choose_threads(int H) {
    if (H <= 64) return 64;
    if (H <= 128) return 128;
    if (H <= 256) return 256;
    return 512;
}


template <typename scalar_t, typename weight_t>
__global__ void rmsnorm_fwd_kernel(
    const scalar_t* __restrict__ x,
    const weight_t* __restrict__ weight,
    scalar_t* __restrict__ y,
    float* __restrict__ rstd,
    int T,
    int H,
    float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    const scalar_t* x_row = x + row * H;
    scalar_t* y_row = y + row * H;

    float local_sum = 0.0f;

    // 计算 sum(x^2)，每个 thread 负责若干列。
    for (int col = tid; col < H; col += blockDim.x) {
        float xv = load_as_float<scalar_t>(x_row + col);
        local_sum += xv * xv;
    }

    // 固定 block reduction。
    float sum = block_reduce_sum(local_sum);

    float row_rstd = rsqrtf(sum / static_cast<float>(H) + eps);

    if (tid == 0) {
        rstd[row] = row_rstd;
    }

    __syncthreads();

    // 写出 y = x * rstd * weight。
    for (int col = tid; col < H; col += blockDim.x) {
        float xv = load_as_float<scalar_t>(x_row + col);
        float wv = load_as_float<weight_t>(weight + col);
        float out = xv * row_rstd * wv;
        store_from_float<scalar_t>(y_row + col, out);
    }
}


template <typename scalar_t, typename weight_t>
__global__ void rmsnorm_bwd_dx_kernel(
    const scalar_t* __restrict__ dy,
    const scalar_t* __restrict__ x,
    const weight_t* __restrict__ weight,
    const float* __restrict__ rstd,
    scalar_t* __restrict__ dx,
    int T,
    int H
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    const scalar_t* dy_row = dy + row * H;
    const scalar_t* x_row = x + row * H;
    scalar_t* dx_row = dx + row * H;

    float local_dot = 0.0f;

    for (int col = tid; col < H; col += blockDim.x) {
        float dyv = load_as_float<scalar_t>(dy_row + col);
        float xv = load_as_float<scalar_t>(x_row + col);
        float wv = load_as_float<weight_t>(weight + col);
        local_dot += dyv * wv * xv;
    }

    float dot = block_reduce_sum(local_dot);

    float r = rstd[row];
    float coeff = dot * r * r * r / static_cast<float>(H);

    for (int col = tid; col < H; col += blockDim.x) {
        float dyv = load_as_float<scalar_t>(dy_row + col);
        float xv = load_as_float<scalar_t>(x_row + col);
        float wv = load_as_float<weight_t>(weight + col);

        float out = r * dyv * wv - xv * coeff;
        store_from_float<scalar_t>(dx_row + col, out);
    }
}


template <typename scalar_t>
__global__ void rmsnorm_partial_dw_kernel(
    const scalar_t* __restrict__ dy,
    const scalar_t* __restrict__ x,
    const float* __restrict__ rstd,
    const bool* __restrict__ mask,
    float* __restrict__ partial_dw,
    int T,
    int H
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = T * H;

    if (idx >= total) return;

    int row = idx / H;

    if (!mask[row]) {
        partial_dw[idx] = 0.0f;
        return;
    }

    float dyv = load_as_float<scalar_t>(dy + idx);
    float xv = load_as_float<scalar_t>(x + idx);
    float r = rstd[row];

    partial_dw[idx] = dyv * xv * r;
}


__global__ void rmsnorm_reduce_dw_kernel(
    const float* __restrict__ partial_dw,
    float* __restrict__ dw,
    int T,
    int H
) {
    int h = blockIdx.x;
    int tid = threadIdx.x;

    float local_sum = 0.0f;

    for (int t = tid; t < T; t += blockDim.x) {
        local_sum += partial_dw[t * H + h];
    }

    float sum = block_reduce_sum(local_sum);

    if (tid == 0) {
        dw[h] = sum;
    }
}


void rmsnorm_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor y,
    torch::Tensor rstd,
    double eps
) {
    int T = x.size(0);
    int H = x.size(1);
    int threads = choose_threads(H);
    size_t smem = threads * sizeof(float);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "rmsnorm_forward_cuda", [&] {
        using x_t = scalar_t;
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, weight.scalar_type(), "rmsnorm_forward_weight_cuda", [&] {
            using w_t = scalar_t;
            rmsnorm_fwd_kernel<x_t, w_t><<<T, threads, smem, stream>>>(
                x.data_ptr<x_t>(),
                weight.data_ptr<w_t>(),
                y.data_ptr<x_t>(),
                rstd.data_ptr<float>(),
                T,
                H,
                static_cast<float>(eps)
            );
        });
    });
}


void rmsnorm_backward_dx_cuda(
    torch::Tensor dy,
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor rstd,
    torch::Tensor dx
) {
    int T = x.size(0);
    int H = x.size(1);
    int threads = choose_threads(H);
    size_t smem = threads * sizeof(float);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "rmsnorm_backward_dx_cuda", [&] {
        using x_t = scalar_t;
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, weight.scalar_type(), "rmsnorm_backward_dx_weight_cuda", [&] {
            using w_t = scalar_t;
            rmsnorm_bwd_dx_kernel<x_t, w_t><<<T, threads, smem, stream>>>(
                dy.data_ptr<x_t>(),
                x.data_ptr<x_t>(),
                weight.data_ptr<w_t>(),
                rstd.data_ptr<float>(),
                dx.data_ptr<x_t>(),
                T,
                H
            );
        });
    });
}


void rmsnorm_backward_partial_dw_cuda(
    torch::Tensor dy,
    torch::Tensor x,
    torch::Tensor rstd,
    torch::Tensor mask,
    torch::Tensor partial_dw
) {
    int T = x.size(0);
    int H = x.size(1);
    int total = T * H;

    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "rmsnorm_partial_dw_cuda", [&] {
        rmsnorm_partial_dw_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            dy.data_ptr<scalar_t>(),
            x.data_ptr<scalar_t>(),
            rstd.data_ptr<float>(),
            mask.data_ptr<bool>(),
            partial_dw.data_ptr<float>(),
            T,
            H
        );
    });
}


void rmsnorm_backward_reduce_dw_cuda(
    torch::Tensor partial_dw,
    torch::Tensor dw
) {
    int T = partial_dw.size(0);
    int H = partial_dw.size(1);

    int threads = choose_threads(T);
    size_t smem = threads * sizeof(float);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    rmsnorm_reduce_dw_kernel<<<H, threads, smem, stream>>>(
        partial_dw.data_ptr<float>(),
        dw.data_ptr<float>(),
        T,
        H
    );
}
