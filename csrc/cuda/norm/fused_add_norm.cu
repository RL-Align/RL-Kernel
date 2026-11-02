// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// Muse-Glimmer-30B [F1]: post-norm, residual add, pre-norm, as one kernel.
//
// Mirrors rl_engine/reference/norm/fused_add_norm.py, which holds the frozen
// semantics and the reasons for them. Two things in particular are copied from
// there and are not incidental:
//
//   * both decoder norms are x * rstd * (1 + w), with the 1 added in FP32;
//   * the sum is narrowed to the input dtype before the pre-norm reads it, and
//     the post-norm result is narrowed before the add. Eager HF does both, so
//     the kernel does both; the stored residual_out is the narrowed value and
//     the second reduction reads that back rather than the FP32 sum.
//
// One block per row, strided columns, block_reduce_sum in a fixed tree, so a
// row's result does not depend on how many rows share the launch. That is
// what makes Axis-A batch invariance hold here; it is not the same tree the
// Triton kernel's tl.sum lowers to, and this file does not claim CUDA ==
// Triton bitwise for that reason (see the Triton module docstring).
//
// Element-wise expressions are written in the same association order as the
// Triton kernel so that, when the two are compared, the reduction is the only
// thing left to differ. FP contraction is left at the build's default like the
// rest of csrc/cuda.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <assert.h>

namespace {

template <typename scalar_t>
__device__ __forceinline__ float load_as_float(const scalar_t* ptr) {
    return static_cast<float>(*ptr);
}
template <>
__device__ __forceinline__ float load_as_float<at::Half>(const at::Half* ptr) {
    return __half2float(*reinterpret_cast<const __half*>(ptr));
}
template <>
__device__ __forceinline__ float load_as_float<at::BFloat16>(const at::BFloat16* ptr) {
    return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(ptr));
}

template <typename scalar_t>
__device__ __forceinline__ void store_from_float(scalar_t* ptr, float v) {
    *ptr = static_cast<scalar_t>(v);
}
template <>
__device__ __forceinline__ void store_from_float<at::Half>(at::Half* ptr, float v) {
    *reinterpret_cast<__half*>(ptr) = __float2half(v);
}
template <>
__device__ __forceinline__ void store_from_float<at::BFloat16>(at::BFloat16* ptr, float v) {
    *reinterpret_cast<__nv_bfloat16*>(ptr) = __float2bfloat16(v);
}

// Round a float through the storage dtype and back. This is the narrowing the
// reference does with .to(dtype).to(float32); for float it is the identity.
template <typename scalar_t>
__device__ __forceinline__ float narrow_to(float v) {
    scalar_t tmp;
    store_from_float<scalar_t>(&tmp, v);
    return load_as_float<scalar_t>(&tmp);
}

// Fixed-order tree reduction. Unlike the copy in rmsnorm.cu this ends with a
// barrier after the result is read out: the kernels below call it twice per
// row, and without the trailing sync a fast thread would start writing smem for
// the second reduction while a slow one is still reading smem[0] from the first.
__device__ __forceinline__ float block_reduce_sum(float v) {
    extern __shared__ float smem[];
    const int tid = threadIdx.x;
    assert((blockDim.x & (blockDim.x - 1)) == 0);

    smem[tid] = v;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    const float total = smem[0];
    __syncthreads();
    return total;
}

int choose_threads(int H) {
    if (H <= 64) return 64;
    if (H <= 128) return 128;
    if (H <= 256) return 256;
    return 512;
}

template <typename scalar_t, typename weight_t>
__global__ void fused_add_norm_fwd_kernel(
    const scalar_t* __restrict__ hidden,
    const scalar_t* __restrict__ residual,
    const weight_t* __restrict__ post_w,
    const weight_t* __restrict__ pre_w,
    scalar_t* __restrict__ res_out,
    scalar_t* __restrict__ pre_out,
    float* __restrict__ rstd_post,
    float* __restrict__ rstd_pre,
    int H,
    float post_eps,
    float pre_eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const scalar_t* h_row = hidden + static_cast<size_t>(row) * H;
    const scalar_t* r_row = residual + static_cast<size_t>(row) * H;
    scalar_t* ro_row = res_out + static_cast<size_t>(row) * H;
    scalar_t* po_row = pre_out + static_cast<size_t>(row) * H;

    // Post-norm statistic over hidden.
    float local = 0.0f;
    for (int col = tid; col < H; col += blockDim.x) {
        const float hv = load_as_float<scalar_t>(h_row + col);
        local += hv * hv;
    }
    const float rp = rsqrtf(block_reduce_sum(local) / static_cast<float>(H) + post_eps);
    if (tid == 0) rstd_post[row] = rp;

    // post = hidden * rstd * (1 + w), narrowed; res = residual + post, narrowed.
    // Each thread writes the columns it will read back below, so no barrier is
    // needed between the write and the re-read.
    for (int col = tid; col < H; col += blockDim.x) {
        const float hv = load_as_float<scalar_t>(h_row + col);
        const float gain = 1.0f + load_as_float<weight_t>(post_w + col);
        const float post = narrow_to<scalar_t>(hv * rp * gain);
        const float res = load_as_float<scalar_t>(r_row + col) + post;
        store_from_float<scalar_t>(ro_row + col, res);
    }

    // Pre-norm statistic over the narrowed residual, read back from storage.
    local = 0.0f;
    for (int col = tid; col < H; col += blockDim.x) {
        const float rv = load_as_float<scalar_t>(ro_row + col);
        local += rv * rv;
    }
    const float rq = rsqrtf(block_reduce_sum(local) / static_cast<float>(H) + pre_eps);
    if (tid == 0) rstd_pre[row] = rq;

    for (int col = tid; col < H; col += blockDim.x) {
        const float rv = load_as_float<scalar_t>(ro_row + col);
        const float gain = 1.0f + load_as_float<weight_t>(pre_w + col);
        store_from_float<scalar_t>(po_row + col, rv * rq * gain);
    }
}

template <typename scalar_t, typename weight_t>
__global__ void fused_add_norm_bwd_kernel(
    const scalar_t* __restrict__ g_res,
    const scalar_t* __restrict__ g_pre,
    const scalar_t* __restrict__ hidden,
    const scalar_t* __restrict__ res_out,
    const weight_t* __restrict__ post_w,
    const weight_t* __restrict__ pre_w,
    const float* __restrict__ rstd_post,
    const float* __restrict__ rstd_pre,
    float* __restrict__ d_hidden,
    float* __restrict__ d_residual,
    float* __restrict__ partial_post_w,
    float* __restrict__ partial_pre_w,
    int H
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const size_t base = static_cast<size_t>(row) * H;
    const float rp = rstd_post[row];
    const float rq = rstd_pre[row];

    // Pre-norm backward: dot over the row first.
    float local = 0.0f;
    for (int col = tid; col < H; col += blockDim.x) {
        const float gp = load_as_float<scalar_t>(g_pre + base + col);
        const float gain = 1.0f + load_as_float<weight_t>(pre_w + col);
        const float ro = load_as_float<scalar_t>(res_out + base + col);
        local += (gp * gain) * ro;
    }
    const float dot_pre = block_reduce_sum(local) / static_cast<float>(H);

    // d_res_out is live on two paths: downstream cotangent plus the pre-norm's
    // input gradient. Store it, and accumulate the post-norm dot in the same pass.
    local = 0.0f;
    for (int col = tid; col < H; col += blockDim.x) {
        const float gp = load_as_float<scalar_t>(g_pre + base + col);
        const float gain = 1.0f + load_as_float<weight_t>(pre_w + col);
        const float ro = load_as_float<scalar_t>(res_out + base + col);
        const float scaled = gp * gain;
        const float d_ro = load_as_float<scalar_t>(g_res + base + col)
                         + (rq * scaled - ro * rq * rq * rq * dot_pre);
        d_residual[base + col] = d_ro;
        partial_pre_w[base + col] = gp * ro * rq;

        const float hv = load_as_float<scalar_t>(hidden + base + col);
        const float pgain = 1.0f + load_as_float<weight_t>(post_w + col);
        local += (d_ro * pgain) * hv;
    }
    const float dot_post = block_reduce_sum(local) / static_cast<float>(H);

    for (int col = tid; col < H; col += blockDim.x) {
        const float d_ro = d_residual[base + col];
        const float hv = load_as_float<scalar_t>(hidden + base + col);
        const float pgain = 1.0f + load_as_float<weight_t>(post_w + col);
        const float scaled = d_ro * pgain;
        d_hidden[base + col] = rp * scaled - hv * rp * rp * rp * dot_post;
        partial_post_w[base + col] = d_ro * hv * rp;
    }
}

}  // namespace

void fused_add_norm_forward_cuda(
    torch::Tensor hidden, torch::Tensor residual,
    torch::Tensor post_w, torch::Tensor pre_w,
    torch::Tensor res_out, torch::Tensor pre_out,
    torch::Tensor rstd_post, torch::Tensor rstd_pre,
    double post_eps, double pre_eps
) {
    const int T = hidden.size(0);
    const int H = hidden.size(1);
    const int threads = choose_threads(H);
    const size_t smem = threads * sizeof(float);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, hidden.scalar_type(), "fused_add_norm_forward_cuda", [&] {
        using x_t = scalar_t;
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, post_w.scalar_type(), "fused_add_norm_forward_weight_cuda", [&] {
            using w_t = scalar_t;
            fused_add_norm_fwd_kernel<x_t, w_t><<<T, threads, smem, stream>>>(
                hidden.data_ptr<x_t>(), residual.data_ptr<x_t>(),
                post_w.data_ptr<w_t>(), pre_w.data_ptr<w_t>(),
                res_out.data_ptr<x_t>(), pre_out.data_ptr<x_t>(),
                rstd_post.data_ptr<float>(), rstd_pre.data_ptr<float>(),
                H, static_cast<float>(post_eps), static_cast<float>(pre_eps));
        });
    });
}

void fused_add_norm_backward_cuda(
    torch::Tensor g_res, torch::Tensor g_pre,
    torch::Tensor hidden, torch::Tensor res_out,
    torch::Tensor post_w, torch::Tensor pre_w,
    torch::Tensor rstd_post, torch::Tensor rstd_pre,
    torch::Tensor d_hidden, torch::Tensor d_residual,
    torch::Tensor partial_post_w, torch::Tensor partial_pre_w
) {
    const int T = hidden.size(0);
    const int H = hidden.size(1);
    const int threads = choose_threads(H);
    const size_t smem = threads * sizeof(float);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, hidden.scalar_type(), "fused_add_norm_backward_cuda", [&] {
        using x_t = scalar_t;
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, post_w.scalar_type(), "fused_add_norm_backward_weight_cuda", [&] {
            using w_t = scalar_t;
            fused_add_norm_bwd_kernel<x_t, w_t><<<T, threads, smem, stream>>>(
                g_res.data_ptr<x_t>(), g_pre.data_ptr<x_t>(),
                hidden.data_ptr<x_t>(), res_out.data_ptr<x_t>(),
                post_w.data_ptr<w_t>(), pre_w.data_ptr<w_t>(),
                rstd_post.data_ptr<float>(), rstd_pre.data_ptr<float>(),
                d_hidden.data_ptr<float>(), d_residual.data_ptr<float>(),
                partial_post_w.data_ptr<float>(), partial_pre_w.data_ptr<float>(),
                H);
        });
    });
}
