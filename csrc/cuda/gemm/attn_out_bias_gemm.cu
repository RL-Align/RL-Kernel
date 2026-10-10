// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
// csrc/cuda/gemm/attn_out_bias_gemm.cu
//
// Qwen-Image WS1 attn-out bias GEMM, CUDA correctness-anchor kernels
// (contract version "attn-out-bias-gemm-tree-v1", see
// docs/operators/attn-out-bias-gemm.md):
//
//   forward : y = single_cast( tree(x @ W^T) + b )   -- per-thread full tree
//   dx      : tree(dY @ W)                           -- same tree kernel
//   dW      : ascending-row LEFT FOLD (contract c-prime; no tree, no partials)
//
// Numeric contract, frozen:
//   * K/N-dim reduction: R splits into 32-wide leaves (short tail allowed);
//     each leaf is an ascending-k fp32 chain starting from +0.0; leaves merge
//     through a mid-split tree T(l,r) = T(l,m) + T(m,r). The tree depends
//     only on R, so row outputs are batch-invariant by construction.
//   * Multiply-add discipline: one correctly-rounded FP32 FMA on BOTH sides
//     (__fmaf_rn here; the FP32 reference uses torch.addcmul, an
//     oracle-verified correctly-rounded FMA), so the kernel matches the
//     reference byte for byte in BOTH dtypes -- no tolerance path.
//   * RNE everywhere; bias added once in fp32 after the complete tree; the
//     single fp32 -> output-dtype cast happens at the final store.
//   * No split-K, no atomics, no tensor-core mma (its internal accumulation
//     tree is not specified by PTX and cannot be frozen).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <type_traits>

namespace {

using nv_bf16 = __nv_bfloat16;

constexpr int kLeafWidth = 32;

__device__ __forceinline__ nv_bf16 cast_out_bf16(float value) {
    return __float2bfloat16(value);  // RNE
}

__device__ __forceinline__ float leaf_chain(
    const float* __restrict__ a_row, const float* __restrict__ b_row, int k0, int k1) {
    float acc = 0.0f;
    for (int k = k0; k < k1; ++k) {
        acc = __fmaf_rn(a_row[k], b_row[k], acc);
    }
    return acc;
}

// Mid-split tree over leaves [lo, hi); leaf_count = ceil(R / 32).
__device__ float tree_reduce(
    const float* __restrict__ a_row,
    const float* __restrict__ b_row,
    int reduction,
    int leaves,
    int lo,
    int hi) {
    if (hi - lo == 1) {
        const int k0 = lo * kLeafWidth;
        const int k1 = min(k0 + kLeafWidth, reduction);
        return leaf_chain(a_row, b_row, k0, k1);
    }
    const int mid = lo + (hi - lo) / 2;
    return tree_reduce(a_row, b_row, reduction, leaves, lo, mid) +
           tree_reduce(a_row, b_row, reduction, leaves, mid, hi);
}

template <typename out_t>
__global__ void attn_out_bias_gemm_fwd_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    out_t* __restrict__ y,
    int rows,
    int out_dim,
    int in_dim,
    int leaves) {
    const long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<long>(rows) * out_dim) {
        return;
    }
    const int s = static_cast<int>(idx / out_dim);
    const int n = static_cast<int>(idx % out_dim);
    float value = tree_reduce(
        x + static_cast<long>(s) * in_dim, w + static_cast<long>(n) * in_dim, in_dim, leaves, 0,
        leaves);
    if (bias != nullptr) {
        value = value + bias[n];  // bias once, fp32, after the complete tree
    }
    if constexpr (std::is_same<out_t, nv_bf16>::value) {
        y[idx] = cast_out_bf16(value);  // the single fp32 -> bf16 cast
    } else {
        y[idx] = value;
    }
}

__global__ void attn_out_dw_left_fold_kernel(
    const float* __restrict__ grad,
    const float* __restrict__ x,
    float* __restrict__ dw,
    int rows,
    int out_dim,
    int in_dim) {
    const long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<long>(out_dim) * in_dim) {
        return;
    }
    const int n = static_cast<int>(idx / in_dim);
    const int k = static_cast<int>(idx % in_dim);
    float acc = 0.0f;
    for (int s = 0; s < rows; ++s) {  // ascending-row left fold
        acc = __fmaf_rn(grad[static_cast<long>(s) * out_dim + n],
                        x[static_cast<long>(s) * in_dim + k], acc);
    }
    dw[idx] = acc;
}

void check_common(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32 (upcast on host)");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

int leaves_of(int64_t reduction) {
    TORCH_CHECK(reduction >= 0, "reduction length must be >= 0");
    return static_cast<int>((reduction + kLeafWidth - 1) / kLeafWidth);
}

}  // namespace

torch::Tensor attn_out_bias_gemm_cuda_forward(
    torch::Tensor x, torch::Tensor weight, c10::optional<torch::Tensor> bias, bool bf16_out) {
    const c10::cuda::CUDAGuard device_guard(x.device());
    check_common(x, "x");
    check_common(weight, "weight");
    TORCH_CHECK(weight.device() == x.device(), "weight must live on x's device");
    TORCH_CHECK(x.dim() == 2, "x must be 2-D [S, K]");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2-D [N, K]");
    TORCH_CHECK(x.size(1) == weight.size(1), "x K must match weight K");
    if (bias.has_value()) {
        check_common(*bias, "bias");
        TORCH_CHECK((*bias).numel() == weight.size(0), "bias must have N elements");
        TORCH_CHECK((*bias).device() == x.device(), "bias must live on x's device");
    }
    const int rows = static_cast<int>(x.size(0));
    const int out_dim = static_cast<int>(weight.size(0));
    const int in_dim = static_cast<int>(x.size(1));
    const int leaves = leaves_of(in_dim);

    const long total = static_cast<long>(rows) * out_dim;
    const int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const float* bias_ptr = bias.has_value() ? bias->data_ptr<float>() : nullptr;
    // The caller decides the output dtype (bf16 => the single final RNE cast
    // happens in-kernel at the store; fp32 => no cast).
    auto out = torch::empty(
        {x.size(0), weight.size(0)}, x.options().dtype(bf16_out ? at::kBFloat16 : at::kFloat));
    if (total == 0) {
        return out;  // zero-row/grid-0 launches are invalid; skip synchronously
    }
    if (bf16_out) {
        attn_out_bias_gemm_fwd_kernel<nv_bf16><<<blocks, threads, 0, stream>>>(
            x.data_ptr<float>(), weight.data_ptr<float>(), bias_ptr,
            reinterpret_cast<nv_bf16*>(out.data_ptr()), rows, out_dim, in_dim, leaves);
    } else {
        attn_out_bias_gemm_fwd_kernel<float><<<blocks, threads, 0, stream>>>(
            x.data_ptr<float>(), weight.data_ptr<float>(), bias_ptr, out.data_ptr<float>(), rows,
            out_dim, in_dim, leaves);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor attn_out_tree_gemm_cuda(torch::Tensor a, torch::Tensor b) {
    const c10::cuda::CUDAGuard device_guard(a.device());
    check_common(a, "a");
    check_common(b, "b");
    TORCH_CHECK(b.device() == a.device(), "b must live on a's device");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "a and b must be 2-D");
    TORCH_CHECK(a.size(1) == b.size(1), "reduction dims must match");
    const int rows = static_cast<int>(a.size(0));
    const int cols = static_cast<int>(b.size(0));
    const int reduction = static_cast<int>(a.size(1));
    const int leaves = leaves_of(reduction);

    auto out = torch::empty({a.size(0), b.size(0)}, a.options());
    const long total = static_cast<long>(rows) * cols;
    if (total == 0) {
        return out;
    }
    const int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    attn_out_bias_gemm_fwd_kernel<float><<<blocks, threads, 0, stream>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), nullptr, out.data_ptr<float>(), rows, cols,
        reduction, leaves);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor attn_out_dw_left_fold_cuda(torch::Tensor grad, torch::Tensor x) {
    const c10::cuda::CUDAGuard device_guard(grad.device());
    check_common(grad, "grad");
    check_common(x, "x");
    TORCH_CHECK(x.device() == grad.device(), "x must live on grad's device");
    TORCH_CHECK(grad.dim() == 2 && x.dim() == 2, "grad and x must be 2-D");
    TORCH_CHECK(grad.size(0) == x.size(0), "row counts must match");
    const int rows = static_cast<int>(grad.size(0));
    const int out_dim = static_cast<int>(grad.size(1));
    const int in_dim = static_cast<int>(x.size(1));

    auto dw = torch::empty({grad.size(1), x.size(1)}, grad.options());
    const long total = static_cast<long>(out_dim) * in_dim;
    if (total == 0) {
        return dw;
    }
    const int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    attn_out_dw_left_fold_kernel<<<blocks, threads, 0, stream>>>(
        grad.data_ptr<float>(), x.data_ptr<float>(), dw.data_ptr<float>(), rows, out_dim, in_dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return dw;
}
