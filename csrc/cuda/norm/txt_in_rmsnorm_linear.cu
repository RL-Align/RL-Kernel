// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
// csrc/cuda/norm/txt_in_rmsnorm_linear.cu
//
// Qwen-Image WS1 txt_in RMSNorm->Linear, CUDA correctness-anchor kernels
// (contract version "txt-in-rmsnorm-linear-v1", see
// docs/operators/txt-in-rmsnorm-linear.md):
//
//   norm stats : sumsq = TreeH(x*x) (one thread per row, full frozen tree);
//                rstd = three-step sqrt (div.rn -> add.rn -> sqrt.rn ->
//                div.rn) with the pinned eps bit pattern 0x358637BD;
//                xhat = x * rstd ; z = xhat * gamma (two isolated muls --
//                the frozen FP32 seam, no cast)
//   forward    : single_cast( tree(z @ W^T) + b ) -- reuses the attn-out
//                per-thread tree kernel entry (same tree, R = 3584)
//   dx         : dz = tree(dY @ W^T-col-major) (attn-out entry on W.T);
//                dxhat = dz * gamma ; dot = TreeH FMA(dxhat, xhat) ;
//                t1 = dot / 3584 (div.rn) ; t2 = xhat * t1 ;
//                t3 = dxhat - t2 (isolated -- FMS contraction forbidden) ;
//                dx = rstd * t3
//   dgamma     : ascending-row LEFT FOLD over (du, xhat)
//   dW / db    : attn-out left-fold entry / host pure-add fold
//
// Numeric contract, frozen:
//   * H/N-dim reductions: 32-wide leaves (short tail allowed); each leaf an
//     ascending fp32 FMA chain from +0.0; leaves merge through a mid-split
//     tree T(l,r) = T(l,m) + T(m,r). The tree depends only on the reduction
//     length, so row outputs are batch-invariant by construction.
//   * One correctly-rounded FP32 FMA discipline on BOTH sides (__fmaf_rn
//     here; the FP32 reference uses torch.addcmul, an oracle-verified
//     correctly-rounded FMA) -- byte-for-byte equal in BOTH dtypes, no
//     tolerance path. Whole-fp64 1/sqrt is forbidden (it skips the sq32
//     rounding); sqrt.rn(x) == fp32(fp64 sqrt(x)) for fp32 x by the
//     double-rounding theorem for sqrt.
//   * RNE everywhere; the RMSNorm->Linear seam is FP32; a single
//     fp32 -> output-dtype cast happens at the final store.
//   * No atomics, no split-K, no tensor-core mma (unspecified accumulation).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>
#include <vector>

namespace {

constexpr int kLeafWidth = 32;
constexpr float kDivisorF32 = 3584.0f;  // 0x45600000, exact

// 1e-6 with the pinned bit pattern 0x358637BD (single rounding of the
// decimal constant; every backend must materialise these exact bits).
__device__ __forceinline__ float eps_f32() {
    return __uint_as_float(0x358637BDu);
}

__device__ __forceinline__ float leaf_chain(
    const float* __restrict__ a_row, const float* __restrict__ b_row, int k0, int k1) {
    float acc = 0.0f;
    for (int k = k0; k < k1; ++k) {
        acc = __fmaf_rn(a_row[k], b_row[k], acc);
    }
    return acc;
}

// Mid-split tree over leaves [lo, hi); mirrors the attn-out frozen tree.
__device__ float tree_reduce(
    const float* __restrict__ a_row,
    const float* __restrict__ b_row,
    int reduction,
    int lo,
    int hi) {
    if (hi - lo == 1) {
        const int k0 = lo * kLeafWidth;
        const int k1 = min(k0 + kLeafWidth, reduction);
        return leaf_chain(a_row, b_row, k0, k1);
    }
    const int mid = lo + (hi - lo) / 2;
    return tree_reduce(a_row, b_row, reduction, lo, mid) +
           tree_reduce(a_row, b_row, reduction, mid, hi);
}

// sumsq[s] / dot[s]: one thread per row evaluates the full frozen tree.
__global__ void row_tree_reduce_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int rows,
    int hidden,
    int leaves) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= rows) {
        return;
    }
    const float* a_row = a + static_cast<long>(s) * hidden;
    out[s] = tree_reduce(a_row, b + static_cast<long>(s) * hidden, hidden, 0, leaves);
}

// rstd = 1 / sqrt.rn( (sumsq / 3584) + eps ) -- each step one rn op.
__global__ void rstd_kernel(
    const float* __restrict__ sumsq,
    float* __restrict__ rstd,
    int rows) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= rows) {
        return;
    }
    const float var = __fdiv_rn(sumsq[s], kDivisorF32);
    const float t = __fadd_rn(var, eps_f32());
    const float sq = __fsqrt_rn(t);
    rstd[s] = __fdiv_rn(1.0f, sq);
}

// xhat = x * rstd ; z = xhat * gamma (two isolated rn muls -- the seam).
__global__ void xhat_z_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ rstd,
    float* __restrict__ xhat,
    float* __restrict__ z,
    long total,
    int hidden) {
    const long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const int s = static_cast<int>(idx / hidden);
    const int h = static_cast<int>(idx % hidden);
    const float xh = __fmul_rn(x[idx], rstd[s]);
    xhat[idx] = xh;
    z[idx] = __fmul_rn(xh, gamma[h]);
}

__global__ void dxhat_kernel(
    const float* __restrict__ dz,
    const float* __restrict__ gamma,
    float* __restrict__ dxhat,
    long total,
    int hidden) {
    const long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    dxhat[idx] = __fmul_rn(dz[idx], gamma[idx % hidden]);
}

__global__ void t1_kernel(
    const float* __restrict__ dot,
    float* __restrict__ t1,
    int rows) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= rows) {
        return;
    }
    t1[s] = __fdiv_rn(dot[s], kDivisorF32);
}

// t2 = xhat * t1 ; t3 = dxhat - t2 ; dx = rstd * t3 -- all isolated rn ops
// (explicit intrinsics cannot be contracted into an FMS).
__global__ void dx_kernel(
    const float* __restrict__ xhat,
    const float* __restrict__ dxhat,
    const float* __restrict__ t1,
    const float* __restrict__ rstd,
    float* __restrict__ dx,
    long total,
    int hidden) {
    const long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const int s = static_cast<int>(idx / hidden);
    const float t2 = __fmul_rn(xhat[idx], t1[s]);
    const float t3 = __fsub_rn(dxhat[idx], t2);
    dx[idx] = __fmul_rn(rstd[s], t3);
}

// dgamma[h] = fold over rows s (ascending) of fma(du[s,h], xhat[s,h]).
__global__ void dgamma_left_fold_kernel(
    const float* __restrict__ du,
    const float* __restrict__ xhat,
    float* __restrict__ dgamma,
    int rows,
    int hidden) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= hidden) {
        return;
    }
    float acc = 0.0f;
    for (int s = 0; s < rows; ++s) {  // ascending-row left fold
        const long offset = static_cast<long>(s) * hidden + h;
        acc = __fmaf_rn(du[offset], xhat[offset], acc);
    }
    dgamma[h] = acc;
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

void launch_row_tree_reduce(
    const float* a,
    const float* b,
    float* out,
    int64_t rows,
    int64_t hidden,
    cudaStream_t stream) {
    if (rows == 0) {
        return;  // zero-row/grid-0 launches are invalid; skip synchronously
    }
    const int threads = 256;
    const int blocks = static_cast<int>((rows + threads - 1) / threads);
    row_tree_reduce_kernel<<<blocks, threads, 0, stream>>>(
        a, b, out, static_cast<int>(rows), static_cast<int>(hidden), leaves_of(hidden));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

// per-row H-dim tree of FMA(a_row, b_row); pass the same tensor twice for
// the sumsq form (fma(x, x, acc)).
torch::Tensor txt_in_row_tree_reduce_cuda(torch::Tensor a, torch::Tensor b) {
    const c10::cuda::CUDAGuard device_guard(a.device());
    check_common(a, "a");
    check_common(b, "b");
    TORCH_CHECK(b.device() == a.device(), "b must live on a's device");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "a and b must be 2-D");
    TORCH_CHECK(a.sizes() == b.sizes(), "a and b must share one shape");
    auto out = torch::empty({a.size(0)}, a.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_row_tree_reduce(
        a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), a.size(0), a.size(1),
        stream);
    return out;
}

// frozen norm statistics: (xhat, z, rstd) -- all fp32, no seam cast.
std::vector<torch::Tensor> txt_in_norm_stats_cuda(torch::Tensor x, torch::Tensor gamma) {
    const c10::cuda::CUDAGuard device_guard(x.device());
    check_common(x, "x");
    check_common(gamma, "gamma");
    TORCH_CHECK(gamma.device() == x.device(), "gamma must live on x's device");
    TORCH_CHECK(x.dim() == 2, "x must be 2-D [S, H]");
    TORCH_CHECK(gamma.dim() == 1 && gamma.size(0) == x.size(1), "gamma must be [H]");

    const int64_t rows = x.size(0);
    const int64_t hidden = x.size(1);
    auto sumsq = torch::empty({rows}, x.options());
    auto rstd = torch::empty({rows}, x.options());
    auto xhat = torch::empty({rows, hidden}, x.options());
    auto z = torch::empty({rows, hidden}, x.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    launch_row_tree_reduce(
        x.data_ptr<float>(), x.data_ptr<float>(), sumsq.data_ptr<float>(), rows, hidden, stream);

    if (rows > 0) {
        const int threads = 256;
        const int blocks = static_cast<int>((rows + threads - 1) / threads);
        rstd_kernel<<<blocks, threads, 0, stream>>>(
            sumsq.data_ptr<float>(), rstd.data_ptr<float>(), static_cast<int>(rows));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    const long total = static_cast<long>(rows) * static_cast<int>(hidden);
    if (total > 0) {
        const int threads = 256;
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        xhat_z_kernel<<<blocks, threads, 0, stream>>>(
            x.data_ptr<float>(), gamma.data_ptr<float>(), rstd.data_ptr<float>(),
            xhat.data_ptr<float>(), z.data_ptr<float>(), total, static_cast<int>(hidden));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {xhat, z, rstd};
}

// frozen dx chain from a precomputed dz (= tree(dY, W.T); du := dz, identity
// seam -- the caller computes the tree once and also feeds it to dgamma).
torch::Tensor txt_in_dx_cuda(
    torch::Tensor dz,
    torch::Tensor gamma,
    torch::Tensor xhat,
    torch::Tensor rstd) {
    const c10::cuda::CUDAGuard device_guard(dz.device());
    check_common(dz, "dz");
    check_common(gamma, "gamma");
    check_common(xhat, "xhat");
    check_common(rstd, "rstd");
    TORCH_CHECK(dz.dim() == 2, "dz must be 2-D [S, H]");
    const int64_t rows = dz.size(0);
    const int64_t hidden = dz.size(1);
    TORCH_CHECK(gamma.numel() == hidden, "gamma must have H elements");
    TORCH_CHECK(xhat.sizes() == dz.sizes(), "xhat must share dz's shape");
    TORCH_CHECK(rstd.numel() == rows, "rstd must have S elements");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int threads = 256;

    auto dxhat = torch::empty({rows, hidden}, dz.options());
    const long total = static_cast<long>(rows) * static_cast<int>(hidden);
    if (total > 0) {
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        dxhat_kernel<<<blocks, threads, 0, stream>>>(
            dz.data_ptr<float>(), gamma.data_ptr<float>(), dxhat.data_ptr<float>(), total,
            static_cast<int>(hidden));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    auto dot = torch::empty({rows}, dz.options());
    launch_row_tree_reduce(
        dxhat.data_ptr<float>(), xhat.data_ptr<float>(), dot.data_ptr<float>(), rows, hidden,
        stream);

    auto t1 = torch::empty({rows}, dz.options());
    if (rows > 0) {
        const int blocks = static_cast<int>((rows + threads - 1) / threads);
        t1_kernel<<<blocks, threads, 0, stream>>>(
            dot.data_ptr<float>(), t1.data_ptr<float>(), static_cast<int>(rows));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    auto dx = torch::empty({rows, hidden}, dz.options());
    if (total > 0) {
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        dx_kernel<<<blocks, threads, 0, stream>>>(
            xhat.data_ptr<float>(), dxhat.data_ptr<float>(), t1.data_ptr<float>(),
            rstd.data_ptr<float>(), dx.data_ptr<float>(), total, static_cast<int>(hidden));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return dx;
}

// dgamma by the ascending-row left fold (contract c-prime).
torch::Tensor txt_in_dgamma_fold_cuda(torch::Tensor du, torch::Tensor xhat) {
    const c10::cuda::CUDAGuard device_guard(du.device());
    check_common(du, "du");
    check_common(xhat, "xhat");
    TORCH_CHECK(xhat.device() == du.device(), "xhat must live on du's device");
    TORCH_CHECK(du.dim() == 2 && xhat.dim() == 2, "du and xhat must be 2-D");
    TORCH_CHECK(du.sizes() == xhat.sizes(), "du and xhat must share one shape");
    const int64_t rows = du.size(0);
    const int64_t hidden = du.size(1);

    auto dgamma = torch::empty({hidden}, du.options());
    const int threads = 256;
    const int blocks = static_cast<int>((hidden + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dgamma_left_fold_kernel<<<blocks, threads, 0, stream>>>(
        du.data_ptr<float>(), xhat.data_ptr<float>(), dgamma.data_ptr<float>(),
        static_cast<int>(rows), static_cast<int>(hidden));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return dgamma;
}
