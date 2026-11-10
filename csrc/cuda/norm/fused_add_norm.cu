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
// row's result does not depend on how many rows share the launch. That is what
// makes Axis-A batch invariance hold here, and the Triton kernel now folds in
// this same order, so CUDA == Triton holds bitwise on all six outputs. The lane
// counts in choose_threads below are shared with _reduction_shape in the Triton
// module and the two must be changed together. See that module's docstring for
// the other two things that had to agree, the divide and the rsqrt.
//
// Both launchers take a CUDAGuard on the input's device before reading the
// current stream, and check the launch, which rmsnorm.cu only does in its
// row-fold helper.
//
// Element-wise expressions are written in the same association order as the
// Triton kernel, with explicit round-to-nearest intrinsics so that neither side
// contracts a multiply and an add into one FFMA, so that when the two are
// compared the reduction is the only thing left to differ. The Triton side gets
// the same guarantee from enable_fp_fusion=False, which both the NVIDIA and the
// AMD Triton backends accept, so no platform-specific code is added for it.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <assert.h>

namespace {

// Explicit round-to-nearest arithmetic, so nvcc cannot contract a multiply and
// an add into one FFMA. The RFC forbids FP contraction of recomputed values, and
// one contraction here is not merely a rounding difference: at the first
// narrowing point the optimiser fused ``hidden * rstd * (1 + w)`` with
// ``+ residual`` and the declared cast stopped happening. Measured on FP32 at
// 256x32 with the residual on a 1/64 grid, before this change: 1757 of 8192
// outputs differed from the contract's two-rounding form and 0 differed from the
// single-rounding FFMA form. The narrow dtypes hid it because ``narrow_to``
// is a real rounding there; FP32 is the dtype where the cast is a no-op and the
// contraction is therefore invisible to every tolerance comparison.
//
// Per-expression intrinsics rather than ``--fmad=false``, because nvcc_flags in
// build_tools/extensions.py is one list shared by every .cu in the extension and
// would change the numerics of already-merged operators. These intrinsics pin the
// rounding mode and the absence of contraction; they do not pin the denormal mode,
// which --use_fast_math changes. See div_rn below for the measurement and for what
// that leaves exposed.
// The one reciprocal square root per row, as a single hardware instruction.
//
// nvcc's ``rsqrtf`` is libdevice's refined non-FTZ sequence; Triton's ``tl.rsqrt``
// lowers to the bare ``rsqrt.approx.ftz.f32``. The two are different
// approximations and disagree by one ULP on about 12% of inputs, which was the
// last thing standing between the two backends after the reduction orders were
// matched. Measured: with the orders matched, the raw row sums agree on 32 of 32
// rows and only the rsqrt tail differed.
//
// Emitting the same instruction here removes that. It also lines the kernels up
// with the gold rather than away from it: ``torch.rsqrt``, which the reference
// uses, is bitwise equal to ``rsqrt.approx.ftz.f32`` over 8192 sampled inputs,
// while neither matches a correctly-rounded reciprocal square root (25% of inputs
// differ by a ULP). Inline PTX is confined to this file, which is the CUDA
// backend; the RFC's "no platform-specific kernel code" applies to the Triton
// kernel that ROCm also runs.
__device__ __forceinline__ float rsqrt_approx_ftz(float x) {
#if defined(USE_ROCM) || defined(__HIP_DEVICE_COMPILE__)
    // This file is in the shared cuda_sources list, so a ROCm build hipifies it
    // and compiles it with hipcc, which cannot assemble NVIDIA PTX. hipify does
    // not translate the asm block either, so without this guard the whole ROCm
    // extension fails to build -- the repo's other PTX-bearing source,
    // attention/prefix_shared_attention.cu, is kept out of the ROCm list instead,
    // but the rest of this file is portable and was building there before.
    //
    // The CUDA == Triton bitwise claim is therefore a claim about the CUDA build.
    // On ROCm the registry dispatches the Triton kernel and the PyTorch gold for
    // this operator and never selects this one, so the fallback changes no result
    // that anything reads; it only has to compile and be correct.
    return rsqrtf(x);
#else
    float r;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
#endif
}

__device__ __forceinline__ float mul_rn(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float add_rn(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float sub_rn(float a, float b) { return __fsub_rn(a, b); }
// The row mean, as a division whose rounding the build flag cannot reinterpret. A
// plain ``/`` is div.rn.f32 by default, but --use_fast_math sets --prec-div=false
// and lowers it to div.approx.f32, and the Triton side is pinned to div.rn.f32
// through tl.div_rn. Measured: with the four divisions left as ``/`` and the
// extension built with KERNEL_ALIGN_USE_FAST_MATH=1, 30 of the 54 bitwise cases
// fail.
//
// What this does not pin, corrected from an earlier version of this comment that
// claimed the flag could not reinterpret the divide at all: --use_fast_math also
// implies --ftz=true, and nvcc applies that to these intrinsics. Measured on CUDA
// 12.8 at -arch=sm_89 -O3 over a faithful copy of this kernel: default gives
// mul.rn.f32 / add.rn.f32 / sub.rn.f32 / div.rn.f32, and the flag gives the same
// four with .ftz, while fma.rn stays absent in both so the anti-contraction part
// above does hold either way. Triton's PTX for its kernel is non-FTZ --
// add.rn.f32, mul.rn.f32, div.rn.f32, with rsqrt.approx.ftz.f32 as the only .ftz
// in it, which is the instruction below matches on purpose. So under fast math the
// two backends agree on every normal intermediate and differ on a subnormal one,
// where this side flushes to zero. A counter-flag cannot be added for one source
// because nvcc_flags is shared, so the bitwise claim is scoped to a default build
// in the operator doc, and
// test_cuda_and_triton_are_bitwise_equal_on_a_subnormal_intermediate is the case
// that goes red if someone sets the flag.
__device__ __forceinline__ float div_rn(float a, float b) { return __fdiv_rn(a, b); }

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
        local = add_rn(local, mul_rn(hv, hv));
    }
    const float rp = rsqrt_approx_ftz(div_rn(block_reduce_sum(local), static_cast<float>(H)) + post_eps);
    if (tid == 0) rstd_post[row] = rp;

    // post = hidden * rstd * (1 + w), narrowed; res = residual + post, narrowed.
    // Each thread writes the columns it will read back below, so no barrier is
    // needed between the write and the re-read.
    for (int col = tid; col < H; col += blockDim.x) {
        const float hv = load_as_float<scalar_t>(h_row + col);
        const float gain = add_rn(1.0f, load_as_float<weight_t>(post_w + col));
        const float post = narrow_to<scalar_t>(mul_rn(mul_rn(hv, rp), gain));
        const float res = add_rn(load_as_float<scalar_t>(r_row + col), post);
        store_from_float<scalar_t>(ro_row + col, res);
    }

    // Pre-norm statistic over the narrowed residual, read back from storage.
    local = 0.0f;
    for (int col = tid; col < H; col += blockDim.x) {
        const float rv = load_as_float<scalar_t>(ro_row + col);
        local = add_rn(local, mul_rn(rv, rv));
    }
    const float rq = rsqrt_approx_ftz(div_rn(block_reduce_sum(local), static_cast<float>(H)) + pre_eps);
    if (tid == 0) rstd_pre[row] = rq;

    for (int col = tid; col < H; col += blockDim.x) {
        const float rv = load_as_float<scalar_t>(ro_row + col);
        const float gain = add_rn(1.0f, load_as_float<weight_t>(pre_w + col));
        store_from_float<scalar_t>(po_row + col, mul_rn(mul_rn(rv, rq), gain));
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
        const float gain = add_rn(1.0f, load_as_float<weight_t>(pre_w + col));
        const float ro = load_as_float<scalar_t>(res_out + base + col);
        local = add_rn(local, mul_rn(mul_rn(gp, gain), ro));
    }
    const float dot_pre = div_rn(block_reduce_sum(local), static_cast<float>(H));

    // d_res_out is live on two paths: downstream cotangent plus the pre-norm's
    // input gradient. Accumulate them the way eager does: the residual output is
    // a model-dtype tensor, so autograd rounds the arriving gradient to that dtype
    // and adds in that dtype. In FP32 this is one ULP per element and puts the
    // post-norm weight gradient 2.6-2.9x outside the contract row at 512 and 2048
    // rows. Store it, and accumulate the post-norm dot in the same pass.
    local = 0.0f;
    for (int col = tid; col < H; col += blockDim.x) {
        const float gp = load_as_float<scalar_t>(g_pre + base + col);
        const float gain = add_rn(1.0f, load_as_float<weight_t>(pre_w + col));
        const float ro = load_as_float<scalar_t>(res_out + base + col);
        const float scaled = mul_rn(gp, gain);
        const float d_from_pre = narrow_to<scalar_t>(sub_rn(
            mul_rn(rq, scaled),
            mul_rn(mul_rn(mul_rn(mul_rn(ro, rq), rq), rq), dot_pre)));
        const float d_ro = narrow_to<scalar_t>(
            add_rn(load_as_float<scalar_t>(g_res + base + col), d_from_pre));
        d_residual[base + col] = d_ro;
        partial_pre_w[base + col] = mul_rn(mul_rn(gp, ro), rq);

        const float hv = load_as_float<scalar_t>(hidden + base + col);
        const float pgain = add_rn(1.0f, load_as_float<weight_t>(post_w + col));
        local = add_rn(local, mul_rn(mul_rn(d_ro, pgain), hv));
    }
    const float dot_post = div_rn(block_reduce_sum(local), static_cast<float>(H));

    for (int col = tid; col < H; col += blockDim.x) {
        const float d_ro = d_residual[base + col];
        const float hv = load_as_float<scalar_t>(hidden + base + col);
        const float pgain = add_rn(1.0f, load_as_float<weight_t>(post_w + col));
        const float scaled = mul_rn(d_ro, pgain);
        d_hidden[base + col] = sub_rn(
            mul_rn(rp, scaled),
            mul_rn(mul_rn(mul_rn(mul_rn(hv, rp), rp), rp), dot_post));
        partial_post_w[base + col] = mul_rn(mul_rn(d_ro, hv), rp);
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
    // A zero-row batch is a valid no-op -- the outputs are already empty -- but
    // <<<0, ...>>> is not a valid launch. It returns cudaErrorInvalidConfiguration,
    // which is sticky: with no launch check the wrapper returns normally and the
    // error surfaces later against an unrelated call, leaving the whole CUDA
    // context unusable for the rest of the process. Empty batches are routine in
    // RL serving, so return before the launch rather than after the damage.
    if (T == 0) return;
    const int threads = choose_threads(H);
    const size_t smem = threads * sizeof(float);
    const c10::cuda::CUDAGuard device_guard(hidden.device());
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
            C10_CUDA_KERNEL_LAUNCH_CHECK();
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
    if (T == 0) return;  // see the forward launcher
    const int threads = choose_threads(H);
    const size_t smem = threads * sizeof(float);
    const c10::cuda::CUDAGuard device_guard(hidden.device());
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
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    });
}
