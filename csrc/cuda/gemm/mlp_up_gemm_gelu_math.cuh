// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// Pinned GELU arithmetic for the Qwen-Image MLP up projection + GELU row
// (`mlp_up_gemm_gelu`).
//
// This header is the single definition of the operator's transcendental
// epilogue, included by both CUDA paths -- the portable fp32-tree kernel
// (`mlp_up_gemm_gelu.cu`, contract `mlp-up-gemm-gelu-tree`) and the Hopper
// TMA + wgmma kernel (`mlp_up_gemm_gelu_sm90.cu`, contract
// `mlp-up-gemm-gelu-mma`) -- so that the two paths agree byte for byte.
//
// Every operation is pinned with an explicit rounding intrinsic. The compiler
// must not reassociate or contract the sequence: the operator's fp32 reference
// in `rl_engine/kernels/ops/pytorch/linear/mlp_up_gemm_gelu.py` emulates exactly
// these operations in fp64 (correctly rounded), and the byte-equality checks
// compare the pre-activation, the polynomial and the gradient against it. The
// only unpinned step is `tanhf` itself, the operator's single transcendental:
// the reference evaluates it correctly rounded, the device's `tanhf` is within
// its documented 2 ulp, and the GELU value/gate comparisons are therefore
// declared-tolerance ones while everything else is byte-equal.
//
// Frozen fp32 sequence (S = sqrt(2/pi) = 0.79788456, C = 0.044715, K3 = 3C):
//
//   q  = C * x                        one correctly-rounded multiply
//   t  = fma(q, x * x, x)             = x + 0.044715 x^3
//   th = tanh(S * t)
//   y  = (0.5 * x) * (1 + th)         tanh-approximate GELU
//
//   a  = 1 + th
//   b  = 1 - th * th
//   e  = fma(K3, x * x, 1)
//   d  = fma(0.5, a, 0.5 * S * x * b * e)   the derivative of the above
//
// The argument is always the *fp32 pre-activation* (the tree/wgmma accumulator
// after the bias was added once in fp32): the row is fused, so there is no
// intermediate bf16 rounding before the activation, and the single bf16 cast
// happens at the store.

#pragma once

#include <cuda_runtime.h>

namespace mlp_up_math {

//: GELU constants as fp32 bit patterns (the reference uses the same ones).
//: kGeluS is sqrt(2/pi) == M_SQRT2 * M_2_SQRTPI * 0.5, the coefficient
//: PyTorch's own tanh GELU uses on both CPU and CUDA (its M_SQRT1_2 only
//: appears in the erf branch).
constexpr float kGeluC = 0.044715f;
constexpr float kGeluS = 0.7978845608028654f;  // sqrt(2/pi)
constexpr float kGeluK3 = 0.134145f;           // 3 * kGeluC

/// tanh-approximate GELU of an fp32 pre-activation, fp32 out.
__device__ __forceinline__ float gelu_tanh_fp32(float x) {
    const float q = __fmul_rn(kGeluC, x);
    const float t = __fmaf_rn(q, __fmul_rn(x, x), x);
    const float th = tanhf(__fmul_rn(kGeluS, t));
    return __fmul_rn(__fmul_rn(0.5f, x), __fadd_rn(1.0f, th));
}

/// The derivative of :func:`gelu_tanh_fp32` (the backward's gate coefficient).
__device__ __forceinline__ float gelu_tanh_grad_fp32(float x) {
    const float x2 = __fmul_rn(x, x);
    const float q = __fmul_rn(kGeluC, x);
    const float t = __fmaf_rn(q, x2, x);
    const float th = tanhf(__fmul_rn(kGeluS, t));
    const float a = __fadd_rn(1.0f, th);
    const float b = __fsub_rn(1.0f, __fmul_rn(th, th));
    const float e = __fmaf_rn(x2, kGeluK3, 1.0f);
    const float h = __fmul_rn(__fmul_rn(__fmul_rn(kGeluS, x), b), e);
    return __fmaf_rn(0.5f, a, __fmul_rn(0.5f, h));
}

}  // namespace mlp_up_math
