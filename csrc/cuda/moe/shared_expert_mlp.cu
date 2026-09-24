// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// P5-5 (#64) Shared Expert MLP strict kernels: fc1 -> one-round SwiGLU -> fc2.
//
// Numeric profile ``oracle-fp32-serial-v1`` (see rl_engine/moe/oracle.py):
// every accumulation is FP32, serial, ascending-k, with multiply and add
// rounded separately (__fmul_rn / __fadd_rn; never contracted into FMA).
// One thread owns one output element, so batch size and padding cannot
// change a row's bytes (Axis-A bitwise invariance) and there is no
// cross-thread floating-point reduction anywhere.
//
// The one-round SwiGLU core (FP32 math, single BF16 round on the output) is
// shared with P5-2 (#63): the p_s / clamp variant extends the same device
// functions in this translation unit rather than forking the math.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>

namespace {

// out[m, n] = sum_{k ascending} fadd_rn(acc, fmul_rn(a[m, k], b(n, k)))
// A is BF16 [M, K]; B is BF16 [N, K] (TRANS_B = false) or [K, N] (true).
// Output stays FP32; the caller rounds to BF16 where the contract says so.
template <bool TRANS_B>
__global__ void p5_strict_gemm_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    float* __restrict__ out,
    const int64_t m_rows,
    const int64_t n_cols,
    const int64_t k_dim) {
  const int64_t total = m_rows * n_cols;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; idx < total;
       idx += stride) {
    const int64_t m = idx / n_cols;
    const int64_t n = idx - m * n_cols;
    const __nv_bfloat16* a_row = a + m * k_dim;
    float acc = 0.0f;
    for (int64_t k = 0; k < k_dim; ++k) {
      const float av = __bfloat162float(a_row[k]);
      const float bv =
          __bfloat162float(TRANS_B ? b[k * n_cols + n] : b[n * k_dim + k]);
      acc = __fadd_rn(acc, __fmul_rn(av, bv));
    }
    out[idx] = acc;
  }
}

// Matches torch.sigmoid on FP32 CUDA tensors: 1 / (1 + exp(-x)) with
// IEEE div.rn and the accurate expf (no fast-math in this build).
__device__ __forceinline__ float sigmoid_rn(float x) {
  return 1.0f / (1.0f + expf(-x));
}

// One-round SwiGLU core, shared-expert mode (p_s = None: no clamp, no route
// weight). z is the packed FP32 fc1 output [T, 2F] (gate columns then up);
// h is the single BF16 round of SiLU(gate) * up.
__global__ void p5_swiglu_shared_forward_kernel(
    const float* __restrict__ z,
    __nv_bfloat16* __restrict__ h,
    const int64_t n,
    const int64_t width) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; idx < n;
       idx += stride) {
    const int64_t row = idx / width;
    const int64_t col = idx - row * width;
    const int64_t gate_index = row * (2 * width) + col;
    const float g = z[gate_index];
    const float u = z[gate_index + width];
    const float sig = sigmoid_rn(g);
    const float silu = __fmul_rn(g, sig);
    h[idx] = __float2bfloat16(__fmul_rn(silu, u));
  }
}

// Backward of the same graph (p_s = None): recomputes sig/silu from the saved
// FP32 z with the identical instruction sequence, so the bits match the
// forward. dz packs (dgate | dup), each rounded to BF16 exactly once at the
// operator edge (mirrors the oracle's cat(...).to(bfloat16)).
__global__ void p5_swiglu_shared_backward_kernel(
    const __nv_bfloat16* __restrict__ dh,
    const float* __restrict__ z,
    __nv_bfloat16* __restrict__ dz,
    const int64_t n,
    const int64_t width) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; idx < n;
       idx += stride) {
    const int64_t row = idx / width;
    const int64_t col = idx - row * width;
    const int64_t gate_index = row * (2 * width) + col;
    const float g = z[gate_index];
    const float u = z[gate_index + width];
    const float dh32 = __bfloat162float(dh[idx]);
    const float sig = sigmoid_rn(g);
    const float silu = __fmul_rn(g, sig);
    // dsilu = sig * (1 + g * (1 - sig)), each op rounded separately.
    float t = __fsub_rn(1.0f, sig);
    t = __fmul_rn(g, t);
    t = __fadd_rn(1.0f, t);
    const float dsilu = __fmul_rn(sig, t);
    const float dgate = __fmul_rn(__fmul_rn(dh32, u), dsilu);
    const float dup = __fmul_rn(dh32, silu);
    dz[gate_index] = __float2bfloat16(dgate);
    dz[gate_index + width] = __float2bfloat16(dup);
  }
}

void launch_1d(int64_t n, int& threads, int64_t& blocks) {
  threads = 256;
  blocks = (n + threads - 1) / threads;
  if (blocks == 0) {
    blocks = 1;
  }
  if (blocks > 65535) {
    blocks = 65535;  // grid-stride loops cover the rest
  }
}

void check_cuda_2d(const torch::Tensor& t, at::ScalarType dtype, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.dim() == 2, name, " must be 2-D");
  TORCH_CHECK(t.scalar_type() == dtype, name, " must be ", dtype, ", got ", t.scalar_type());
}

}  // namespace

torch::Tensor p5_strict_gemm(torch::Tensor a, torch::Tensor b, bool trans_b) {
  check_cuda_2d(a, at::kBFloat16, "a");
  check_cuda_2d(b, at::kBFloat16, "b");
  TORCH_CHECK(a.device() == b.device(), "a and b must be on the same CUDA device");
  const int64_t m_rows = a.size(0);
  const int64_t k_dim = a.size(1);
  const int64_t n_cols = trans_b ? b.size(1) : b.size(0);
  const int64_t bk = trans_b ? b.size(0) : b.size(1);
  TORCH_CHECK(bk == k_dim, "K mismatch: a has K=", k_dim, ", b has K=", bk);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto out = torch::empty({m_rows, n_cols}, a.options().dtype(at::kFloat));
  const int64_t n = out.numel();
  if (n == 0 || k_dim == 0) {
    return n == 0 ? out : out.zero_();
  }
  int threads = 0;
  int64_t blocks = 0;
  launch_1d(n, threads, blocks);
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto* a_ptr = reinterpret_cast<const __nv_bfloat16*>(a.data_ptr());
  const auto* b_ptr = reinterpret_cast<const __nv_bfloat16*>(b.data_ptr());
  if (trans_b) {
    p5_strict_gemm_kernel<true><<<blocks, threads, 0, stream>>>(
        a_ptr, b_ptr, out.data_ptr<float>(), m_rows, n_cols, k_dim);
  } else {
    p5_strict_gemm_kernel<false><<<blocks, threads, 0, stream>>>(
        a_ptr, b_ptr, out.data_ptr<float>(), m_rows, n_cols, k_dim);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor p5_swiglu_shared_forward(torch::Tensor z) {
  check_cuda_2d(z, at::kFloat, "z");
  TORCH_CHECK(z.size(1) % 2 == 0, "z width must be even (packed gate|up)");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(z));
  const int64_t width = z.size(1) / 2;
  auto h = torch::empty({z.size(0), width}, z.options().dtype(at::kBFloat16));
  const int64_t n = h.numel();
  if (n == 0) {
    return h;
  }
  int threads = 0;
  int64_t blocks = 0;
  launch_1d(n, threads, blocks);
  auto stream = at::cuda::getCurrentCUDAStream();
  p5_swiglu_shared_forward_kernel<<<blocks, threads, 0, stream>>>(
      z.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(h.data_ptr()), n, width);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return h;
}

torch::Tensor p5_swiglu_shared_backward(torch::Tensor dh, torch::Tensor z) {
  check_cuda_2d(dh, at::kBFloat16, "dh");
  check_cuda_2d(z, at::kFloat, "z");
  TORCH_CHECK(dh.device() == z.device(), "dh and z must be on the same CUDA device");
  TORCH_CHECK(z.size(1) % 2 == 0, "z width must be even (packed gate|up)");
  TORCH_CHECK(
      dh.size(0) == z.size(0) && dh.size(1) * 2 == z.size(1),
      "dh shape must match the packed gate/up halves of z");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(z));
  auto dz = torch::empty_like(z, z.options().dtype(at::kBFloat16));
  const int64_t n = dh.numel();
  if (n == 0) {
    return dz;
  }
  int threads = 0;
  int64_t blocks = 0;
  launch_1d(n, threads, blocks);
  auto stream = at::cuda::getCurrentCUDAStream();
  p5_swiglu_shared_backward_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(dh.data_ptr()),
      z.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(dz.data_ptr()),
      n,
      dh.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return dz;
}
