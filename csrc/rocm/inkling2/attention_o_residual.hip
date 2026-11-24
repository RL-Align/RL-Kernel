// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// Inkling 2 Mini attention O projection + residual (INK-17), ROCm-native.
// Compiled only for ROCm builds (the CUDA build keeps
// csrc/cuda/inkling2/attention_o_residual.cu); exports the same C++ symbols.
//
// Determinism contract (identical to the CUDA kernel / Python reference):
//   * forward : acc = bias (FP32, or 0); for h ascending: acc = fma(x[r,h], W[o,h], acc);
//               acc += residual[r,o] (FP32 add AFTER the full projection and bias);
//               one cast to the storage dtype.
//   * d_attention[r,h] = sum_o dy[r,o] * W[o,h]     (o ascending, FP32)
//   * d_weight[o,h]    = sum_r dy[r,o] * x[r,h]     (r ascending, FP32)
//   * d_bias[o]        = sum_r dy[r,o]              (r ascending, FP32)
//   * d_residual       = dy
// Register micro-tiling and LDS staging only change which thread owns an output,
// never its accumulation order, so results are independent of launch shape, batch
// size, and other rows. No float atomics.
//
// Tuned for MI300X (gfx942): wave64, 256-thread blocks, 64x64 output tile, 4x4
// register micro-tile per thread, 16-deep K step staged in LDS (2 x 16 x 65 x 4 B
// ~ 8 KB, so several blocks co-reside per CU). Sized for hidden 2304 (24*128 -> 2304).

#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <c10/core/DeviceGuard.h>
#include <c10/hip/HIPException.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#include <cstdint>
#include <vector>

namespace {

constexpr int kThreads = 256;  // 4 wave64
constexpr int kTile = 64;
constexpr int kTileK = 16;
constexpr int kMicro = 4;
constexpr int kPad = kTile + 1;
static_assert(kThreads == (kTile / kMicro) * (kTile / kMicro), "thread layout");
static_assert(kThreads % 64 == 0, "block must be whole wavefronts");

// C[r,n] = init(n) + sum_k A[r,k] * B(n,k), k ascending, then optional + residual[r,n].
// kBKContig: B stored [N][K] (forward); otherwise [K][N] (backward w.r.t. attention).
template <typename scalar_t, bool kBKContig, bool kHasBias, bool kHasResidual>
__global__ void __launch_bounds__(kThreads) o_residual_gemm_kernel(
    const scalar_t* __restrict__ a,         // [rows, K]
    const scalar_t* __restrict__ b,         // [N, K] or [K, N]
    const scalar_t* __restrict__ bias,      // [N] or null
    const scalar_t* __restrict__ residual,  // [rows, N] or null
    scalar_t* __restrict__ c,               // [rows, N]
    int64_t rows, int64_t K, int64_t N) {
  const int64_t m_base = static_cast<int64_t>(blockIdx.x) * kTile;
  const int64_t n_base = static_cast<int64_t>(blockIdx.y) * kTile;
  __shared__ float as[kTileK][kPad];
  __shared__ float bs[kTileK][kPad];
  const int tid = threadIdx.x;
  const int tx = tid % (kTile / kMicro);
  const int ty = tid / (kTile / kMicro);

  float acc[kMicro][kMicro];
#pragma unroll
  for (int i = 0; i < kMicro; ++i) {
#pragma unroll
    for (int j = 0; j < kMicro; ++j) {
      float init = 0.0f;
      if (kHasBias) {
        const int64_t n = n_base + tx * kMicro + j;
        if (n < N) init = static_cast<float>(bias[n]);
      }
      acc[i][j] = init;
    }
  }

  for (int64_t k0 = 0; k0 < K; k0 += kTileK) {
    for (int idx = tid; idx < kTile * kTileK; idx += kThreads) {
      const int m = idx / kTileK, k = idx % kTileK;  // 16 consecutive k per row
      float v = 0.0f;
      if (m_base + m < rows && k0 + k < K) v = static_cast<float>(a[(m_base + m) * K + k0 + k]);
      as[k][m] = v;
    }
    for (int idx = tid; idx < kTile * kTileK; idx += kThreads) {
      float v = 0.0f;
      int n, k;
      if (kBKContig) {
        n = idx / kTileK;
        k = idx % kTileK;
        if (n_base + n < N && k0 + k < K) v = static_cast<float>(b[(n_base + n) * K + k0 + k]);
      } else {
        k = idx / kTile;
        n = idx % kTile;
        if (n_base + n < N && k0 + k < K) v = static_cast<float>(b[(k0 + k) * N + n_base + n]);
      }
      bs[k][n] = v;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < kTileK; ++k) {  // zero padding past K adds exact +0.0 products
      float av[kMicro], bv[kMicro];
#pragma unroll
      for (int i = 0; i < kMicro; ++i) av[i] = as[k][ty * kMicro + i];
#pragma unroll
      for (int j = 0; j < kMicro; ++j) bv[j] = bs[k][tx * kMicro + j];
#pragma unroll
      for (int i = 0; i < kMicro; ++i)
#pragma unroll
        for (int j = 0; j < kMicro; ++j) acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < kMicro; ++i) {
    const int64_t r = m_base + ty * kMicro + i;
    if (r >= rows) continue;
#pragma unroll
    for (int j = 0; j < kMicro; ++j) {
      const int64_t n = n_base + tx * kMicro + j;
      if (n >= N) continue;
      float value = acc[i][j];
      if (kHasResidual) value += static_cast<float>(residual[r * N + n]);
      c[r * N + n] = static_cast<scalar_t>(value);
    }
  }
}

// d_weight[o,h] = sum_r dy[r,o] * attention[r,h], r ascending.
template <typename scalar_t>
__global__ void __launch_bounds__(kThreads) o_residual_weight_grad_kernel(
    const scalar_t* __restrict__ dy,         // [rows, O]
    const scalar_t* __restrict__ attention,  // [rows, I]
    scalar_t* __restrict__ d_weight,         // [O, I]
    int64_t rows, int64_t I, int64_t O) {
  const int64_t m_base = static_cast<int64_t>(blockIdx.y) * kTile;  // o
  const int64_t n_base = static_cast<int64_t>(blockIdx.x) * kTile;  // h
  __shared__ float as[kTileK][kPad];
  __shared__ float bs[kTileK][kPad];
  const int tid = threadIdx.x;
  const int tx = tid % (kTile / kMicro);
  const int ty = tid / (kTile / kMicro);

  float acc[kMicro][kMicro];
#pragma unroll
  for (int i = 0; i < kMicro; ++i)
#pragma unroll
    for (int j = 0; j < kMicro; ++j) acc[i][j] = 0.0f;

  for (int64_t k0 = 0; k0 < rows; k0 += kTileK) {
    for (int idx = tid; idx < kTile * kTileK; idx += kThreads) {
      const int k = idx / kTile, col = idx % kTile;  // coalesced along o / h
      float va = 0.0f, vb = 0.0f;
      if (k0 + k < rows) {
        if (m_base + col < O) va = static_cast<float>(dy[(k0 + k) * O + m_base + col]);
        if (n_base + col < I) vb = static_cast<float>(attention[(k0 + k) * I + n_base + col]);
      }
      as[k][col] = va;
      bs[k][col] = vb;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < kTileK; ++k) {
      float av[kMicro], bv[kMicro];
#pragma unroll
      for (int i = 0; i < kMicro; ++i) av[i] = as[k][ty * kMicro + i];
#pragma unroll
      for (int j = 0; j < kMicro; ++j) bv[j] = bs[k][tx * kMicro + j];
#pragma unroll
      for (int i = 0; i < kMicro; ++i)
#pragma unroll
        for (int j = 0; j < kMicro; ++j) acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < kMicro; ++i) {
    const int64_t o = m_base + ty * kMicro + i;
    if (o >= O) continue;
#pragma unroll
    for (int j = 0; j < kMicro; ++j) {
      const int64_t h = n_base + tx * kMicro + j;
      if (h < I) d_weight[o * I + h] = static_cast<scalar_t>(acc[i][j]);
    }
  }
}

// d_bias[o] = sum_r dy[r,o], r ascending; one thread per output column (coalesced across threads).
template <typename scalar_t>
__global__ void __launch_bounds__(kThreads) o_residual_bias_grad_kernel(
    const scalar_t* __restrict__ dy, scalar_t* __restrict__ d_bias, int64_t rows, int64_t O) {
  const int64_t o = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (o >= O) return;
  float value = 0.0f;
  for (int64_t r = 0; r < rows; ++r) value += static_cast<float>(dy[r * O + o]);
  d_bias[o] = static_cast<scalar_t>(value);
}

void check_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous(), name, " must be contiguous ROCm");
  TORCH_CHECK(
      tensor.scalar_type() == at::kFloat || tensor.scalar_type() == at::kHalf ||
          tensor.scalar_type() == at::kBFloat16,
      name, " must be float32, float16, or bfloat16");
}

void check_same(const torch::Tensor& lhs, const torch::Tensor& rhs, const char* lhs_name,
                const char* rhs_name) {
  TORCH_CHECK(lhs.device() == rhs.device(), lhs_name, " and ", rhs_name, " must share device");
  TORCH_CHECK(lhs.scalar_type() == rhs.scalar_type(), lhs_name, " and ", rhs_name, " must share dtype");
}

inline unsigned ceil_tiles(int64_t n) { return static_cast<unsigned>((n + kTile - 1) / kTile); }

}  // namespace

torch::Tensor inkling2_attention_o_residual_forward(
    torch::Tensor attention,
    torch::Tensor residual,
    torch::Tensor weight,
    torch::Tensor bias) {
  check_tensor(attention, "attention");
  check_tensor(residual, "residual");
  check_tensor(weight, "weight");
  check_same(attention, residual, "attention", "residual");
  check_same(attention, weight, "attention", "weight");
  TORCH_CHECK(attention.dim() == 2 && residual.sizes() == attention.sizes(),
              "attention and residual must be [rows, hidden]");
  TORCH_CHECK(weight.dim() == 2 && weight.size(1) == attention.size(1) &&
                  weight.size(0) == residual.size(1),
              "weight must be [output, input]");
  TORCH_CHECK(bias.numel() == 0 || (bias.is_cuda() && bias.is_contiguous()),
              "bias must be empty or contiguous ROCm");
  const bool has_bias = bias.numel() != 0;
  if (has_bias) {
    check_tensor(bias, "bias");
    check_same(attention, bias, "attention", "bias");
    TORCH_CHECK(bias.dim() == 1 && bias.size(0) == weight.size(0), "bias size mismatch");
  }
  const c10::OptionalDeviceGuard device_guard(device_of(attention));
  const int64_t rows = attention.size(0), I = attention.size(1), O = weight.size(0);
  auto output = torch::empty({rows, O}, attention.options());
  if (output.numel() == 0) return output;
  const dim3 grid(ceil_tiles(rows), ceil_tiles(O));
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, attention.scalar_type(),
                                  "inkling2_attention_o_residual_forward", [&] {
    if (has_bias) {
      o_residual_gemm_kernel<scalar_t, true, true, true><<<grid, kThreads, 0, stream>>>(
          attention.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(),
          residual.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), rows, I, O);
    } else {
      o_residual_gemm_kernel<scalar_t, true, false, true><<<grid, kThreads, 0, stream>>>(
          attention.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), nullptr,
          residual.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), rows, I, O);
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::vector<torch::Tensor> inkling2_attention_o_residual_backward(
    torch::Tensor dy,
    torch::Tensor attention,
    torch::Tensor residual,
    torch::Tensor weight,
    bool need_bias) {
  check_tensor(dy, "dy");
  check_tensor(attention, "attention");
  check_tensor(residual, "residual");
  check_tensor(weight, "weight");
  check_same(dy, attention, "dy", "attention");
  check_same(attention, residual, "attention", "residual");
  check_same(attention, weight, "attention", "weight");
  TORCH_CHECK(dy.sizes() == residual.sizes(), "dy and residual must share shape");
  const c10::OptionalDeviceGuard device_guard(device_of(attention));
  const int64_t rows = attention.size(0), I = attention.size(1), O = weight.size(0);
  auto d_attention = torch::empty_like(attention);
  auto d_residual = torch::empty_like(residual);
  auto d_weight = torch::empty_like(weight);
  auto d_bias = need_bias ? torch::empty({O}, attention.options()) : torch::Tensor();
  if (attention.numel() == 0 || weight.numel() == 0) return {d_attention, d_residual, d_weight, d_bias};
  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 attn_grid(ceil_tiles(rows), ceil_tiles(I));
  const dim3 weight_grid(ceil_tiles(I), ceil_tiles(O));
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, attention.scalar_type(),
                                  "inkling2_attention_o_residual_backward", [&] {
    o_residual_gemm_kernel<scalar_t, false, false, false><<<attn_grid, kThreads, 0, stream>>>(
        dy.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), nullptr, nullptr,
        d_attention.data_ptr<scalar_t>(), rows, O, I);
    d_residual.copy_(dy);
    o_residual_weight_grad_kernel<scalar_t><<<weight_grid, kThreads, 0, stream>>>(
        dy.data_ptr<scalar_t>(), attention.data_ptr<scalar_t>(), d_weight.data_ptr<scalar_t>(), rows, I, O);
    if (need_bias) {
      const int bias_blocks = static_cast<int>((O + kThreads - 1) / kThreads);
      o_residual_bias_grad_kernel<scalar_t><<<bias_blocks, kThreads, 0, stream>>>(
          dy.data_ptr<scalar_t>(), d_bias.data_ptr<scalar_t>(), rows, O);
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {d_attention, d_residual, d_weight, d_bias};
}
