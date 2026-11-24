// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// Inkling 2 Mini deterministic grouped expert GEMM (INK-27), ROCm-native.
// Compiled only for ROCm builds (the CUDA build keeps csrc/cuda/inkling2/grouped_gemm.cu);
// exports the same C++ symbols.
//
// Determinism contract (identical to the CUDA kernel / Python reference):
//   * forward  y[t,o]  = bias[e,o] + sum_h x[t,h] * W[e,o,h]   (h ascending, FP32 FMA chain)
//   * dx[t,h]          = sum_o dy[t,o] * W[e,o,h]              (o ascending)
//   * dW[e,o,h]        = sum over tokens of expert e, ascending original token index, dy*x
//   * db[e,o]          = same token order over dy
// Tokens are bucketed per expert with a stable sort, so every output element has a
// reduction order that depends only on (expert, K index), never on the launch
// configuration, tile placement, batch size, or other tokens. Tiling
// (register micro-tiles + LDS staging) only changes which thread does the work,
// never the accumulation order. No float atomics. Results are stored with one
// round to the storage dtype.
//
// Tuned for MI300X (gfx942): wave64, 256-thread blocks, 64x64 output tile with
// a 4x4 register micro-tile per thread and a 16-deep K step staged in LDS
// (2 x 16 x 65 x 4 B ~ 8 KB, well under the 64 KB LDS budget so several blocks
// co-reside per CU).

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
constexpr int kTile = 64;      // output tile edge (M and N)
constexpr int kTileK = 16;     // K step staged through LDS
constexpr int kMicro = 4;      // per-thread micro tile edge
constexpr int kPad = kTile + 1;
static_assert(kThreads == (kTile / kMicro) * (kTile / kMicro), "thread layout");
static_assert(kThreads % 64 == 0, "block must be whole wavefronts");

// Maps a flat tile index to its expert via binary search over the tile prefix sums.
__device__ __forceinline__ int find_expert(const int64_t* __restrict__ tile_off, int experts,
                                           int64_t tile) {
  int lo = 0, hi = experts;  // largest e with tile_off[e] <= tile
  while (hi - lo > 1) {
    const int mid = (lo + hi) >> 1;
    if (tile_off[mid] <= tile) lo = mid; else hi = mid;
  }
  return lo;
}

// Shared by forward (B is [N][K], K contiguous) and backward-tokens (B is [K][N], N contiguous).
// Rows of A and of C are tokens, located through the expert-sorted `order`.
template <typename scalar_t, bool kBKContig, bool kHasBias>
__global__ void __launch_bounds__(kThreads) grouped_rows_gemm_kernel(
    const scalar_t* __restrict__ a,        // [tokens, K]
    const scalar_t* __restrict__ weights,  // [experts, d1, d2], d1*d2 == K*N
    const scalar_t* __restrict__ bias,     // [experts, N] or null
    scalar_t* __restrict__ c,              // [tokens, N]
    const int64_t* __restrict__ order,
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ counts,
    const int64_t* __restrict__ tile_off,
    int experts, int64_t K, int64_t N) {
  const int64_t tile = blockIdx.x;
  if (tile >= tile_off[experts]) return;
  const int expert = find_expert(tile_off, experts, tile);
  const int64_t row_base = (tile - tile_off[expert]) * kTile;  // within the expert tokens
  const int64_t count = counts[expert];
  const int64_t off = offsets[expert];
  const int64_t n_base = static_cast<int64_t>(blockIdx.y) * kTile;
  const scalar_t* w = weights + static_cast<int64_t>(expert) * K * N;

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
        if (n < N) init = static_cast<float>(bias[static_cast<int64_t>(expert) * N + n]);
      }
      acc[i][j] = init;
    }
  }

  for (int64_t k0 = 0; k0 < K; k0 += kTileK) {
    for (int idx = tid; idx < kTile * kTileK; idx += kThreads) {
      // A tile: 16 consecutive k per row, coalesced along k.
      const int m = idx / kTileK, k = idx % kTileK;
      float v = 0.0f;
      if (row_base + m < count && k0 + k < K) {
        const int64_t token = order[off + row_base + m];
        v = static_cast<float>(a[token * K + k0 + k]);
      }
      as[k][m] = v;
    }
    for (int idx = tid; idx < kTile * kTileK; idx += kThreads) {
      float v = 0.0f;
      int n, k;
      if (kBKContig) {
        n = idx / kTileK;
        k = idx % kTileK;
        if (n_base + n < N && k0 + k < K) v = static_cast<float>(w[(n_base + n) * K + k0 + k]);
      } else {
        k = idx / kTile;
        n = idx % kTile;
        if (n_base + n < N && k0 + k < K) v = static_cast<float>(w[(k0 + k) * N + n_base + n]);
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
    const int64_t m = row_base + ty * kMicro + i;
    if (m >= count) continue;
    const int64_t token = order[off + m];
#pragma unroll
    for (int j = 0; j < kMicro; ++j) {
      const int64_t n = n_base + tx * kMicro + j;
      if (n < N) c[token * N + n] = static_cast<scalar_t>(acc[i][j]);
    }
  }
}

// dW[e][o][h] = sum_{i in tokens(e), ascending} dy[tok_i][o] * x[tok_i][h]
template <typename scalar_t>
__global__ void __launch_bounds__(kThreads) grouped_weight_grad_kernel(
    const scalar_t* __restrict__ dy,  // [tokens, O]
    const scalar_t* __restrict__ x,   // [tokens, H]
    scalar_t* __restrict__ dw,        // [experts, O, H]
    const int64_t* __restrict__ order,
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ counts,
    int64_t O, int64_t H) {
  const int expert = blockIdx.z;
  const int64_t m_base = static_cast<int64_t>(blockIdx.y) * kTile;  // output index o
  const int64_t n_base = static_cast<int64_t>(blockIdx.x) * kTile;  // hidden index h
  const int64_t count = counts[expert];
  const int64_t off = offsets[expert];

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

  for (int64_t k0 = 0; k0 < count; k0 += kTileK) {
    for (int idx = tid; idx < kTile * kTileK; idx += kThreads) {
      const int k = idx / kTile, col = idx % kTile;  // coalesced along o / h
      float va = 0.0f, vb = 0.0f;
      if (k0 + k < count) {
        const int64_t token = order[off + k0 + k];
        if (m_base + col < O) va = static_cast<float>(dy[token * O + m_base + col]);
        if (n_base + col < H) vb = static_cast<float>(x[token * H + n_base + col]);
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
      if (h < H) dw[(static_cast<int64_t>(expert) * O + o) * H + h] = static_cast<scalar_t>(acc[i][j]);
    }
  }
}

// db[e][o] = sum_{i in tokens(e), ascending} dy[tok_i][o]; one thread per (e, o).
template <typename scalar_t>
__global__ void __launch_bounds__(kThreads) grouped_bias_grad_kernel(
    const scalar_t* __restrict__ dy,
    scalar_t* __restrict__ db,
    const int64_t* __restrict__ order,
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ counts,
    int64_t experts, int64_t O) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= experts * O) return;
  const int64_t expert = index / O;
  const int64_t o = index - expert * O;
  const int64_t off = offsets[expert];
  float value = 0.0f;
  for (int64_t i = 0; i < counts[expert]; ++i) {
    value += static_cast<float>(dy[order[off + i] * O + o]);
  }
  db[index] = static_cast<scalar_t>(value);
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
  TORCH_CHECK(lhs.scalar_type() == rhs.scalar_type(), lhs_name, " and ", rhs_name,
              " must share dtype");
}

struct Buckets {
  torch::Tensor order, offsets, counts, tile_off;
  int64_t max_tiles;
};

// Stable expert bucketing from deterministic torch ops (stable sort, integer scatter_add).
Buckets make_buckets(const torch::Tensor& expert_ids, int64_t experts) {
  Buckets b;
  auto opts = expert_ids.options().dtype(torch::kInt64);
  b.order = std::get<1>(expert_ids.sort(/*stable=*/true, /*dim=*/0, /*descending=*/false));
  b.counts = torch::zeros({experts}, opts).scatter_add_(0, expert_ids, torch::ones_like(expert_ids));
  auto csum = b.counts.cumsum(0);
  b.offsets = csum - b.counts;
  auto tiles = (b.counts + (kTile - 1)).floor_divide(kTile);
  b.tile_off = torch::cat({torch::zeros({1}, opts), tiles.cumsum(0)});
  b.max_tiles = (expert_ids.numel() + kTile - 1) / kTile + experts;
  return b;
}

void check_indices(const torch::Tensor& expert_indices) {
  TORCH_CHECK(expert_indices.is_cuda() && expert_indices.is_contiguous(),
              "expert_indices must be contiguous ROCm");
  TORCH_CHECK(expert_indices.scalar_type() == at::kInt || expert_indices.scalar_type() == at::kLong,
              "expert_indices must be int32 or int64");
}

}  // namespace

torch::Tensor inkling2_grouped_gemm_forward(
    torch::Tensor tokens,
    torch::Tensor expert_indices,
    torch::Tensor expert_weights,
    torch::Tensor bias) {
  check_tensor(tokens, "tokens");
  check_tensor(expert_weights, "expert_weights");
  check_same(tokens, expert_weights, "tokens", "expert_weights");
  check_indices(expert_indices);
  TORCH_CHECK(tokens.dim() == 2, "tokens must be [tokens, hidden]");
  TORCH_CHECK(expert_weights.dim() == 3, "expert_weights must be [experts, output, hidden]");
  TORCH_CHECK(expert_indices.dim() == 1 && expert_indices.size(0) == tokens.size(0),
              "expert_indices must have one entry per token");
  TORCH_CHECK(expert_weights.size(2) == tokens.size(1), "hidden size mismatch");
  TORCH_CHECK(bias.numel() == 0 || (bias.is_cuda() && bias.is_contiguous()),
              "bias must be empty or contiguous ROCm");
  const bool has_bias = bias.numel() != 0;
  if (has_bias) {
    check_tensor(bias, "bias");
    check_same(tokens, bias, "tokens", "bias");
    TORCH_CHECK(bias.sizes() == torch::IntArrayRef({expert_weights.size(0), expert_weights.size(1)}),
                "bias must be [experts, output]");
  }

  const c10::OptionalDeviceGuard device_guard(device_of(tokens));
  auto expert_ids = expert_indices.to(torch::kInt64);
  if (expert_ids.numel() != 0) {
    TORCH_CHECK(expert_ids.min().item<int64_t>() >= 0 &&
                    expert_ids.max().item<int64_t>() < expert_weights.size(0),
                "expert index is outside the available expert range");
  }
  const int64_t experts = expert_weights.size(0), O = expert_weights.size(1), H = tokens.size(1);
  auto output = torch::empty({tokens.size(0), O}, tokens.options());
  if (output.numel() == 0) return output;
  if (H == 0) {  // empty reduction: bias only (or zero)
    if (has_bias) output.copy_(bias.index_select(0, expert_ids));
    else output.zero_();
    return output;
  }
  auto b = make_buckets(expert_ids, experts);
  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(static_cast<unsigned>(b.max_tiles), static_cast<unsigned>((O + kTile - 1) / kTile));
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, tokens.scalar_type(),
                                  "inkling2_grouped_gemm_forward", [&] {
    if (has_bias) {
      grouped_rows_gemm_kernel<scalar_t, true, true><<<grid, kThreads, 0, stream>>>(
          tokens.data_ptr<scalar_t>(), expert_weights.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(),
          output.data_ptr<scalar_t>(), b.order.data_ptr<int64_t>(), b.offsets.data_ptr<int64_t>(),
          b.counts.data_ptr<int64_t>(), b.tile_off.data_ptr<int64_t>(), static_cast<int>(experts), H, O);
    } else {
      grouped_rows_gemm_kernel<scalar_t, true, false><<<grid, kThreads, 0, stream>>>(
          tokens.data_ptr<scalar_t>(), expert_weights.data_ptr<scalar_t>(), nullptr,
          output.data_ptr<scalar_t>(), b.order.data_ptr<int64_t>(), b.offsets.data_ptr<int64_t>(),
          b.counts.data_ptr<int64_t>(), b.tile_off.data_ptr<int64_t>(), static_cast<int>(experts), H, O);
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::vector<torch::Tensor> inkling2_grouped_gemm_backward(
    torch::Tensor dy,
    torch::Tensor tokens,
    torch::Tensor expert_indices,
    torch::Tensor expert_weights,
    bool need_bias) {
  check_tensor(dy, "dy");
  check_tensor(tokens, "tokens");
  check_tensor(expert_weights, "expert_weights");
  check_same(dy, tokens, "dy", "tokens");
  check_same(tokens, expert_weights, "tokens", "expert_weights");
  check_indices(expert_indices);
  TORCH_CHECK(dy.dim() == 2 && dy.size(0) == tokens.size(0) && dy.size(1) == expert_weights.size(1),
              "dy shape must match grouped GEMM output");

  const c10::OptionalDeviceGuard device_guard(device_of(tokens));
  auto expert_ids = expert_indices.to(torch::kInt64);
  const int64_t experts = expert_weights.size(0), O = expert_weights.size(1), H = tokens.size(1);
  auto d_tokens = torch::empty_like(tokens);
  auto d_weights = torch::empty_like(expert_weights);
  auto d_bias = need_bias ? torch::empty({experts, O}, tokens.options()) : torch::Tensor();
  if (tokens.numel() == 0 || expert_weights.numel() == 0) return {d_tokens, d_weights, d_bias};
  TORCH_CHECK(expert_ids.min().item<int64_t>() >= 0 && expert_ids.max().item<int64_t>() < experts,
              "expert index is outside the available expert range");

  auto b = make_buckets(expert_ids, experts);
  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 token_grid(static_cast<unsigned>(b.max_tiles), static_cast<unsigned>((H + kTile - 1) / kTile));
  const dim3 weight_grid(static_cast<unsigned>((H + kTile - 1) / kTile),
                         static_cast<unsigned>((O + kTile - 1) / kTile), static_cast<unsigned>(experts));
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, tokens.scalar_type(),
                                  "inkling2_grouped_gemm_backward", [&] {
    grouped_rows_gemm_kernel<scalar_t, false, false><<<token_grid, kThreads, 0, stream>>>(
        dy.data_ptr<scalar_t>(), expert_weights.data_ptr<scalar_t>(), nullptr,
        d_tokens.data_ptr<scalar_t>(), b.order.data_ptr<int64_t>(), b.offsets.data_ptr<int64_t>(),
        b.counts.data_ptr<int64_t>(), b.tile_off.data_ptr<int64_t>(), static_cast<int>(experts), O, H);
    grouped_weight_grad_kernel<scalar_t><<<weight_grid, kThreads, 0, stream>>>(
        dy.data_ptr<scalar_t>(), tokens.data_ptr<scalar_t>(), d_weights.data_ptr<scalar_t>(),
        b.order.data_ptr<int64_t>(), b.offsets.data_ptr<int64_t>(), b.counts.data_ptr<int64_t>(), O, H);
    if (need_bias) {
      const int blocks = static_cast<int>((experts * O + kThreads - 1) / kThreads);
      grouped_bias_grad_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
          dy.data_ptr<scalar_t>(), d_bias.data_ptr<scalar_t>(), b.order.data_ptr<int64_t>(),
          b.offsets.data_ptr<int64_t>(), b.counts.data_ptr<int64_t>(), experts, O);
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {d_tokens, d_weights, d_bias};
}
