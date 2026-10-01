// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#ifndef LATENT_TILE_Y
#define LATENT_TILE_Y 32
#endif
#ifndef LATENT_THREADS_Y
#define LATENT_THREADS_Y 8
#endif

namespace {
constexpr int tile_y = LATENT_TILE_Y;
constexpr int tile_x = tile_y;
constexpr int threads_x = tile_x;
constexpr int threads_y = LATENT_THREADS_Y;
static_assert(tile_y % threads_y == 0 && threads_x * threads_y <= 1024);
template <typename Bits> struct Pair { Bits lo, hi; };

template <typename Bits, bool Unpack>
__global__ void latent_permute(const Pair<Bits>* __restrict__ input, Pair<Bits>* __restrict__ output, int64_t num_channels, int64_t latent_height, int64_t latent_width) {
  __shared__ Pair<Bits> shared_pairs[tile_y][tile_x + 1];
  const int thread_x = threadIdx.x, thread_y = threadIdx.y;
  const int64_t token_cols = latent_width / 2, pairs_per_token = num_channels * 2;
  const int64_t batch_pair_offset = static_cast<int64_t>(blockIdx.z) * num_channels * latent_height * token_cols;
  const int64_t token_col_start = static_cast<int64_t>(blockIdx.x) * tile_x;
  const int64_t token_row = blockIdx.y;
  for (int64_t token_pair_start = 0; token_pair_start < pairs_per_token; token_pair_start += tile_y) {
    #pragma unroll
    for (int thread_y_offset = 0; thread_y_offset < tile_y; thread_y_offset += threads_y) {
      if constexpr (Unpack) {
        const int64_t token_col = token_col_start + thread_y + thread_y_offset;
        const int64_t pair_in_token = token_pair_start + thread_x;
        if (token_col < token_cols && pair_in_token < pairs_per_token) {
          const int64_t input_offset = batch_pair_offset + (token_row * token_cols + token_col) * pairs_per_token + pair_in_token;
          shared_pairs[thread_y + thread_y_offset][thread_x] = input[input_offset];
        }
      } else {
        const int64_t token_col = token_col_start + thread_x;
        const int64_t pair_in_token = token_pair_start + thread_y + thread_y_offset;
        if (token_col < token_cols && pair_in_token < pairs_per_token) {
          const int64_t channel = pair_in_token / 2;
          const int64_t spatial_row = token_row * 2 + pair_in_token % 2;
          const int64_t input_offset = batch_pair_offset + (channel * latent_height + spatial_row) * token_cols + token_col;
          shared_pairs[thread_y + thread_y_offset][thread_x] = input[input_offset];
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int thread_y_offset = 0; thread_y_offset < tile_x; thread_y_offset += threads_y) {
      if constexpr (Unpack) {
        const int64_t token_col = token_col_start + thread_x;
        const int64_t pair_in_token = token_pair_start + thread_y + thread_y_offset;
        if (token_col < token_cols && pair_in_token < pairs_per_token) {
          const int64_t channel = pair_in_token / 2;
          const int64_t spatial_row = token_row * 2 + pair_in_token % 2;
          const int64_t output_offset = batch_pair_offset + (channel * latent_height + spatial_row) * token_cols + token_col;
          output[output_offset] = shared_pairs[thread_x][thread_y + thread_y_offset];
        }
      } else {
        const int64_t token_col = token_col_start + thread_y + thread_y_offset;
        const int64_t pair_in_token = token_pair_start + thread_x;
        if (token_col < token_cols && pair_in_token < pairs_per_token) {
          const int64_t output_offset = batch_pair_offset + (token_row * token_cols + token_col) * pairs_per_token + pair_in_token;
          output[output_offset] = shared_pairs[thread_x][thread_y + thread_y_offset];
        }
      }
    }
    if (token_pair_start + tile_y < pairs_per_token) {
      __syncthreads();
    }
  }
}

template <typename Bits, bool Unpack>
void launch(const torch::Tensor& x, torch::Tensor& y, int64_t B, int64_t C, int64_t H, int64_t W) {
  const dim3 grid((W / 2 + tile_x - 1) / tile_x, H / 2, B);
  const dim3 block(threads_x, threads_y);
  latent_permute<Bits, Unpack><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const Pair<Bits>*>(x.data_ptr()), reinterpret_cast<Pair<Bits>*>(y.data_ptr()), C, H, W);
}
}

torch::Tensor latent_pack_unpack_cuda(torch::Tensor x, int64_t B, int64_t C, int64_t H, int64_t W, bool unpack) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "expected contiguous CUDA input");
  TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16, "expected fp32, fp16, or bf16");
  TORCH_CHECK(B >= 0 && B <= 65535 && C > 0 && H > 0 && W > 0 && H % 2 == 0 && W % 2 == 0, "invalid latent dimensions");
  TORCH_CHECK(C <= INT32_MAX / 4 && H <= INT32_MAX && W <= INT32_MAX, "latent dimensions exceed indexing limits");
  const int64_t P = (H / 2) * (W / 2);
  TORCH_CHECK(P <= INT32_MAX && H / 2 <= 65535, "latent dimensions exceed launch limits");
  TORCH_CHECK(x.numel() / (C * 4) == B * P && x.numel() % (C * 4) == 0, "input element count does not match latent dimensions");
  if (unpack) {
    TORCH_CHECK(x.dim() == 3 && x.size(0) == B && x.size(1) == P && x.size(2) == C * 4, "expected packed [B,P,4*C] input");
  } else {
    const bool nchw = x.dim() == 4 && x.size(1) == C;
    const bool singleton = x.dim() == 5 && ((x.size(1) == C && x.size(2) == 1) || (x.size(1) == 1 && x.size(2) == C));
    TORCH_CHECK((nchw || singleton) && x.size(0) == B && x.size(-2) == H && x.size(-1) == W, "expected spatial latent input");
  }
  const c10::cuda::CUDAGuard guard(x.device());
  torch::Tensor y;
  if (unpack) {
    y = torch::empty({B, C, 1, H, W}, x.options());
  } else {
    y = torch::empty({B, P, C * 4}, x.options());
  }
  if (B == 0) {
    return y;
  }
  if (x.scalar_type() == at::kFloat) {
    if (unpack) {
      launch<uint32_t, true>(x, y, B, C, H, W);
    } else {
      launch<uint32_t, false>(x, y, B, C, H, W);
    }
  } else {
    if (unpack) {
      launch<uint16_t, true>(x, y, B, C, H, W);
    } else {
      launch<uint16_t, false>(x, y, B, C, H, W);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
