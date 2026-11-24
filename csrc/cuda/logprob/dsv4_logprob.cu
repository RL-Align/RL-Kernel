// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// DSv4 selected-token log-probability, numeric profile dsv4-logprob-v1. Byte-equal to the
// golden rl_engine.kernels.ops.pytorch.logprob_dsv4 (its docstring is the arithmetic
// contract). The logits come from DeepGEMM bf16_gemm_nt with FP32 output (Python side).
//
// Every FP32 step is spelled __f*_rn, so nothing is contracted into an FMA; expf / logf are
// libdevice, like PyTorch's CUDA exp / log.
//
// Kernels
//   tile_stats: one warp per (row, tile): tile max, then 32 lane sums of expf(z' - m) in
//               column order, then a halving reduction. Writes (m, s) [n, tiles].
//   merge:      one warp per row: the 64 tile partials -> LSE and the target's logp.
//   backward:   elementwise dL/dz = g * (onehot - p) [/ T], rounded once to BF16.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>

#include <cstdint>
#include <vector>

#if defined(__USE_FAST_MATH__)
#error "dsv4_logprob.cu must not be built with --use_fast_math (dsv4-logprob-v1 is bitwise)"
#endif

namespace {

constexpr int kTiles = 64;
constexpr int kWarps = 8;
constexpr unsigned kFull = 0xffffffffu;

__device__ __forceinline__ float scaled(float z, bool scale, float t) {
  return scale ? __fdiv_rn(z, t) : z;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(kFull, v, o));
  return v;
}

__device__ __forceinline__ float warp_halve(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_down_sync(kFull, v, o));
  return v;  // lane 0 holds the result
}

// grid (n, ceil(tiles / kWarps)), block kWarps * 32.
__global__ void __launch_bounds__(kWarps * 32)
    tile_stats_kernel(const float* __restrict__ z, int64_t ld, int tiles, int tile_len,
                      bool scale, float t, float* __restrict__ m_out,
                      float* __restrict__ s_out) {
  const int lane = threadIdx.x & 31;
  const int tile = blockIdx.y * kWarps + (threadIdx.x >> 5);
  if (tile >= tiles) return;  // warp-uniform
  const int64_t row = blockIdx.x;
  const float* src = z + row * ld + static_cast<int64_t>(tile) * tile_len;

  float m = -INFINITY;
  for (int c = lane; c < tile_len; c += 32) m = fmaxf(m, scaled(__ldg(src + c), scale, t));
  m = warp_max(m);
  float s = 0.f;
  for (int c = lane; c < tile_len; c += 32)
    s = __fadd_rn(s, expf(__fsub_rn(scaled(__ldg(src + c), scale, t), m)));
  s = warp_halve(s);
  if (lane == 0) {
    m_out[row * tiles + tile] = m;
    s_out[row * tiles + tile] = s;
  }
}

// One warp per row; block kWarps * 32 covers kWarps rows.
__global__ void __launch_bounds__(kWarps * 32)
    merge_kernel(const float* __restrict__ m, const float* __restrict__ s,
                 const float* __restrict__ zt, int64_t n, bool scale, float t,
                 float* __restrict__ logp, float* __restrict__ lse) {
  const int lane = threadIdx.x & 31;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarps + (threadIdx.x >> 5);
  if (row >= n) return;
  const float* mr = m + row * kTiles;
  const float* sr = s + row * kTiles;
  const float ma = mr[lane], mb = mr[lane + 32];
  const float big = warp_max(fmaxf(ma, mb));
  const float ea = __fmul_rn(sr[lane], expf(__fsub_rn(ma, big)));
  const float eb = __fmul_rn(sr[lane + 32], expf(__fsub_rn(mb, big)));
  const float sum = warp_halve(__fadd_rn(ea, eb));
  if (lane == 0) {
    const float l = __fadd_rn(big, logf(sum));
    lse[row] = l;
    logp[row] = __fsub_rn(scaled(zt[row], scale, t), l);
  }
}

// grid (ceil(cols / (256 * 4)), n), block 256; four columns per thread.
__global__ void __launch_bounds__(256)
    backward_kernel(const float* __restrict__ z, int64_t ld, int64_t cols, int64_t col_offset,
                    const int64_t* __restrict__ targets, const float* __restrict__ lse,
                    const float* __restrict__ grad, bool scale, float t,
                    __nv_bfloat16* __restrict__ out) {
  const int64_t row = blockIdx.y;
  const int64_t c0 = (static_cast<int64_t>(blockIdx.x) * 256 + threadIdx.x) * 4;
  if (c0 >= cols) return;
  const float l = lse[row], g = grad[row];
  const int64_t target = targets[row] - col_offset;
  const float* src = z + row * ld;
  __nv_bfloat16* dst = out + row * cols;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int64_t c = c0 + i;
    if (c >= cols) break;
    const float p = expf(__fsub_rn(scaled(src[c], scale, t), l));
    float d = c == target ? __fsub_rn(1.f, p) : -p;
    d = __fmul_rn(g, d);
    if (scale) d = __fdiv_rn(d, t);
    dst[c] = __float2bfloat16_rn(d);
  }
}

void check_logits(const torch::Tensor& z) {
  TORCH_CHECK(z.is_cuda() && z.scalar_type() == torch::kFloat32 && z.dim() == 2,
              "dsv4_logprob: logits must be CUDA FP32 [n, V]");
  TORCH_CHECK(z.stride(1) == 1, "dsv4_logprob: logits rows must be contiguous");
}

bool scale_of(double temperature) {
  TORCH_CHECK(temperature > 0.0, "dsv4_logprob: temperature must be positive");
  return static_cast<float>(temperature) != 1.f;
}

}  // namespace

std::vector<torch::Tensor> dsv4_logprob_tile_stats(torch::Tensor z, int64_t tile_len,
                                                   double temperature) {
  check_logits(z);
  TORCH_CHECK(tile_len > 0 && z.size(1) % tile_len == 0,
              "dsv4_logprob_tile_stats: columns must be whole tiles");
  const c10::cuda::CUDAGuard guard(z.device());
  const int64_t n = z.size(0);
  const int tiles = static_cast<int>(z.size(1) / tile_len);
  auto opts = z.options();
  auto m = torch::empty({n, tiles}, opts), s = torch::empty({n, tiles}, opts);
  if (n == 0) return {m, s};
  TORCH_CHECK(n <= 2147483647LL, "dsv4_logprob_tile_stats: too many rows");
  const bool scale = scale_of(temperature);
  dim3 grid(static_cast<unsigned>(n), (tiles + kWarps - 1) / kWarps);
  tile_stats_kernel<<<grid, kWarps * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      z.data_ptr<float>(), z.stride(0), tiles, static_cast<int>(tile_len), scale,
      static_cast<float>(temperature), m.data_ptr<float>(), s.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {m, s};
}

std::vector<torch::Tensor> dsv4_logprob_merge(torch::Tensor m, torch::Tensor s, torch::Tensor zt,
                                              double temperature) {
  for (const auto* x : {&m, &s})
    TORCH_CHECK(x->is_cuda() && x->scalar_type() == torch::kFloat32 && x->dim() == 2 &&
                    x->size(1) == kTiles && x->is_contiguous(),
                "dsv4_logprob_merge: partials must be contiguous CUDA FP32 [n, 64]");
  TORCH_CHECK(s.size(0) == m.size(0), "dsv4_logprob_merge: partial shapes differ");
  TORCH_CHECK(zt.is_cuda() && zt.scalar_type() == torch::kFloat32 && zt.is_contiguous() &&
                  zt.numel() == m.size(0),
              "dsv4_logprob_merge: target logits must be contiguous CUDA FP32 [n]");
  const c10::cuda::CUDAGuard guard(m.device());
  const int64_t n = m.size(0);
  auto logp = torch::empty({n}, m.options()), lse = torch::empty({n}, m.options());
  if (n == 0) return {logp, lse};
  const bool scale = scale_of(temperature);
  const auto blocks = static_cast<unsigned>((n + kWarps - 1) / kWarps);
  merge_kernel<<<blocks, kWarps * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      m.data_ptr<float>(), s.data_ptr<float>(), zt.data_ptr<float>(), n, scale,
      static_cast<float>(temperature), logp.data_ptr<float>(), lse.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {logp, lse};
}

torch::Tensor dsv4_logprob_backward(torch::Tensor z, torch::Tensor targets, torch::Tensor lse,
                                    torch::Tensor grad, double temperature, int64_t col_offset) {
  check_logits(z);
  const int64_t n = z.size(0), cols = z.size(1);
  TORCH_CHECK(targets.is_cuda() && targets.scalar_type() == torch::kInt64 &&
                  targets.is_contiguous() && targets.numel() == n,
              "dsv4_logprob_backward: targets must be contiguous CUDA int64 [n]");
  for (const auto* x : {&lse, &grad})
    TORCH_CHECK(x->is_cuda() && x->scalar_type() == torch::kFloat32 && x->is_contiguous() &&
                    x->numel() == n,
                "dsv4_logprob_backward: lse / grad must be contiguous CUDA FP32 [n]");
  const c10::cuda::CUDAGuard guard(z.device());
  auto out = torch::empty({n, cols}, z.options().dtype(torch::kBFloat16));
  if (n == 0 || cols == 0) return out;
  TORCH_CHECK(n <= 65535, "dsv4_logprob_backward: at most 65535 rows per call");
  const bool scale = scale_of(temperature);
  dim3 grid(static_cast<unsigned>((cols + 1023) / 1024), static_cast<unsigned>(n));
  backward_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      z.data_ptr<float>(), z.stride(0), cols, col_offset, targets.data_ptr<int64_t>(),
      lse.data_ptr<float>(), grad.data_ptr<float>(), scale, static_cast<float>(temperature),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
