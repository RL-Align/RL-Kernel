// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// DSv4 KV compressor, indexer entries (head 128): compress -> RMSNorm -> RoPE ->
// FP8 entry into the paged indexer K cache (SM90).
//
// Byte-equal to rl_engine.kernels.ops.pytorch.compressor.compress_indexer (see
// its docstring for the contract). One 32-thread block per token; non-boundary
// tokens and tokens without a cache slot exit at once. Each thread owns 4
// consecutive channels and holds the whole window in registers:
//
//   window   positions pos-W+1 .. pos (W = coff * ratio), negatives skipped;
//            with overlap the older `ratio` tokens read channel half 0
//   softmax  m = max s_j; S = sum_j expf(s_j - m) ascending; x = (sum_j kv_j *
//            e_j ascending) / S; every step an explicit round-to-nearest op
//   RMSNorm  4 squares per thread in order, a warp butterfly, warp partials in
//            ascending order;
//            r = 1 / sqrt(v / D + eps)
//   RoPE     GPT-J pairs on the last rope_dim channels at (pos / ratio) * ratio,
//            (both pairs of a thread are its own), products rounded, no FMA
//   store    BF16 round, power-of-two FP8 scale, e4m3 satfinite bytes and the
//            FP32 scale into the cache block (D FP8 rows, then FP32 scales)
//
// Each entry reads only its own request's state rows (through its block
// table), so the output is batch-invariant.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

#if defined(__USE_FAST_MATH__) || defined(__FAST_MATH__)
#error "compress_fp8.cu must not be built with fast-math (byte-exact contract)"
#endif

namespace {

template <typename T>
__device__ __forceinline__ float to_f32(T v);
template <>
__device__ __forceinline__ float to_f32<float>(float v) { return v; }
template <>
__device__ __forceinline__ float to_f32<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

// D/4 threads per token, 4 consecutive channels per thread; the whole window
// (W tokens of kv and score) is loaded once into registers.
template <int D, int W, typename CS, typename RW>
__global__ void __launch_bounds__(D / 4) compress_fp8_entry_kernel(
    const float* __restrict__ state, int64_t state_block_stride, int64_t state_token_stride,
    int state_block_size, const int* __restrict__ block_table, int64_t block_table_stride,
    const int* __restrict__ token_to_req, const int64_t* __restrict__ positions,
    const RW* __restrict__ rms_weight, float eps, const CS* __restrict__ cos_sin,
    int64_t cs_stride, int rope_dim, uint8_t* __restrict__ k_cache, int64_t k_block_stride,
    int k_block_size, const int64_t* __restrict__ k_slots, int ratio, int coff) {
  constexpr int kThreads = D / 4;
  constexpr int kWarps = (kThreads + 31) / 32;
  __shared__ float s_partial[kWarps];
  __shared__ uint32_t s_amax[kWarps];

  const int token = blockIdx.x;
  const int64_t pos = positions[token];
  if ((pos + 1) % ratio != 0) return;
  const int64_t slot = k_slots[token];
  if (slot < 0) return;
  const int t = threadIdx.x;
  const int lane = t & 31, warp = t >> 5;
  const int c0 = 4 * t;
  const int width = coff * D;
  const int* table = block_table + static_cast<int64_t>(token_to_req[token]) * block_table_stride;

  float4 kv[W], sc[W];
  bool valid[W];
#pragma unroll
  for (int j = 0; j < W; ++j) {
    const int64_t p = pos - W + 1 + j;
    valid[j] = p >= 0;
    if (valid[j]) {
      const float* r = state +
                       static_cast<int64_t>(table[p / state_block_size]) * state_block_stride +
                       (p % state_block_size) * state_token_stride;
      const int half = (coff == 2 && j >= ratio) ? D : 0;
      kv[j] = __ldg(reinterpret_cast<const float4*>(r + half + c0));
      sc[j] = __ldg(reinterpret_cast<const float4*>(r + width + half + c0));
    } else {
      kv[j] = sc[j] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
  }

  // Per channel: softmax over the window, then the weighted sum (ascending j).
  float x[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    auto comp = [i](const float4& v) { return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w; };
    float m = -INFINITY;
#pragma unroll
    for (int j = 0; j < W; ++j) {
      const float s = comp(sc[j]);
      if (valid[j] && (isnan(s) || s > m)) m = s;  // NaN propagates, as torch.amax does
    }
    float e[W];
    float total = 0.0f;
#pragma unroll
    for (int j = 0; j < W; ++j) {
      e[j] = expf(__fsub_rn(comp(sc[j]), m));
      if (valid[j]) total = __fadd_rn(total, e[j]);
    }
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < W; ++j) {
      if (valid[j]) acc = __fadd_rn(acc, __fmul_rn(comp(kv[j]), e[j]));
    }
    x[i] = __fdiv_rn(acc, total);
  }

  // RMSNorm: 4 squares in order, warp butterfly, warp partials in order.
  float sq = __fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(x[0], x[0]), __fmul_rn(x[1], x[1])),
                                 __fmul_rn(x[2], x[2])),
                       __fmul_rn(x[3], x[3]));
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sq = __fadd_rn(sq, __shfl_xor_sync(0xFFFFFFFFU, sq, offset));
  }
  if (lane == 0) s_partial[warp] = sq;
  __syncthreads();
  float sum = s_partial[0];
#pragma unroll
  for (int w = 1; w < kWarps; ++w) sum = __fadd_rn(sum, s_partial[w]);
  const float var = __fdiv_rn(sum, static_cast<float>(D));
  const float rr = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(var, eps)));
  float n[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) n[i] = __fmul_rn(__fmul_rn(x[i], rr), to_f32(rms_weight[c0 + i]));

  // GPT-J RoPE: pairs (c0, c0+1) and (c0+2, c0+3) live in this thread.
  const int nope = D - rope_dim;
  if (c0 >= nope) {
    const CS* cs = cos_sin + ((pos / ratio) * ratio) * cs_stride;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int k = (c0 - nope) / 2 + h;
      const float cv = to_f32(cs[k]), sv = to_f32(cs[rope_dim / 2 + k]);
      const float a = n[2 * h], b = n[2 * h + 1];
      n[2 * h] = __fsub_rn(__fmul_rn(a, cv), __fmul_rn(b, sv));
      n[2 * h + 1] = __fadd_rn(__fmul_rn(b, cv), __fmul_rn(a, sv));
    }
  }

  // BF16 round; power-of-two FP8 scale from the row amax (NaN ignored).
  float v[4];
  uint32_t mag = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const __nv_bfloat16 b = __float2bfloat16_rn(n[i]);
    v[i] = __bfloat162float(b);
    const uint32_t mi = __bfloat16_as_ushort(b) & 0x7FFFU;
    mag = max(mag, mi > 0x7F80U ? 0U : mi);
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    mag = max(mag, __shfl_xor_sync(0xFFFFFFFFU, mag, offset));
  }
  if (lane == 0) s_amax[warp] = mag;
  __syncthreads();
  uint32_t amax = s_amax[0];
#pragma unroll
  for (int w = 1; w < kWarps; ++w) amax = max(amax, s_amax[w]);
  const float s = __fmul_rn(fmaxf(__uint_as_float(amax << 16), 1e-4f), 1.0f / 448.0f);
  const uint32_t e = ((__float_as_uint(s) + 0x7FFFFFU) >> 23) & 0xFFU;
  const float inverse = __uint_as_float((254U - e) << 23);

  uint8_t* block = k_cache + (slot / k_block_size) * k_block_stride;
  const int64_t in_block = slot % k_block_size;
  uint32_t packed = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    packed |= static_cast<uint32_t>(
                  __nv_cvt_float_to_fp8(__fmul_rn(v[i], inverse), __NV_SATFINITE, __NV_E4M3))
              << (8 * i);
  }
  *reinterpret_cast<uint32_t*>(block + in_block * D + c0) = packed;
  if (t == 0) {
    reinterpret_cast<float*>(block + static_cast<int64_t>(k_block_size) * D)[in_block] =
        __uint_as_float(e << 23);
  }
}

template <int D, int W, typename CS, typename RW>
void launch(const torch::Tensor& state, const torch::Tensor& block_table,
            const torch::Tensor& token_to_req, const torch::Tensor& positions,
            const torch::Tensor& rms_weight, double eps, const torch::Tensor& cos_sin,
            torch::Tensor& k_cache, int64_t k_block_size, const torch::Tensor& k_slots,
            int64_t ratio, int64_t coff, cudaStream_t stream) {
  compress_fp8_entry_kernel<D, W, CS, RW>
      <<<static_cast<unsigned>(positions.size(0)), D / 4, 0, stream>>>(
          state.data_ptr<float>(), state.stride(0), state.stride(1),
          static_cast<int>(state.size(1)), block_table.data_ptr<int>(), block_table.stride(0),
          token_to_req.data_ptr<int>(), positions.data_ptr<int64_t>(),
          reinterpret_cast<const RW*>(rms_weight.data_ptr()), static_cast<float>(eps),
          reinterpret_cast<const CS*>(cos_sin.data_ptr()), cos_sin.stride(0),
          static_cast<int>(cos_sin.size(1)), k_cache.data_ptr<uint8_t>(), k_cache.stride(0),
          static_cast<int>(k_block_size), k_slots.data_ptr<int64_t>(), static_cast<int>(ratio),
          static_cast<int>(coff));
}

}  // namespace

void compress_indexer_fp8_cuda(torch::Tensor state_cache, torch::Tensor block_table,
                               torch::Tensor token_to_req, torch::Tensor positions,
                               torch::Tensor rms_weight, double eps,
                               torch::Tensor cos_sin_cache, torch::Tensor k_cache,
                               int64_t k_block_size, torch::Tensor k_slots, int64_t ratio,
                               bool overlap) {
  constexpr int D = 128;
  const int64_t coff = overlap ? 2 : 1;
  TORCH_CHECK(state_cache.is_cuda() && state_cache.scalar_type() == torch::kFloat32 &&
                  state_cache.dim() == 3 && state_cache.stride(2) == 1 &&
                  state_cache.size(2) == 2 * coff * D,
              "compress_indexer: state_cache must be FP32 [blocks, block_size, 2 * coff * 128]");
  const int64_t tokens = positions.size(0);
  TORCH_CHECK(positions.is_cuda() && positions.scalar_type() == torch::kInt64 &&
                  positions.dim() == 1 && positions.is_contiguous(),
              "compress_indexer: positions must be contiguous int64 [tokens]");
  for (const auto* t : {&token_to_req, &block_table}) {
    TORCH_CHECK(t->is_cuda() && t->scalar_type() == torch::kInt32,
                "compress_indexer: token_to_req / block_table must be int32");
  }
  TORCH_CHECK(token_to_req.dim() == 1 && token_to_req.size(0) == tokens &&
                  token_to_req.is_contiguous() && block_table.dim() == 2 &&
                  block_table.stride(1) == 1,
              "compress_indexer: token_to_req [tokens], block_table [requests, blocks]");
  TORCH_CHECK(k_slots.is_cuda() && k_slots.scalar_type() == torch::kInt64 &&
                  k_slots.dim() == 1 && k_slots.size(0) == tokens && k_slots.is_contiguous(),
              "compress_indexer: k_slots must be contiguous int64 [tokens]");
  TORCH_CHECK(k_cache.is_cuda() && k_cache.scalar_type() == torch::kUInt8 &&
                  k_cache.dim() == 2 && k_cache.stride(1) == 1 &&
                  k_cache.size(1) >= k_block_size * (D + 4) && k_block_size > 0,
              "compress_indexer: k_cache must be uint8 [blocks, >= block_size * 132]");
  TORCH_CHECK(rms_weight.is_cuda() && rms_weight.dim() == 1 && rms_weight.size(0) == D &&
                  rms_weight.is_contiguous() &&
                  (rms_weight.scalar_type() == torch::kFloat32 ||
                   rms_weight.scalar_type() == torch::kBFloat16),
              "compress_indexer: rms_weight must be FP32/BF16 [128]");
  TORCH_CHECK(cos_sin_cache.is_cuda() && cos_sin_cache.dim() == 2 &&
                  cos_sin_cache.stride(1) == 1 && cos_sin_cache.size(1) % 2 == 0 &&
                  cos_sin_cache.size(1) <= D &&
                  (cos_sin_cache.scalar_type() == torch::kFloat32 ||
                   cos_sin_cache.scalar_type() == torch::kBFloat16),
              "compress_indexer: cos_sin_cache must be FP32/BF16 [positions, rope_dim]");
  TORCH_CHECK(ratio * coff == 8,
              "compress_indexer: the indexer compressor is C4 with overlap (window of 8)");
  TORCH_CHECK(state_cache.stride(1) % 4 == 0 && state_cache.stride(0) % 4 == 0 &&
                  reinterpret_cast<uintptr_t>(state_cache.data_ptr()) % 16 == 0,
              "compress_indexer: state rows must be 16-byte aligned");
  if (tokens == 0) return;

  const c10::cuda::CUDAGuard guard(state_cache.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const bool cs32 = cos_sin_cache.scalar_type() == torch::kFloat32;
  const bool w32 = rms_weight.scalar_type() == torch::kFloat32;
#define RLK_COMPRESS_LAUNCH(CS, RW)                                                          \
  launch<D, 8, CS, RW>(state_cache, block_table, token_to_req, positions, rms_weight, eps,      \
                    cos_sin_cache, k_cache, k_block_size, k_slots, ratio, coff, stream)
  if (cs32 && w32) RLK_COMPRESS_LAUNCH(float, float);
  else if (cs32) RLK_COMPRESS_LAUNCH(float, __nv_bfloat16);
  else if (w32) RLK_COMPRESS_LAUNCH(__nv_bfloat16, float);
  else RLK_COMPRESS_LAUNCH(__nv_bfloat16, __nv_bfloat16);
#undef RLK_COMPRESS_LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
