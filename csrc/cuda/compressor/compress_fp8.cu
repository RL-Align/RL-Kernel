// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// DSv4 KV compressor: compress -> RMSNorm -> RoPE -> FP8 cache entry (SM90).
//
// Byte-equal to rl_engine.kernels.ops.pytorch.compressor (compress_indexer and
// compress_main; see that module for the contract). One block of D/4 threads
// per token; non-boundary tokens and tokens without a cache slot exit at once.
// Each thread owns 4 consecutive channels:
//
//   window   positions pos-W+1 .. pos (W = coff * ratio), negatives skipped;
//            with overlap the older `ratio` tokens read channel half 0. W = 8
//            (C4) is held in registers; W = 128 (C128) is read in two passes
//   softmax  m = max s_j; S = sum_j expf(s_j - m) ascending; x = (sum_j kv_j *
//            e_j ascending) / S; every step an explicit round-to-nearest op
//   RMSNorm  4 squares per thread in order, a warp butterfly, warp partials in
//            ascending order; r = 1 / sqrt(v / D + eps); n = (x * r) * w
//   RoPE     GPT-J pairs on the last rope_dim channels at (pos / ratio) * ratio
//            (both pairs of a thread are its own), products rounded, no FMA
//   store    BF16 round, then
//              indexer (D = 128): one power-of-two scale over the row; D FP8
//                bytes per entry, FP32 scales after the block's rows
//              fp8_ds_mla (D = 512): 448 nope values in 7 blocks of 64 with a
//                UE8M0 byte each, 64 RoPE values kept BF16; 576-byte rows, then
//                8 scale bytes (7 + zero pad) per entry after the block's rows
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

enum class Format { kIndexer, kDsMla };

template <typename T>
__device__ __forceinline__ float to_f32(T v);
template <>
__device__ __forceinline__ float to_f32<float>(float v) { return v; }
template <>
__device__ __forceinline__ float to_f32<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

__device__ __forceinline__ float comp(const float4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

// Biased exponent of 2^ceil(log2(max(amax, 1e-4) / 448)), from the bits.
__device__ __forceinline__ uint32_t pow2_exponent(uint32_t amax_bf16_bits) {
  const float s = __fmul_rn(fmaxf(__uint_as_float(amax_bf16_bits << 16), 1e-4f), 1.0f / 448.0f);
  return ((__float_as_uint(s) + 0x7FFFFFU) >> 23) & 0xFFU;
}

__device__ __forceinline__ uint32_t fp8x4(const float v[4], float inverse) {
  uint32_t packed = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    packed |= static_cast<uint32_t>(
                  __nv_cvt_float_to_fp8(__fmul_rn(v[i], inverse), __NV_SATFINITE, __NV_E4M3))
              << (8 * i);
  }
  return packed;
}

struct StateView {
  const float* state;
  int64_t block_stride, token_stride;
  int block_size;
  const int* table;
  __device__ __forceinline__ const float* row(int64_t p) const {
    return state + static_cast<int64_t>(table[p / block_size]) * block_stride +
           (p % block_size) * token_stride;
  }
};

// Windowed softmax + weighted sum for this thread's 4 channels.
template <int D, int W>
__device__ __forceinline__ void compress_channels(const StateView& sv, int64_t pos, int ratio,
                                                  int coff, int c0, float x[4]) {
  const int width = coff * D;
  if constexpr (W <= 16) {
    float4 kv[W], sc[W];
    bool valid[W];
#pragma unroll
    for (int j = 0; j < W; ++j) {
      const int64_t p = pos - W + 1 + j;
      valid[j] = p >= 0;
      if (valid[j]) {
        const float* r = sv.row(p);
        const int half = (coff == 2 && j >= ratio) ? D : 0;
        kv[j] = __ldg(reinterpret_cast<const float4*>(r + half + c0));
        sc[j] = __ldg(reinterpret_cast<const float4*>(r + width + half + c0));
      } else {
        kv[j] = sc[j] = make_float4(0.f, 0.f, 0.f, 0.f);
      }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float m = -INFINITY;
#pragma unroll
      for (int j = 0; j < W; ++j) {
        const float s = comp(sc[j], i);
        if (valid[j] && (isnan(s) || s > m)) m = s;  // NaN propagates, as torch.amax does
      }
      float total = 0.0f, acc = 0.0f;
#pragma unroll
      for (int j = 0; j < W; ++j) {
        const float e = expf(__fsub_rn(comp(sc[j], i), m));
        if (valid[j]) {
          total = __fadd_rn(total, e);
          acc = __fadd_rn(acc, __fmul_rn(comp(kv[j], i), e));
        }
      }
      x[i] = __fdiv_rn(acc, total);
    }
  } else {
    float m[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    for (int j = 0; j < W; ++j) {
      const int64_t p = pos - W + 1 + j;
      if (p < 0) continue;
      const int half = (coff == 2 && j >= ratio) ? D : 0;
      const float4 s4 = __ldg(reinterpret_cast<const float4*>(sv.row(p) + width + half + c0));
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float s = comp(s4, i);
        if (isnan(s) || s > m[i]) m[i] = s;
      }
    }
    float total[4] = {0.f, 0.f, 0.f, 0.f}, acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int j = 0; j < W; ++j) {
      const int64_t p = pos - W + 1 + j;
      if (p < 0) continue;
      const int half = (coff == 2 && j >= ratio) ? D : 0;
      const float* r = sv.row(p);
      const float4 s4 = __ldg(reinterpret_cast<const float4*>(r + width + half + c0));
      const float4 k4 = __ldg(reinterpret_cast<const float4*>(r + half + c0));
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float e = expf(__fsub_rn(comp(s4, i), m[i]));
        total[i] = __fadd_rn(total[i], e);
        acc[i] = __fadd_rn(acc[i], __fmul_rn(comp(k4, i), e));
      }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) x[i] = __fdiv_rn(acc[i], total[i]);
  }
}

template <int D, int W, Format F, typename CS, typename RW>
__global__ void __launch_bounds__(D / 4) compress_fp8_kernel(
    StateView sv_base, int64_t block_table_stride, const int* __restrict__ token_to_req,
    const int64_t* __restrict__ positions, const RW* __restrict__ rms_weight, float eps,
    const CS* __restrict__ cos_sin, int64_t cs_stride, int rope_dim,
    uint8_t* __restrict__ k_cache, int64_t k_block_stride, int k_block_size,
    const int64_t* __restrict__ k_slots, int ratio, int coff) {
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
  StateView sv = sv_base;
  sv.table += static_cast<int64_t>(token_to_req[token]) * block_table_stride;

  float x[4];
  compress_channels<D, W>(sv, pos, ratio, coff, c0, x);

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
  const bool rotates = c0 >= nope;
  if (rotates) {
    const CS* cs = cos_sin + ((pos / ratio) * ratio) * cs_stride;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int k = (c0 - nope) / 2 + h;
      const float cv = to_f32(cs[k]), sn = to_f32(cs[rope_dim / 2 + k]);
      const float a = n[2 * h], b = n[2 * h + 1];
      n[2 * h] = __fsub_rn(__fmul_rn(a, cv), __fmul_rn(b, sn));
      n[2 * h + 1] = __fadd_rn(__fmul_rn(b, cv), __fmul_rn(a, sn));
    }
  }

  // BF16 round; |value| bit patterns with NaN mapped to 0.
  float v[4];
  uint16_t bits[4];
  uint32_t mag = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const __nv_bfloat16 b = __float2bfloat16_rn(n[i]);
    v[i] = __bfloat162float(b);
    bits[i] = __bfloat16_as_ushort(b);
    const uint32_t mi = bits[i] & 0x7FFFU;
    mag = max(mag, mi > 0x7F80U ? 0U : mi);
  }

  uint8_t* block = k_cache + (slot / k_block_size) * k_block_stride;
  const int64_t in_block = slot % k_block_size;
  if constexpr (F == Format::kIndexer) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      mag = max(mag, __shfl_xor_sync(0xFFFFFFFFU, mag, offset));
    }
    if (lane == 0) s_amax[warp] = mag;
    __syncthreads();
    uint32_t amax = s_amax[0];
#pragma unroll
    for (int w = 1; w < kWarps; ++w) amax = max(amax, s_amax[w]);
    const uint32_t e = pow2_exponent(amax);
    *reinterpret_cast<uint32_t*>(block + in_block * D + c0) =
        fp8x4(v, __uint_as_float((254U - e) << 23));
    if (t == 0) {
      reinterpret_cast<float*>(block + static_cast<int64_t>(k_block_size) * D)[in_block] =
          __uint_as_float(e << 23);
    }
  } else {
    constexpr int kRow = 576, kQuant = 64;
    // 64-channel quant blocks are 16-lane halves of a warp.
#pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
      mag = max(mag, __shfl_xor_sync(0xFFFFFFFFU, mag, offset));
    }
    uint8_t* row = block + in_block * kRow;
    uint8_t* scales = block + static_cast<int64_t>(k_block_size) * kRow + in_block * 8;
    if (!rotates) {
      const uint32_t e = pow2_exponent(mag);
      *reinterpret_cast<uint32_t*>(row + c0) = fp8x4(v, __uint_as_float((254U - e) << 23));
      if (c0 % kQuant == 0) scales[c0 / kQuant] = static_cast<uint8_t>(e);
    } else {
      uint2 pair;
      pair.x = static_cast<uint32_t>(bits[0]) | (static_cast<uint32_t>(bits[1]) << 16);
      pair.y = static_cast<uint32_t>(bits[2]) | (static_cast<uint32_t>(bits[3]) << 16);
      *reinterpret_cast<uint2*>(row + nope + 2 * (c0 - nope)) = pair;
      if (c0 == nope) scales[nope / kQuant] = 0;  // pad byte
    }
  }
}

template <int D, int W, Format F, typename CS, typename RW>
void launch(const torch::Tensor& state, const torch::Tensor& block_table,
            const torch::Tensor& token_to_req, const torch::Tensor& positions,
            const torch::Tensor& rms_weight, double eps, const torch::Tensor& cos_sin,
            torch::Tensor& k_cache, int64_t k_block_size, const torch::Tensor& k_slots,
            int64_t ratio, int64_t coff, cudaStream_t stream) {
  StateView sv{state.data_ptr<float>(), state.stride(0), state.stride(1),
               static_cast<int>(state.size(1)), block_table.data_ptr<int>()};
  compress_fp8_kernel<D, W, F, CS, RW>
      <<<static_cast<unsigned>(positions.size(0)), D / 4, 0, stream>>>(
          sv, block_table.stride(0), token_to_req.data_ptr<int>(), positions.data_ptr<int64_t>(),
          reinterpret_cast<const RW*>(rms_weight.data_ptr()), static_cast<float>(eps),
          reinterpret_cast<const CS*>(cos_sin.data_ptr()), cos_sin.stride(0),
          static_cast<int>(cos_sin.size(1)), k_cache.data_ptr<uint8_t>(), k_cache.stride(0),
          static_cast<int>(k_block_size), k_slots.data_ptr<int64_t>(), static_cast<int>(ratio),
          static_cast<int>(coff));
}

template <int D, int W, Format F>
void dispatch_dtypes(const torch::Tensor& state, const torch::Tensor& block_table,
                     const torch::Tensor& token_to_req, const torch::Tensor& positions,
                     const torch::Tensor& rms_weight, double eps, const torch::Tensor& cos_sin,
                     torch::Tensor& k_cache, int64_t k_block_size, const torch::Tensor& k_slots,
                     int64_t ratio, int64_t coff, cudaStream_t stream) {
  const bool cs32 = cos_sin.scalar_type() == torch::kFloat32;
  const bool w32 = rms_weight.scalar_type() == torch::kFloat32;
#define RLK_COMPRESS_LAUNCH(CS, RW)                                                         \
  launch<D, W, F, CS, RW>(state, block_table, token_to_req, positions, rms_weight, eps,     \
                          cos_sin, k_cache, k_block_size, k_slots, ratio, coff, stream)
  if (cs32 && w32) RLK_COMPRESS_LAUNCH(float, float);
  else if (cs32) RLK_COMPRESS_LAUNCH(float, __nv_bfloat16);
  else if (w32) RLK_COMPRESS_LAUNCH(__nv_bfloat16, float);
  else RLK_COMPRESS_LAUNCH(__nv_bfloat16, __nv_bfloat16);
#undef RLK_COMPRESS_LAUNCH
}

void check_common(const torch::Tensor& state_cache, const torch::Tensor& block_table,
                  const torch::Tensor& token_to_req, const torch::Tensor& positions,
                  const torch::Tensor& rms_weight, const torch::Tensor& cos_sin_cache,
                  const torch::Tensor& k_cache, int64_t k_block_size,
                  const torch::Tensor& k_slots, int64_t d, int64_t coff, int64_t row_bytes) {
  TORCH_CHECK(state_cache.is_cuda() && state_cache.scalar_type() == torch::kFloat32 &&
                  state_cache.dim() == 3 && state_cache.stride(2) == 1 &&
                  state_cache.size(2) == 2 * coff * d,
              "compressor: state_cache must be FP32 [blocks, block_size, 2 * coff * head_dim]");
  TORCH_CHECK(state_cache.stride(1) % 4 == 0 && state_cache.stride(0) % 4 == 0 &&
                  reinterpret_cast<uintptr_t>(state_cache.data_ptr()) % 16 == 0,
              "compressor: state rows must be 16-byte aligned");
  const int64_t tokens = positions.size(0);
  TORCH_CHECK(positions.is_cuda() && positions.scalar_type() == torch::kInt64 &&
                  positions.dim() == 1 && positions.is_contiguous(),
              "compressor: positions must be contiguous int64 [tokens]");
  TORCH_CHECK(token_to_req.is_cuda() && token_to_req.scalar_type() == torch::kInt32 &&
                  token_to_req.dim() == 1 && token_to_req.size(0) == tokens &&
                  token_to_req.is_contiguous(),
              "compressor: token_to_req must be contiguous int32 [tokens]");
  TORCH_CHECK(block_table.is_cuda() && block_table.scalar_type() == torch::kInt32 &&
                  block_table.dim() == 2 && block_table.stride(1) == 1,
              "compressor: block_table must be int32 [requests, blocks]");
  TORCH_CHECK(k_slots.is_cuda() && k_slots.scalar_type() == torch::kInt64 &&
                  k_slots.dim() == 1 && k_slots.size(0) == tokens && k_slots.is_contiguous(),
              "compressor: k_slots must be contiguous int64 [tokens]");
  TORCH_CHECK(k_cache.is_cuda() && k_cache.scalar_type() == torch::kUInt8 &&
                  k_cache.dim() == 2 && k_cache.stride(1) == 1 && k_block_size > 0 &&
                  k_cache.size(1) >= k_block_size * row_bytes,
              "compressor: k_cache must be uint8 [blocks, >= block_size * entry bytes]");
  TORCH_CHECK(rms_weight.is_cuda() && rms_weight.dim() == 1 && rms_weight.size(0) == d &&
                  rms_weight.is_contiguous() &&
                  (rms_weight.scalar_type() == torch::kFloat32 ||
                   rms_weight.scalar_type() == torch::kBFloat16),
              "compressor: rms_weight must be FP32/BF16 [head_dim]");
  TORCH_CHECK(cos_sin_cache.is_cuda() && cos_sin_cache.dim() == 2 &&
                  cos_sin_cache.stride(1) == 1 && cos_sin_cache.size(1) % 8 == 0 &&
                  cos_sin_cache.size(1) <= d &&
                  (cos_sin_cache.scalar_type() == torch::kFloat32 ||
                   cos_sin_cache.scalar_type() == torch::kBFloat16),
              "compressor: cos_sin_cache must be FP32/BF16 [positions, rope_dim]");
}

}  // namespace

void compress_indexer_fp8_cuda(torch::Tensor state_cache, torch::Tensor block_table,
                               torch::Tensor token_to_req, torch::Tensor positions,
                               torch::Tensor rms_weight, double eps,
                               torch::Tensor cos_sin_cache, torch::Tensor k_cache,
                               int64_t k_block_size, torch::Tensor k_slots, int64_t ratio,
                               bool overlap) {
  const int64_t coff = overlap ? 2 : 1;
  TORCH_CHECK(ratio * coff == 8,
              "compress_indexer: the indexer compressor is C4 with overlap (window of 8)");
  check_common(state_cache, block_table, token_to_req, positions, rms_weight, cos_sin_cache,
               k_cache, k_block_size, k_slots, 128, coff, 128 + 4);
  if (positions.size(0) == 0) return;
  const c10::cuda::CUDAGuard guard(state_cache.device());
  dispatch_dtypes<128, 8, Format::kIndexer>(state_cache, block_table, token_to_req, positions,
                                            rms_weight, eps, cos_sin_cache, k_cache,
                                            k_block_size, k_slots, ratio, coff,
                                            at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void compress_main_ds_mla_cuda(torch::Tensor state_cache, torch::Tensor block_table,
                               torch::Tensor token_to_req, torch::Tensor positions,
                               torch::Tensor rms_weight, double eps,
                               torch::Tensor cos_sin_cache, torch::Tensor k_cache,
                               int64_t k_block_size, torch::Tensor k_slots, int64_t ratio) {
  TORCH_CHECK(ratio == 4 || ratio == 128, "compress_main: ratio must be 4 or 128");
  const int64_t coff = ratio == 4 ? 2 : 1;
  check_common(state_cache, block_table, token_to_req, positions, rms_weight, cos_sin_cache,
               k_cache, k_block_size, k_slots, 512, coff, 576 + 8);
  TORCH_CHECK(cos_sin_cache.size(1) == 64, "compress_main: rope_dim must be 64 (448 + 64)");
  if (positions.size(0) == 0) return;
  const c10::cuda::CUDAGuard guard(state_cache.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (ratio == 4) {
    dispatch_dtypes<512, 8, Format::kDsMla>(state_cache, block_table, token_to_req, positions,
                                            rms_weight, eps, cos_sin_cache, k_cache,
                                            k_block_size, k_slots, ratio, coff, stream);
  } else {
    dispatch_dtypes<512, 128, Format::kDsMla>(state_cache, block_table, token_to_req,
                                              positions, rms_weight, eps, cos_sin_cache,
                                              k_cache, k_block_size, k_slots, ratio, coff,
                                              stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
