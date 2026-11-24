// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// DSv4 indexer query: RoPE + per-(token, head) FP8 quantization (SM90).
//
// Byte-equal to rl_engine.kernels.ops.pytorch.indexer.indexer_q_rope_fp8, which
// reproduces vLLM's CuTe-DSL IndexerQFp8Kernel (the rollout's kernel):
//
//   rope   rot0 = fma(q0, cos, -(q1 * sin)),  rot1 = fma(q0, sin, q1 * cos),
//          on GPT-J pairs of the last rope_dim values, rounded to BF16
//   amax   max |BF16 value| over the row, NaN ignored
//   scale  s = max(amax, 1e-4) * fp32(1/448), rounded up to 2^(e-127) with
//          e = ((bits(s) + 0x7FFFFF) >> 23) & 0xFF
//   q_fp8  e4m3 satfinite of value * 2^-(e-127)
//   w_out  (fp32(w) * fp32(softmax_scale * head_scale)) * 2^(e-127)
//
// Every FP32 step is an explicit round-to-nearest intrinsic, so neither FMA
// contraction nor fast-math can move a byte. Eight threads own a row (16
// values each) and reduce its amax with shuffles inside their group; a group
// handles 2 heads of one token (1 for odd head counts). A row's bytes depend on nothing outside
// the row, whatever the grouping (batch-invariant).

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <type_traits>

#if defined(__USE_FAST_MATH__) || defined(__FAST_MATH__)
#error "indexer_q_rope_fp8.cu must not be built with fast-math (byte-exact contract)"
#endif

namespace {

constexpr int kHeadDim = 128;
constexpr int kLanes = 8;  // threads per (token, head) row
constexpr int kVals = kHeadDim / kLanes;  // 16 values per thread
constexpr int kThreads = 256;
constexpr int kRowsPerBlock = kThreads / kLanes;

__device__ __forceinline__ uint16_t float_to_bf16_bits(float x) {
  return __bfloat16_as_ushort(__float2bfloat16_rn(x));
}

template <typename CS>
__device__ __forceinline__ float load_cs(const CS* p);
template <>
__device__ __forceinline__ float load_cs<float>(const float* p) { return __ldg(p); }
template <>
__device__ __forceinline__ float load_cs<__nv_bfloat16>(const __nv_bfloat16* p) {
  return __bfloat162float(__ldg(p));
}

// A group of 8 threads owns kHeads heads of one token (the token's rotary
// values are loaded once). Values stay packed as BF16 pairs in 32-bit words:
// word i holds values 2i (low half) and 2i + 1 (high half).
template <typename CS, typename Pos, int kHeads>
__global__ void __launch_bounds__(kThreads) indexer_q_rope_fp8_kernel(
    const __nv_bfloat16* __restrict__ q, int64_t q_stride0, int64_t q_stride1,
    const Pos* __restrict__ positions, const CS* __restrict__ cos_sin, int64_t cs_stride0,
    int rope_dim, const __nv_bfloat16* __restrict__ weights, int64_t w_stride0,
    uint8_t* __restrict__ q_fp8, float* __restrict__ weights_out, int64_t groups, int heads,
    float combined_scale) {
  constexpr int kWords = kVals / 2;
  const int64_t group = static_cast<int64_t>(blockIdx.x) * kRowsPerBlock + threadIdx.x / kLanes;
  const int lane = threadIdx.x % kLanes;
  const bool active = group < groups;  // inactive lanes still join the shuffles
  const int tiles = heads / kHeads;
  const int64_t token = active ? group / tiles : 0;
  const int head0 = active ? static_cast<int>(group % tiles) * kHeads : 0;
  const int nope = kHeadDim - rope_dim;
  const int first = lane * kVals;

  uint32_t words[kHeads][kWords];
#pragma unroll
  for (int h = 0; h < kHeads; ++h) {
    uint4 a = make_uint4(0, 0, 0, 0), b = a;
    if (active) {
      const uint4* src = reinterpret_cast<const uint4*>(
          q + token * q_stride0 + static_cast<int64_t>(head0 + h) * q_stride1 + first);
      a = __ldg(src);
      b = __ldg(src + 1);
    }
    words[h][0] = a.x, words[h][1] = a.y, words[h][2] = a.z, words[h][3] = a.w;
    words[h][4] = b.x, words[h][5] = b.y, words[h][6] = b.z, words[h][7] = b.w;
  }

  // RoPE on whole 16-value chunks (nope is a multiple of 16): pair k is word k.
  if (active && first >= nope) {
    const int half = rope_dim / 2;
    const int pair0 = (first - nope) / 2;
    const CS* cs_row = cos_sin + static_cast<int64_t>(positions[token]) * cs_stride0;
    float cs_c[kWords], cs_s[kWords];
#pragma unroll
    for (int k = 0; k < kWords; ++k) {
      cs_c[k] = load_cs(cs_row + pair0 + k);
      cs_s[k] = load_cs(cs_row + half + pair0 + k);
    }
#pragma unroll
    for (int h = 0; h < kHeads; ++h) {
#pragma unroll
      for (int k = 0; k < kWords; ++k) {
        const float q0 = __uint_as_float(words[h][k] << 16);
        const float q1 = __uint_as_float(words[h][k] & 0xFFFF0000U);
        const float r0 = __fmaf_rn(q0, cs_c[k], -__fmul_rn(q1, cs_s[k]));
        const float r1 = __fmaf_rn(q0, cs_s[k], __fmul_rn(q1, cs_c[k]));
        words[h][k] = static_cast<uint32_t>(float_to_bf16_bits(r0)) |
                      (static_cast<uint32_t>(float_to_bf16_bits(r1)) << 16);
      }
    }
  }

#pragma unroll
  for (int h = 0; h < kHeads; ++h) {
    // amax over |BF16| bit patterns, two halves at a time; NaN (> 0x7F80)
    // does not count.
    uint32_t amax2 = 0;
#pragma unroll
    for (int k = 0; k < kWords; ++k) {
      const uint32_t mag = words[h][k] & 0x7FFF7FFFU;
      amax2 = __vmaxu2(amax2, mag & ~__vcmpgtu2(mag, 0x7F807F80U));
    }
    uint32_t amax = max(amax2 & 0xFFFFU, amax2 >> 16);
#pragma unroll
    for (int offset = 1; offset < kLanes; offset <<= 1) {
      amax = max(amax, __shfl_xor_sync(0xFFFFFFFFU, amax, offset));
    }
    const float s = __fmul_rn(fmaxf(__uint_as_float(amax << 16), 1e-4f), 1.0f / 448.0f);
    const uint32_t e = ((__float_as_uint(s) + 0x7FFFFFU) >> 23) & 0xFFU;
    const float scale = __uint_as_float(e << 23);
    const float inverse = __uint_as_float((254U - e) << 23);
    if (!active) continue;

    uint32_t packed[kWords / 2];
#pragma unroll
    for (int i = 0; i < kWords / 2; ++i) {
      uint32_t out = 0;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const uint32_t w = words[h][2 * i + j];
        const float v0 = __fmul_rn(__uint_as_float(w << 16), inverse);
        const float v1 = __fmul_rn(__uint_as_float(w & 0xFFFF0000U), inverse);
        const __nv_fp8x2_storage_t two =
            __nv_cvt_float2_to_fp8x2(make_float2(v0, v1), __NV_SATFINITE, __NV_E4M3);
        out |= static_cast<uint32_t>(two) << (16 * j);
      }
      packed[i] = out;
    }
    const int64_t row = token * heads + head0 + h;
    *reinterpret_cast<uint4*>(q_fp8 + row * kHeadDim + first) =
        make_uint4(packed[0], packed[1], packed[2], packed[3]);
    if (lane == h) {
      const float w = __bfloat162float(weights[token * w_stride0 + head0 + h]);
      weights_out[row] = __fmul_rn(__fmul_rn(w, combined_scale), scale);
    }
  }
}

template <typename CS, typename Pos>
void launch(const torch::Tensor& q, const torch::Tensor& positions, const torch::Tensor& cs,
            const torch::Tensor& weights, torch::Tensor& q_fp8, torch::Tensor& weights_out,
            float combined, cudaStream_t stream) {
  const int heads = static_cast<int>(q.size(1));
  auto run = [&](auto heads_per_group) {
    constexpr int kHeads = decltype(heads_per_group)::value;
    const int64_t groups = q.size(0) * (heads / kHeads);
    const auto blocks = static_cast<unsigned>((groups + kRowsPerBlock - 1) / kRowsPerBlock);
    indexer_q_rope_fp8_kernel<CS, Pos, kHeads><<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()), q.stride(0),
        q.stride(1), positions.data_ptr<Pos>(), reinterpret_cast<const CS*>(cs.data_ptr()),
        cs.stride(0), static_cast<int>(cs.size(1)),
        reinterpret_cast<const __nv_bfloat16*>(weights.data_ptr<at::BFloat16>()),
        weights.stride(0), reinterpret_cast<uint8_t*>(q_fp8.data_ptr()),
        weights_out.data_ptr<float>(), groups, heads, combined);
  };
  // Grouping heads only changes which thread computes a row, not its bytes.
  if (heads % 2 == 0) {
    run(std::integral_constant<int, 2>{});
  } else {
    run(std::integral_constant<int, 1>{});
  }
}

}  // namespace

std::vector<torch::Tensor> indexer_q_rope_fp8_forward(torch::Tensor q, torch::Tensor positions,
                                                      torch::Tensor cos_sin_cache,
                                                      torch::Tensor weights, double softmax_scale,
                                                      double head_scale) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kBFloat16 && q.dim() == 3,
              "indexer_q_rope_fp8: q must be BF16 [tokens, heads, 128] on CUDA");
  TORCH_CHECK(q.size(2) == kHeadDim && q.stride(2) == 1 && q.stride(1) % 8 == 0 &&
                  q.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0,
              "indexer_q_rope_fp8: q rows must be 128 contiguous, 16-byte aligned BF16 values");
  const int64_t tokens = q.size(0), heads = q.size(1);
  TORCH_CHECK(positions.is_cuda() && positions.dim() == 1 && positions.size(0) == tokens &&
                  (positions.scalar_type() == torch::kInt64 ||
                   positions.scalar_type() == torch::kInt32),
              "indexer_q_rope_fp8: positions must be int32/int64 [tokens] on CUDA");
  TORCH_CHECK(cos_sin_cache.is_cuda() && cos_sin_cache.dim() == 2 && cos_sin_cache.stride(1) == 1 &&
                  (cos_sin_cache.scalar_type() == torch::kFloat32 ||
                   cos_sin_cache.scalar_type() == torch::kBFloat16),
              "indexer_q_rope_fp8: cos_sin_cache must be FP32/BF16 [max_position, rope_dim]");
  const int64_t rope_dim = cos_sin_cache.size(1);
  TORCH_CHECK(rope_dim > 0 && rope_dim <= kHeadDim && rope_dim % 16 == 0,
              "indexer_q_rope_fp8: rope_dim must be a multiple of 16 in (0, 128]");
  TORCH_CHECK(weights.is_cuda() && weights.scalar_type() == torch::kBFloat16 &&
                  weights.dim() == 2 && weights.size(0) == tokens && weights.size(1) == heads &&
                  weights.stride(1) == 1,
              "indexer_q_rope_fp8: weights must be BF16 [tokens, heads] with unit head stride");
  positions = positions.contiguous();

  auto q_fp8 = torch::empty(q.sizes(), q.options().dtype(torch::kFloat8_e4m3fn));
  auto weights_out = torch::empty({tokens, heads}, q.options().dtype(torch::kFloat32));
  if (tokens * heads == 0) return {q_fp8, weights_out};

  const c10::cuda::CUDAGuard guard(q.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // Python's float(softmax_scale * head_scale) rounded to FP32, as vLLM passes it.
  const float combined = static_cast<float>(softmax_scale * head_scale);
  const bool fp32_cs = cos_sin_cache.scalar_type() == torch::kFloat32;
  const bool pos64 = positions.scalar_type() == torch::kInt64;
  if (fp32_cs && pos64) launch<float, int64_t>(q, positions, cos_sin_cache, weights, q_fp8, weights_out, combined, stream);
  else if (fp32_cs) launch<float, int32_t>(q, positions, cos_sin_cache, weights, q_fp8, weights_out, combined, stream);
  else if (pos64) launch<__nv_bfloat16, int64_t>(q, positions, cos_sin_cache, weights, q_fp8, weights_out, combined, stream);
  else launch<__nv_bfloat16, int32_t>(q, positions, cos_sin_cache, weights, q_fp8, weights_out, combined, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {q_fp8, weights_out};
}
