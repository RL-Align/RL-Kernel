// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// DSv4 attention query / output side (SM90). Byte-equal to
// rl_engine.kernels.ops.pytorch.attention_dsv4 (see qo.py for the contract).
//
// dsv4_q_norm_rope_kv_insert: one warp per (token, row), 16 values per lane.
//   Rows 0 .. padded_heads-1 are query heads (weightless RMSNorm: 16 squares in
//   order per lane, a 32-lane butterfly, r = 1 / sqrt(v / 512 + eps); RoPE;
//   BF16; heads >= num_heads written as zeros). Row padded_heads is the KV row:
//   RoPE, then its fp8_ds_mla entry (448 FP8 in UE8M0 blocks of 64 = 4 lanes,
//   64 BF16 RoPE values) stored at its sliding-window cache slot.
// dsv4_inv_rope_fp8: one warp per (token, head): inverse RoPE in FP32, then each
//   128-value block (8 lanes) quantized with a power-of-two scale
//   (max(amax, 1e-10) * fp32(1/448), ceiling from the bits) into the group's
//   DeepGEMM A operand; FP32 scales written MN-major (column stride T_pad).
//
// Every row depends only on itself: batch-invariant.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

#if defined(__USE_FAST_MATH__) || defined(__FAST_MATH__)
#error "dsv4_qo.cu must not be built with fast-math (byte-exact contract)"
#endif

namespace {

constexpr int kHeadDim = 512, kNope = 448, kRope = 64, kLane = 16;
constexpr int kWarpsPerBlock = 8;

__device__ __forceinline__ void load16(const __nv_bfloat16* src, float v[kLane]) {
  const uint4 a = __ldg(reinterpret_cast<const uint4*>(src));
  const uint4 b = __ldg(reinterpret_cast<const uint4*>(src) + 1);
  const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    v[2 * i] = __uint_as_float(w[i] << 16);
    v[2 * i + 1] = __uint_as_float(w[i] & 0xFFFF0000U);
  }
}

__device__ __forceinline__ uint16_t bf16_bits(float x) {
  return __bfloat16_as_ushort(__float2bfloat16_rn(x));
}

__device__ __forceinline__ void store16_bf16(__nv_bfloat16* dst, const float v[kLane]) {
  uint32_t w[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    w[i] = static_cast<uint32_t>(bf16_bits(v[2 * i])) |
           (static_cast<uint32_t>(bf16_bits(v[2 * i + 1])) << 16);
  }
  reinterpret_cast<uint4*>(dst)[0] = make_uint4(w[0], w[1], w[2], w[3]);
  reinterpret_cast<uint4*>(dst)[1] = make_uint4(w[4], w[5], w[6], w[7]);
}

// GPT-J RoPE on this lane's 8 pairs (lane covers dims [16 * lane, 16 * lane + 16)).
__device__ __forceinline__ void rope16(float v[kLane], const float* cs, int lane, bool inverse) {
  const int k0 = (lane * kLane - kNope) / 2;
#pragma unroll
  for (int p = 0; p < kLane / 2; ++p) {
    const float c = __ldg(cs + k0 + p), s = __ldg(cs + kRope / 2 + k0 + p);
    const float e = v[2 * p], o = v[2 * p + 1];
    if (inverse) {
      v[2 * p] = __fadd_rn(__fmul_rn(e, c), __fmul_rn(o, s));
      v[2 * p + 1] = __fsub_rn(__fmul_rn(o, c), __fmul_rn(e, s));
    } else {
      v[2 * p] = __fsub_rn(__fmul_rn(e, c), __fmul_rn(o, s));
      v[2 * p + 1] = __fadd_rn(__fmul_rn(e, s), __fmul_rn(o, c));
    }
  }
}

// Biased exponent of 2^ceil(log2(max(amax, floor) / 448)), from the bits.
__device__ __forceinline__ uint32_t pow2_exponent(float amax, float floor) {
  const float s = __fmul_rn(fmaxf(amax, floor), 1.0f / 448.0f);
  return ((__float_as_uint(s) + 0x7FFFFFU) >> 23) & 0xFFU;
}

__device__ __forceinline__ float lane_amax(const float v[kLane]) {
  uint32_t m = 0;  // |x| bit patterns of FP32 values, NaN mapped to 0
#pragma unroll
  for (int i = 0; i < kLane; ++i) {
    const uint32_t b = __float_as_uint(v[i]) & 0x7FFFFFFFU;
    m = max(m, b > 0x7F800000U ? 0U : b);
  }
  return __uint_as_float(m);
}

__device__ __forceinline__ void fp8x16(const float v[kLane], float inverse, uint8_t* dst) {
  uint32_t w[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    w[i] = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      w[i] |= static_cast<uint32_t>(__nv_cvt_float_to_fp8(__fmul_rn(v[4 * i + j], inverse),
                                                          __NV_SATFINITE, __NV_E4M3))
              << (8 * j);
    }
  }
  *reinterpret_cast<uint4*>(dst) = make_uint4(w[0], w[1], w[2], w[3]);
}

__global__ void __launch_bounds__(kWarpsPerBlock * 32) q_norm_rope_kv_insert_kernel(
    const __nv_bfloat16* __restrict__ q, int64_t q_stride0, int64_t q_stride1,
    const __nv_bfloat16* __restrict__ kv, int64_t kv_stride0, const int64_t* __restrict__ positions,
    const float* __restrict__ cos_sin, int64_t cs_stride, __nv_bfloat16* __restrict__ q_out,
    uint8_t* __restrict__ cache, int64_t cache_block_stride, int cache_block_size,
    const int64_t* __restrict__ slots, int64_t tokens, int num_heads, int padded_heads, float eps) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + threadIdx.x / 32;
  const int rows_per_token = padded_heads + 1;
  if (row >= tokens * rows_per_token) return;  // whole warp exits together
  const int lane = threadIdx.x & 31;
  const int64_t token = row / rows_per_token;
  const int head = static_cast<int>(row % rows_per_token);
  const bool is_kv = head == padded_heads;
  const int d0 = lane * kLane;
  const float* cs = cos_sin + positions[token] * cs_stride;

  if (!is_kv) {
    __nv_bfloat16* dst = q_out + (token * padded_heads + head) * kHeadDim + d0;
    if (head >= num_heads) {
      reinterpret_cast<uint4*>(dst)[0] = make_uint4(0, 0, 0, 0);
      reinterpret_cast<uint4*>(dst)[1] = make_uint4(0, 0, 0, 0);
      return;
    }
    float v[kLane];
    load16(q + token * q_stride0 + head * q_stride1 + d0, v);
    float sq = __fmul_rn(v[0], v[0]);
#pragma unroll
    for (int i = 1; i < kLane; ++i) sq = __fadd_rn(sq, __fmul_rn(v[i], v[i]));
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      sq = __fadd_rn(sq, __shfl_xor_sync(0xFFFFFFFFU, sq, offset));
    }
    const float r = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(sq, 512.0f), eps)));
#pragma unroll
    for (int i = 0; i < kLane; ++i) v[i] = __fmul_rn(v[i], r);
    if (d0 >= kNope) rope16(v, cs, lane, false);
    store16_bf16(dst, v);
    return;
  }

  const int64_t slot = slots[token];
  if (slot < 0) return;
  float v[kLane];
  load16(kv + token * kv_stride0 + d0, v);
  uint8_t* block = cache + (slot / cache_block_size) * cache_block_stride;
  const int64_t in_block = slot % cache_block_size;
  uint8_t* entry = block + in_block * 576;
  uint8_t* scales = block + static_cast<int64_t>(cache_block_size) * 576 + in_block * 8;
  if (d0 >= kNope) {
    rope16(v, cs, lane, false);
    uint16_t b[kLane];
#pragma unroll
    for (int i = 0; i < kLane; ++i) b[i] = bf16_bits(v[i]);
    uint32_t w[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) w[i] = b[2 * i] | (static_cast<uint32_t>(b[2 * i + 1]) << 16);
    uint4* dst = reinterpret_cast<uint4*>(entry + kNope + 2 * (d0 - kNope));
    dst[0] = make_uint4(w[0], w[1], w[2], w[3]);
    dst[1] = make_uint4(w[4], w[5], w[6], w[7]);
    if (d0 == kNope) scales[7] = 0;  // pad byte
  }
  // 64-value quant blocks are groups of 4 lanes (all lanes join the shuffles).
  float amax = lane_amax(v);
  amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFU, amax, 1));
  amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFU, amax, 2));
  if (d0 < kNope) {
    const uint32_t e = pow2_exponent(amax, 1e-4f);
    fp8x16(v, __uint_as_float((254U - e) << 23), entry + d0);
    if ((lane & 3) == 0) scales[d0 / 64] = static_cast<uint8_t>(e);
  }
}

__global__ void __launch_bounds__(kWarpsPerBlock * 32) inv_rope_fp8_kernel(
    const __nv_bfloat16* __restrict__ o, int64_t o_stride0, int64_t o_stride1,
    const int64_t* __restrict__ positions, const float* __restrict__ cos_sin, int64_t cs_stride,
    uint8_t* __restrict__ a, float* __restrict__ scales, int64_t tokens, int heads,
    int heads_per_group, int64_t scale_stride) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + threadIdx.x / 32;
  if (row >= tokens * heads) return;
  const int lane = threadIdx.x & 31;
  const int64_t token = row / heads;
  const int head = static_cast<int>(row % heads);
  const int group = head / heads_per_group, in_group = head % heads_per_group;
  const int d0 = lane * kLane;
  float v[kLane];
  load16(o + token * o_stride0 + head * o_stride1 + d0, v);
  if (d0 >= kNope) rope16(v, cos_sin + positions[token] * cs_stride, lane, true);
  // 128-value blocks: groups of 8 lanes.
  float amax = lane_amax(v);
#pragma unroll
  for (int offset = 1; offset < 8; offset <<= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFU, amax, offset));
  }
  const uint32_t e = pow2_exponent(amax, 1e-10f);
  const int64_t k = static_cast<int64_t>(heads_per_group) * kHeadDim;
  const int64_t col = static_cast<int64_t>(in_group) * kHeadDim + d0;
  fp8x16(v, __uint_as_float((254U - e) << 23),
         a + (static_cast<int64_t>(group) * tokens + token) * k + col);
  if ((lane & 7) == 0) {
    scales[(static_cast<int64_t>(group) * (k / 128) + col / 128) * scale_stride + token] =
        __uint_as_float(e << 23);
  }
}

}  // namespace

torch::Tensor dsv4_q_norm_rope_kv_insert(torch::Tensor q, torch::Tensor kv,
                                         torch::Tensor positions, torch::Tensor cos_sin_cache,
                                         torch::Tensor cache, int64_t cache_block_size,
                                         torch::Tensor slots, int64_t padded_heads, double eps) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kBFloat16 && q.dim() == 3 &&
                  q.size(2) == kHeadDim && q.stride(2) == 1 && q.stride(1) % 8 == 0 &&
                  q.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0,
              "dsv4_q_norm_rope: q must be BF16 [T, H, 512] with 16-byte aligned rows");
  const int64_t tokens = q.size(0);
  TORCH_CHECK(padded_heads >= q.size(1), "dsv4_q_norm_rope: padded_heads < heads");
  TORCH_CHECK(kv.is_cuda() && kv.scalar_type() == torch::kBFloat16 && kv.dim() == 2 &&
                  kv.size(0) == tokens && kv.size(1) == kHeadDim && kv.stride(1) == 1 &&
                  kv.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(kv.data_ptr()) % 16 == 0,
              "dsv4_q_norm_rope: kv must be BF16 [T, 512] with 16-byte aligned rows");
  TORCH_CHECK(positions.is_cuda() && positions.scalar_type() == torch::kInt64 &&
                  positions.dim() == 1 && positions.size(0) == tokens && positions.is_contiguous(),
              "dsv4_q_norm_rope: positions must be contiguous int64 [T]");
  TORCH_CHECK(cos_sin_cache.is_cuda() && cos_sin_cache.scalar_type() == torch::kFloat32 &&
                  cos_sin_cache.dim() == 2 && cos_sin_cache.size(1) == kRope &&
                  cos_sin_cache.stride(1) == 1,
              "dsv4_q_norm_rope: cos_sin_cache must be FP32 [positions, 64]");
  TORCH_CHECK(cache.is_cuda() && cache.scalar_type() == torch::kUInt8 && cache.dim() == 2 &&
                  cache.stride(1) == 1 && cache.size(1) >= cache_block_size * (576 + 8),
              "dsv4_q_norm_rope: cache must be uint8 [blocks, >= block_size * 584]");
  TORCH_CHECK(slots.is_cuda() && slots.scalar_type() == torch::kInt64 && slots.dim() == 1 &&
                  slots.size(0) == tokens && slots.is_contiguous(),
              "dsv4_q_norm_rope: slots must be contiguous int64 [T]");
  auto q_out = torch::empty({tokens, padded_heads, kHeadDim}, q.options());
  const int64_t rows = tokens * (padded_heads + 1);
  if (rows == 0 || tokens == 0) return q_out;
  const c10::cuda::CUDAGuard guard(q.device());
  q_norm_rope_kv_insert_kernel<<<static_cast<unsigned>((rows + kWarpsPerBlock - 1) / kWarpsPerBlock),
                                 kWarpsPerBlock * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()), q.stride(0), q.stride(1),
      reinterpret_cast<const __nv_bfloat16*>(kv.data_ptr()), kv.stride(0),
      positions.data_ptr<int64_t>(), cos_sin_cache.data_ptr<float>(), cos_sin_cache.stride(0),
      reinterpret_cast<__nv_bfloat16*>(q_out.data_ptr()), cache.data_ptr<uint8_t>(),
      cache.stride(0), static_cast<int>(cache_block_size), slots.data_ptr<int64_t>(), tokens,
      static_cast<int>(q.size(1)), static_cast<int>(padded_heads), static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return q_out;
}

std::vector<torch::Tensor> dsv4_inv_rope_fp8(torch::Tensor o, torch::Tensor positions,
                                             torch::Tensor cos_sin_cache, int64_t groups) {
  TORCH_CHECK(o.is_cuda() && o.scalar_type() == torch::kBFloat16 && o.dim() == 3 &&
                  o.size(2) == kHeadDim && o.stride(2) == 1 && o.stride(1) % 8 == 0 &&
                  o.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(o.data_ptr()) % 16 == 0,
              "dsv4_inv_rope_fp8: o must be BF16 [T, H, 512] with 16-byte aligned rows");
  const int64_t tokens = o.size(0), heads = o.size(1);
  TORCH_CHECK(groups > 0 && heads % groups == 0, "dsv4_inv_rope_fp8: heads % groups != 0");
  TORCH_CHECK(positions.is_cuda() && positions.scalar_type() == torch::kInt64 &&
                  positions.dim() == 1 && positions.size(0) == tokens && positions.is_contiguous(),
              "dsv4_inv_rope_fp8: positions must be contiguous int64 [T]");
  TORCH_CHECK(cos_sin_cache.is_cuda() && cos_sin_cache.scalar_type() == torch::kFloat32 &&
                  cos_sin_cache.dim() == 2 && cos_sin_cache.size(1) == kRope &&
                  cos_sin_cache.stride(1) == 1,
              "dsv4_inv_rope_fp8: cos_sin_cache must be FP32 [positions, 64]");
  const int64_t k = heads / groups * kHeadDim;
  const int64_t t_pad = (tokens + 3) / 4 * 4;  // DeepGEMM SM90 scale column stride
  auto a = torch::empty({groups, tokens, k}, o.options().dtype(torch::kFloat8_e4m3fn));
  auto scale_storage = torch::empty({groups, k / 128, t_pad}, o.options().dtype(torch::kFloat32));
  auto scales = scale_storage.transpose(1, 2).narrow(1, 0, tokens);  // [G, T, K/128], MN-major
  if (tokens == 0) return {a, scales};
  const c10::cuda::CUDAGuard guard(o.device());
  const int64_t rows = tokens * heads;
  inv_rope_fp8_kernel<<<static_cast<unsigned>((rows + kWarpsPerBlock - 1) / kWarpsPerBlock),
                        kWarpsPerBlock * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(o.data_ptr()), o.stride(0), o.stride(1),
      positions.data_ptr<int64_t>(), cos_sin_cache.data_ptr<float>(), cos_sin_cache.stride(0),
      reinterpret_cast<uint8_t*>(a.data_ptr()), scale_storage.data_ptr<float>(), tokens,
      static_cast<int>(heads), static_cast<int>(heads / groups), t_pad);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {a, scales};
}
