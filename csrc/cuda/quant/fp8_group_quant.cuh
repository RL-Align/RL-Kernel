// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// Device helpers for per-token-group FP8 (E4M3) quantization, shared by
// fp8_group_quant.cu and the fused clamped-SwiGLU + quant kernel. See
// fp8_group_quant.cu for the semantics; every step that could see a subnormal
// is integer bit math or explicit non-.ftz PTX.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>

#if defined(__USE_FAST_MATH__)
#error "fp8_group_quant requires precise FP32 math; disable --use_fast_math"
#endif

namespace rlk_fp8q {

constexpr int kGroup = 128;
constexpr int kLanes = 8;              // threads per group
constexpr int kVec = kGroup / kLanes;  // 16 values per thread
constexpr uint32_t kAbsMask = 0x7fffffffu;
constexpr uint32_t kInfBits = 0x7f800000u;
constexpr uint32_t kEpsBits = 0x2edbe6ffu;         // 1e-10f
constexpr uint32_t kScaleFloorBits = 0x2edbe6ffu;  // 1e-10f
constexpr float kFp8Max = 448.0f;

template <typename T>
__device__ __forceinline__ float to_float(T v);
template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}
template <>
__device__ __forceinline__ float to_float<__half>(__half v) {
  return __half2float(v);
}

__device__ __forceinline__ float div_rn_no_ftz(float a, float b) {
  float r;
  asm("div.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

// |x| as bits; NaN maps to 0 so it never wins the max (fmaxf semantics).
__device__ __forceinline__ uint32_t abs_bits_ignoring_nan(float v) {
  const uint32_t a = __float_as_uint(v) & kAbsMask;
  return a > kInfBits ? 0u : a;
}

__device__ __forceinline__ float ue8m0_scale(uint32_t absmax_bits) {
  // absmax >= eps > 0 and finite or +inf; IEEE division by 448.
  float y = div_rn_no_ftz(__uint_as_float(absmax_bits), kFp8Max);
  uint32_t bits = __float_as_uint(y);
  bits = bits < kScaleFloorBits ? kScaleFloorBits : bits;
  const uint32_t exponent = (bits >> 23) & 0xffu;
  const uint32_t mantissa = bits & 0x7fffffu;
  return __uint_as_float((exponent + (mantissa != 0u ? 1u : 0u)) << 23);
}

__device__ __forceinline__ float plain_scale(uint32_t absmax_bits) {
  float y = div_rn_no_ftz(__uint_as_float(absmax_bits), kFp8Max);
  const uint32_t bits = __float_as_uint(y);
  return bits < kScaleFloorBits ? __uint_as_float(kScaleFloorBits) : y;
}

__device__ __forceinline__ uint8_t quantize(float x, float scale) {
  float v = div_rn_no_ftz(x, scale);
  // fmax(v, -448) then fmin(., 448): a NaN v becomes -448.
  v = (v != v || v < -kFp8Max) ? -kFp8Max : v;
  v = v > kFp8Max ? kFp8Max : v;
  return static_cast<uint8_t>(__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3));
}

// One group of 128 values held by 8 lanes (16 each, lanes aligned to an octet
// of the warp). Returns the group scale and writes the FP8 bytes to `out`.
__device__ __forceinline__ float quantize_group(const float (&vals)[kVec], uint8_t (&out)[kVec],
                                                bool scale_ue8m0) {
  uint32_t m = kEpsBits;
#pragma unroll
  for (int i = 0; i < kVec; ++i) m = max(m, abs_bits_ignoring_nan(vals[i]));
  const unsigned mask = 0xffu << (threadIdx.x & 24u);
  m = max(m, __shfl_xor_sync(mask, m, 4));
  m = max(m, __shfl_xor_sync(mask, m, 2));
  m = max(m, __shfl_xor_sync(mask, m, 1));
  const float scale = scale_ue8m0 ? ue8m0_scale(m) : plain_scale(m);
#pragma unroll
  for (int i = 0; i < kVec; ++i) out[i] = quantize(vals[i], scale);
  return scale;
}

}  // namespace rlk_fp8q
