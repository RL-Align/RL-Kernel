// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// Per-token-group FP8 (E4M3) activation quantization for DeepGEMM (DSv4, SM90).
//
// Bit-exact with rl_engine.kernels.ops.pytorch.quant.fp8_group_quant, which
// reproduces vLLM's per_token_group_fp8_quant (group 128, eps 1e-10, UE8M0):
//
//   absmax = fmax(eps, max |x|)        NaN does not count
//   s      = absmax / 448              IEEE division
//   s      = fmax(s, 1e-10)
//   s      = 2^ceil(log2(s))           UE8M0, from the FP32 bits
//   q      = e4m3(fmin(fmax(x / s, -448), 448))    NaN -> -448
//
// The absmax is an integer max over |x| bit patterns (NaN mapped to zero), the
// division is explicit non-.ftz PTX and the clamps are written out, so neither
// --use_fast_math nor libm rounding can move an output byte. Each group of 128
// values is quantized independently by a fixed set of 8 threads, so the bytes do
// not depend on the grid or on how many rows are in flight (batch-invariant).
//
// Scales are FP32 in DeepGEMM's SM90 layout: logical [rows, groups],
// column-major, column stride = rows rounded up to a multiple of 4.
//
// fp8_group_quant_tokens_forward quantizes x [T, C] along tokens for the wgrad
// operand x.T: output q [C, T_pad], scales [C, T_pad / 128], with T_pad a
// multiple of 128 and the padded tokens zero. A block transposes a 128-token x
// 32-channel tile through shared memory so global reads stay coalesced.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>

#if defined(__USE_FAST_MATH__)
#error "fp8_group_quant requires precise FP32 math; disable --use_fast_math"
#endif

namespace {

constexpr int kGroup = 128;
constexpr int kLanes = 8;              // threads per group
constexpr int kVec = kGroup / kLanes;  // 16 values per thread
constexpr int kGroupsPerBlock = 32;
constexpr int kThreads = kGroupsPerBlock * kLanes;  // 256
constexpr int kTileChannels = 32;                   // token-direction tile width
constexpr uint32_t kAbsMask = 0x7fffffffu;
constexpr uint32_t kInfBits = 0x7f800000u;
constexpr uint32_t kEpsBits = 0x2edbe6ffu;          // 1e-10f
constexpr uint32_t kScaleFloorBits = 0x2edbe6ffu;   // 1e-10f
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

// Shared by both directions: 8 lanes, 16 values each. Returns the group scale.
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

template <typename T>
__global__ void __launch_bounds__(kThreads) fp8_group_quant_rows_kernel(
    const T* __restrict__ x, uint8_t* __restrict__ q, float* __restrict__ scales,
    int64_t rows, int64_t groups, int64_t scale_stride, bool scale_ue8m0) {
  const int64_t group = static_cast<int64_t>(blockIdx.x) * kGroupsPerBlock + threadIdx.x / kLanes;
  const int lane = threadIdx.x % kLanes;
  // Whole octets share a group, so this exit keeps every shuffle mask complete.
  if (group >= rows * groups) return;
  const int64_t row = group / groups;
  const int64_t col = group % groups;
  const int64_t offset = group * kGroup + lane * kVec;

  alignas(16) T in[kVec];
  const uint4* src = reinterpret_cast<const uint4*>(x + offset);
  uint4* dst = reinterpret_cast<uint4*>(in);
  dst[0] = src[0];
  dst[1] = src[1];
  float vals[kVec];
#pragma unroll
  for (int i = 0; i < kVec; ++i) vals[i] = to_float(in[i]);

  alignas(16) uint8_t out[kVec];
  const float scale = quantize_group(vals, out, scale_ue8m0);
  *reinterpret_cast<uint4*>(q + offset) = *reinterpret_cast<const uint4*>(out);
  if (lane == 0) scales[col * scale_stride + row] = scale;
}

template <typename T>
__global__ void __launch_bounds__(kThreads) fp8_group_quant_tokens_kernel(
    const T* __restrict__ x, uint8_t* __restrict__ q, float* __restrict__ scales,
    int64_t tokens, int64_t channels, int64_t padded_tokens, int64_t scale_stride,
    bool scale_ue8m0) {
  __shared__ float tile[kTileChannels][kGroup + 1];
  const int64_t c0 = static_cast<int64_t>(blockIdx.x) * kTileChannels;
  const int64_t t0 = static_cast<int64_t>(blockIdx.y) * kGroup;

  for (int i = threadIdx.x; i < kGroup * kTileChannels; i += kThreads) {
    const int t = i / kTileChannels;
    const int c = i % kTileChannels;
    float v = 0.0f;
    if (t0 + t < tokens && c0 + c < channels) v = to_float(x[(t0 + t) * channels + c0 + c]);
    tile[c][t] = v;
  }
  __syncthreads();

  const int c = threadIdx.x / kLanes;
  const int lane = threadIdx.x % kLanes;
  float vals[kVec];
#pragma unroll
  for (int i = 0; i < kVec; ++i) vals[i] = tile[c][lane * kVec + i];
  alignas(16) uint8_t out[kVec];
  const float scale = quantize_group(vals, out, scale_ue8m0);
  if (c0 + c >= channels) return;
  *reinterpret_cast<uint4*>(q + (c0 + c) * padded_tokens + t0 + lane * kVec) =
      *reinterpret_cast<const uint4*>(out);
  if (lane == 0) scales[(t0 / kGroup) * scale_stride + c0 + c] = scale;
}

int64_t tma_aligned_rows(int64_t rows) { return (rows + 3) / 4 * 4; }

void check_input(const torch::Tensor& x) {
  TORCH_CHECK(x.is_cuda(), "fp8_group_quant: x must be a CUDA tensor");
  TORCH_CHECK(x.dim() == 2, "fp8_group_quant: x must be 2-D");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16 || x.scalar_type() == torch::kHalf,
              "fp8_group_quant: x must be BF16 or FP16");
  TORCH_CHECK(x.is_contiguous(), "fp8_group_quant: x must be contiguous");
}

}  // namespace

std::vector<torch::Tensor> fp8_group_quant_forward(torch::Tensor x, bool scale_ue8m0) {
  check_input(x);
  const int64_t rows = x.size(0);
  const int64_t cols = x.size(1);
  TORCH_CHECK(cols > 0 && cols % kGroup == 0,
              "fp8_group_quant: the last dimension must be a positive multiple of 128");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
              "fp8_group_quant: x must be 16-byte aligned");
  const int64_t groups = cols / kGroup;
  const c10::cuda::CUDAGuard guard(x.device());
  auto q = torch::empty({rows, cols}, x.options().dtype(torch::kFloat8_e4m3fn));
  const int64_t stride = tma_aligned_rows(rows);
  auto scales = torch::empty_strided({rows, groups}, {1, stride}, x.options().dtype(torch::kFloat));
  const int64_t total = rows * groups;
  if (total == 0) return {q, scales};
  const int64_t blocks = (total + kGroupsPerBlock - 1) / kGroupsPerBlock;
  TORCH_CHECK(blocks <= INT32_MAX, "fp8_group_quant: input too large");
  auto stream = at::cuda::getCurrentCUDAStream();
  auto* q_ptr = reinterpret_cast<uint8_t*>(q.data_ptr());
  if (x.scalar_type() == torch::kBFloat16) {
    fp8_group_quant_rows_kernel<__nv_bfloat16><<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), q_ptr, scales.data_ptr<float>(),
        rows, groups, stride, scale_ue8m0);
  } else {
    fp8_group_quant_rows_kernel<__half><<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
        reinterpret_cast<const __half*>(x.data_ptr()), q_ptr, scales.data_ptr<float>(),
        rows, groups, stride, scale_ue8m0);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {q, scales};
}

std::vector<torch::Tensor> fp8_group_quant_tokens_forward(torch::Tensor x, bool scale_ue8m0) {
  check_input(x);
  const int64_t tokens = x.size(0);
  const int64_t channels = x.size(1);
  TORCH_CHECK(tokens > 0 && channels > 0,
              "fp8_group_quant_tokens: x must have at least one token and one channel");
  const int64_t padded = (tokens + kGroup - 1) / kGroup * kGroup;
  const int64_t groups = padded / kGroup;
  const c10::cuda::CUDAGuard guard(x.device());
  auto q = torch::empty({channels, padded}, x.options().dtype(torch::kFloat8_e4m3fn));
  const int64_t stride = tma_aligned_rows(channels);
  auto scales =
      torch::empty_strided({channels, groups}, {1, stride}, x.options().dtype(torch::kFloat));
  const int64_t channel_tiles = (channels + kTileChannels - 1) / kTileChannels;
  TORCH_CHECK(channel_tiles <= INT32_MAX && groups <= 65535,
              "fp8_group_quant_tokens: input too large");
  const dim3 grid(static_cast<unsigned>(channel_tiles), static_cast<unsigned>(groups));
  auto stream = at::cuda::getCurrentCUDAStream();
  auto* q_ptr = reinterpret_cast<uint8_t*>(q.data_ptr());
  if (x.scalar_type() == torch::kBFloat16) {
    fp8_group_quant_tokens_kernel<__nv_bfloat16><<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), q_ptr, scales.data_ptr<float>(),
        tokens, channels, padded, stride, scale_ue8m0);
  } else {
    fp8_group_quant_tokens_kernel<__half><<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const __half*>(x.data_ptr()), q_ptr, scales.data_ptr<float>(),
        tokens, channels, padded, stride, scale_ue8m0);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {q, scales};
}
