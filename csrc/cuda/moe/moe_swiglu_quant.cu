// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// Standalone clamp-SwiGLU * p_s -> MXFP8 for the P5 routed experts, byte-equal to the fc1
// epilogue of sm90_fused_moe_mlp.cu (FC1_HQ). The LoRA path needs it: fc1 runs as fc1_z
// (FP32 z), the LoRA delta is added to z, then this kernel quantizes h for fc3. With a
// zero delta its bytes equal the fused fast path (tested). It also writes h as BF16 (the
// LoRA fc2 input).
//
//   h     = BF16(SiLU(min(g, 10)) * clamp(u, -10, 10) * p)       per column
//   code  = E8M0 of the 32-column block amax: floor(log2(amax)) - 8, clamped, 127 for 0
//   q     = e4m3 satfinite(h * 2^(127 - code))

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <vector>

#if defined(__USE_FAST_MATH__)
#error "moe_swiglu_quant.cu must not be built with --use_fast_math (byte-exact contract)"
#endif

namespace {

constexpr float kGateClampMax = 10.0f;
constexpr float kUpClampMin = -10.0f;
constexpr float kUpClampMax = 10.0f;

// The fused epilogue's helpers, verbatim.
__device__ __forceinline__ float swiglu_bf16(float g, float u, float p) {
  g = fminf(g, kGateClampMax);
  u = fminf(fmaxf(u, kUpClampMin), kUpClampMax);
  const float sig = 1.0f / (1.0f + expf(-g));
  const float h = (g * sig) * u * p;
  return __bfloat162float(__float2bfloat16(h));
}

__device__ __forceinline__ int e8m0_code_from_amax(float amax) {
  if (amax == 0.0f) return 127;
  int ex;
  (void)frexpf(fmaxf(amax, 1.17549435e-38f), &ex);
  int e = (ex - 1) - 8;
  e = max(-127, min(127, e));
  return e + 127;
}

__device__ __forceinline__ float e8m0_to_f32(uint32_t code) {
  return __uint_as_float(code ? (code << 23) : 0x00400000u);
}

// One warp per (row, 32-column block): lane c owns column 32 b + c, so the z reads are
// coalesced; the block amax is a warp max (order-free). z_add (optional, FP32
// [tokens, 2F]) is added to z at row token[row] in FP32 before the SwiGLU (the LoRA fc1
// delta, shared by a token's rows).
__global__ void __launch_bounds__(256)
    swiglu_quant_kernel(const float* __restrict__ z, const float* __restrict__ z_add,
                        const int64_t* __restrict__ token, const float* __restrict__ p_s,
                        int64_t rows, int f, uint8_t* __restrict__ codes,
                        uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ h_out) {
  const int blocks = f / 32;
  const int lane = threadIdx.x & 31;
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * (blockDim.x / 32) + (threadIdx.x >> 5);
  if (idx >= rows * blocks) return;  // warp-uniform
  const int64_t row = idx / blocks;
  const int col = static_cast<int>(idx - row * blocks) * 32 + lane;
  const float* zr = z + row * 2 * f;
  float g = zr[col], u = zr[f + col];
  if (z_add != nullptr) {
    const float* ar = z_add + token[row] * 2 * f;
    g = __fadd_rn(g, ar[col]);
    u = __fadd_rn(u, ar[f + col]);
  }
  const float h = swiglu_bf16(g, u, p_s[row]);
  float amax = fabsf(h);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  const int code = e8m0_code_from_amax(amax);
  const float inv = e8m0_to_f32(254 - code);
  codes[row * f + col] =
      static_cast<uint8_t>(__nv_cvt_float_to_fp8(h * inv, __NV_SATFINITE, __NV_E4M3));
  h_out[row * f + col] = __float2bfloat16_rn(h);
  if (lane == 0) scales[row * blocks + col / 32] = static_cast<uint8_t>(code);
}

}  // namespace

// z FP32 [M, 2F] (gate | up), p_s FP32 [M] -> (codes uint8 [M, F], scales uint8 [M, F/32],
// h BF16 [M, F]). Optional z_add FP32 [T, 2F] with token int64 [M]: z[r] += z_add[token[r]].
std::vector<torch::Tensor> moe_swiglu_quant(torch::Tensor z, torch::Tensor p_s,
                                            c10::optional<torch::Tensor> z_add,
                                            c10::optional<torch::Tensor> token) {
  TORCH_CHECK(z.is_cuda() && z.scalar_type() == torch::kFloat32 && z.dim() == 2 &&
                  z.is_contiguous() && z.size(1) % 64 == 0,
              "moe_swiglu_quant: z must be contiguous CUDA FP32 [M, 2F], F % 32 == 0");
  const int64_t m = z.size(0);
  const int f = static_cast<int>(z.size(1) / 2);
  TORCH_CHECK(p_s.is_cuda() && p_s.scalar_type() == torch::kFloat32 && p_s.is_contiguous() &&
                  p_s.numel() == m,
              "moe_swiglu_quant: p_s must be contiguous CUDA FP32 [M]");
  const float* add_ptr = nullptr;
  const int64_t* tok_ptr = nullptr;
  if (z_add.has_value()) {
    TORCH_CHECK(token.has_value(), "moe_swiglu_quant: z_add needs token");
    const auto& a = *z_add;
    const auto& t = *token;
    TORCH_CHECK(a.is_cuda() && a.scalar_type() == torch::kFloat32 && a.is_contiguous() &&
                    a.dim() == 2 && a.size(1) == 2 * f,
                "moe_swiglu_quant: z_add must be contiguous CUDA FP32 [T, 2F]");
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kInt64 && t.is_contiguous() &&
                    t.numel() == m,
                "moe_swiglu_quant: token must be contiguous CUDA int64 [M]");
    add_ptr = a.data_ptr<float>();
    tok_ptr = t.data_ptr<int64_t>();
  }
  const c10::cuda::CUDAGuard guard(z.device());
  auto u8 = z.options().dtype(torch::kUInt8);
  auto codes = torch::empty({m, f}, u8), scales = torch::empty({m, f / 32}, u8);
  auto h = torch::empty({m, f}, z.options().dtype(torch::kBFloat16));
  const int64_t work = m * (f / 32);
  if (work == 0) return {codes, scales, h};
  swiglu_quant_kernel<<<static_cast<unsigned>((work + 7) / 8), 256, 0,
                        at::cuda::getCurrentCUDAStream()>>>(
      z.data_ptr<float>(), add_ptr, tok_ptr, p_s.data_ptr<float>(), m, f,
      codes.data_ptr<uint8_t>(),
      scales.data_ptr<uint8_t>(), reinterpret_cast<__nv_bfloat16*>(h.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {codes, scales, h};
}
