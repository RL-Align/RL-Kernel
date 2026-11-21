// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// Clamped SwiGLU of the DSv4 shared expert (forward fused with FP8 group
// quantization, and backward). Bit-exact with
// rl_engine.kernels.ops.pytorch.activation.silu_and_mul_clamp, whose forward
// reproduces vLLM's silu_and_mul_with_clamp on z = [gate | up] (BF16):
//
//   g   = fminf(gate, limit)              NaN gate -> limit
//   u   = fmaxf(fminf(up, limit), -limit)
//   act = bf16(g / (1 + expf(-g)))
//   h   = bf16(act * (u + 0.0f))
//
// and then quantizes h exactly like fp8_group_quant (vLLM's
// per_token_group_fp8_quant), so the fused kernel emits the same bytes as the
// two separate rollout kernels. Backward (ours; vLLM has none):
//
//   d = expf(-g) + 1 ; sig = 1 / d ; dsilu = sig * ((g * (1 - sig)) + 1)
//   dgate = gate < limit          ? (dh * u) * dsilu : 0
//   dup   = -limit < up < limit   ? dh * bf16(g / d) : 0
//
// Every FP32 step is an explicit _rn intrinsic or non-.ftz division, so FMA
// contraction cannot change a byte; expf is libdevice expf, as in vLLM and
// torch.exp. Each output depends only on its own row, so both kernels are
// batch-invariant.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include "../quant/fp8_group_quant.cuh"

#include <cstdint>

namespace {

using namespace rlk_fp8q;

constexpr int kGroupsPerBlock = 32;
constexpr int kThreads = kGroupsPerBlock * kLanes;  // 256
constexpr int kBwdVec = 8;                          // 8 BF16 = 16 bytes

struct Clamped {
  float g;
  float u;
};

__device__ __forceinline__ Clamped clamp_inputs(float gate, float up, float limit) {
  return {fminf(gate, limit), fmaxf(fminf(up, limit), -limit)};
}

__device__ __forceinline__ float round_bf16(float v) {
  return __bfloat162float(__float2bfloat16_rn(v));
}

__device__ __forceinline__ float swiglu_value(float gate, float up, float limit) {
  const Clamped c = clamp_inputs(gate, up, limit);
  const float denominator = __fadd_rn(1.0f, expf(-c.g));
  const float act = round_bf16(div_rn_no_ftz(c.g, denominator));
  return round_bf16(__fmul_rn(act, __fadd_rn(c.u, 0.0f)));
}

__global__ void __launch_bounds__(kThreads) silu_and_mul_clamp_quant_kernel(
    const __nv_bfloat16* __restrict__ z, uint8_t* __restrict__ q, float* __restrict__ scales,
    __nv_bfloat16* __restrict__ h, int64_t rows, int64_t ffn, int64_t scale_stride, float limit,
    bool scale_ue8m0) {
  const int64_t groups = ffn / kGroup;
  const int64_t group = static_cast<int64_t>(blockIdx.x) * kGroupsPerBlock + threadIdx.x / kLanes;
  const int lane = threadIdx.x % kLanes;
  // Whole octets share a group, so this exit keeps every shuffle mask complete.
  if (group >= rows * groups) return;
  const int64_t row = group / groups;
  const int64_t col = group % groups;
  const int64_t out_offset = row * ffn + col * kGroup + lane * kVec;
  const __nv_bfloat16* gate_ptr = z + row * 2 * ffn + col * kGroup + lane * kVec;
  const __nv_bfloat16* up_ptr = gate_ptr + ffn;

  alignas(16) __nv_bfloat16 gate[kVec];
  alignas(16) __nv_bfloat16 up[kVec];
  reinterpret_cast<uint4*>(gate)[0] = reinterpret_cast<const uint4*>(gate_ptr)[0];
  reinterpret_cast<uint4*>(gate)[1] = reinterpret_cast<const uint4*>(gate_ptr)[1];
  reinterpret_cast<uint4*>(up)[0] = reinterpret_cast<const uint4*>(up_ptr)[0];
  reinterpret_cast<uint4*>(up)[1] = reinterpret_cast<const uint4*>(up_ptr)[1];

  float vals[kVec];
#pragma unroll
  for (int i = 0; i < kVec; ++i)
    vals[i] = swiglu_value(__bfloat162float(gate[i]), __bfloat162float(up[i]), limit);

  if (h != nullptr) {
    alignas(16) __nv_bfloat16 hv[kVec];
#pragma unroll
    for (int i = 0; i < kVec; ++i) hv[i] = __float2bfloat16_rn(vals[i]);  // exact: already BF16
    reinterpret_cast<uint4*>(h + out_offset)[0] = reinterpret_cast<const uint4*>(hv)[0];
    reinterpret_cast<uint4*>(h + out_offset)[1] = reinterpret_cast<const uint4*>(hv)[1];
  }

  alignas(16) uint8_t out[kVec];
  const float scale = quantize_group(vals, out, scale_ue8m0);
  *reinterpret_cast<uint4*>(q + out_offset) = *reinterpret_cast<const uint4*>(out);
  if (lane == 0) scales[col * scale_stride + row] = scale;
}

__global__ void silu_and_mul_clamp_backward_kernel(const __nv_bfloat16* __restrict__ dh,
                                                   const __nv_bfloat16* __restrict__ z,
                                                   __nv_bfloat16* __restrict__ dz, int64_t rows,
                                                   int64_t ffn, float limit) {
  const int64_t chunks_per_row = ffn / kBwdVec;
  const int64_t total = rows * chunks_per_row;
  for (int64_t chunk = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; chunk < total;
       chunk += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t row = chunk / chunks_per_row;
    const int64_t col = (chunk % chunks_per_row) * kBwdVec;
    alignas(16) __nv_bfloat16 g_in[kBwdVec];
    alignas(16) __nv_bfloat16 u_in[kBwdVec];
    alignas(16) __nv_bfloat16 d_in[kBwdVec];
    *reinterpret_cast<uint4*>(g_in) = *reinterpret_cast<const uint4*>(z + row * 2 * ffn + col);
    *reinterpret_cast<uint4*>(u_in) =
        *reinterpret_cast<const uint4*>(z + row * 2 * ffn + ffn + col);
    *reinterpret_cast<uint4*>(d_in) = *reinterpret_cast<const uint4*>(dh + row * ffn + col);

    alignas(16) __nv_bfloat16 dgate[kBwdVec];
    alignas(16) __nv_bfloat16 dup[kBwdVec];
#pragma unroll
    for (int i = 0; i < kBwdVec; ++i) {
      const float gate = __bfloat162float(g_in[i]);
      const float up = __bfloat162float(u_in[i]);
      const float d = __bfloat162float(d_in[i]);
      const Clamped c = clamp_inputs(gate, up, limit);
      const float denominator = __fadd_rn(expf(-c.g), 1.0f);
      const float sig = div_rn_no_ftz(1.0f, denominator);
      const float dsilu =
          __fmul_rn(sig, __fadd_rn(__fmul_rn(c.g, __fsub_rn(1.0f, sig)), 1.0f));
      const float act = round_bf16(div_rn_no_ftz(c.g, denominator));
      const float dg = gate < limit ? __fmul_rn(__fmul_rn(d, c.u), dsilu) : 0.0f;
      const float du = (up > -limit && up < limit) ? __fmul_rn(d, act) : 0.0f;
      dgate[i] = __float2bfloat16_rn(dg);
      dup[i] = __float2bfloat16_rn(du);
    }
    *reinterpret_cast<uint4*>(dz + row * 2 * ffn + col) = *reinterpret_cast<const uint4*>(dgate);
    *reinterpret_cast<uint4*>(dz + row * 2 * ffn + ffn + col) =
        *reinterpret_cast<const uint4*>(dup);
  }
}

void check_bf16_2d(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.dim() == 2, name, " must be 2-D");
  TORCH_CHECK(t.scalar_type() == torch::kBFloat16, name, " must be BF16");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, name,
              " must be 16-byte aligned");
}

}  // namespace

std::vector<torch::Tensor> silu_and_mul_clamp_quant_forward(torch::Tensor z, double limit,
                                                            bool scale_ue8m0, bool write_h) {
  check_bf16_2d(z, "silu_and_mul_clamp: z");
  const int64_t rows = z.size(0);
  TORCH_CHECK(z.size(1) % (2 * kGroup) == 0,
              "silu_and_mul_clamp: z's last dimension must be a multiple of 256 (F % 128 == 0)");
  const int64_t ffn = z.size(1) / 2;
  const int64_t groups = ffn / kGroup;
  const c10::cuda::CUDAGuard guard(z.device());
  auto q = torch::empty({rows, ffn}, z.options().dtype(torch::kFloat8_e4m3fn));
  const int64_t stride = (rows + 3) / 4 * 4;
  auto scales = torch::empty_strided({rows, groups}, {1, stride}, z.options().dtype(torch::kFloat));
  auto h = write_h ? torch::empty({rows, ffn}, z.options()) : torch::Tensor();
  const int64_t total = rows * groups;
  if (total == 0) return {q, scales, h};
  const int64_t blocks = (total + kGroupsPerBlock - 1) / kGroupsPerBlock;
  TORCH_CHECK(blocks <= INT32_MAX, "silu_and_mul_clamp: input too large");
  silu_and_mul_clamp_quant_kernel<<<static_cast<unsigned>(blocks), kThreads, 0,
                                    at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(z.data_ptr()),
      reinterpret_cast<uint8_t*>(q.data_ptr()), scales.data_ptr<float>(),
      write_h ? reinterpret_cast<__nv_bfloat16*>(h.data_ptr()) : nullptr, rows, ffn, stride,
      static_cast<float>(limit), scale_ue8m0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {q, scales, h};
}

torch::Tensor silu_and_mul_clamp_backward_cuda(torch::Tensor dh, torch::Tensor z, double limit) {
  check_bf16_2d(dh, "silu_and_mul_clamp_backward: dh");
  check_bf16_2d(z, "silu_and_mul_clamp_backward: z");
  const int64_t rows = z.size(0);
  const int64_t ffn = z.size(1) / 2;
  TORCH_CHECK(z.size(1) % 2 == 0 && dh.size(0) == rows && dh.size(1) == ffn,
              "silu_and_mul_clamp_backward: dh must be [T, F] for z [T, 2F]");
  TORCH_CHECK(ffn % kBwdVec == 0, "silu_and_mul_clamp_backward: F must be a multiple of 8");
  const c10::cuda::CUDAGuard guard(z.device());
  auto dz = torch::empty_like(z);
  const int64_t total = rows * (ffn / kBwdVec);
  if (total == 0) return dz;
  const int threads = 256;
  const int64_t max_blocks = static_cast<int64_t>(at::cuda::getCurrentDeviceProperties()->multiProcessorCount) * 16;
  const int64_t blocks = std::min((total + threads - 1) / threads, max_blocks);
  silu_and_mul_clamp_backward_kernel<<<static_cast<unsigned>(blocks), threads, 0,
                                       at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(dh.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(z.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(dz.data_ptr()), rows, ffn, static_cast<float>(limit));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return dz;
}
