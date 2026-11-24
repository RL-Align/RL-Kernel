// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// DSv4 cache helpers (SM90): exact copies and dequantization, no arithmetic
// that can round.
//
// dsv4_ds_mla_gather: out[i] = dequant(cache entry at slots[i]) as BF16 [512]
//   (zeros where slots[i] == -1; rows with slots[i] < -1 are left untouched). An fp8_ds_mla entry is 448 e4m3 values in
//   UE8M0 blocks of 64 (value * 2^(e - 127) is exact in FP32 and in BF16)
//   followed by 64 BF16 values, copied as is. Equal to
//   rl_engine.kernels.ops.pytorch.compressor.dequant_ds_mla. One warp per row.
// dsv4_save_states: state[slot] = [kv | score + ape[pos % ratio]] in FP32 (one
//   IEEE add), skipping slots < 0. Equal to the golden save_states.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace {

constexpr int kRow = 576, kNope = 448, kScales = 8, kDim = 512, kWarps = 8;

__global__ void __launch_bounds__(kWarps * 32) ds_mla_gather_kernel(
    const uint8_t* __restrict__ cache, int64_t block_stride, int block_size,
    const int64_t* __restrict__ slots, int64_t rows, __nv_bfloat16* __restrict__ out,
    int64_t out_stride) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarps + threadIdx.x / 32;
  if (row >= rows) return;
  const int lane = threadIdx.x & 31;
  const int d0 = lane * 16;
  uint4* dst = reinterpret_cast<uint4*>(out + row * out_stride + d0);
  const int64_t slot = slots[row];
  if (slot < -1) return;  // leave this row to another gather
  if (slot < 0) {
    dst[0] = make_uint4(0, 0, 0, 0);
    dst[1] = make_uint4(0, 0, 0, 0);
    return;
  }
  const uint8_t* block = cache + (slot / block_size) * block_stride;
  const int64_t i = slot % block_size;
  const uint8_t* entry = block + i * kRow;
  if (d0 >= kNope) {  // BF16 RoPE values: copy
    const uint4* src = reinterpret_cast<const uint4*>(entry + kNope + 2 * (d0 - kNope));
    dst[0] = __ldg(src);
    dst[1] = __ldg(src + 1);
    return;
  }
  const uint8_t e = block[static_cast<int64_t>(block_size) * kRow + i * kScales + d0 / 64];
  const float factor = __uint_as_float(static_cast<uint32_t>(e) << 23);
  const uint4 q = __ldg(reinterpret_cast<const uint4*>(entry + d0));
  const uint32_t words[4] = {q.x, q.y, q.z, q.w};
  uint32_t o[8];
#pragma unroll
  for (int w = 0; w < 4; ++w) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const __nv_fp8x2_storage_t two = static_cast<__nv_fp8x2_storage_t>(words[w] >> (16 * h));
      const __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2(two, __NV_E4M3);
      const float2 f = __half22float2(__half2(hr));
      const uint16_t lo = __bfloat16_as_ushort(__float2bfloat16_rn(__fmul_rn(f.x, factor)));
      const uint16_t hi = __bfloat16_as_ushort(__float2bfloat16_rn(__fmul_rn(f.y, factor)));
      o[2 * w + h] = static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
    }
  }
  dst[0] = make_uint4(o[0], o[1], o[2], o[3]);
  dst[1] = make_uint4(o[4], o[5], o[6], o[7]);
}

__global__ void save_states_kernel(const float* __restrict__ kv, int64_t kv_stride,
                                   const float* __restrict__ score, int64_t score_stride,
                                   const float* __restrict__ ape, const int64_t* __restrict__ positions,
                                   const int64_t* __restrict__ slots, float* __restrict__ state,
                                   int64_t state_row, int width, int ratio) {
  const int64_t t = blockIdx.x;
  const int64_t slot = slots[t];
  if (slot < 0) return;
  const float* a = ape + (positions[t] % ratio) * width;
  float* dst = state + slot * state_row;
  for (int c = threadIdx.x; c < width; c += blockDim.x) {
    dst[c] = kv[t * kv_stride + c];
    dst[width + c] = __fadd_rn(score[t * score_stride + c], a[c]);
  }
}

}  // namespace

void dsv4_ds_mla_gather(torch::Tensor cache, int64_t block_size, torch::Tensor slots,
                        torch::Tensor out) {
  TORCH_CHECK(cache.is_cuda() && cache.scalar_type() == torch::kUInt8 && cache.dim() == 2 &&
                  cache.stride(1) == 1 && cache.size(1) >= block_size * (kRow + kScales) &&
                  cache.stride(0) % 16 == 0,
              "ds_mla_gather: cache must be uint8 [blocks, >= block_size * 584], 16-byte rows");
  TORCH_CHECK(slots.is_cuda() && slots.scalar_type() == torch::kInt64 && slots.dim() == 1 &&
                  slots.is_contiguous(),
              "ds_mla_gather: slots must be contiguous int64 [n]");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kBFloat16 && out.dim() == 2 &&
                  out.size(0) == slots.size(0) && out.size(1) == kDim && out.stride(1) == 1 &&
                  out.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0,
              "ds_mla_gather: out must be BF16 [n, 512] with 16-byte aligned rows");
  const int64_t rows = slots.size(0);
  if (rows == 0) return;
  const c10::cuda::CUDAGuard guard(cache.device());
  ds_mla_gather_kernel<<<static_cast<unsigned>((rows + kWarps - 1) / kWarps), kWarps * 32, 0,
                         at::cuda::getCurrentCUDAStream()>>>(
      cache.data_ptr<uint8_t>(), cache.stride(0), static_cast<int>(block_size),
      slots.data_ptr<int64_t>(), rows, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      out.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dsv4_save_states(torch::Tensor kv, torch::Tensor score, torch::Tensor ape,
                      torch::Tensor positions, torch::Tensor state, torch::Tensor slots) {
  const int64_t tokens = positions.size(0);
  const int64_t width = ape.size(1);
  TORCH_CHECK(kv.scalar_type() == torch::kFloat32 && score.scalar_type() == torch::kFloat32 &&
                  kv.dim() == 2 && score.dim() == 2 && kv.size(0) == tokens &&
                  score.size(0) == tokens && kv.size(1) == width && score.size(1) == width &&
                  kv.stride(1) == 1 && score.stride(1) == 1,
              "save_states: kv / score must be FP32 [T, width] with unit column stride");
  TORCH_CHECK(ape.scalar_type() == torch::kFloat32 && ape.is_contiguous(),
              "save_states: ape must be contiguous FP32 [ratio, width]");
  TORCH_CHECK(positions.scalar_type() == torch::kInt64 && positions.is_contiguous() &&
                  slots.scalar_type() == torch::kInt64 && slots.is_contiguous() &&
                  slots.size(0) == tokens,
              "save_states: positions / slots must be contiguous int64 [T]");
  TORCH_CHECK(state.scalar_type() == torch::kFloat32 && state.dim() == 3 &&
                  state.size(2) == 2 * width && state.stride(2) == 1 &&
                  state.stride(1) == 2 * width && state.stride(0) == state.size(1) * 2 * width,
              "save_states: state must be contiguous FP32 [blocks, block_size, 2 * width]");
  if (tokens == 0) return;
  const c10::cuda::CUDAGuard guard(state.device());
  save_states_kernel<<<static_cast<unsigned>(tokens), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      kv.data_ptr<float>(), kv.stride(0), score.data_ptr<float>(), score.stride(0),
      ape.data_ptr<float>(), positions.data_ptr<int64_t>(), slots.data_ptr<int64_t>(),
      state.data_ptr<float>(), 2 * width, static_cast<int>(width), static_cast<int>(ape.size(0)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
