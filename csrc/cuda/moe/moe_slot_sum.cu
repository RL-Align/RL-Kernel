// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// DSv4 MoE combine / backward fan-in (SM90): for each token t,
//   out[t] = cast(((rows[inv[t,0]] + rows[inv[t,1]] + ... + rows[inv[t,5]])
//                  + shared[t]) + residual[t])
// in FP32, slot order, the slot-0 row copied (not added to zero), shared and
// residual optional and added once each, then rounded to the output dtype (RNE).
// Byte-equal to rl_engine.kernels.ops.pytorch.moe.route_plan.slot_sum (P6
// fixed_order_combine + shared_residual_merge, and fused_dx_fanin backward).
// Each thread owns 8 consecutive columns of one token: no reduction across
// threads, so a token's bytes depend only on its own rows.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>

#include <cstdint>

namespace {

constexpr int kTopK = 6;
constexpr int kThreads = 256;
constexpr int kVec = 8;

__device__ __forceinline__ void load8(const float* p, float (&v)[kVec]) {
  const float4 a = __ldg(reinterpret_cast<const float4*>(p));
  const float4 b = __ldg(reinterpret_cast<const float4*>(p) + 1);
  v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
  v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}

__device__ __forceinline__ void load8(const __nv_bfloat16* p, float (&v)[kVec]) {
  const uint4 q = __ldg(reinterpret_cast<const uint4*>(p));
  const uint32_t w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    v[2 * i] = __uint_as_float(w[i] << 16);
    v[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
  }
}

__device__ __forceinline__ void store8(float* p, const float (&v)[kVec]) {
  reinterpret_cast<float4*>(p)[0] = make_float4(v[0], v[1], v[2], v[3]);
  reinterpret_cast<float4*>(p)[1] = make_float4(v[4], v[5], v[6], v[7]);
}

__device__ __forceinline__ void store8(__nv_bfloat16* p, const float (&v)[kVec]) {
  uint32_t o[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint16_t lo = __bfloat16_as_ushort(__float2bfloat16_rn(v[2 * i]));
    const uint16_t hi = __bfloat16_as_ushort(__float2bfloat16_rn(v[2 * i + 1]));
    o[i] = static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
  }
  *reinterpret_cast<uint4*>(p) = make_uint4(o[0], o[1], o[2], o[3]);
}

template <typename RowT, typename SharedT, typename OutT>
__global__ void __launch_bounds__(kThreads) slot_sum_kernel(
    const RowT* __restrict__ rows, int64_t row_stride, const int64_t* __restrict__ inv,
    const SharedT* __restrict__ shared, int64_t shared_stride,
    const __nv_bfloat16* __restrict__ residual, int64_t residual_stride, int hidden,
    OutT* __restrict__ out, int64_t out_stride) {
  const int64_t t = blockIdx.x;
  const int col = (blockIdx.y * kThreads + threadIdx.x) * kVec;
  if (col >= hidden) return;
  float acc[kVec], v[kVec];
  load8(rows + inv[t * kTopK] * row_stride + col, acc);
#pragma unroll
  for (int s = 1; s < kTopK; ++s) {
    load8(rows + inv[t * kTopK + s] * row_stride + col, v);
#pragma unroll
    for (int i = 0; i < kVec; ++i) acc[i] = __fadd_rn(acc[i], v[i]);
  }
  if (shared != nullptr) {
    load8(shared + t * shared_stride + col, v);
#pragma unroll
    for (int i = 0; i < kVec; ++i) acc[i] = __fadd_rn(acc[i], v[i]);
  }
  if (residual != nullptr) {
    load8(residual + t * residual_stride + col, v);
#pragma unroll
    for (int i = 0; i < kVec; ++i) acc[i] = __fadd_rn(acc[i], v[i]);
  }
  store8(out + t * out_stride + col, acc);
}

void check_rows(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.dim() == 2 && t.stride(1) == 1 &&
                  (t.scalar_type() == torch::kFloat32 || t.scalar_type() == torch::kBFloat16),
              "moe_slot_sum: ", name, " must be FP32/BF16 [n, H] with unit column stride");
  const int64_t align = t.scalar_type() == torch::kFloat32 ? 4 : 8;
  TORCH_CHECK(t.size(0) == 0 || (t.stride(0) % align == 0 &&
                                 reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0),
              "moe_slot_sum: ", name, " rows must be 16-byte aligned");
}

template <typename RowT, typename SharedT, typename OutT>
void launch(const torch::Tensor& rows, const torch::Tensor& inv, const SharedT* shared,
            int64_t shared_stride, const __nv_bfloat16* residual, int64_t residual_stride,
            torch::Tensor& out, cudaStream_t stream) {
  const int64_t tokens = inv.size(0);
  const int hidden = static_cast<int>(out.size(1));
  const dim3 grid(static_cast<unsigned>(tokens),
                  static_cast<unsigned>((hidden + kThreads * kVec - 1) / (kThreads * kVec)));
  slot_sum_kernel<RowT, SharedT, OutT><<<grid, kThreads, 0, stream>>>(
      reinterpret_cast<const RowT*>(rows.data_ptr()), rows.stride(0), inv.data_ptr<int64_t>(),
      shared, shared_stride, residual, residual_stride, hidden,
      reinterpret_cast<OutT*>(out.data_ptr()), out.stride(0));
}

template <typename RowT, typename OutT>
void dispatch_shared(const torch::Tensor& rows, const torch::Tensor& inv,
                     const std::optional<torch::Tensor>& shared, const __nv_bfloat16* residual,
                     int64_t residual_stride, torch::Tensor& out, cudaStream_t stream) {
  if (shared.has_value() && shared->scalar_type() == torch::kFloat32) {
    launch<RowT, float, OutT>(rows, inv, shared->data_ptr<float>(), shared->stride(0), residual,
                              residual_stride, out, stream);
  } else {
    const auto* p = shared.has_value()
                        ? reinterpret_cast<const __nv_bfloat16*>(shared->data_ptr())
                        : nullptr;
    launch<RowT, __nv_bfloat16, OutT>(rows, inv, p, shared.has_value() ? shared->stride(0) : 0,
                                      residual, residual_stride, out, stream);
  }
}

template <typename RowT>
void dispatch_out(const torch::Tensor& rows, const torch::Tensor& inv,
                  const std::optional<torch::Tensor>& shared, const __nv_bfloat16* residual,
                  int64_t residual_stride, torch::Tensor& out, cudaStream_t stream) {
  if (out.scalar_type() == torch::kFloat32) {
    dispatch_shared<RowT, float>(rows, inv, shared, residual, residual_stride, out, stream);
  } else {
    dispatch_shared<RowT, __nv_bfloat16>(rows, inv, shared, residual, residual_stride, out,
                                         stream);
  }
}

}  // namespace

void moe_slot_sum(torch::Tensor rows, torch::Tensor inv, std::optional<torch::Tensor> shared,
                  std::optional<torch::Tensor> residual, torch::Tensor out) {
  check_rows(rows, "rows");
  check_rows(out, "out");
  TORCH_CHECK(inv.is_cuda() && inv.scalar_type() == torch::kInt64 && inv.is_contiguous() &&
                  inv.dim() == 2 && inv.size(1) == kTopK,
              "moe_slot_sum: inv must be contiguous int64 [T, 6]");
  const int64_t tokens = inv.size(0);
  TORCH_CHECK(out.size(0) == tokens && out.size(1) == rows.size(1) && out.size(1) % kVec == 0,
              "moe_slot_sum: out must be [T, H] with H % 8 == 0 and H equal to rows'");
  if (shared.has_value()) {
    check_rows(*shared, "shared");
    TORCH_CHECK(shared->size(0) == tokens && shared->size(1) == out.size(1),
                "moe_slot_sum: shared must be [T, H]");
  }
  const __nv_bfloat16* residual_ptr = nullptr;
  int64_t residual_stride = 0;
  if (residual.has_value()) {
    check_rows(*residual, "residual");
    TORCH_CHECK(residual->scalar_type() == torch::kBFloat16 && residual->size(0) == tokens &&
                    residual->size(1) == out.size(1),
                "moe_slot_sum: residual must be BF16 [T, H]");
    residual_ptr = reinterpret_cast<const __nv_bfloat16*>(residual->data_ptr());
    residual_stride = residual->stride(0);
  }
  if (tokens == 0) return;
  const c10::cuda::CUDAGuard guard(out.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (rows.scalar_type() == torch::kFloat32) {
    dispatch_out<float>(rows, inv, shared, residual_ptr, residual_stride, out, stream);
  } else {
    dispatch_out<__nv_bfloat16>(rows, inv, shared, residual_ptr, residual_stride, out, stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
