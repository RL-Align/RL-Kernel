// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// DSv4 MoE router (SM90): sqrt-softplus scores, top-6 / hash routing, weights,
// and the dX backward through the frozen gate. Byte-equal to the golden
// rl_engine.kernels.ops.pytorch.router.dsv4_router (see its docstring for the
// arithmetic contract). Every FP32 step is spelled with __f*_rn so nothing is
// contracted into an FMA; expf / log1pf are libdevice, like PyTorch's CUDA ops.
//
// dsv4_router_topk: one warp per token. Each lane holds experts lane + 32 j;
//   six rounds of a warp arg-max over (key descending, id ascending).
// dsv4_router_backward_dx: one block per token. Warp 0 turns dL/dweights into
//   the six logit gradients; then every thread sums six gate rows (BF16) for
//   its columns in slot order.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>

#include <climits>
#include <cstdint>

namespace {

constexpr int kTopK = 6;
constexpr int kWarpsPerBlock = 4;
constexpr float kNanKey = -1e30f;

__device__ __forceinline__ float sqrt_softplus(float x) {
  const float sp = __fadd_rn(fmaxf(x, 0.f), log1pf(expf(-fabsf(x))));
  return __fsqrt_rn(sp);
}

__device__ __forceinline__ float sigmoid_fixed(float x) {
  if (x >= 0.f) return __fdiv_rn(1.f, __fadd_rn(1.f, expf(-x)));
  const float e = expf(x);
  return __fdiv_rn(e, __fadd_rn(1.f, e));
}

__device__ __forceinline__ bool better(float ka, int ia, float kb, int ib) {
  return ka > kb || (ka == kb && ia < ib);
}

// Weights from six selected logits: every lane ends with the same S, lane k < 6
// returns out_k (others return 0).
__device__ __forceinline__ float slot_weights(float s_mine, int lane, float c, float* total) {
  float w = isnan(s_mine) ? 0.f : s_mine;
  float sum = __shfl_sync(0xffffffffu, w, 0);
#pragma unroll
  for (int k = 1; k < kTopK; ++k) sum = __fadd_rn(sum, __shfl_sync(0xffffffffu, w, k));
  *total = sum;
  const float f = __fdiv_rn(c, sum > 0.f ? sum : 1.f);
  return lane < kTopK ? __fmul_rn(w, f) : 0.f;
}

template <int kPerLane, typename HashT>
__global__ void __launch_bounds__(kWarpsPerBlock * 32) router_topk_kernel(
    const float* __restrict__ logits, int64_t logits_stride, const float* __restrict__ bias,
    const HashT* __restrict__ input_ids, const HashT* __restrict__ tid2eid, int64_t rows,
    float c, float* __restrict__ weights, int32_t* __restrict__ ids) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + threadIdx.x / 32;
  if (row >= rows) return;
  const int lane = threadIdx.x & 31;
  const float* x = logits + row * logits_stride;

  int my_id = 0;  // lane k < 6 ends with slot k's expert
  if (tid2eid != nullptr) {
    if (lane < kTopK) {
      const int64_t token = static_cast<int64_t>(input_ids[row]);
      my_id = static_cast<int>(tid2eid[token * kTopK + lane]);
    }
  } else {
    float key[kPerLane];
#pragma unroll
    for (int j = 0; j < kPerLane; ++j) {
      const int e = lane + 32 * j;
      const float k = __fadd_rn(sqrt_softplus(x[e]), bias[e]);
      key[j] = isnan(k) ? kNanKey : k;
    }
    unsigned removed = 0;
#pragma unroll
    for (int r = 0; r < kTopK; ++r) {
      float bk = -INFINITY;
      int bi = INT_MAX;
#pragma unroll
      for (int j = 0; j < kPerLane; ++j) {
        const int e = lane + 32 * j;
        if (!((removed >> j) & 1u) && better(key[j], e, bk, bi)) {
          bk = key[j];
          bi = e;
        }
      }
#pragma unroll
      for (int m = 16; m > 0; m >>= 1) {
        const float ok = __shfl_xor_sync(0xffffffffu, bk, m);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, m);
        if (better(ok, oi, bk, bi)) {
          bk = ok;
          bi = oi;
        }
      }
      if ((bi & 31) == lane) removed |= 1u << (bi >> 5);
      if (lane == r) my_id = bi;
    }
  }
  const float s = lane < kTopK ? sqrt_softplus(x[my_id]) : 0.f;
  float total;
  const float out = slot_weights(s, lane, c, &total);
  if (lane < kTopK) {
    weights[row * kTopK + lane] = out;
    ids[row * kTopK + lane] = my_id;
  }
}

template <typename OutT>
__device__ __forceinline__ void store8(OutT* dst, const float (&v)[8]);

template <>
__device__ __forceinline__ void store8<float>(float* dst, const float (&v)[8]) {
  reinterpret_cast<float4*>(dst)[0] = make_float4(v[0], v[1], v[2], v[3]);
  reinterpret_cast<float4*>(dst)[1] = make_float4(v[4], v[5], v[6], v[7]);
}

template <>
__device__ __forceinline__ void store8<__nv_bfloat16>(__nv_bfloat16* dst, const float (&v)[8]) {
  uint32_t o[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint16_t lo = __bfloat16_as_ushort(__float2bfloat16_rn(v[2 * i]));
    const uint16_t hi = __bfloat16_as_ushort(__float2bfloat16_rn(v[2 * i + 1]));
    o[i] = static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
  }
  *reinterpret_cast<uint4*>(dst) = make_uint4(o[0], o[1], o[2], o[3]);
}

template <typename OutT>
__global__ void __launch_bounds__(256) router_backward_dx_kernel(
    const float* __restrict__ logits, int64_t logits_stride, const int32_t* __restrict__ ids,
    const float* __restrict__ d_weights, const __nv_bfloat16* __restrict__ gate, int hidden,
    float c, OutT* __restrict__ dx, int64_t dx_stride) {
  __shared__ float dsel[kTopK];
  __shared__ int sel[kTopK];
  const int64_t row = blockIdx.x;
  if (threadIdx.x < 32) {
    const int lane = threadIdx.x;
    int id = 0;
    float x = 0.f, s = 0.f, d = 0.f;
    if (lane < kTopK) {
      id = ids[row * kTopK + lane];
      x = logits[row * logits_stride + id];
      s = sqrt_softplus(x);
      d = d_weights[row * kTopK + lane];
    }
    float total;
    const float out = slot_weights(s, lane, c, &total);
    const float p = __fmul_rn(d, out);
    float g = __shfl_sync(0xffffffffu, p, 0);
#pragma unroll
    for (int k = 1; k < kTopK; ++k) g = __fadd_rn(g, __shfl_sync(0xffffffffu, p, k));
    if (lane < kTopK) {
      const float cd = __fmul_rn(c, d);
      const float dw = total > 0.f ? __fdiv_rn(__fsub_rn(cd, g), total) : cd;
      const bool live = !isnan(s) && s > 0.f && !isinf(s);
      dsel[lane] = live ? __fdiv_rn(__fmul_rn(dw, sigmoid_fixed(x)), __fmul_rn(s, 2.f)) : 0.f;
      sel[lane] = id;
    }
  }
  __syncthreads();
  float dk[kTopK];
  const __nv_bfloat16* rows[kTopK];
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    dk[k] = dsel[k];
    rows[k] = gate + static_cast<int64_t>(sel[k]) * hidden;
  }
  for (int col = threadIdx.x * 8; col < hidden; col += blockDim.x * 8) {
    float acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = 0.f;
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      const uint4 q = __ldg(reinterpret_cast<const uint4*>(rows[k] + col));
      const uint32_t words[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float lo = __uint_as_float(words[i] << 16);
        const float hi = __uint_as_float(words[i] & 0xffff0000u);
        acc[2 * i] = __fadd_rn(acc[2 * i], __fmul_rn(dk[k], lo));
        acc[2 * i + 1] = __fadd_rn(acc[2 * i + 1], __fmul_rn(dk[k], hi));
      }
    }
    store8<OutT>(dx + row * dx_stride + col, acc);
  }
}

template <int kPerLane>
void launch_topk(const torch::Tensor& logits, const float* bias,
                 const std::optional<torch::Tensor>& input_ids,
                 const std::optional<torch::Tensor>& tid2eid, float c, torch::Tensor& weights,
                 torch::Tensor& ids, cudaStream_t stream) {
  const int64_t rows = logits.size(0);
  const unsigned grid = static_cast<unsigned>((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
  const float* x = logits.data_ptr<float>();
  float* w = weights.data_ptr<float>();
  int32_t* i = ids.data_ptr<int32_t>();
  if (tid2eid.has_value() && tid2eid->scalar_type() == torch::kInt64) {
    router_topk_kernel<kPerLane, int64_t><<<grid, kWarpsPerBlock * 32, 0, stream>>>(
        x, logits.stride(0), bias, input_ids->data_ptr<int64_t>(), tid2eid->data_ptr<int64_t>(),
        rows, c, w, i);
  } else if (tid2eid.has_value()) {
    router_topk_kernel<kPerLane, int32_t><<<grid, kWarpsPerBlock * 32, 0, stream>>>(
        x, logits.stride(0), bias, input_ids->data_ptr<int32_t>(), tid2eid->data_ptr<int32_t>(),
        rows, c, w, i);
  } else {
    router_topk_kernel<kPerLane, int32_t><<<grid, kWarpsPerBlock * 32, 0, stream>>>(
        x, logits.stride(0), bias, nullptr, nullptr, rows, c, w, i);
  }
}

}  // namespace

void dsv4_router_topk(torch::Tensor logits, std::optional<torch::Tensor> bias,
                      std::optional<torch::Tensor> input_ids, std::optional<torch::Tensor> tid2eid,
                      double routed_scaling_factor, torch::Tensor weights, torch::Tensor ids) {
  TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == torch::kFloat32 && logits.dim() == 2 &&
                  logits.stride(1) == 1,
              "router_topk: logits must be FP32 [T, E] with unit column stride");
  const int64_t rows = logits.size(0);
  const int64_t experts = logits.size(1);
  TORCH_CHECK(experts == 256 || experts == 384, "router_topk: E must be 256 or 384");
  TORCH_CHECK(weights.scalar_type() == torch::kFloat32 && weights.is_contiguous() &&
                  weights.dim() == 2 && weights.size(0) == rows && weights.size(1) == kTopK,
              "router_topk: weights must be contiguous FP32 [T, 6]");
  TORCH_CHECK(ids.scalar_type() == torch::kInt32 && ids.is_contiguous() && ids.dim() == 2 &&
                  ids.size(0) == rows && ids.size(1) == kTopK,
              "router_topk: ids must be contiguous int32 [T, 6]");
  const float* bias_ptr = nullptr;
  if (tid2eid.has_value()) {
    TORCH_CHECK(input_ids.has_value(), "router_topk: hash routing requires input_ids");
    TORCH_CHECK(tid2eid->is_cuda() && tid2eid->is_contiguous() && tid2eid->dim() == 2 &&
                    tid2eid->size(1) == kTopK &&
                    (tid2eid->scalar_type() == torch::kInt32 ||
                     tid2eid->scalar_type() == torch::kInt64),
                "router_topk: tid2eid must be contiguous int32/int64 [vocab, 6]");
    TORCH_CHECK(input_ids->is_cuda() && input_ids->is_contiguous() && input_ids->dim() == 1 &&
                    input_ids->size(0) == rows &&
                    input_ids->scalar_type() == tid2eid->scalar_type(),
                "router_topk: input_ids must be contiguous [T] with tid2eid's dtype");
  } else {
    TORCH_CHECK(bias.has_value() && bias->is_cuda() && bias->scalar_type() == torch::kFloat32 &&
                    bias->is_contiguous() && bias->numel() == experts,
                "router_topk: scored routing requires a contiguous FP32 bias [E]");
    bias_ptr = bias->data_ptr<float>();
  }
  if (rows == 0) return;
  const c10::cuda::CUDAGuard guard(logits.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const float c = static_cast<float>(routed_scaling_factor);
  if (experts == 256) {
    launch_topk<8>(logits, bias_ptr, input_ids, tid2eid, c, weights, ids, stream);
  } else {
    launch_topk<12>(logits, bias_ptr, input_ids, tid2eid, c, weights, ids, stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dsv4_router_backward_dx(torch::Tensor logits, torch::Tensor ids, torch::Tensor d_weights,
                             torch::Tensor gate_weight, double routed_scaling_factor,
                             torch::Tensor dx) {
  TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == torch::kFloat32 && logits.dim() == 2 &&
                  logits.stride(1) == 1,
              "router_backward_dx: logits must be FP32 [T, E] with unit column stride");
  const int64_t rows = logits.size(0);
  TORCH_CHECK(ids.scalar_type() == torch::kInt32 && ids.is_contiguous() && ids.dim() == 2 &&
                  ids.size(0) == rows && ids.size(1) == kTopK,
              "router_backward_dx: ids must be contiguous int32 [T, 6]");
  TORCH_CHECK(d_weights.scalar_type() == torch::kFloat32 && d_weights.is_contiguous() &&
                  d_weights.dim() == 2 && d_weights.size(0) == rows &&
                  d_weights.size(1) == kTopK,
              "router_backward_dx: d_weights must be contiguous FP32 [T, 6]");
  TORCH_CHECK(gate_weight.scalar_type() == torch::kBFloat16 && gate_weight.is_contiguous() &&
                  gate_weight.dim() == 2 && gate_weight.size(0) == logits.size(1) &&
                  gate_weight.size(1) % 8 == 0,
              "router_backward_dx: gate_weight must be contiguous BF16 [E, H], H % 8 == 0");
  const int64_t hidden = gate_weight.size(1);
  TORCH_CHECK((dx.scalar_type() == torch::kFloat32 || dx.scalar_type() == torch::kBFloat16) &&
                  dx.dim() == 2 && dx.size(0) == rows && dx.size(1) == hidden &&
                  dx.stride(1) == 1 && dx.stride(0) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(dx.data_ptr()) % 16 == 0,
              "router_backward_dx: dx must be FP32/BF16 [T, H] with 16-byte aligned rows");
  if (rows == 0) return;
  const c10::cuda::CUDAGuard guard(logits.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const float c = static_cast<float>(routed_scaling_factor);
  const auto* gate = reinterpret_cast<const __nv_bfloat16*>(gate_weight.data_ptr());
  if (dx.scalar_type() == torch::kFloat32) {
    router_backward_dx_kernel<float><<<static_cast<unsigned>(rows), 256, 0, stream>>>(
        logits.data_ptr<float>(), logits.stride(0), ids.data_ptr<int32_t>(),
        d_weights.data_ptr<float>(), gate, static_cast<int>(hidden), c, dx.data_ptr<float>(),
        dx.stride(0));
  } else {
    router_backward_dx_kernel<__nv_bfloat16><<<static_cast<unsigned>(rows), 256, 0, stream>>>(
        logits.data_ptr<float>(), logits.stride(0), ids.data_ptr<int32_t>(),
        d_weights.data_ptr<float>(), gate, static_cast<int>(hidden), c,
        reinterpret_cast<__nv_bfloat16*>(dx.data_ptr()), dx.stride(0));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
