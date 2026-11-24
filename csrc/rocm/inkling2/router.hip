// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// Inkling 2 Mini deterministic sigmoid top-k router (INK-24), ROCm-native.
// Compiled only for ROCm builds; csrc/cuda/inkling2/router.cu keeps the CUDA path
// and this file exports the same inkling2_router_forward symbol.
//
// Determinism contract: one wave64 per token. Lane l scans experts l, l+64, ... in
// ascending id order and keeps a local sorted top-k under the strict total order
// (selection score descending, NaN above everything, expert id ascending). The
// waves then run top_k rounds of a butterfly argmax over the lane heads. Because the
// order is total, the merged result equals the sequential CUDA/reference result
// independent of the lane split. The selected unbiased probabilities are summed in
// selected rank order 0..k-1, then divided (FLT_MIN floor) and scaled, matching the
// Python reference. No atomics; every output depends only on its own token row.
// Tuned for MI300X (gfx942): 64-thread blocks, FP32 math, register-resident lists.

#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPException.h>
#include <c10/hip/HIPGuard.h>
#include <hip/hip_runtime.h>

#include <cfloat>
#include <cstdint>
#include <torch/extension.h>
#include <tuple>

namespace {
constexpr int kMaxTopK = 8;
constexpr int kWave = 64;

// True when candidate a strictly precedes candidate b. Invalid (id < 0) is worst.
__device__ __forceinline__ bool better(float sa, int ia, float sb, int ib) {
  if (ia < 0) return false;
  if (ib < 0) return true;
  const bool na = sa != sa;
  const bool nb = sb != sb;
  if (na != nb) return na;
  if (!na && sa != sb) return sa > sb;
  return ia < ib;
}

template <typename scalar_t>
__global__ void __launch_bounds__(kWave) router_kernel(
    const scalar_t* __restrict__ logits,
    const float* __restrict__ bias,
    int64_t* __restrict__ indices,
    float* __restrict__ weights,
    int64_t experts,
    int top_k,
    float route_scale) {
  const int64_t token = blockIdx.x;
  const int lane = threadIdx.x;
  const scalar_t* row = logits + token * experts;

  float sc[kMaxTopK];
  float pr[kMaxTopK];
  int id[kMaxTopK];
#pragma unroll
  for (int i = 0; i < kMaxTopK; ++i) {
    sc[i] = 0.0f;
    pr[i] = 0.0f;
    id[i] = -1;
  }

  for (int64_t e = lane; e < experts; e += kWave) {
    const float logit = static_cast<float>(row[e]);
    const float p = 1.0f / (1.0f + expf(-logit));
    const float s = p + bias[e];
    const int eid = static_cast<int>(e);
    // Rank of the new candidate among this lane's entries (first strictly worse slot).
    int at = kMaxTopK;
#pragma unroll
    for (int i = kMaxTopK - 1; i >= 0; --i) {
      if (i < top_k && better(s, eid, sc[i], id[i])) at = i;
    }
#pragma unroll
    for (int i = kMaxTopK - 1; i >= 0; --i) {
      if (i > at) {
        sc[i] = sc[i - 1];
        pr[i] = pr[i - 1];
        id[i] = id[i - 1];
      } else if (i == at) {
        sc[i] = s;
        pr[i] = p;
        id[i] = eid;
      }
    }
  }

  float sel_prob[kMaxTopK];
  int sel_id[kMaxTopK];
#pragma unroll
  for (int r = 0; r < kMaxTopK; ++r) {
    sel_prob[r] = 0.0f;
    sel_id[r] = -1;
    if (r < top_k) {
      float ws = sc[0], wp = pr[0];
      int wi = id[0];
#pragma unroll
      for (int off = 32; off >= 1; off >>= 1) {
        const float os = __shfl_xor(ws, off, kWave);
        const float op = __shfl_xor(wp, off, kWave);
        const int oi = __shfl_xor(wi, off, kWave);
        if (better(os, oi, ws, wi)) {
          ws = os;
          wp = op;
          wi = oi;
        }
      }
      sel_prob[r] = wp;
      sel_id[r] = wi;
      // The winning lane pops its head.
      if (id[0] >= 0 && id[0] == wi) {
#pragma unroll
        for (int i = 0; i < kMaxTopK - 1; ++i) {
          sc[i] = sc[i + 1];
          pr[i] = pr[i + 1];
          id[i] = id[i + 1];
        }
        sc[kMaxTopK - 1] = 0.0f;
        pr[kMaxTopK - 1] = 0.0f;
        id[kMaxTopK - 1] = -1;
      }
    }
  }

  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < kMaxTopK; ++i) {
    if (i < top_k) sum += sel_prob[i];
  }
  const float denominator = sum < FLT_MIN ? FLT_MIN : sum;
#pragma unroll
  for (int i = 0; i < kMaxTopK; ++i) {
    if (i < top_k && lane == i) {
      indices[token * top_k + i] = sel_id[i];
      weights[token * top_k + i] = (sel_prob[i] / denominator) * route_scale;
    }
  }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> inkling2_router_forward(
    torch::Tensor logits,
    torch::Tensor selection_bias,
    int64_t top_k,
    double route_scale) {
  TORCH_CHECK(logits.is_cuda() && logits.is_contiguous(), "logits must be contiguous ROCm");
  TORCH_CHECK(selection_bias.is_cuda() && selection_bias.is_contiguous(),
              "selection_bias must be contiguous ROCm");
  TORCH_CHECK(selection_bias.device() == logits.device(), "selection_bias must share the logits device");
  TORCH_CHECK(logits.dim() == 2, "logits must have shape [tokens, experts]");
  TORCH_CHECK(selection_bias.dim() == 1 && selection_bias.size(0) == logits.size(1),
              "selection_bias must have shape [experts]");
  TORCH_CHECK(top_k > 0 && top_k <= kMaxTopK && top_k <= logits.size(1),
              "top_k must be in [1, 8] and no larger than experts");
  TORCH_CHECK(logits.size(1) <= INT32_MAX, "too many experts");
  TORCH_CHECK(logits.scalar_type() == at::kFloat || logits.scalar_type() == at::kHalf ||
                  logits.scalar_type() == at::kBFloat16,
              "logits must be float32, float16, or bfloat16");
  TORCH_CHECK(selection_bias.is_floating_point(), "selection_bias must be floating point");

  c10::cuda::CUDAGuard guard(logits.device());
  auto bias = selection_bias.to(torch::kFloat32).contiguous();
  auto indices = torch::empty({logits.size(0), top_k}, logits.options().dtype(torch::kInt64));
  auto weights = torch::empty({logits.size(0), top_k}, logits.options().dtype(torch::kFloat32));
  if (logits.size(0) == 0) return {indices, weights};
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, logits.scalar_type(),
                                  "inkling2_router_forward", [&] {
    router_kernel<scalar_t><<<logits.size(0), kWave, 0, stream>>>(
        logits.data_ptr<scalar_t>(), bias.data_ptr<float>(), indices.data_ptr<int64_t>(),
        weights.data_ptr<float>(), logits.size(1), static_cast<int>(top_k),
        static_cast<float>(route_scale));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {indices, weights};
}
