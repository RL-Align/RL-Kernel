// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
// Inkling 2 Mini deterministic sigmoid top-k router (INK-24).

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cfloat>
#include <tuple>

namespace {
constexpr int kMaxTopK = 8;
constexpr int kThreads = 32;

template <typename scalar_t>
__global__ void router_kernel(
    const scalar_t* __restrict__ logits,
    const float* __restrict__ bias,
    int64_t* __restrict__ indices,
    float* __restrict__ weights,
    int64_t tokens,
    int64_t experts,
    int top_k,
    float route_scale) {
  const int64_t token = blockIdx.x;
  if (token >= tokens || threadIdx.x != 0) return;

  float selected_scores[kMaxTopK];
  float selected_probs[kMaxTopK];
  int64_t selected_ids[kMaxTopK];
  for (int i = 0; i < kMaxTopK; ++i) {
    selected_scores[i] = 0.0f;
    selected_probs[i] = 0.0f;
    selected_ids[i] = -1;
  }

  for (int64_t expert = 0; expert < experts; ++expert) {
    const float logit = static_cast<float>(logits[token * experts + expert]);
    const float probability = 1.0f / (1.0f + expf(-logit));
    const float score = probability + bias[expert];
    int insert_at = top_k;
    for (int i = 0; i < top_k; ++i) {
      // Experts are visited in ascending ID order, which preserves ties.
      if (selected_ids[i] < 0 || score > selected_scores[i] ||
          (isnan(score) && !isnan(selected_scores[i]))) {
        insert_at = i;
        break;
      }
    }
    if (insert_at == top_k) continue;
    for (int i = top_k - 1; i > insert_at; --i) {
      selected_scores[i] = selected_scores[i - 1];
      selected_probs[i] = selected_probs[i - 1];
      selected_ids[i] = selected_ids[i - 1];
    }
    selected_scores[insert_at] = score;
    selected_probs[insert_at] = probability;
    selected_ids[insert_at] = expert;
  }

  float sum = 0.0f;
  for (int i = 0; i < top_k; ++i) sum += selected_probs[i];
  const float denominator = sum < FLT_MIN ? FLT_MIN : sum;
  for (int i = 0; i < top_k; ++i) {
    indices[token * top_k + i] = selected_ids[i];
    weights[token * top_k + i] = (selected_probs[i] / denominator) * route_scale;
  }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> inkling2_router_forward(
    torch::Tensor logits,
    torch::Tensor selection_bias,
    int64_t top_k,
    double route_scale) {
  TORCH_CHECK(logits.is_cuda() && logits.is_contiguous(), "logits must be contiguous CUDA");
  TORCH_CHECK(selection_bias.is_cuda() && selection_bias.is_contiguous(),
              "selection_bias must be contiguous CUDA");
  TORCH_CHECK(selection_bias.device() == logits.device(), "selection_bias must share the logits device");
  TORCH_CHECK(logits.dim() == 2, "logits must have shape [tokens, experts]");
  TORCH_CHECK(selection_bias.dim() == 1 && selection_bias.size(0) == logits.size(1),
              "selection_bias must have shape [experts]");
  TORCH_CHECK(top_k > 0 && top_k <= kMaxTopK && top_k <= logits.size(1),
              "top_k must be in [1, 8] and no larger than experts");
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
    router_kernel<scalar_t><<<logits.size(0), kThreads, 0, stream>>>(
        logits.data_ptr<scalar_t>(), bias.data_ptr<float>(), indices.data_ptr<int64_t>(),
        weights.data_ptr<float>(), logits.size(0), logits.size(1), static_cast<int>(top_k),
        static_cast<float>(route_scale));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {indices, weights};
}
