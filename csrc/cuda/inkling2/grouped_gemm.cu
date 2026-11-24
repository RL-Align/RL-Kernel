// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
// Inkling 2 Mini deterministic grouped expert GEMM (INK-27).

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <vector>

namespace {

constexpr int kThreads = 256;

template <typename scalar_t>
__global__ void grouped_gemm_forward_kernel(
    const scalar_t* __restrict__ tokens,
    const int64_t* __restrict__ expert_ids,
    const scalar_t* __restrict__ expert_weights,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int64_t token_count,
    int64_t hidden_size,
    int64_t expert_count,
    int64_t output_size) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = token_count * output_size;
  if (index >= total) return;
  const int64_t token = index / output_size;
  const int64_t output_index = index - token * output_size;
  const int64_t expert = expert_ids[token];
  float value = bias == nullptr ? 0.0f : static_cast<float>(bias[expert * output_size + output_index]);
  for (int64_t h = 0; h < hidden_size; ++h) {
    value += static_cast<float>(tokens[token * hidden_size + h]) *
        static_cast<float>(expert_weights[(expert * output_size + output_index) * hidden_size + h]);
  }
  output[index] = static_cast<scalar_t>(value);
}

template <typename scalar_t>
__global__ void grouped_gemm_backward_tokens_kernel(
    const scalar_t* __restrict__ dy,
    const int64_t* __restrict__ expert_ids,
    const scalar_t* __restrict__ expert_weights,
    scalar_t* __restrict__ d_tokens,
    int64_t token_count,
    int64_t hidden_size,
    int64_t output_size) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = token_count * hidden_size;
  if (index >= total) return;
  const int64_t token = index / hidden_size;
  const int64_t hidden_index = index - token * hidden_size;
  const int64_t expert = expert_ids[token];
  float value = 0.0f;
  for (int64_t output_index = 0; output_index < output_size; ++output_index) {
    value += static_cast<float>(dy[token * output_size + output_index]) *
        static_cast<float>(expert_weights[(expert * output_size + output_index) * hidden_size + hidden_index]);
  }
  d_tokens[index] = static_cast<scalar_t>(value);
}

template <typename scalar_t>
__global__ void grouped_gemm_backward_weights_kernel(
    const scalar_t* __restrict__ dy,
    const scalar_t* __restrict__ tokens,
    const int64_t* __restrict__ expert_ids,
    scalar_t* __restrict__ d_expert_weights,
    int64_t token_count,
    int64_t hidden_size,
    int64_t expert_count,
    int64_t output_size) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = expert_count * output_size * hidden_size;
  if (index >= total) return;
  const int64_t hidden_index = index % hidden_size;
  const int64_t output_index = (index / hidden_size) % output_size;
  const int64_t expert = index / (output_size * hidden_size);
  float value = 0.0f;
  for (int64_t token = 0; token < token_count; ++token) {
    if (expert_ids[token] != expert) continue;
    value += static_cast<float>(dy[token * output_size + output_index]) *
        static_cast<float>(tokens[token * hidden_size + hidden_index]);
  }
  d_expert_weights[index] = static_cast<scalar_t>(value);
}

template <typename scalar_t>
__global__ void grouped_gemm_backward_bias_kernel(
    const scalar_t* __restrict__ dy,
    const int64_t* __restrict__ expert_ids,
    scalar_t* __restrict__ d_bias,
    int64_t token_count,
    int64_t expert_count,
    int64_t output_size) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = expert_count * output_size;
  if (index >= total) return;
  const int64_t expert = index / output_size;
  const int64_t output_index = index - expert * output_size;
  float value = 0.0f;
  for (int64_t token = 0; token < token_count; ++token) {
    if (expert_ids[token] == expert) {
      value += static_cast<float>(dy[token * output_size + output_index]);
    }
  }
  d_bias[index] = static_cast<scalar_t>(value);
}

static void check_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous(), name, " must be contiguous CUDA");
  TORCH_CHECK(
      tensor.scalar_type() == at::kFloat || tensor.scalar_type() == at::kHalf ||
          tensor.scalar_type() == at::kBFloat16,
      name,
      " must be float32, float16, or bfloat16");
}

static void check_same(const torch::Tensor& lhs, const torch::Tensor& rhs, const char* lhs_name,
                       const char* rhs_name) {
  TORCH_CHECK(lhs.device() == rhs.device(), lhs_name, " and ", rhs_name, " must share device");
  TORCH_CHECK(lhs.scalar_type() == rhs.scalar_type(), lhs_name, " and ", rhs_name, " must share dtype");
}

}  // namespace

torch::Tensor inkling2_grouped_gemm_forward(
    torch::Tensor tokens,
    torch::Tensor expert_indices,
    torch::Tensor expert_weights,
    torch::Tensor bias) {
  check_tensor(tokens, "tokens");
  check_tensor(expert_weights, "expert_weights");
  check_same(tokens, expert_weights, "tokens", "expert_weights");
  TORCH_CHECK(expert_indices.is_cuda() && expert_indices.is_contiguous(),
              "expert_indices must be contiguous CUDA");
  TORCH_CHECK(expert_indices.scalar_type() == at::kInt || expert_indices.scalar_type() == at::kLong,
              "expert_indices must be int32 or int64");
  TORCH_CHECK(tokens.dim() == 2, "tokens must be [tokens, hidden]");
  TORCH_CHECK(expert_weights.dim() == 3, "expert_weights must be [experts, output, hidden]");
  TORCH_CHECK(expert_indices.dim() == 1 && expert_indices.size(0) == tokens.size(0),
              "expert_indices must have one entry per token");
  TORCH_CHECK(expert_weights.size(2) == tokens.size(1), "hidden size mismatch");
  TORCH_CHECK(bias.numel() == 0 || bias.is_cuda() && bias.is_contiguous(),
              "bias must be empty or contiguous CUDA");
  if (bias.numel() != 0) {
    check_tensor(bias, "bias");
    check_same(tokens, bias, "tokens", "bias");
    TORCH_CHECK(bias.sizes() == torch::IntArrayRef({expert_weights.size(0), expert_weights.size(1)}),
                "bias must be [experts, output]");
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(tokens));
  auto expert_ids = expert_indices.to(torch::kInt64);
  if (expert_ids.numel() != 0) {
    TORCH_CHECK(expert_ids.min().item<int64_t>() >= 0 &&
                    expert_ids.max().item<int64_t>() < expert_weights.size(0),
                "expert index is outside the available expert range");
  }
  auto output = torch::empty({tokens.size(0), expert_weights.size(1)}, tokens.options());
  const int64_t total = output.numel();
  if (total == 0) return output;
  auto stream = at::cuda::getCurrentCUDAStream();
  const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, tokens.scalar_type(),
                                  "inkling2_grouped_gemm_forward", [&] {
    grouped_gemm_forward_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
        tokens.data_ptr<scalar_t>(), expert_ids.data_ptr<int64_t>(), expert_weights.data_ptr<scalar_t>(),
        bias.numel() == 0 ? nullptr : bias.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
        tokens.size(0), tokens.size(1), expert_weights.size(0), expert_weights.size(1));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::vector<torch::Tensor> inkling2_grouped_gemm_backward(
    torch::Tensor dy,
    torch::Tensor tokens,
    torch::Tensor expert_indices,
    torch::Tensor expert_weights,
    bool need_bias) {
  check_tensor(dy, "dy");
  check_tensor(tokens, "tokens");
  check_tensor(expert_weights, "expert_weights");
  check_same(dy, tokens, "dy", "tokens");
  check_same(tokens, expert_weights, "tokens", "expert_weights");
  TORCH_CHECK(expert_indices.is_cuda() && expert_indices.is_contiguous(),
              "expert_indices must be contiguous CUDA");
  TORCH_CHECK(expert_indices.scalar_type() == at::kInt || expert_indices.scalar_type() == at::kLong,
              "expert_indices must be int32 or int64");
  TORCH_CHECK(dy.dim() == 2 && dy.size(0) == tokens.size(0) && dy.size(1) == expert_weights.size(1),
              "dy shape must match grouped GEMM output");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(tokens));
  auto expert_ids = expert_indices.to(torch::kInt64);
  auto d_tokens = torch::empty_like(tokens);
  auto d_weights = torch::empty_like(expert_weights);
  auto d_bias = need_bias ? torch::empty({expert_weights.size(0), expert_weights.size(1)}, tokens.options())
                          : torch::Tensor();
  auto stream = at::cuda::getCurrentCUDAStream();
  const int token_blocks = static_cast<int>((tokens.numel() + kThreads - 1) / kThreads);
  const int weight_blocks = static_cast<int>((expert_weights.numel() + kThreads - 1) / kThreads);
  const int bias_blocks = static_cast<int>((expert_weights.size(0) * expert_weights.size(1) + kThreads - 1) / kThreads);
  if (tokens.numel() == 0 || expert_weights.numel() == 0) return {d_tokens, d_weights, d_bias};

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, tokens.scalar_type(),
                                  "inkling2_grouped_gemm_backward", [&] {
    grouped_gemm_backward_tokens_kernel<scalar_t><<<token_blocks, kThreads, 0, stream>>>(
        dy.data_ptr<scalar_t>(), expert_ids.data_ptr<int64_t>(), expert_weights.data_ptr<scalar_t>(),
        d_tokens.data_ptr<scalar_t>(), tokens.size(0), tokens.size(1), expert_weights.size(1));
    grouped_gemm_backward_weights_kernel<scalar_t><<<weight_blocks, kThreads, 0, stream>>>(
        dy.data_ptr<scalar_t>(), tokens.data_ptr<scalar_t>(), expert_ids.data_ptr<int64_t>(),
        d_weights.data_ptr<scalar_t>(), tokens.size(0), tokens.size(1), expert_weights.size(0),
        expert_weights.size(1));
    if (need_bias) {
      grouped_gemm_backward_bias_kernel<scalar_t><<<bias_blocks, kThreads, 0, stream>>>(
          dy.data_ptr<scalar_t>(), expert_ids.data_ptr<int64_t>(), d_bias.data_ptr<scalar_t>(),
          tokens.size(0), expert_weights.size(0), expert_weights.size(1));
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {d_tokens, d_weights, d_bias};
}
