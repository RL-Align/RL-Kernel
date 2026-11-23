// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
// Inkling 2 Mini softcapped expert gate/up projection (INK-26).

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

__device__ __forceinline__ float softcap_f32(float value, float cap) {
  return cap * tanhf(value / cap);
}

__device__ __forceinline__ float softcap_grad_f32(float value, float cap) {
  const float t = tanhf(value / cap);
  return 1.0f - t * t;
}

__device__ __forceinline__ float silu_f32(float value) {
  const float sigmoid = 1.0f / (1.0f + expf(-value));
  return value * sigmoid;
}

__device__ __forceinline__ float silu_grad_f32(float value) {
  const float sigmoid = 1.0f / (1.0f + expf(-value));
  return sigmoid * (1.0f + value * (1.0f - sigmoid));
}

template <typename scalar_t>
__global__ void gate_up_forward_kernel(
    const scalar_t* __restrict__ hidden,
    const scalar_t* __restrict__ gate_weight,
    const scalar_t* __restrict__ up_weight,
    const float* __restrict__ gate_bias,
    const float* __restrict__ up_bias,
    scalar_t* __restrict__ output,
    float* __restrict__ gate_pre,
    float* __restrict__ up_pre,
    int64_t rows,
    int64_t hidden_size,
    int64_t intermediate_size,
    float gate_cap,
    float up_cap) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = rows * intermediate_size;
  if (index >= total) return;
  const int64_t row = index / intermediate_size;
  const int64_t intermediate = index - row * intermediate_size;

  float gate = gate_bias == nullptr ? 0.0f : gate_bias[intermediate];
  float up = up_bias == nullptr ? 0.0f : up_bias[intermediate];
  for (int64_t h = 0; h < hidden_size; ++h) {
    const float value = static_cast<float>(hidden[row * hidden_size + h]);
    gate += value * static_cast<float>(gate_weight[intermediate * hidden_size + h]);
    up += value * static_cast<float>(up_weight[intermediate * hidden_size + h]);
  }
  gate_pre[index] = gate;
  up_pre[index] = up;
  const float gate_capped = softcap_f32(gate, gate_cap);
  const float up_capped = softcap_f32(up, up_cap);
  output[index] = static_cast<scalar_t>(silu_f32(gate_capped) * up_capped);
}

template <typename scalar_t>
__global__ void gate_up_backward_hidden_kernel(
    const scalar_t* __restrict__ dy,
    const scalar_t* __restrict__ gate_weight,
    const scalar_t* __restrict__ up_weight,
    const float* __restrict__ gate_pre,
    const float* __restrict__ up_pre,
    scalar_t* __restrict__ d_hidden,
    int64_t rows,
    int64_t hidden_size,
    int64_t intermediate_size,
    float gate_cap,
    float up_cap) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = rows * hidden_size;
  if (index >= total) return;
  const int64_t row = index / hidden_size;
  const int64_t hidden_index = index - row * hidden_size;

  float gradient = 0.0f;
  for (int64_t intermediate = 0; intermediate < intermediate_size; ++intermediate) {
    const int64_t offset = row * intermediate_size + intermediate;
    const float gate_capped = softcap_f32(gate_pre[offset], gate_cap);
    const float up_capped = softcap_f32(up_pre[offset], up_cap);
    const float d_gate = static_cast<float>(dy[offset]) * up_capped *
        silu_grad_f32(gate_capped) * softcap_grad_f32(gate_pre[offset], gate_cap);
    const float d_up = static_cast<float>(dy[offset]) * silu_f32(gate_capped) *
        softcap_grad_f32(up_pre[offset], up_cap);
    gradient += d_gate * static_cast<float>(gate_weight[intermediate * hidden_size + hidden_index]);
    gradient += d_up * static_cast<float>(up_weight[intermediate * hidden_size + hidden_index]);
  }
  d_hidden[index] = static_cast<scalar_t>(gradient);
}

template <typename scalar_t>
__global__ void gate_up_backward_weight_kernel(
    const scalar_t* __restrict__ dy,
    const scalar_t* __restrict__ hidden,
    const float* __restrict__ gate_pre,
    const float* __restrict__ up_pre,
    scalar_t* __restrict__ d_gate_weight,
    scalar_t* __restrict__ d_up_weight,
    int64_t rows,
    int64_t hidden_size,
    int64_t intermediate_size,
    float gate_cap,
    float up_cap) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = intermediate_size * hidden_size;
  if (index >= total) return;
  const int64_t intermediate = index / hidden_size;
  const int64_t hidden_index = index - intermediate * hidden_size;

  float d_gate_weight_value = 0.0f;
  float d_up_weight_value = 0.0f;
  for (int64_t row = 0; row < rows; ++row) {
    const int64_t offset = row * intermediate_size + intermediate;
    const float gate_capped = softcap_f32(gate_pre[offset], gate_cap);
    const float up_capped = softcap_f32(up_pre[offset], up_cap);
    const float d_gate = static_cast<float>(dy[offset]) * up_capped *
        silu_grad_f32(gate_capped) * softcap_grad_f32(gate_pre[offset], gate_cap);
    const float d_up = static_cast<float>(dy[offset]) * silu_f32(gate_capped) *
        softcap_grad_f32(up_pre[offset], up_cap);
    const float hidden_value = static_cast<float>(hidden[row * hidden_size + hidden_index]);
    d_gate_weight_value += d_gate * hidden_value;
    d_up_weight_value += d_up * hidden_value;
  }
  d_gate_weight[index] = static_cast<scalar_t>(d_gate_weight_value);
  d_up_weight[index] = static_cast<scalar_t>(d_up_weight_value);
}

template <typename scalar_t>
__global__ void gate_up_backward_bias_kernel(
    const scalar_t* __restrict__ dy,
    const float* __restrict__ gate_pre,
    const float* __restrict__ up_pre,
    scalar_t* __restrict__ d_gate_bias,
    scalar_t* __restrict__ d_up_bias,
    int64_t rows,
    int64_t intermediate_size,
    float gate_cap,
    float up_cap) {
  const int64_t intermediate = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (intermediate >= intermediate_size) return;

  float d_gate_bias_value = 0.0f;
  float d_up_bias_value = 0.0f;
  for (int64_t row = 0; row < rows; ++row) {
    const int64_t offset = row * intermediate_size + intermediate;
    const float gate_capped = softcap_f32(gate_pre[offset], gate_cap);
    const float up_capped = softcap_f32(up_pre[offset], up_cap);
    const float d_gate = static_cast<float>(dy[offset]) * up_capped *
        silu_grad_f32(gate_capped) * softcap_grad_f32(gate_pre[offset], gate_cap);
    const float d_up = static_cast<float>(dy[offset]) * silu_f32(gate_capped) *
        softcap_grad_f32(up_pre[offset], up_cap);
    d_gate_bias_value += d_gate;
    d_up_bias_value += d_up;
  }
  d_gate_bias[intermediate] = static_cast<scalar_t>(d_gate_bias_value);
  d_up_bias[intermediate] = static_cast<scalar_t>(d_up_bias_value);
}

static void check_input(const torch::Tensor& tensor, const char* name) {
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

std::vector<torch::Tensor> inkling2_expert_gate_up_forward(
    torch::Tensor hidden,
    torch::Tensor gate_weight,
    torch::Tensor up_weight,
    torch::Tensor gate_bias,
    torch::Tensor up_bias,
    double gate_cap,
    double up_cap) {
  check_input(hidden, "hidden");
  check_input(gate_weight, "gate_weight");
  check_input(up_weight, "up_weight");
  check_same(hidden, gate_weight, "hidden", "gate_weight");
  check_same(hidden, up_weight, "hidden", "up_weight");
  TORCH_CHECK(hidden.dim() == 2, "hidden must be [rows, hidden]");
  TORCH_CHECK(gate_weight.dim() == 2 && up_weight.sizes() == gate_weight.sizes(),
              "gate/up weights must be [intermediate, hidden]");
  TORCH_CHECK(hidden.size(1) == gate_weight.size(1), "hidden size mismatch");
  TORCH_CHECK(gate_cap > 0.0 && up_cap > 0.0, "softcaps must be positive");
  TORCH_CHECK(gate_bias.numel() == 0 || gate_bias.dim() == 1, "gate_bias must be [intermediate]");
  TORCH_CHECK(up_bias.numel() == 0 || up_bias.dim() == 1, "up_bias must be [intermediate]");
  if (gate_bias.numel() != 0) {
    check_input(gate_bias, "gate_bias");
    check_same(hidden, gate_bias, "hidden", "gate_bias");
    TORCH_CHECK(gate_bias.size(0) == gate_weight.size(0), "gate_bias size mismatch");
  }
  if (up_bias.numel() != 0) {
    check_input(up_bias, "up_bias");
    check_same(hidden, up_bias, "hidden", "up_bias");
    TORCH_CHECK(up_bias.size(0) == gate_weight.size(0), "up_bias size mismatch");
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(hidden));
  auto output = torch::empty({hidden.size(0), gate_weight.size(0)}, hidden.options());
  auto gate_pre = torch::empty({hidden.size(0), gate_weight.size(0)},
                               hidden.options().dtype(torch::kFloat32));
  auto up_pre = torch::empty_like(gate_pre);
  auto gate_bias_f32 = gate_bias.numel() == 0 ? torch::Tensor() : gate_bias.to(torch::kFloat32);
  auto up_bias_f32 = up_bias.numel() == 0 ? torch::Tensor() : up_bias.to(torch::kFloat32);
  const int64_t total = hidden.size(0) * gate_weight.size(0);
  if (total == 0) return {output, gate_pre, up_pre};

  const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, hidden.scalar_type(),
                                  "inkling2_expert_gate_up_forward", [&] {
    gate_up_forward_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
        hidden.data_ptr<scalar_t>(), gate_weight.data_ptr<scalar_t>(), up_weight.data_ptr<scalar_t>(),
        gate_bias_f32.numel() == 0 ? nullptr : gate_bias_f32.data_ptr<float>(),
        up_bias_f32.numel() == 0 ? nullptr : up_bias_f32.data_ptr<float>(),
        output.data_ptr<scalar_t>(), gate_pre.data_ptr<float>(), up_pre.data_ptr<float>(),
        hidden.size(0), hidden.size(1), gate_weight.size(0), static_cast<float>(gate_cap),
        static_cast<float>(up_cap));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, gate_pre, up_pre};
}

std::vector<torch::Tensor> inkling2_expert_gate_up_backward(
    torch::Tensor dy,
    torch::Tensor hidden,
    torch::Tensor gate_weight,
    torch::Tensor up_weight,
    torch::Tensor gate_pre,
    torch::Tensor up_pre,
    double gate_cap,
    double up_cap,
    bool need_gate_bias,
    bool need_up_bias) {
  check_input(dy, "dy");
  check_input(hidden, "hidden");
  check_input(gate_weight, "gate_weight");
  check_input(up_weight, "up_weight");
  TORCH_CHECK(gate_pre.is_cuda() && up_pre.is_cuda() && gate_pre.is_contiguous() && up_pre.is_contiguous(),
              "saved preactivations must be contiguous CUDA");
  check_same(dy, hidden, "dy", "hidden");
  check_same(hidden, gate_weight, "hidden", "gate_weight");
  check_same(hidden, up_weight, "hidden", "up_weight");
  TORCH_CHECK(gate_pre.scalar_type() == at::kFloat && up_pre.scalar_type() == at::kFloat,
              "saved preactivations must be float32");
  TORCH_CHECK(dy.sizes() == gate_pre.sizes() && gate_pre.sizes() == up_pre.sizes(),
              "dy and saved preactivations must share shape");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(hidden));
  auto d_hidden = torch::empty_like(hidden);
  auto d_gate_weight = torch::empty_like(gate_weight);
  auto d_up_weight = torch::empty_like(up_weight);
  auto d_gate_bias = need_gate_bias ? torch::empty({gate_weight.size(0)}, hidden.options()) : torch::Tensor();
  auto d_up_bias = need_up_bias ? torch::empty({gate_weight.size(0)}, hidden.options()) : torch::Tensor();
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t hidden_total = hidden.numel();
  const int64_t weight_total = gate_weight.numel();
  const int64_t intermediate_size = gate_weight.size(0);
  const int hidden_blocks = static_cast<int>((hidden_total + kThreads - 1) / kThreads);
  const int weight_blocks = static_cast<int>((weight_total + kThreads - 1) / kThreads);
  const int bias_blocks = static_cast<int>((intermediate_size + kThreads - 1) / kThreads);
  if (hidden_total == 0 || weight_total == 0) {
    return {d_hidden, d_gate_weight, d_up_weight, d_gate_bias, d_up_bias};
  }
  auto gate_bias_workspace = need_gate_bias ? d_gate_bias : torch::empty({intermediate_size}, hidden.options());
  auto up_bias_workspace = need_up_bias ? d_up_bias : torch::empty({intermediate_size}, hidden.options());

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, hidden.scalar_type(),
                                  "inkling2_expert_gate_up_backward", [&] {
    gate_up_backward_hidden_kernel<scalar_t><<<hidden_blocks, kThreads, 0, stream>>>(
        dy.data_ptr<scalar_t>(), gate_weight.data_ptr<scalar_t>(), up_weight.data_ptr<scalar_t>(),
        gate_pre.data_ptr<float>(), up_pre.data_ptr<float>(), d_hidden.data_ptr<scalar_t>(),
        hidden.size(0), hidden.size(1), intermediate_size, static_cast<float>(gate_cap),
        static_cast<float>(up_cap));
    gate_up_backward_weight_kernel<scalar_t><<<weight_blocks, kThreads, 0, stream>>>(
        dy.data_ptr<scalar_t>(), hidden.data_ptr<scalar_t>(), gate_pre.data_ptr<float>(),
        up_pre.data_ptr<float>(), d_gate_weight.data_ptr<scalar_t>(), d_up_weight.data_ptr<scalar_t>(),
        hidden.size(0), hidden.size(1), intermediate_size, static_cast<float>(gate_cap),
        static_cast<float>(up_cap));
    if (need_gate_bias || need_up_bias) {
      auto gate_bias_ptr = gate_bias_workspace.data_ptr<scalar_t>();
      auto up_bias_ptr = up_bias_workspace.data_ptr<scalar_t>();
      gate_up_backward_bias_kernel<scalar_t><<<bias_blocks, kThreads, 0, stream>>>(
          dy.data_ptr<scalar_t>(), gate_pre.data_ptr<float>(), up_pre.data_ptr<float>(), gate_bias_ptr,
          up_bias_ptr, hidden.size(0), intermediate_size, static_cast<float>(gate_cap),
          static_cast<float>(up_cap));
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {d_hidden, d_gate_weight, d_up_weight, d_gate_bias, d_up_bias};
}
