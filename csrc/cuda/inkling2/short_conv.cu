// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// Inkling 2 Mini kernel-size-four causal short convolution (INK-10).
// The tap loop is deliberately fixed and accumulates in FP32.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

constexpr int kKernelSize = 4;
constexpr int kStateSize = 3;
constexpr int kThreads = 256;

template <typename scalar_t>
__global__ void short_conv_forward_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ state,
    scalar_t* __restrict__ output,
    int64_t batch,
    int64_t time,
    int64_t hidden) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = batch * time * hidden;
  if (linear >= total) {
    return;
  }

  const int64_t h = linear % hidden;
  const int64_t t = (linear / hidden) % time;
  const int64_t b = linear / (time * hidden);
  float acc = 0.0f;
#pragma unroll
  for (int tap = 0; tap < kKernelSize; ++tap) {
    const int64_t source_t = t + tap;
    const float source = source_t < kStateSize
        ? static_cast<float>(state[(b * hidden + h) * kStateSize + source_t])
        : static_cast<float>(x[(b * time + source_t - kStateSize) * hidden + h]);
    acc += source * static_cast<float>(weight[h * kKernelSize + tap]);
  }
  output[linear] = static_cast<scalar_t>(acc);
}

template <typename scalar_t>
__global__ void short_conv_state_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ next_state,
    int64_t batch,
    int64_t time,
    int64_t hidden) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = batch * hidden * kStateSize;
  if (linear >= total) {
    return;
  }
  const int64_t state_slot = linear % kStateSize;
  const int64_t h = (linear / kStateSize) % hidden;
  const int64_t b = linear / (hidden * kStateSize);
  const int64_t source_t = time - kStateSize + state_slot;
  next_state[linear] = source_t >= 0
      ? x[(b * time + source_t) * hidden + h]
      : scalar_t(0);
}

template <typename scalar_t>
__global__ void short_conv_backward_dx_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ state,
    scalar_t* __restrict__ grad_x,
    scalar_t* __restrict__ grad_state,
    int64_t batch,
    int64_t time,
    int64_t hidden) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_x = batch * time * hidden;
  const int64_t total_state = batch * kStateSize * hidden;
  if (linear >= total_x + total_state) {
    return;
  }

  const bool is_state = linear >= total_x;
  const int64_t local = is_state ? linear - total_x : linear;
  const int64_t h = local % hidden;
  const int64_t position = (local / hidden) % (is_state ? kStateSize : time);
  const int64_t b = local / (hidden * (is_state ? kStateSize : time));
  float acc = 0.0f;
#pragma unroll
  for (int tap = 0; tap < kKernelSize; ++tap) {
    const int64_t output_t = is_state ? position - tap : position + kStateSize - tap;
    if (output_t >= 0 && output_t < time) {
      acc += static_cast<float>(grad_output[(b * time + output_t) * hidden + h]) *
             static_cast<float>(weight[h * kKernelSize + tap]);
    }
  }
  if (is_state) {
    grad_state[(b * hidden + h) * kStateSize + position] = static_cast<scalar_t>(acc);
  } else {
    grad_x[(b * time + position) * hidden + h] = static_cast<scalar_t>(acc);
  }
}

template <typename scalar_t>
__global__ void short_conv_backward_dw_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ state,
    scalar_t* __restrict__ grad_weight,
    int64_t batch,
    int64_t time,
    int64_t hidden) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = hidden * kKernelSize;
  if (linear >= total) {
    return;
  }
  const int64_t tap = linear % kKernelSize;
  const int64_t h = linear / kKernelSize;
  float acc = 0.0f;
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t t = 0; t < time; ++t) {
      const int64_t source_t = t + tap;
      const float source = source_t < kStateSize
          ? static_cast<float>(state[(b * hidden + h) * kStateSize + source_t])
          : static_cast<float>(x[(b * time + source_t - kStateSize) * hidden + h]);
      acc += static_cast<float>(grad_output[(b * time + t) * hidden + h]) * source;
    }
  }
  grad_weight[linear] = static_cast<scalar_t>(acc);
}

void check_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(
      tensor.scalar_type() == at::kFloat || tensor.scalar_type() == at::kHalf ||
          tensor.scalar_type() == at::kBFloat16,
      name, " must be float32, float16, or bfloat16");
}

void check_common(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& state) {
  check_tensor(x, "x");
  check_tensor(weight, "weight");
  check_tensor(state, "state");
  TORCH_CHECK(x.dim() == 3, "x must have shape [batch, time, hidden]");
  TORCH_CHECK(weight.sizes() == torch::IntArrayRef({x.size(2), kKernelSize}),
              "weight must have shape [hidden, 4]");
  TORCH_CHECK(state.sizes() == torch::IntArrayRef({x.size(0), x.size(2), kStateSize}),
              "state must have shape [batch, hidden, 3]");
  TORCH_CHECK(weight.device() == x.device() && state.device() == x.device(),
              "x, weight, and state must share a CUDA device");
  TORCH_CHECK(weight.scalar_type() == x.scalar_type() && state.scalar_type() == x.scalar_type(),
              "x, weight, and state must share a dtype");
}

torch::Tensor launch_output(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& state) {
  auto output = torch::empty_like(x);
  const int64_t total = x.numel();
  const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, x.scalar_type(), "inkling2_short_conv_forward", [&] {
        short_conv_forward_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
            x.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), state.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(), x.size(0), x.size(1), x.size(2));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace

torch::Tensor inkling2_short_conv_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor state) {
  check_common(x, weight, state);
  c10::cuda::CUDAGuard guard(x.device());
  return launch_output(x, weight, state);
}

std::vector<torch::Tensor> inkling2_short_conv_forward_with_state(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor state) {
  check_common(x, weight, state);
  c10::cuda::CUDAGuard guard(x.device());
  auto output = launch_output(x, weight, state);
  auto next_state = torch::empty_like(state);
  const int64_t total = next_state.numel();
  const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, x.scalar_type(), "inkling2_short_conv_state", [&] {
        short_conv_state_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
            x.data_ptr<scalar_t>(), next_state.data_ptr<scalar_t>(), x.size(0), x.size(1),
            x.size(2));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, next_state};
}

std::vector<torch::Tensor> inkling2_short_conv_backward(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor state) {
  check_common(x, weight, state);
  check_tensor(grad_output, "grad_output");
  TORCH_CHECK(grad_output.sizes() == x.sizes(), "grad_output and x must have matching shapes");
  TORCH_CHECK(grad_output.scalar_type() == x.scalar_type(), "grad_output and x must share dtype");
  c10::cuda::CUDAGuard guard(x.device());
  auto grad_x = torch::empty_like(x);
  auto grad_weight = torch::empty_like(weight);
  auto grad_state = torch::empty_like(state);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t dx_total = x.numel() + state.numel();
  const int dx_blocks = static_cast<int>((dx_total + kThreads - 1) / kThreads);
  const int dw_blocks = static_cast<int>((weight.numel() + kThreads - 1) / kThreads);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, x.scalar_type(), "inkling2_short_conv_backward", [&] {
        short_conv_backward_dx_kernel<scalar_t><<<dx_blocks, kThreads, 0, stream>>>(
            grad_output.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
            state.data_ptr<scalar_t>(), grad_x.data_ptr<scalar_t>(), grad_state.data_ptr<scalar_t>(),
            x.size(0), x.size(1), x.size(2));
        short_conv_backward_dw_kernel<scalar_t><<<dw_blocks, kThreads, 0, stream>>>(
            grad_output.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(), state.data_ptr<scalar_t>(),
            grad_weight.data_ptr<scalar_t>(), x.size(0), x.size(1), x.size(2));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_x, grad_weight, grad_state};
}
