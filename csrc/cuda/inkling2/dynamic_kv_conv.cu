// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
// Inkling 2 Mini dynamic K/V convolution (INK-16).

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
constexpr int kKVStreams = 2;
constexpr int kThreads = 256;

template <typename scalar_t>
__device__ __forceinline__ float input_at(
    const scalar_t* x, const scalar_t* state, int64_t b, int64_t t, int64_t s, int64_t h,
    int64_t time, int64_t hidden) {
  if (t < kStateSize) {
    return static_cast<float>(state[((b * kKVStreams + s) * hidden + h) * kStateSize + t]);
  }
  return static_cast<float>(x[((b * time + t - kStateSize) * kKVStreams + s) * hidden + h]);
}

template <typename scalar_t>
__global__ void forward_kernel(
    const scalar_t* x, const scalar_t* coefficients, const scalar_t* state, scalar_t* output,
    int64_t batch, int64_t time, int64_t hidden) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = batch * time * kKVStreams * hidden;
  if (linear >= total) return;
  const int64_t h = linear % hidden;
  const int64_t s = (linear / hidden) % kKVStreams;
  const int64_t t = (linear / (hidden * kKVStreams)) % time;
  const int64_t b = linear / (hidden * kKVStreams * time);
  float acc = 0.0f;
#pragma unroll
  for (int tap = 0; tap < kKernelSize; ++tap) {
    const float source = input_at(x, state, b, t + tap, s, h, time, hidden);
    const int64_t coeff = (((b * time + t) * kKVStreams + s) * hidden + h) * kKernelSize + tap;
    acc += source * static_cast<float>(coefficients[coeff]);
  }
  output[linear] = static_cast<scalar_t>(acc);
}

template <typename scalar_t>
__global__ void state_kernel(const scalar_t* x, scalar_t* next_state, int64_t batch, int64_t time,
                             int64_t hidden) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = batch * kKVStreams * hidden * kStateSize;
  if (linear >= total) return;
  const int64_t p = linear % kStateSize;
  const int64_t h = (linear / kStateSize) % hidden;
  const int64_t s = (linear / (kStateSize * hidden)) % kKVStreams;
  const int64_t b = linear / (kStateSize * hidden * kKVStreams);
  const int64_t t = time - kStateSize + p;
  next_state[linear] = t >= 0
      ? x[((b * time + t) * kKVStreams + s) * hidden + h]
      : scalar_t(0);
}

template <typename scalar_t>
__global__ void backward_x_state_kernel(
    const scalar_t* grad_output, const scalar_t* coefficients, scalar_t* grad_x,
    scalar_t* grad_state, int64_t batch, int64_t time, int64_t hidden) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t x_total = batch * time * kKVStreams * hidden;
  const int64_t state_total = batch * kKVStreams * hidden * kStateSize;
  if (linear >= x_total + state_total) return;
  const bool is_state = linear >= x_total;
  const int64_t local = is_state ? linear - x_total : linear;
  const int64_t h = local % hidden;
  const int64_t s = (local / hidden) % kKVStreams;
  const int64_t position = (local / (hidden * kKVStreams)) % (is_state ? kStateSize : time);
  const int64_t b = local / (hidden * kKVStreams * (is_state ? kStateSize : time));
  float acc = 0.0f;
#pragma unroll
  for (int tap = 0; tap < kKernelSize; ++tap) {
    const int64_t output_t = is_state ? position - tap : position + kStateSize - tap;
    if (output_t >= 0 && output_t < time) {
      const int64_t out = ((b * time + output_t) * kKVStreams + s) * hidden + h;
      const int64_t coeff = (((b * time + output_t) * kKVStreams + s) * hidden + h) * kKernelSize + tap;
      acc += static_cast<float>(grad_output[out]) * static_cast<float>(coefficients[coeff]);
    }
  }
  if (is_state) {
    grad_state[((b * kKVStreams + s) * hidden + h) * kStateSize + position] = static_cast<scalar_t>(acc);
  } else {
    grad_x[((b * time + position) * kKVStreams + s) * hidden + h] = static_cast<scalar_t>(acc);
  }
}

template <typename scalar_t>
__global__ void backward_coeff_kernel(
    const scalar_t* grad_output, const scalar_t* x, const scalar_t* state, scalar_t* grad_coeff,
    int64_t batch, int64_t time, int64_t hidden) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = batch * time * kKVStreams * hidden * kKernelSize;
  if (linear >= total) return;
  const int tap = linear % kKernelSize;
  const int64_t h = (linear / kKernelSize) % hidden;
  const int64_t s = (linear / (kKernelSize * hidden)) % kKVStreams;
  const int64_t t = (linear / (kKernelSize * hidden * kKVStreams)) % time;
  const int64_t b = linear / (kKernelSize * hidden * kKVStreams * time);
  const int64_t out = ((b * time + t) * kKVStreams + s) * hidden + h;
  grad_coeff[linear] = static_cast<scalar_t>(
      static_cast<float>(grad_output[out]) * input_at(x, state, b, t + tap, s, h, time, hidden));
}

void check_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == at::kFloat || tensor.scalar_type() == at::kHalf ||
                  tensor.scalar_type() == at::kBFloat16,
              name, " must be float32, float16, or bfloat16");
}

void check_common(const torch::Tensor& x, const torch::Tensor& coefficients,
                  const torch::Tensor& state) {
  check_tensor(x, "x");
  check_tensor(coefficients, "coefficients");
  check_tensor(state, "state");
  TORCH_CHECK(x.dim() == 4 && x.size(2) == kKVStreams,
              "x must have shape [batch, time, 2, hidden]");
  TORCH_CHECK(coefficients.sizes() == torch::IntArrayRef(
                  {x.size(0), x.size(1), kKVStreams, x.size(3), kKernelSize}),
              "coefficients must have shape [batch, time, 2, hidden, 4]");
  TORCH_CHECK(state.sizes() == torch::IntArrayRef(
                  {x.size(0), kKVStreams, x.size(3), kStateSize}),
              "state must have shape [batch, 2, hidden, 3]");
  TORCH_CHECK(x.device() == coefficients.device() && x.device() == state.device(),
              "x, coefficients, and state must share a CUDA device");
  TORCH_CHECK(x.scalar_type() == coefficients.scalar_type() && x.scalar_type() == state.scalar_type(),
              "x, coefficients, and state must share a dtype");
}

}  // namespace

torch::Tensor inkling2_dynamic_kv_conv_forward(torch::Tensor x, torch::Tensor coefficients,
                                                torch::Tensor state) {
  check_common(x, coefficients, state);
  c10::cuda::CUDAGuard guard(x.device());
  auto output = torch::empty_like(x);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int blocks = static_cast<int>((x.numel() + kThreads - 1) / kThreads);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(),
                                  "inkling2_dynamic_kv_conv_forward", [&] {
    forward_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
        x.data_ptr<scalar_t>(), coefficients.data_ptr<scalar_t>(), state.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(), x.size(0), x.size(1), x.size(3));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::vector<torch::Tensor> inkling2_dynamic_kv_conv_forward_with_state(
    torch::Tensor x, torch::Tensor coefficients, torch::Tensor state) {
  check_common(x, coefficients, state);
  c10::cuda::CUDAGuard guard(x.device());
  auto output = inkling2_dynamic_kv_conv_forward(x, coefficients, state);
  auto next_state = torch::empty_like(state);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int blocks = static_cast<int>((next_state.numel() + kThreads - 1) / kThreads);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(),
                                  "inkling2_dynamic_kv_conv_state", [&] {
    state_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
        x.data_ptr<scalar_t>(), next_state.data_ptr<scalar_t>(), x.size(0), x.size(1), x.size(3));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, next_state};
}

std::vector<torch::Tensor> inkling2_dynamic_kv_conv_backward(
    torch::Tensor grad_output, torch::Tensor x, torch::Tensor coefficients, torch::Tensor state) {
  check_common(x, coefficients, state);
  check_tensor(grad_output, "grad_output");
  TORCH_CHECK(grad_output.sizes() == x.sizes(), "grad_output and x must have matching shapes");
  TORCH_CHECK(grad_output.scalar_type() == x.scalar_type(), "grad_output and x must share dtype");
  c10::cuda::CUDAGuard guard(x.device());
  auto grad_x = torch::empty_like(x);
  auto grad_coefficients = torch::empty_like(coefficients);
  auto grad_state = torch::empty_like(state);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int dx_blocks = static_cast<int>((x.numel() + state.numel() + kThreads - 1) / kThreads);
  const int dc_blocks = static_cast<int>((coefficients.numel() + kThreads - 1) / kThreads);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(),
                                  "inkling2_dynamic_kv_conv_backward", [&] {
    backward_x_state_kernel<scalar_t><<<dx_blocks, kThreads, 0, stream>>>(
        grad_output.data_ptr<scalar_t>(), coefficients.data_ptr<scalar_t>(), grad_x.data_ptr<scalar_t>(),
        grad_state.data_ptr<scalar_t>(), x.size(0), x.size(1), x.size(3));
    backward_coeff_kernel<scalar_t><<<dc_blocks, kThreads, 0, stream>>>(
        grad_output.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(), state.data_ptr<scalar_t>(),
        grad_coefficients.data_ptr<scalar_t>(), x.size(0), x.size(1), x.size(3));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_x, grad_coefficients, grad_state};
}
