// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
// Inkling 2 Mini attention O projection and residual (INK-17).

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
__global__ void attention_o_residual_forward_kernel(
    const scalar_t* __restrict__ attention,
    const scalar_t* __restrict__ residual,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int64_t rows,
    int64_t input_size,
    int64_t output_size) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = rows * output_size;
  if (index >= total) return;
  const int64_t row = index / output_size;
  const int64_t output_index = index - row * output_size;
  float value = bias == nullptr ? 0.0f : static_cast<float>(bias[output_index]);
  for (int64_t h = 0; h < input_size; ++h) {
    value += static_cast<float>(attention[row * input_size + h]) *
        static_cast<float>(weight[output_index * input_size + h]);
  }
  value += static_cast<float>(residual[row * output_size + output_index]);
  output[index] = static_cast<scalar_t>(value);
}

template <typename scalar_t>
__global__ void attention_o_residual_backward_attention_kernel(
    const scalar_t* __restrict__ dy,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ d_attention,
    int64_t rows,
    int64_t input_size,
    int64_t output_size) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = rows * input_size;
  if (index >= total) return;
  const int64_t row = index / input_size;
  const int64_t input_index = index - row * input_size;
  float value = 0.0f;
  for (int64_t output_index = 0; output_index < output_size; ++output_index) {
    value += static_cast<float>(dy[row * output_size + output_index]) *
        static_cast<float>(weight[output_index * input_size + input_index]);
  }
  d_attention[index] = static_cast<scalar_t>(value);
}

template <typename scalar_t>
__global__ void attention_o_residual_backward_weight_kernel(
    const scalar_t* __restrict__ dy,
    const scalar_t* __restrict__ attention,
    scalar_t* __restrict__ d_weight,
    int64_t rows,
    int64_t input_size,
    int64_t output_size) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = output_size * input_size;
  if (index >= total) return;
  const int64_t input_index = index % input_size;
  const int64_t output_index = index / input_size;
  float value = 0.0f;
  for (int64_t row = 0; row < rows; ++row) {
    value += static_cast<float>(dy[row * output_size + output_index]) *
        static_cast<float>(attention[row * input_size + input_index]);
  }
  d_weight[index] = static_cast<scalar_t>(value);
}

template <typename scalar_t>
__global__ void attention_o_residual_backward_bias_kernel(
    const scalar_t* __restrict__ dy,
    scalar_t* __restrict__ d_bias,
    int64_t rows,
    int64_t output_size) {
  const int64_t output_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (output_index >= output_size) return;
  float value = 0.0f;
  for (int64_t row = 0; row < rows; ++row) {
    value += static_cast<float>(dy[row * output_size + output_index]);
  }
  d_bias[output_index] = static_cast<scalar_t>(value);
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

torch::Tensor inkling2_attention_o_residual_forward(
    torch::Tensor attention,
    torch::Tensor residual,
    torch::Tensor weight,
    torch::Tensor bias) {
  check_tensor(attention, "attention");
  check_tensor(residual, "residual");
  check_tensor(weight, "weight");
  check_same(attention, residual, "attention", "residual");
  check_same(attention, weight, "attention", "weight");
  TORCH_CHECK(attention.dim() == 2 && residual.sizes() == attention.sizes(),
              "attention and residual must be [rows, hidden]");
  TORCH_CHECK(weight.dim() == 2 && weight.size(1) == attention.size(1) &&
                  weight.size(0) == residual.size(1),
              "weight must be [output, input]");
  TORCH_CHECK(bias.numel() == 0 || bias.is_cuda() && bias.is_contiguous(),
              "bias must be empty or contiguous CUDA");
  if (bias.numel() != 0) {
    check_tensor(bias, "bias");
    check_same(attention, bias, "attention", "bias");
    TORCH_CHECK(bias.dim() == 1 && bias.size(0) == weight.size(0), "bias size mismatch");
  }
  const at::cuda::OptionalCUDAGuard device_guard(device_of(attention));
  auto output = torch::empty({attention.size(0), weight.size(0)}, attention.options());
  if (output.numel() == 0) return output;
  const int blocks = static_cast<int>((output.numel() + kThreads - 1) / kThreads);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, attention.scalar_type(),
                                  "inkling2_attention_o_residual_forward", [&] {
    attention_o_residual_forward_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
        attention.data_ptr<scalar_t>(), residual.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
        bias.numel() == 0 ? nullptr : bias.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
        attention.size(0), attention.size(1), weight.size(0));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::vector<torch::Tensor> inkling2_attention_o_residual_backward(
    torch::Tensor dy,
    torch::Tensor attention,
    torch::Tensor residual,
    torch::Tensor weight,
    bool need_bias) {
  check_tensor(dy, "dy");
  check_tensor(attention, "attention");
  check_tensor(residual, "residual");
  check_tensor(weight, "weight");
  check_same(dy, attention, "dy", "attention");
  check_same(attention, residual, "attention", "residual");
  check_same(attention, weight, "attention", "weight");
  TORCH_CHECK(dy.sizes() == residual.sizes(), "dy and residual must share shape");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(attention));
  auto d_attention = torch::empty_like(attention);
  auto d_residual = torch::empty_like(residual);
  auto d_weight = torch::empty_like(weight);
  auto d_bias = need_bias ? torch::empty({weight.size(0)}, attention.options()) : torch::Tensor();
  const int64_t input_total = attention.numel();
  const int64_t weight_total = weight.numel();
  if (input_total == 0 || weight_total == 0) return {d_attention, d_residual, d_weight, d_bias};
  const int input_blocks = static_cast<int>((input_total + kThreads - 1) / kThreads);
  const int weight_blocks = static_cast<int>((weight_total + kThreads - 1) / kThreads);
  const int bias_blocks = static_cast<int>((weight.size(0) + kThreads - 1) / kThreads);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, attention.scalar_type(),
                                  "inkling2_attention_o_residual_backward", [&] {
    attention_o_residual_backward_attention_kernel<scalar_t><<<input_blocks, kThreads, 0, stream>>>(
        dy.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), d_attention.data_ptr<scalar_t>(),
        attention.size(0), attention.size(1), weight.size(0));
    d_residual.copy_(dy);
    attention_o_residual_backward_weight_kernel<scalar_t><<<weight_blocks, kThreads, 0, stream>>>(
        dy.data_ptr<scalar_t>(), attention.data_ptr<scalar_t>(), d_weight.data_ptr<scalar_t>(),
        attention.size(0), attention.size(1), weight.size(0));
    if (need_bias) {
      attention_o_residual_backward_bias_kernel<scalar_t><<<bias_blocks, kThreads, 0, stream>>>(
          dy.data_ptr<scalar_t>(), d_bias.data_ptr<scalar_t>(), attention.size(0), weight.size(0));
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {d_attention, d_residual, d_weight, d_bias};
}
