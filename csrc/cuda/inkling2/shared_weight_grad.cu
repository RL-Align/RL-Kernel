// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
// Inkling 2 Mini deterministic shared-loop weight gradient fold (INK-21).

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
__global__ void shared_weight_grad_forward_kernel(
    const scalar_t* __restrict__ gradients,
    scalar_t* __restrict__ output,
    int64_t visits,
    int64_t elements,
    int64_t full_rows,
    int64_t shard_start,
    int64_t output_rows) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = output_rows * elements;
  if (index >= total) return;
  const int64_t row = index / elements;
  const int64_t element = index - row * elements;
  float value = 0.0f;
  for (int64_t visit = 0; visit < visits; ++visit) {
    value += static_cast<float>(gradients[(visit * full_rows + row + shard_start) * elements + element]);
  }
  output[index] = static_cast<scalar_t>(value);
}

template <typename scalar_t>
__global__ void shared_weight_grad_backward_kernel(
    const scalar_t* __restrict__ grad_output,
    scalar_t* __restrict__ grad_gradients,
    int64_t visits,
    int64_t elements,
    int64_t shard_start,
    int64_t output_rows,
    int64_t full_rows) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = visits * full_rows * elements;
  if (index >= total) return;
  const int64_t element = index % elements;
  const int64_t row = (index / elements) % full_rows;
  if (row < shard_start || row >= shard_start + output_rows) {
    grad_gradients[index] = static_cast<scalar_t>(0.0f);
    return;
  }
  const int64_t output_index = (row - shard_start) * elements + element;
  grad_gradients[index] = grad_output[output_index];
}

static void check_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous(), name, " must be contiguous CUDA");
  TORCH_CHECK(tensor.scalar_type() == at::kFloat || tensor.scalar_type() == at::kHalf ||
                  tensor.scalar_type() == at::kBFloat16,
              name, " must be float32, float16, or bfloat16");
}

}  // namespace

torch::Tensor inkling2_shared_weight_grad_forward(
    torch::Tensor gradients,
    int64_t shard_start,
    int64_t shard_end) {
  check_tensor(gradients, "gradients");
  TORCH_CHECK(gradients.dim() >= 2, "gradients must be [visits, ...]");
  TORCH_CHECK(gradients.size(0) > 0, "gradients must contain at least one visit");
  const int64_t visits = gradients.size(0);
  const int64_t full_rows = gradients.size(1);
  TORCH_CHECK(shard_start >= 0 && shard_end >= shard_start && shard_end <= full_rows,
              "invalid shard range");
  const int64_t output_rows = shard_end - shard_start;
  const int64_t elements = gradients.numel() / (visits * full_rows);
  std::vector<int64_t> output_shape(gradients.sizes().begin() + 1, gradients.sizes().end());
  output_shape[0] = output_rows;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(gradients));
  auto output = torch::empty(output_shape, gradients.options());
  if (output.numel() == 0) return output;
  const int blocks = static_cast<int>((output.numel() + kThreads - 1) / kThreads);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, gradients.scalar_type(),
                                  "inkling2_shared_weight_grad_forward", [&] {
    shared_weight_grad_forward_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
        gradients.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), visits, elements, full_rows, shard_start,
        output_rows);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor inkling2_shared_weight_grad_backward(
    torch::Tensor grad_output,
    int64_t visits,
    int64_t full_rows,
    int64_t elements,
    int64_t shard_start,
    int64_t shard_end) {
  check_tensor(grad_output, "grad_output");
  TORCH_CHECK(visits > 0 && full_rows >= 0 && elements >= 0, "invalid shared gradient shape");
  TORCH_CHECK(shard_start >= 0 && shard_end >= shard_start && shard_end <= full_rows,
              "invalid shard range");
  TORCH_CHECK(grad_output.numel() == (shard_end - shard_start) * elements,
              "grad_output shape mismatch");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(grad_output));
  auto grad_gradients = torch::empty({visits, full_rows, elements}, grad_output.options());
  if (grad_gradients.numel() == 0) return grad_gradients;
  const int blocks = static_cast<int>((grad_gradients.numel() + kThreads - 1) / kThreads);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, grad_output.scalar_type(),
                                  "inkling2_shared_weight_grad_backward", [&] {
    shared_weight_grad_backward_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
        grad_output.data_ptr<scalar_t>(), grad_gradients.data_ptr<scalar_t>(), visits, elements,
        shard_start, shard_end - shard_start, full_rows);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_gradients;
}
