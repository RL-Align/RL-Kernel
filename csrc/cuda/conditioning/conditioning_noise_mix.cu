// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace {

constexpr int kThreads = 256;

// Explicit PTX rounding without .ftz preserves this operator's arithmetic
// contract even when the shared extension is built with --use_fast_math.
__device__ __forceinline__ float add_rn(float a, float b) {
  float result;
  asm("add.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
  return result;
}

__device__ __forceinline__ float sub_rn(float a, float b) {
  float result;
  asm("sub.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
  return result;
}

__device__ __forceinline__ float mul_rn(float a, float b) {
  float result;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
  return result;
}

template <typename scalar_t>
__device__ __forceinline__ float rounded(float value) {
  return static_cast<float>(static_cast<scalar_t>(value));
}

template <typename scalar_t>
__global__ void mix_forward_kernel(
    const scalar_t* sample,
    const scalar_t* timestep,
    const scalar_t* noise,
    scalar_t* out,
    int64_t count,
    int64_t row_size,
    bool scalar_time) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  const float t = static_cast<float>(timestep[scalar_time ? 0 : i / row_size]);
  // Preserve each eager PyTorch dtype boundary, including BF16/FP16.
  const float u = rounded<scalar_t>(sub_rn(1.0f, t));
  const float clean = rounded<scalar_t>(mul_rn(t, static_cast<float>(sample[i])));
  const float random = rounded<scalar_t>(mul_rn(u, static_cast<float>(noise[i])));
  out[i] = static_cast<scalar_t>(add_rn(clean, random));
}

template <typename scalar_t>
__global__ void mix_backward_kernel(
    const scalar_t* grad,
    const scalar_t* timestep,
    scalar_t* d_sample,
    scalar_t* d_noise,
    int64_t count,
    int64_t row_size,
    bool scalar_time) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  const float t = static_cast<float>(timestep[scalar_time ? 0 : i / row_size]);
  const float u = rounded<scalar_t>(sub_rn(1.0f, t));
  const float g = static_cast<float>(grad[i]);
  d_sample[i] = static_cast<scalar_t>(mul_rn(g, t));
  d_noise[i] = static_cast<scalar_t>(mul_rn(g, u));
}

void check_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(!tensor.is_neg(), name, " must resolve the negative view bit before native access");
  TORCH_CHECK(
      tensor.scalar_type() == at::kFloat || tensor.scalar_type() == at::kHalf ||
          tensor.scalar_type() == at::kBFloat16,
      name, " must be fp32, fp16, or bf16");
}

void check_timestep(const torch::Tensor& sample, const torch::Tensor& timestep) {
  check_tensor(sample, "sample/grad");
  check_tensor(timestep, "timestep");
  TORCH_CHECK(sample.dim() >= 1 && sample.numel() > 0,
              "sample/grad must be non-empty with a batch dimension");
  TORCH_CHECK(timestep.dim() == 1 &&
                  (timestep.numel() == 1 || timestep.numel() == sample.size(0)),
              "timestep must be a scalar vector or one value per batch sample");
  TORCH_CHECK(timestep.device() == sample.device(), "timestep must share sample/grad device");
  TORCH_CHECK(timestep.scalar_type() == sample.scalar_type(),
              "timestep must share sample/grad dtype");
  TORCH_CHECK(!timestep.requires_grad(), "timestep gradients are unsupported");
}

}  // namespace

torch::Tensor conditioning_noise_mix_forward(
    torch::Tensor sample, torch::Tensor timestep, torch::Tensor noise) {
  check_timestep(sample, timestep);
  check_tensor(noise, "noise");
  TORCH_CHECK(sample.device() == noise.device(), "sample and noise must share device");
  TORCH_CHECK(sample.sizes() == noise.sizes(), "sample and noise must share shape");
  TORCH_CHECK(sample.scalar_type() == noise.scalar_type(), "sample and noise must share dtype");
  const c10::cuda::CUDAGuard guard(sample.device());
  auto out = torch::empty_like(sample);
  const int64_t count = sample.numel();
  const int64_t blocks = (count + kThreads - 1) / kThreads;
  TORCH_CHECK(blocks <= 2147483647, "tensor is too large for the launch grid");
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, sample.scalar_type(), "conditioning_noise_mix_forward", [&] {
        mix_forward_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
            sample.data_ptr<scalar_t>(), timestep.data_ptr<scalar_t>(), noise.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(), count, count / sample.size(0), timestep.numel() == 1);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::vector<torch::Tensor> conditioning_noise_mix_backward(
    torch::Tensor grad, torch::Tensor timestep) {
  check_timestep(grad, timestep);
  const c10::cuda::CUDAGuard guard(grad.device());
  auto d_sample = torch::empty_like(grad);
  auto d_noise = torch::empty_like(grad);
  const int64_t count = grad.numel();
  const int64_t blocks = (count + kThreads - 1) / kThreads;
  TORCH_CHECK(blocks <= 2147483647, "tensor is too large for the launch grid");
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, grad.scalar_type(), "conditioning_noise_mix_backward", [&] {
        mix_backward_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
            grad.data_ptr<scalar_t>(), timestep.data_ptr<scalar_t>(),
            d_sample.data_ptr<scalar_t>(), d_noise.data_ptr<scalar_t>(),
            count, count / grad.size(0), timestep.numel() == 1);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {d_sample, d_noise};
}
