// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
// Inkling 2 Mini isolated visit-state clone (INK-22).

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

constexpr int kThreads = 256;

template <typename scalar_t>
__global__ void visit_state_clone_kernel(
    const scalar_t* __restrict__ source,
    scalar_t* __restrict__ destination,
    int64_t elements) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) destination[index] = source[index];
}

}  // namespace

torch::Tensor inkling2_visit_state_clone(torch::Tensor source) {
  TORCH_CHECK(source.is_cuda() && source.is_contiguous(),
              "visit state must be contiguous CUDA");
  TORCH_CHECK(source.scalar_type() == at::kFloat || source.scalar_type() == at::kHalf ||
                  source.scalar_type() == at::kBFloat16,
              "visit state clone supports float32, float16, and bfloat16");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(source));
  auto destination = torch::empty_like(source);
  if (source.numel() == 0) return destination;
  const int blocks = static_cast<int>((source.numel() + kThreads - 1) / kThreads);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, source.scalar_type(),
                                  "inkling2_visit_state_clone", [&] {
    visit_state_clone_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
        source.data_ptr<scalar_t>(), destination.data_ptr<scalar_t>(), source.numel());
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return destination;
}
