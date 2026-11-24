// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// Inkling 2 Mini deterministic shared-loop weight gradient fold (INK-21), ROCm-native.
// Compiled only for ROCm builds; the CUDA build uses csrc/cuda/inkling2/shared_weight_grad.cu.
//
// Determinism contract: every output element is accumulated by exactly one thread,
// in FP32, as ((0 + g[0]) + g[1]) + g[2] ... over the visit axis in the stored
// (declared) order, then cast once to the storage dtype. There are no atomics, no
// cross-thread reductions and no split-K, so the result is independent of launch
// configuration, of the other rows' contents and of the shard window (a shard's
// rows equal the same rows of the full fold bit for bit). This matches the Python
// reference accumulate_shared_weight_gradients exactly. The visit order is the
// stacking order chosen by the caller; the kernel never reorders visits.
//
// Tuned for MI300X (gfx942, wave64): 256-thread blocks (4 waves), each thread owns
// kVec contiguous output elements so loads/stores are coalesced and vectorizable,
// and the shard offset is folded into the base pointer (no per-element division).

#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPException.h>
#include <c10/hip/HIPGuard.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <torch/extension.h>

#include <vector>

namespace {

constexpr int kThreads = 256;  // multiple of the 64-lane wavefront
constexpr int kVec = 4;        // contiguous output elements per thread
static_assert(kThreads % 64 == 0, "block must be wave64-aligned");

template <typename scalar_t>
__global__ void shared_weight_grad_forward_kernel(
    const scalar_t* __restrict__ gradients,
    scalar_t* __restrict__ output,
    int64_t visits,
    int64_t visit_stride,  // full_rows * elements
    int64_t base_offset,   // shard_start * elements
    int64_t total) {       // output_rows * elements
  const int64_t first = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * kVec;
  if (first >= total) return;
  float acc[kVec];
#pragma unroll
  for (int v = 0; v < kVec; ++v) acc[v] = 0.0f;
  const bool full = first + kVec <= total;
  for (int64_t visit = 0; visit < visits; ++visit) {
    const scalar_t* src = gradients + visit * visit_stride + base_offset + first;
    if (full) {
#pragma unroll
      for (int v = 0; v < kVec; ++v) acc[v] += static_cast<float>(src[v]);
    } else {
      for (int v = 0; v < kVec && first + v < total; ++v) acc[v] += static_cast<float>(src[v]);
    }
  }
  for (int v = 0; v < kVec && first + v < total; ++v) {
    output[first + v] = static_cast<scalar_t>(acc[v]);
  }
}

template <typename scalar_t>
__global__ void shared_weight_grad_backward_kernel(
    const scalar_t* __restrict__ grad_output,
    scalar_t* __restrict__ grad_gradients,
    int64_t visit_stride,  // full_rows * elements
    int64_t lo,            // shard_start * elements
    int64_t hi,            // shard_end * elements
    int64_t total) {       // visits * visit_stride
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) return;
  const int64_t within = index % visit_stride;
  grad_gradients[index] =
      (within >= lo && within < hi) ? grad_output[within - lo] : static_cast<scalar_t>(0.0f);
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
  const int64_t visit_numel = gradients.numel() / visits;
  const int64_t elements = full_rows > 0 ? visit_numel / full_rows : 0;
  std::vector<int64_t> output_shape(gradients.sizes().begin() + 1, gradients.sizes().end());
  output_shape[0] = output_rows;
  const c10::cuda::OptionalCUDAGuard device_guard(device_of(gradients));
  auto output = torch::empty(output_shape, gradients.options());
  if (output.numel() == 0) return output;
  const int64_t total = output.numel();
  const int64_t threads_needed = (total + kVec - 1) / kVec;
  const int blocks = static_cast<int>((threads_needed + kThreads - 1) / kThreads);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, gradients.scalar_type(),
                                  "inkling2_shared_weight_grad_forward", [&] {
    shared_weight_grad_forward_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
        gradients.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), visits, visit_numel,
        shard_start * elements, total);
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
  const c10::cuda::OptionalCUDAGuard device_guard(device_of(grad_output));
  auto grad_gradients = torch::empty({visits, full_rows, elements}, grad_output.options());
  if (grad_gradients.numel() == 0) return grad_gradients;
  const int64_t total = grad_gradients.numel();
  const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, grad_output.scalar_type(),
                                  "inkling2_shared_weight_grad_backward", [&] {
    shared_weight_grad_backward_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
        grad_output.data_ptr<scalar_t>(), grad_gradients.data_ptr<scalar_t>(),
        full_rows * elements, shard_start * elements, shard_end * elements, total);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_gradients;
}
