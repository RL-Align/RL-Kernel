// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// T06 arithmetic prototype: dweights[T,6] -> ds[T,256].
// This is NOT the p3-op-abi.v4 provider. Sealed-state validation, status/echo,
// and official oracle integration belong to the pending T01 start kit.

#include <torch/extension.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace {

constexpr int kSlots = 6;
constexpr int kExperts = 256;

__global__ void route_backward_kernel(
    const float* __restrict__ dweights,
    const int32_t* __restrict__ ids,
    const float* __restrict__ p,
    const float* __restrict__ z,
    const bool* __restrict__ row_active,
    float* __restrict__ ds) {
  // One block owns one token; thread e owns ds[token, e] at 256 threads.
  const int64_t token = blockIdx.x;
  const int thread = threadIdx.x;
  float* output_row = ds + token * kExperts;

  // This branch is uniform across the block. Do not read padding's saved data:
  // it may contain sentinel IDs, Z=0, or NaN, none of which enters arithmetic.
  if (!row_active[token]) {
    for (int expert = thread; expert < kExperts; expert += blockDim.x) {
      output_row[expert] = 0.0f;
    }
    return;
  }

  __shared__ float da[kSlots];
  __shared__ int32_t selected[kSlots];

  // Only six terms: spell out the prescribed tree rather than a warp reduce.
  // Explicit RN intrinsics also prevent multiplying g*p and adding from
  // contracting into FMA (which would round once instead of twice).
  if (thread == 0) {
    const int64_t offset = token * kSlots;
    float g[kSlots];
    float gp[kSlots];
    #pragma unroll
    for (int slot = 0; slot < kSlots; ++slot) {
      g[slot] = dweights[offset + slot];
      gp[slot] = __fmul_rn(g[slot], p[offset + slot]);
      selected[slot] = ids[offset + slot];
    }
    const float c01 = __fadd_rn(gp[0], gp[1]);
    const float c23 = __fadd_rn(gp[2], gp[3]);
    const float c45 = __fadd_rn(gp[4], gp[5]);
    const float c = __fadd_rn(__fadd_rn(c01, c23), c45);
    const float scale_over_z = __fdiv_rn(1.5f, z[token]);
    #pragma unroll
    for (int slot = 0; slot < kSlots; ++slot) {
      da[slot] = __fmul_rn(scale_over_z, __fsub_rn(g[slot], c));
    }
  }
  __syncthreads();  // All warps can now read the six gradients and expert IDs.

  // Gather the matching slots into each output instead of racing to scatter.
  // Every output has one writer, including zeros. Never read the old buffer.
  for (int expert = thread; expert < kExperts; expert += blockDim.x) {
    float acc = 0.0f;
    #pragma unroll
    for (int slot = 0; slot < kSlots; ++slot) {
      if (selected[slot] == expert) {
        acc = __fadd_rn(acc, da[slot]);  // Duplicate IDs: slot 0 -> 5.
      }
    }
    output_row[expert] = acc == 0.0f ? 0.0f : acc;
  }
}

void check_tensor(
    const torch::Tensor& tensor,
    at::ScalarType dtype,
    at::IntArrayRef shape,
    const c10::Device& device,
    const char* name) {
  TORCH_CHECK(tensor.is_cuda() && tensor.device() == device,
              name, " must be on the same CUDA device as dweights");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has incorrect dtype");
  TORCH_CHECK(tensor.sizes() == shape, name, " has incorrect shape");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

// Internal experiment entry point. The Python harness checks active values.
// Value checks synchronize and are deliberately outside the arithmetic kernel.
void route_backward_core_out(
    const torch::Tensor& dweights,
    const torch::Tensor& ids,
    const torch::Tensor& p,
    const torch::Tensor& z,
    const torch::Tensor& row_active,
    torch::Tensor out,
    int64_t threads) {
  TORCH_CHECK(dweights.is_cuda(), "dweights must be CUDA");
  TORCH_CHECK(dweights.dim() == 2, "dweights must have shape [T, 6]");
  TORCH_CHECK(threads == 128 || threads == 256, "threads must be 128 or 256");
  const int64_t tokens = dweights.size(0);
  const auto device = dweights.device();
  check_tensor(dweights, at::kFloat, {tokens, kSlots}, device, "dweights");
  check_tensor(ids, at::kInt, {tokens, kSlots}, device, "ids");
  check_tensor(p, at::kFloat, {tokens, kSlots}, device, "p");
  check_tensor(z, at::kFloat, {tokens}, device, "z");
  check_tensor(row_active, at::kBool, {tokens}, device, "row_active");
  check_tensor(out, at::kFloat, {tokens, kExperts}, device, "out");
  for (const auto& input : {dweights, ids, p, z, row_active}) {
    at::assert_no_overlap(out, input);
  }
  if (tokens == 0) {
    return;
  }

  const c10::cuda::CUDAGuard guard(device);
  TORCH_CHECK(tokens <= at::cuda::getCurrentDeviceProperties()->maxGridSize[0],
              "T exceeds the one-block-per-token grid limit");
  const auto stream = at::cuda::getCurrentCUDAStream();
  route_backward_kernel<<<static_cast<unsigned int>(tokens), threads, 0, stream>>>(
      dweights.data_ptr<float>(), ids.data_ptr<int32_t>(), p.data_ptr<float>(),
      z.data_ptr<float>(), row_active.data_ptr<bool>(), out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
