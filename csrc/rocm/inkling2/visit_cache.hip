// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// Inkling 2 Mini isolated visit-state clone (INK-22), ROCm-native.
// Compiled only for ROCm builds; the CUDA build uses csrc/cuda/inkling2/visit_cache.cu.
//
// Determinism contract: the clone is an exact byte copy (no arithmetic, no dtype
// conversion), so NaN payloads, -0.0 and denormals survive bit for bit. The
// destination is a fresh allocation, so a cached visit state can never alias or
// be contaminated by another visit, pass, or the caller's live tensor.
//
// Tuned for MI300X (gfx942, wave64): 16-byte vector copies with 256-thread blocks
// (4 waves); a scalar byte tail handles sizes that are not a multiple of 16 and
// pointers that are not 16-byte aligned.

#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPException.h>
#include <c10/hip/HIPGuard.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <torch/extension.h>

namespace {

constexpr int kThreads = 256;  // multiple of the 64-lane wavefront
static_assert(kThreads % 64 == 0, "block must be wave64-aligned");

struct alignas(16) Vec16 {
  uint32_t x, y, z, w;
};

__global__ void visit_state_copy_vec_kernel(const Vec16* __restrict__ source,
                                            Vec16* __restrict__ destination,
                                            int64_t vectors) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < vectors) destination[index] = source[index];
}

__global__ void visit_state_copy_bytes_kernel(const uint8_t* __restrict__ source,
                                              uint8_t* __restrict__ destination,
                                              int64_t bytes) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < bytes) destination[index] = source[index];
}

inline int blocks_for(int64_t items) {
  return static_cast<int>((items + kThreads - 1) / kThreads);
}

}  // namespace

torch::Tensor inkling2_visit_state_clone(torch::Tensor source) {
  TORCH_CHECK(source.is_cuda() && source.is_contiguous(),
              "visit state must be contiguous CUDA");
  TORCH_CHECK(source.scalar_type() == at::kFloat || source.scalar_type() == at::kHalf ||
                  source.scalar_type() == at::kBFloat16,
              "visit state clone supports float32, float16, and bfloat16");
  const c10::cuda::OptionalCUDAGuard device_guard(device_of(source));
  auto destination = torch::empty_like(source);
  if (source.numel() == 0) return destination;
  const int64_t bytes = source.numel() * source.element_size();
  const auto* src = static_cast<const uint8_t*>(source.data_ptr());
  auto* dst = static_cast<uint8_t*>(destination.data_ptr());
  auto stream = at::cuda::getCurrentCUDAStream();

  int64_t vector_bytes = 0;
  if ((reinterpret_cast<uintptr_t>(src) % 16 == 0) &&
      (reinterpret_cast<uintptr_t>(dst) % 16 == 0)) {
    const int64_t vectors = bytes / 16;
    vector_bytes = vectors * 16;
    if (vectors > 0) {
      visit_state_copy_vec_kernel<<<blocks_for(vectors), kThreads, 0, stream>>>(
          reinterpret_cast<const Vec16*>(src), reinterpret_cast<Vec16*>(dst), vectors);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }
  const int64_t tail = bytes - vector_bytes;
  if (tail > 0) {
    visit_state_copy_bytes_kernel<<<blocks_for(tail), kThreads, 0, stream>>>(
        src + vector_bytes, dst + vector_bytes, tail);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return destination;
}
