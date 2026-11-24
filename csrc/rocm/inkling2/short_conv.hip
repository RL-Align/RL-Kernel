// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// Inkling 2 Mini kernel-size-four causal short convolution (INK-10), native
// ROCm implementation. Compiled only for ROCm builds and tuned for MI300X
// (gfx942, wave64); the shared CUDA source is csrc/cuda/inkling2/short_conv.cu.
//
// Determinism contract: forward taps are accumulated in FP32 from the oldest
// to the newest tap in a fixed, unrolled order, exactly like the Python
// reference, and cast once to the storage dtype. Every output element depends
// only on its own channel's history, so results are independent of launch
// configuration, batch size and other rows. The weight gradient is a fixed
// two-level reduction (constant row-group striding, then a fixed-order sum of
// the row-group partials) with no atomics, so it is bitwise reproducible and
// independent of the grid. Exports the same symbols as the CUDA source.

#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPException.h>
#include <c10/hip/HIPGuard.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <vector>
#include <torch/extension.h>

// HIP defaults to -ffp-contract=fast; the tap products must round before the add.
#pragma clang fp contract(off)

namespace {

constexpr int kKernelSize = 4;
constexpr int kStateSize = 3;
constexpr int kWave = 64;
constexpr int kThreads = 256;  // 4 waves
// Weight-gradient reduction geometry: 64 channels x 16 row groups.
constexpr int kDwRowGroups = 16;

template <typename scalar_t>
__device__ __forceinline__ float load_f(const scalar_t* p) {
  return static_cast<float>(*p);
}

// Thread per (b, t, h) with h fastest: every wave reads/writes contiguous channels.
template <typename scalar_t>
__global__ __launch_bounds__(kThreads) void short_conv_forward_kernel(
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
        ? load_f(state + (b * hidden + h) * kStateSize + source_t)
        : load_f(x + (b * time + source_t - kStateSize) * hidden + h);
    // Separate multiply and add keep the reference rounding (no FMA contraction).
    const float prod = __fmul_rn(source, load_f(weight + h * kKernelSize + tap));
    acc = __fadd_rn(acc, prod);
  }
  output[linear] = static_cast<scalar_t>(acc);
}

template <typename scalar_t>
__global__ __launch_bounds__(kThreads) void short_conv_state_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ state,
    scalar_t* __restrict__ next_state,
    int64_t batch,
    int64_t time,
    int64_t hidden) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = batch * hidden * kStateSize;
  if (linear >= total) {
    return;
  }
  const int64_t slot = linear % kStateSize;
  const int64_t h = (linear / kStateSize) % hidden;
  const int64_t b = linear / (hidden * kStateSize);
  // History is [state(3), x(time)]; the new state is its last three entries.
  const int64_t hist = time + slot;
  next_state[linear] = hist < kStateSize
      ? state[(b * hidden + h) * kStateSize + hist]
      : x[(b * time + hist - kStateSize) * hidden + h];
}

template <typename scalar_t>
__global__ __launch_bounds__(kThreads) void short_conv_backward_dx_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ weight,
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
  const int64_t rows = is_state ? kStateSize : time;
  const int64_t h = local % hidden;
  const int64_t position = (local / hidden) % rows;
  const int64_t b = local / (hidden * rows);
  float acc = 0.0f;
#pragma unroll
  for (int tap = 0; tap < kKernelSize; ++tap) {
    const int64_t output_t = is_state ? position - tap : position + kStateSize - tap;
    if (output_t >= 0 && output_t < time) {
      acc = __fadd_rn(
          acc,
          __fmul_rn(
              load_f(grad_output + (b * time + output_t) * hidden + h),
              load_f(weight + h * kKernelSize + tap)));
    }
  }
  if (is_state) {
    // The layouts differ: grad_state is [batch, hidden, 3].
    grad_state[(b * hidden + h) * kStateSize + position] = static_cast<scalar_t>(acc);
  } else {
    grad_x[(b * time + position) * hidden + h] = static_cast<scalar_t>(acc);
  }
}

// Block = 64 channels x kDwRowGroups row groups. Row group g reduces rows
// g, g+G, g+2G, ... of the flattened (batch, time) axis in ascending order,
// then group partials are summed in ascending group order.
template <typename scalar_t>
__global__ __launch_bounds__(kWave* kDwRowGroups) void short_conv_backward_dw_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ state,
    scalar_t* __restrict__ grad_weight,
    int64_t batch,
    int64_t time,
    int64_t hidden) {
  __shared__ float partial[kDwRowGroups][kKernelSize][kWave];
  const int lane = threadIdx.x % kWave;
  const int group = threadIdx.x / kWave;
  const int64_t h = static_cast<int64_t>(blockIdx.x) * kWave + lane;
  float acc[kKernelSize] = {0.f, 0.f, 0.f, 0.f};
  if (h < hidden) {
    const int64_t rows = batch * time;
    for (int64_t r = group; r < rows; r += kDwRowGroups) {
      const int64_t b = r / time;
      const int64_t t = r - b * time;
      const float go = load_f(grad_output + r * hidden + h);
#pragma unroll
      for (int tap = 0; tap < kKernelSize; ++tap) {
        const int64_t source_t = t + tap;
        const float source = source_t < kStateSize
            ? load_f(state + (b * hidden + h) * kStateSize + source_t)
            : load_f(x + (b * time + source_t - kStateSize) * hidden + h);
        acc[tap] = __fadd_rn(acc[tap], __fmul_rn(go, source));
      }
    }
  }
#pragma unroll
  for (int tap = 0; tap < kKernelSize; ++tap) {
    partial[group][tap][lane] = acc[tap];
  }
  __syncthreads();
  if (group == 0 && h < hidden) {
#pragma unroll
    for (int tap = 0; tap < kKernelSize; ++tap) {
      float sum = 0.0f;
#pragma unroll
      for (int g = 0; g < kDwRowGroups; ++g) {
        sum = __fadd_rn(sum, partial[g][tap][lane]);
      }
      grad_weight[h * kKernelSize + tap] = static_cast<scalar_t>(sum);
    }
  }
}

void check_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a ROCm (HIP) tensor");
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
              "x, weight, and state must share a device");
  TORCH_CHECK(weight.scalar_type() == x.scalar_type() && state.scalar_type() == x.scalar_type(),
              "x, weight, and state must share a dtype");
}

inline int blocks_for(int64_t total) {
  return static_cast<int>((total + kThreads - 1) / kThreads);
}

torch::Tensor launch_output(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& state) {
  auto output = torch::empty_like(x);
  const int64_t total = x.numel();
  if (total == 0) {
    return output;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, x.scalar_type(), "inkling2_short_conv_forward", [&] {
        short_conv_forward_kernel<scalar_t><<<blocks_for(total), kThreads, 0, stream>>>(
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
  if (total > 0) {
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, x.scalar_type(), "inkling2_short_conv_state", [&] {
          short_conv_state_kernel<scalar_t><<<blocks_for(total), kThreads, 0, stream>>>(
              x.data_ptr<scalar_t>(), state.data_ptr<scalar_t>(), next_state.data_ptr<scalar_t>(),
              x.size(0), x.size(1), x.size(2));
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
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
  auto grad_weight = torch::zeros_like(weight);
  auto grad_state = torch::zeros_like(state);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t dx_total = x.numel() + state.numel();
  const int64_t hidden = x.size(2);
  if (dx_total > 0 && hidden > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, x.scalar_type(), "inkling2_short_conv_backward", [&] {
          short_conv_backward_dx_kernel<scalar_t><<<blocks_for(dx_total), kThreads, 0, stream>>>(
              grad_output.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
              grad_x.data_ptr<scalar_t>(), grad_state.data_ptr<scalar_t>(), x.size(0), x.size(1),
              hidden);
          const int dw_blocks = static_cast<int>((hidden + kWave - 1) / kWave);
          short_conv_backward_dw_kernel<scalar_t><<<dw_blocks, kWave * kDwRowGroups, 0, stream>>>(
              grad_output.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(), state.data_ptr<scalar_t>(),
              grad_weight.data_ptr<scalar_t>(), x.size(0), x.size(1), hidden);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {grad_x, grad_weight, grad_state};
}
