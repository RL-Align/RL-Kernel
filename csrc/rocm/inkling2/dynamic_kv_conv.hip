// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// Inkling 2 Mini dynamic (token-dependent) K/V convolution (INK-16), native
// ROCm implementation. Compiled only for ROCm builds and tuned for MI300X
// (gfx942, wave64); the shared CUDA source is csrc/cuda/inkling2/dynamic_kv_conv.cu.
//
// Determinism contract: each output element is the FP32 sum of four tap
// products accumulated from the oldest to the newest tap in a fixed, unrolled
// order (multiply and add rounded separately, no FMA contraction), cast once to
// the storage dtype, matching the Python reference bit for bit. Coefficients are
// per token, so no gradient reduction crosses threads: every gradient element is
// an independent fixed-order sum, there are no atomics, and results do not depend
// on launch configuration, batch size or other rows. Exports the same symbols as
// the CUDA source.

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

// The empty asm pins the rounded product in a register so the backend cannot fold the
// following add into an FMA (HIP maps __fmul_rn to a plain multiply).
__device__ __forceinline__ float mul_nofuse(float a, float b) {
  float p = a * b;
  asm volatile("" : "+v"(p));
  return p;
}


constexpr int kKernelSize = 4;
constexpr int kStateSize = 3;
constexpr int kKVStreams = 2;
constexpr int kThreads = 256;  // 4 waves of 64

template <typename scalar_t>
__device__ __forceinline__ float input_at(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ state,
    int64_t b,
    int64_t t,
    int64_t s,
    int64_t h,
    int64_t time,
    int64_t hidden) {
  if (t < kStateSize) {
    return static_cast<float>(state[((b * kKVStreams + s) * hidden + h) * kStateSize + t]);
  }
  return static_cast<float>(x[((b * time + t - kStateSize) * kKVStreams + s) * hidden + h]);
}

// Thread per (b, t, s, h) with h fastest so waves access contiguous channels.
template <typename scalar_t>
__global__ __launch_bounds__(kThreads) void forward_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ coefficients,
    const scalar_t* __restrict__ state,
    scalar_t* __restrict__ output,
    int64_t batch,
    int64_t time,
    int64_t hidden) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = batch * time * kKVStreams * hidden;
  if (linear >= total) {
    return;
  }
  const int64_t h = linear % hidden;
  const int64_t s = (linear / hidden) % kKVStreams;
  const int64_t t = (linear / (hidden * kKVStreams)) % time;
  const int64_t b = linear / (hidden * kKVStreams * time);
  const scalar_t* coeff = coefficients + linear * kKernelSize;
  float acc = 0.0f;
#pragma unroll
  for (int tap = 0; tap < kKernelSize; ++tap) {
    const float source = input_at(x, state, b, t + tap, s, h, time, hidden);
    acc = __fadd_rn(acc, mul_nofuse(source, static_cast<float>(coeff[tap])));
  }
  output[linear] = static_cast<scalar_t>(acc);
}

template <typename scalar_t>
__global__ __launch_bounds__(kThreads) void state_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ state,
    scalar_t* __restrict__ next_state,
    int64_t batch,
    int64_t time,
    int64_t hidden) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = batch * kKVStreams * hidden * kStateSize;
  if (linear >= total) {
    return;
  }
  const int64_t p = linear % kStateSize;
  const int64_t h = (linear / kStateSize) % hidden;
  const int64_t s = (linear / (kStateSize * hidden)) % kKVStreams;
  const int64_t b = linear / (kStateSize * hidden * kKVStreams);
  // History is [state(3), x(time)]; the new state is its last three entries.
  const int64_t hist = time + p;
  next_state[linear] = hist < kStateSize
      ? state[((b * kKVStreams + s) * hidden + h) * kStateSize + hist]
      : x[((b * time + hist - kStateSize) * kKVStreams + s) * hidden + h];
}

template <typename scalar_t>
__global__ __launch_bounds__(kThreads) void backward_x_state_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ coefficients,
    scalar_t* __restrict__ grad_x,
    scalar_t* __restrict__ grad_state,
    int64_t batch,
    int64_t time,
    int64_t hidden) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t x_total = batch * time * kKVStreams * hidden;
  const int64_t state_total = batch * kKVStreams * hidden * kStateSize;
  if (linear >= x_total + state_total) {
    return;
  }
  const bool is_state = linear >= x_total;
  const int64_t local = is_state ? linear - x_total : linear;
  const int64_t rows = is_state ? kStateSize : time;
  const int64_t h = local % hidden;
  const int64_t s = (local / hidden) % kKVStreams;
  const int64_t position = (local / (hidden * kKVStreams)) % rows;
  const int64_t b = local / (hidden * kKVStreams * rows);
  float acc = 0.0f;
#pragma unroll
  for (int tap = 0; tap < kKernelSize; ++tap) {
    const int64_t output_t = is_state ? position - tap : position + kStateSize - tap;
    if (output_t >= 0 && output_t < time) {
      const int64_t out = ((b * time + output_t) * kKVStreams + s) * hidden + h;
      acc = __fadd_rn(
          acc,
          mul_nofuse(
              static_cast<float>(grad_output[out]),
              static_cast<float>(coefficients[out * kKernelSize + tap])));
    }
  }
  if (is_state) {
    // grad_state is [batch, 2, hidden, 3], which differs from the x layout.
    grad_state[((b * kKVStreams + s) * hidden + h) * kStateSize + position] =
        static_cast<scalar_t>(acc);
  } else {
    grad_x[((b * time + position) * kKVStreams + s) * hidden + h] = static_cast<scalar_t>(acc);
  }
}

template <typename scalar_t>
__global__ __launch_bounds__(kThreads) void backward_coeff_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ state,
    scalar_t* __restrict__ grad_coeff,
    int64_t batch,
    int64_t time,
    int64_t hidden) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = batch * time * kKVStreams * hidden * kKernelSize;
  if (linear >= total) {
    return;
  }
  const int tap = static_cast<int>(linear % kKernelSize);
  const int64_t out = linear / kKernelSize;
  const int64_t h = out % hidden;
  const int64_t s = (out / hidden) % kKVStreams;
  const int64_t t = (out / (hidden * kKVStreams)) % time;
  const int64_t b = out / (hidden * kKVStreams * time);
  grad_coeff[linear] = static_cast<scalar_t>(
      mul_nofuse(static_cast<float>(grad_output[out]), input_at(x, state, b, t + tap, s, h, time, hidden)));
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
    const torch::Tensor& coefficients,
    const torch::Tensor& state) {
  check_tensor(x, "x");
  check_tensor(coefficients, "coefficients");
  check_tensor(state, "state");
  TORCH_CHECK(x.dim() == 4 && x.size(2) == kKVStreams,
              "x must have shape [batch, time, 2, hidden]");
  TORCH_CHECK(
      coefficients.sizes() ==
          torch::IntArrayRef({x.size(0), x.size(1), kKVStreams, x.size(3), kKernelSize}),
      "coefficients must have shape [batch, time, 2, hidden, 4]");
  TORCH_CHECK(
      state.sizes() == torch::IntArrayRef({x.size(0), kKVStreams, x.size(3), kStateSize}),
      "state must have shape [batch, 2, hidden, 3]");
  TORCH_CHECK(x.device() == coefficients.device() && x.device() == state.device(),
              "x, coefficients, and state must share a device");
  TORCH_CHECK(
      x.scalar_type() == coefficients.scalar_type() && x.scalar_type() == state.scalar_type(),
      "x, coefficients, and state must share a dtype");
}

inline int blocks_for(int64_t total) {
  return static_cast<int>((total + kThreads - 1) / kThreads);
}

torch::Tensor launch_forward(
    const torch::Tensor& x,
    const torch::Tensor& coefficients,
    const torch::Tensor& state) {
  auto output = torch::empty_like(x);
  if (x.numel() == 0) {
    return output;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, x.scalar_type(), "inkling2_dynamic_kv_conv_forward", [&] {
        forward_kernel<scalar_t><<<blocks_for(x.numel()), kThreads, 0, stream>>>(
            x.data_ptr<scalar_t>(), coefficients.data_ptr<scalar_t>(), state.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(), x.size(0), x.size(1), x.size(3));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace

torch::Tensor inkling2_dynamic_kv_conv_forward(
    torch::Tensor x,
    torch::Tensor coefficients,
    torch::Tensor state) {
  check_common(x, coefficients, state);
  c10::cuda::CUDAGuard guard(x.device());
  return launch_forward(x, coefficients, state);
}

std::vector<torch::Tensor> inkling2_dynamic_kv_conv_forward_with_state(
    torch::Tensor x,
    torch::Tensor coefficients,
    torch::Tensor state) {
  check_common(x, coefficients, state);
  c10::cuda::CUDAGuard guard(x.device());
  auto output = launch_forward(x, coefficients, state);
  auto next_state = torch::empty_like(state);
  if (next_state.numel() > 0) {
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, x.scalar_type(), "inkling2_dynamic_kv_conv_state", [&] {
          state_kernel<scalar_t><<<blocks_for(next_state.numel()), kThreads, 0, stream>>>(
              x.data_ptr<scalar_t>(), state.data_ptr<scalar_t>(), next_state.data_ptr<scalar_t>(),
              x.size(0), x.size(1), x.size(3));
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {output, next_state};
}

std::vector<torch::Tensor> inkling2_dynamic_kv_conv_backward(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor coefficients,
    torch::Tensor state) {
  check_common(x, coefficients, state);
  check_tensor(grad_output, "grad_output");
  TORCH_CHECK(grad_output.sizes() == x.sizes(), "grad_output and x must have matching shapes");
  TORCH_CHECK(grad_output.scalar_type() == x.scalar_type(), "grad_output and x must share dtype");
  c10::cuda::CUDAGuard guard(x.device());
  auto grad_x = torch::empty_like(x);
  auto grad_coefficients = torch::empty_like(coefficients);
  auto grad_state = torch::zeros_like(state);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t dx_total = x.numel() + state.numel();
  if (dx_total > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, x.scalar_type(), "inkling2_dynamic_kv_conv_backward", [&] {
          backward_x_state_kernel<scalar_t><<<blocks_for(dx_total), kThreads, 0, stream>>>(
              grad_output.data_ptr<scalar_t>(), coefficients.data_ptr<scalar_t>(),
              grad_x.data_ptr<scalar_t>(), grad_state.data_ptr<scalar_t>(), x.size(0), x.size(1),
              x.size(3));
          if (coefficients.numel() > 0) {
            backward_coeff_kernel<scalar_t>
                <<<blocks_for(coefficients.numel()), kThreads, 0, stream>>>(
                    grad_output.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(),
                    state.data_ptr<scalar_t>(), grad_coefficients.data_ptr<scalar_t>(), x.size(0),
                    x.size(1), x.size(3));
          }
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {grad_x, grad_coefficients, grad_state};
}
