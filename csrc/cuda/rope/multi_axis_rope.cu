// SPDX-License-Identifier: Apache-2.0
// Qwen-Image adjacent-pair rotation. Reads strided BSHD inputs/tables directly.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>

namespace {
constexpr int64_t kHeadDim = 128;
constexpr int64_t kPairsPerHead = kHeadDim / 2;

struct Strides {
  int64_t xb, xs, xh, xd;
  int64_t cb, cs, cd, sb, ss, sd;
};

template <typename scalar_t>
__global__ void rotate_pairs(const scalar_t* x, const float* cos, const float* sin,
                             scalar_t* y, int64_t n, int64_t seq, int64_t heads,
                             Strides strides, bool transpose) {
  const int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += step) {
    const int64_t pair = i % kPairsPerHead;
    const int64_t row = i / kPairsPerHead;
    const int64_t head = row % heads;
    const int64_t token = row / heads;
    const int64_t batch = token / seq;
    const int64_t position = token % seq;
    const int64_t xi = batch * strides.xb + position * strides.xs +
                       head * strides.xh + 2 * pair * strides.xd;
    const float c = cos[batch * strides.cb + position * strides.cs + pair * strides.cd];
    const float sn = sin[batch * strides.sb + position * strides.ss + pair * strides.sd];
    const float s = transpose ? -sn : sn;
    const float a = static_cast<float>(x[xi]);
    const float b = static_cast<float>(x[xi + strides.xd]);
    y[2 * i] = static_cast<scalar_t>(__fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s)));
    y[2 * i + 1] = static_cast<scalar_t>(__fadd_rn(__fmul_rn(b, c), __fmul_rn(a, s)));
  }
}

at::Tensor multi_axis_rope_apply(const at::Tensor& x, const at::Tensor& cos,
                                const at::Tensor& sin, bool transpose) {
  TORCH_CHECK(x.is_cuda() && cos.is_cuda() && sin.is_cuda(), "multi_axis_rope: CUDA tensors required");
  TORCH_CHECK(x.device() == cos.device() && x.device() == sin.device(), "multi_axis_rope: device mismatch");
  TORCH_CHECK(x.dim() == 4 && x.size(3) == kHeadDim && x.size(2) > 0,
              "multi_axis_rope: x must be [B, S, H, 128], H > 0");
  TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf ||
              x.scalar_type() == at::kBFloat16, "multi_axis_rope: unsupported input dtype");
  TORCH_CHECK(cos.scalar_type() == at::kFloat && sin.scalar_type() == at::kFloat,
              "multi_axis_rope: tables must be fp32");
  TORCH_CHECK(cos.sizes() == sin.sizes(), "multi_axis_rope: table shape mismatch");
  TORCH_CHECK((cos.dim() == 2 && cos.size(0) == x.size(1) && cos.size(1) == kPairsPerHead) ||
              (cos.dim() == 3 && cos.size(0) == x.size(0) && cos.size(1) == x.size(1) && cos.size(2) == kPairsPerHead),
              "multi_axis_rope: tables must be [S, 64] or [B, S, 64]");
  TORCH_CHECK(!cos.requires_grad() && !sin.requires_grad(), "multi_axis_rope: table gradients unsupported");
  const c10::cuda::CUDAGuard guard(x.device());
  auto y = at::empty(x.sizes(), x.options());
  const int64_t n = x.numel() / 2;
  if (n == 0) return y;
  const Strides strides{x.stride(0), x.stride(1), x.stride(2), x.stride(3),
      cos.dim() == 3 ? cos.stride(0) : 0, cos.stride(-2), cos.stride(-1),
      sin.dim() == 3 ? sin.stride(0) : 0, sin.stride(-2), sin.stride(-1)};
  const int blocks = static_cast<int>(std::min<int64_t>((n + 255) / 256, 65535));
  auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
      x.scalar_type(), "multi_axis_rope", [&] {
    rotate_pairs<scalar_t><<<blocks, 256, 0, stream>>>(x.data_ptr<scalar_t>(), cos.data_ptr<float>(),
        sin.data_ptr<float>(), y.data_ptr<scalar_t>(), n, x.size(1), x.size(2), strides, transpose);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
}  // namespace

// The focused validator and normal package build load this same registration.
TORCH_LIBRARY_FRAGMENT(rl_kernel_native, m) {
  m.def("multi_axis_rope(Tensor x, Tensor cos, Tensor sin, bool transpose) -> Tensor");
}
TORCH_LIBRARY_IMPL(rl_kernel_native, CUDA, m) {
  m.impl("multi_axis_rope", &multi_axis_rope_apply);
}
