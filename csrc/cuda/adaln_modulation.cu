// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
// One CTA per logical row; H uses a fixed lower-plus-upper pairwise tree.
// Shared modulation gradients fold token 0..S-1 in FP32, without atomics.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>

#if defined(__CUDA_FAST_MATH__)
#error "adaln_modulation requires CUDA compilation without --use_fast_math"
#endif

namespace {

constexpr int THREADS = 256;

__device__ float fixed_sum(float* shared, int width) {
  __syncthreads();
  for (int stride = width / 2; stride > 0; stride /= 2) {
    for (int i = threadIdx.x; i < stride; i += blockDim.x) {
      shared[i] = __fadd_rn(shared[i], shared[i + stride]);
    }
    __syncthreads();
  }
  return shared[0];
}

template <typename scalar_t>
__global__ void forward_kernel(
    const scalar_t* x, const scalar_t* modulation, scalar_t* y, scalar_t* gate,
    int seq, int hidden, int width, float eps) {
  extern __shared__ float shared[];
  const int row = blockIdx.x;
  const int batch = row / seq;
  for (int col = threadIdx.x; col < width; col += blockDim.x) {
    shared[col] = col < hidden ? static_cast<float>(x[row * hidden + col]) : 0.0f;
  }
  const float mean = __fdiv_rn(fixed_sum(shared, width), hidden);
  for (int col = threadIdx.x; col < width; col += blockDim.x) {
    const float centered = col < hidden
        ? __fsub_rn(static_cast<float>(x[row * hidden + col]), mean) : 0.0f;
    shared[col] = __fmul_rn(centered, centered);
  }
  const float variance = __fdiv_rn(fixed_sum(shared, width), hidden);
  const float rstd = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(variance, eps)));
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const float centered = __fsub_rn(static_cast<float>(x[row * hidden + col]), mean);
    const float norm = __fmul_rn(centered, rstd);
    const float scale = static_cast<float>(modulation[batch * 3 * hidden + hidden + col]);
    const float shift = static_cast<float>(modulation[batch * 3 * hidden + col]);
    const float value = __fadd_rn(__fmul_rn(norm, __fadd_rn(1.0f, scale)), shift);
    y[row * hidden + col] = static_cast<scalar_t>(value);
    if (row % seq == 0) gate[batch * hidden + col] = modulation[batch * 3 * hidden + 2 * hidden + col];
  }
}

template <typename scalar_t>
__global__ void backward_rows_kernel(
    const scalar_t* x, const scalar_t* modulation, const scalar_t* dy,
    scalar_t* dx, float* partial_shift, float* partial_scale,
    int seq, int hidden, int width, float eps) {
  extern __shared__ float shared[];
  const int row = blockIdx.x;
  const int batch = row / seq;
  for (int col = threadIdx.x; col < width; col += blockDim.x) {
    shared[col] = col < hidden ? static_cast<float>(x[row * hidden + col]) : 0.0f;
  }
  const float mean = __fdiv_rn(fixed_sum(shared, width), hidden);
  for (int col = threadIdx.x; col < width; col += blockDim.x) {
    const float centered = col < hidden
        ? __fsub_rn(static_cast<float>(x[row * hidden + col]), mean) : 0.0f;
    shared[col] = __fmul_rn(centered, centered);
  }
  const float variance = __fdiv_rn(fixed_sum(shared, width), hidden);
  const float rstd = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(variance, eps)));
  for (int col = threadIdx.x; col < width; col += blockDim.x) {
    if (col < hidden) {
      const float grad = static_cast<float>(dy[row * hidden + col]);
      const float scale = static_cast<float>(modulation[batch * 3 * hidden + hidden + col]);
      shared[col] = __fmul_rn(grad, __fadd_rn(1.0f, scale));
    } else shared[col] = 0.0f;
  }
  const float mean_dnorm = __fdiv_rn(fixed_sum(shared, width), hidden);
  for (int col = threadIdx.x; col < width; col += blockDim.x) {
    if (col < hidden) {
      const float centered = __fsub_rn(static_cast<float>(x[row * hidden + col]), mean);
      const float norm = __fmul_rn(centered, rstd);
      const float grad = static_cast<float>(dy[row * hidden + col]);
      const float scale = static_cast<float>(modulation[batch * 3 * hidden + hidden + col]);
      shared[col] = __fmul_rn(__fmul_rn(grad, __fadd_rn(1.0f, scale)), norm);
    } else shared[col] = 0.0f;
  }
  const float mean_dnorm_norm = __fdiv_rn(fixed_sum(shared, width), hidden);
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const float grad = static_cast<float>(dy[row * hidden + col]);
    const float scale = static_cast<float>(modulation[batch * 3 * hidden + hidden + col]);
    const float centered = __fsub_rn(static_cast<float>(x[row * hidden + col]), mean);
    const float norm = __fmul_rn(centered, rstd);
    const float dnorm = __fmul_rn(grad, __fadd_rn(1.0f, scale));
    const float value = __fmul_rn(
        __fsub_rn(__fsub_rn(dnorm, mean_dnorm), __fmul_rn(norm, mean_dnorm_norm)), rstd);
    dx[row * hidden + col] = static_cast<scalar_t>(value);
    partial_shift[row * hidden + col] = grad;
    partial_scale[row * hidden + col] = __fmul_rn(grad, norm);
  }
}

template <typename scalar_t>
__global__ void backward_reduce_kernel(
    const float* partial_shift, const float* partial_scale, const scalar_t* dg,
    scalar_t* dm, int seq, int hidden) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int batch = blockIdx.y;
  if (col >= hidden) return;
  float shift = 0.0f, scale = 0.0f;
#pragma unroll 1
  for (int token = 0; token < seq; ++token) {
    const int offset = (batch * seq + token) * hidden + col;
    shift = __fadd_rn(shift, partial_shift[offset]);
    scale = __fadd_rn(scale, partial_scale[offset]);
  }
  dm[batch * 3 * hidden + col] = static_cast<scalar_t>(shift);
  dm[batch * 3 * hidden + hidden + col] = static_cast<scalar_t>(scale);
  dm[batch * 3 * hidden + 2 * hidden + col] = dg[batch * hidden + col];
}

void check(const torch::Tensor& x, const torch::Tensor& modulation, double eps) {
  TORCH_CHECK(x.is_cuda() && modulation.is_cuda(), "AdaLN inputs must be CUDA tensors");
  TORCH_CHECK(x.is_contiguous() && modulation.is_contiguous(), "AdaLN inputs must be contiguous");
  TORCH_CHECK(x.dim() == 3 && x.size(0) > 0 && x.size(1) > 0 && x.size(2) > 0,
              "x must have nonempty shape [B,S,H]");
  TORCH_CHECK(modulation.dim() == 2 && modulation.size(0) == x.size(0) &&
              modulation.size(1) == 3 * x.size(2), "modulation must have shape [B,3H]");
  TORCH_CHECK(x.device() == modulation.device() && x.scalar_type() == modulation.scalar_type(),
              "AdaLN inputs must share device and dtype");
  TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kBFloat16,
              "AdaLN inputs must be FP32 or BF16");
  TORCH_CHECK(x.size(2) <= 4096, "AdaLN CUDA supports H <= 4096");
  TORCH_CHECK(std::isfinite(eps) && eps >= 0.0, "eps must be finite and nonnegative");
}

}  // namespace

std::vector<torch::Tensor> adaln_modulation_forward_cuda(
    torch::Tensor x, torch::Tensor modulation, double eps) {
  check(x, modulation, eps);
  c10::cuda::CUDAGuard guard(x.device());
  const int batch = x.size(0), seq = x.size(1), hidden = x.size(2);
  int width = 1;
  while (width < hidden) width <<= 1;
  auto y = torch::empty_like(x);
  auto gate = torch::empty({batch, 1, hidden}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND(at::kBFloat16, x.scalar_type(), "adaln_forward", [&] {
    forward_kernel<scalar_t><<<batch * seq, THREADS, width * sizeof(float), stream>>>(
        x.data_ptr<scalar_t>(), modulation.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(),
        gate.data_ptr<scalar_t>(), seq, hidden, width, static_cast<float>(eps));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y, gate};
}

std::vector<torch::Tensor> adaln_modulation_backward_cuda(
    torch::Tensor dy, torch::Tensor dg, torch::Tensor x,
    torch::Tensor modulation, double eps) {
  check(x, modulation, eps);
  TORCH_CHECK(dy.is_cuda() && dy.is_contiguous() && dy.sizes() == x.sizes() &&
              dy.scalar_type() == x.scalar_type() && dy.device() == x.device(),
              "dy must match x");
  TORCH_CHECK(dg.is_cuda() && dg.is_contiguous() && dg.dim() == 3 &&
              dg.size(0) == x.size(0) && dg.size(1) == 1 && dg.size(2) == x.size(2) &&
              dg.scalar_type() == x.scalar_type() && dg.device() == x.device(),
              "dg must match x device/dtype and have shape [B,1,H]");
  c10::cuda::CUDAGuard guard(x.device());
  const int batch = x.size(0), seq = x.size(1), hidden = x.size(2);
  int width = 1;
  while (width < hidden) width <<= 1;
  auto dx = torch::empty_like(x);
  auto dm = torch::empty_like(modulation);
  auto partial_shift = torch::empty(x.sizes(), x.options().dtype(torch::kFloat32));
  auto partial_scale = torch::empty_like(partial_shift);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND(at::kBFloat16, x.scalar_type(), "adaln_backward", [&] {
    backward_rows_kernel<scalar_t><<<batch * seq, THREADS, width * sizeof(float), stream>>>(
        x.data_ptr<scalar_t>(), modulation.data_ptr<scalar_t>(), dy.data_ptr<scalar_t>(),
        dx.data_ptr<scalar_t>(), partial_shift.data_ptr<float>(), partial_scale.data_ptr<float>(),
        seq, hidden, width, static_cast<float>(eps));
    backward_reduce_kernel<scalar_t><<<dim3((hidden + THREADS - 1) / THREADS, batch),
                                        THREADS, 0, stream>>>(
        partial_shift.data_ptr<float>(), partial_scale.data_ptr<float>(),
        dg.data_ptr<scalar_t>(), dm.data_ptr<scalar_t>(), seq, hidden);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dx, dm};
}
