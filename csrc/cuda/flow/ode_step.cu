// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// MiniMax-H3 deterministic rectified-flow Euler step (RFC #420).
//
//     x0     = xt + sigma * v
//     r      = sigma_next / sigma
//     x_next = r * xt + (1 - r) * x0
//
// Two things are load-bearing:
//   1. Every step uses an explicitly-rounded intrinsic.  nvcc contracts a*b+c
//      into an FMA by default, fusing two roundings into one; that would break
//      op-for-op equality with the PyTorch reference.
//   2. The blend keeps the declared expression order.  The algebraically equal
//      xt + (sigma - sigma_next) * v reassociates the sum and is a different
//      fp32 program (ablation probe H13).  Do not "simplify" it.
//
// Element-wise with a per-row sigma broadcast: no cross-row reduction, so batch
// size, batch position and unrelated rows cannot change a row's bytes.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace {

constexpr int kOdeStepThreads = 256;

// x0 = xt + sigma * v ; x_next = r * xt + (1 - r) * x0, one rounding per op.
__device__ __forceinline__ void ode_step_f32(
    const float xt, const float v, const float sigma, const float sigma_next,
    float& x_next_out, float& x0_out) {
  const float r = __fdiv_rn(sigma_next, sigma);
  const float scaled = __fmul_rn(sigma, v);
  const float x0 = __fadd_rn(xt, scaled);
  const float one_minus_r = __fadd_rn(1.0f, -r);
  const float a = __fmul_rn(r, xt);
  const float b = __fmul_rn(one_minus_r, x0);
  x_next_out = __fadd_rn(a, b);
  x0_out = x0;
}

// g_x0 = g0 + (1 - r) * g_next ; g_xt = r * g_next + g_x0 ; g_v = sigma * g_x0
__device__ __forceinline__ void ode_step_grad_f32(
    const float gn, const float g0, const float sigma, const float sigma_next,
    float& g_xt_out, float& g_v_out) {
  const float r = __fdiv_rn(sigma_next, sigma);
  const float one_minus_r = __fadd_rn(1.0f, -r);
  const float from_next = __fmul_rn(one_minus_r, gn);
  const float g_x0 = __fadd_rn(g0, from_next);
  const float r_times_gn = __fmul_rn(r, gn);
  g_xt_out = __fadd_rn(r_times_gn, g_x0);
  g_v_out = __fmul_rn(sigma, g_x0);
}

template <typename scalar_t, bool kRowSigma>
__global__ void ode_step_forward_kernel(
    const scalar_t* __restrict__ xt,
    const scalar_t* __restrict__ v,
    const float* __restrict__ sigma,
    const float* __restrict__ sigma_next,
    scalar_t* __restrict__ x_next,
    scalar_t* __restrict__ x0,
    const int64_t n,
    const int64_t row_width) {
  const int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (idx >= n) {
    return;
  }
  float s;
  float sn;
  if (kRowSigma) {
    const int64_t row = idx / row_width;
    s = sigma[row];
    sn = sigma_next[row];
  } else {
    s = sigma[0];
    sn = sigma_next[0];
  }
  float xn;
  float x0v;
  ode_step_f32(static_cast<float>(xt[idx]), static_cast<float>(v[idx]), s, sn, xn, x0v);
  x_next[idx] = static_cast<scalar_t>(xn);
  x0[idx] = static_cast<scalar_t>(x0v);
}

template <typename scalar_t, bool kRowSigma>
__global__ void ode_step_backward_kernel(
    const scalar_t* __restrict__ grad_next,
    const scalar_t* __restrict__ grad_x0,
    const float* __restrict__ sigma,
    const float* __restrict__ sigma_next,
    scalar_t* __restrict__ grad_xt,
    scalar_t* __restrict__ grad_v,
    const int64_t n,
    const int64_t row_width) {
  const int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (idx >= n) {
    return;
  }
  float s;
  float sn;
  if (kRowSigma) {
    const int64_t row = idx / row_width;
    s = sigma[row];
    sn = sigma_next[row];
  } else {
    s = sigma[0];
    sn = sigma_next[0];
  }
  float g_xt;
  float g_v;
  ode_step_grad_f32(
      static_cast<float>(grad_next[idx]), static_cast<float>(grad_x0[idx]), s, sn,
      g_xt, g_v);
  grad_xt[idx] = static_cast<scalar_t>(g_xt);
  grad_v[idx] = static_cast<scalar_t>(g_v);
}

void check_sigma(const torch::Tensor& sigma, const char* name,
                 const int64_t rows, const torch::Tensor& xt) {
  TORCH_CHECK(sigma.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(sigma.scalar_type() == at::kFloat, name, " must be float32");
  TORCH_CHECK(sigma.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(sigma.device() == xt.device(), name, " must share xt's device");
  TORCH_CHECK(
      sigma.numel() == 1 || sigma.numel() == rows,
      name,
      " must be either a scalar or exactly one value per packed row (",
      rows, "), got numel=", sigma.numel(),
      ". Refusing to broadcast implicitly: video and audio schedulers own "
      "separate sigma grids.");
}

struct RowGeometry {
  int64_t rows;
  int64_t width;
};

RowGeometry row_geometry(const torch::Tensor& t) {
  const int64_t width = t.dim() >= 1 ? t.size(-1) : 1;
  return RowGeometry{t.numel() / std::max<int64_t>(width, 1), width};
}

void check_common(const torch::Tensor& xt, const torch::Tensor& v) {
  TORCH_CHECK(xt.is_cuda(), "xt must be a CUDA tensor");
  TORCH_CHECK(xt.is_contiguous(), "xt must be contiguous");
  TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
  TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
  TORCH_CHECK(xt.device() == v.device(), "xt and v must share a device");
  TORCH_CHECK(xt.sizes() == v.sizes(), "xt and v must share shape");
  TORCH_CHECK(xt.scalar_type() == v.scalar_type(), "xt and v must share dtype");
}

}  // namespace

std::vector<torch::Tensor> ode_step_forward_cuda(
    torch::Tensor xt,
    torch::Tensor v,
    torch::Tensor sigma,
    torch::Tensor sigma_next) {
  check_common(xt, v);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(xt));
  const RowGeometry geo = row_geometry(xt);
  check_sigma(sigma, "sigma", geo.rows, xt);
  check_sigma(sigma_next, "sigma_next", geo.rows, xt);

  auto x_next = torch::empty_like(xt);
  auto x0 = torch::empty_like(xt);
  const int64_t n = xt.numel();
  if (n == 0) {
    return {x_next, x0};
  }
  const bool row_sigma = sigma.numel() != 1;
  const dim3 block(kOdeStepThreads);
  const dim3 grid((n + kOdeStepThreads - 1) / kOdeStepThreads);
  auto stream = at::cuda::getCurrentCUDAStream();

#define RLK_ODE_STEP_FORWARD_DISPATCH(ROW_SIGMA)                                   \
  AT_DISPATCH_FLOATING_TYPES_AND2(                                                 \
      at::ScalarType::Half, at::ScalarType::BFloat16, xt.scalar_type(),            \
      "ode_step_forward_cuda", [&] {                                               \
        ode_step_forward_kernel<scalar_t, ROW_SIGMA><<<grid, block, 0, stream>>>(  \
            xt.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),                       \
            sigma.data_ptr<float>(), sigma_next.data_ptr<float>(),                 \
            x_next.data_ptr<scalar_t>(), x0.data_ptr<scalar_t>(), n, geo.width);   \
      })

  if (row_sigma) {
    RLK_ODE_STEP_FORWARD_DISPATCH(true);
  } else {
    RLK_ODE_STEP_FORWARD_DISPATCH(false);
  }
#undef RLK_ODE_STEP_FORWARD_DISPATCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {x_next, x0};
}

std::vector<torch::Tensor> ode_step_backward_cuda(
    torch::Tensor grad_next,
    torch::Tensor grad_x0,
    torch::Tensor xt,
    torch::Tensor sigma,
    torch::Tensor sigma_next) {
  TORCH_CHECK(grad_next.is_cuda() && grad_next.is_contiguous(),
              "grad_next must be a contiguous CUDA tensor");
  TORCH_CHECK(grad_x0.is_cuda() && grad_x0.is_contiguous(),
              "grad_x0 must be a contiguous CUDA tensor");
  // The kernel launches on xt's device and dereferences both gradient pointers.
  // A gradient left on another device would only fail — or, with peer access
  // enabled, silently read remote memory — at execution time, so reject it here.
  TORCH_CHECK(grad_next.device() == xt.device() && grad_x0.device() == xt.device(),
              "upstream gradients must share xt's device");
  TORCH_CHECK(grad_next.sizes() == xt.sizes() && grad_x0.sizes() == xt.sizes(),
              "upstream gradients must share xt's shape");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(xt));
  const RowGeometry geo = row_geometry(xt);
  check_sigma(sigma, "sigma", geo.rows, xt);
  check_sigma(sigma_next, "sigma_next", geo.rows, xt);

  auto grad_xt = torch::empty_like(xt);
  auto grad_v = torch::empty_like(xt);
  const int64_t n = xt.numel();
  if (n == 0) {
    return {grad_xt, grad_v};
  }
  const bool row_sigma = sigma.numel() != 1;
  const dim3 block(kOdeStepThreads);
  const dim3 grid((n + kOdeStepThreads - 1) / kOdeStepThreads);
  auto stream = at::cuda::getCurrentCUDAStream();

#define RLK_ODE_STEP_BACKWARD_DISPATCH(ROW_SIGMA)                                  \
  AT_DISPATCH_FLOATING_TYPES_AND2(                                                 \
      at::ScalarType::Half, at::ScalarType::BFloat16, xt.scalar_type(),            \
      "ode_step_backward_cuda", [&] {                                              \
        ode_step_backward_kernel<scalar_t, ROW_SIGMA><<<grid, block, 0, stream>>>( \
            grad_next.data_ptr<scalar_t>(), grad_x0.data_ptr<scalar_t>(),          \
            sigma.data_ptr<float>(), sigma_next.data_ptr<float>(),                 \
            grad_xt.data_ptr<scalar_t>(), grad_v.data_ptr<scalar_t>(), n,          \
            geo.width);                                                            \
      })

  if (row_sigma) {
    RLK_ODE_STEP_BACKWARD_DISPATCH(true);
  } else {
    RLK_ODE_STEP_BACKWARD_DISPATCH(false);
  }
#undef RLK_ODE_STEP_BACKWARD_DISPATCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_xt, grad_v};
}
