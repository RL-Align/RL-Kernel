// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// Batch-invariant SiLU / SwiGLU CUDA kernels (WS1 elementwise activations).
//
// Semantics match NativeSiLUOp / NativeSwiGLUOp:
//   silu(x)      = x * sigmoid(x)                 (math in fp32)
//   swiglu(g, u) = silu(g) * u                    (math in fp32)
//
// Pure elementwise / token-local: no cross-row reduction, so batch size and
// padding cannot change a row's result (Axis-A bitwise invariance).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace {

__device__ __forceinline__ float silu_f32(float x) {
  // sigmoid(x) = 1 / (1 + exp(-x)); use expf for device fp32.
  const float s = 1.0f / (1.0f + expf(-x));
  return x * s;
}

__device__ __forceinline__ float silu_grad_f32(float x) {
  // d/dx [x * s] = s + x * s * (1 - s) = s * (1 + x * (1 - s)), s = sigmoid(x)
  const float s = 1.0f / (1.0f + expf(-x));
  return s * (1.0f + x * (1.0f - s));
}

template <typename scalar_t>
__global__ void silu_forward_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ y,
    const int64_t n) {
  const int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (idx >= n) {
    return;
  }
  const float xv = static_cast<float>(x[idx]);
  y[idx] = static_cast<scalar_t>(silu_f32(xv));
}

template <typename scalar_t>
__global__ void silu_backward_kernel(
    const scalar_t* __restrict__ dy,
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ dx,
    const int64_t n) {
  const int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (idx >= n) {
    return;
  }
  const float dyv = static_cast<float>(dy[idx]);
  const float xv = static_cast<float>(x[idx]);
  dx[idx] = static_cast<scalar_t>(dyv * silu_grad_f32(xv));
}

template <typename scalar_t>
__global__ void swiglu_forward_kernel(
    const scalar_t* __restrict__ gate,
    const scalar_t* __restrict__ up,
    scalar_t* __restrict__ y,
    const int64_t n) {
  const int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (idx >= n) {
    return;
  }
  const float gv = static_cast<float>(gate[idx]);
  const float uv = static_cast<float>(up[idx]);
  y[idx] = static_cast<scalar_t>(silu_f32(gv) * uv);
}

template <typename scalar_t>
__global__ void swiglu_backward_kernel(
    const scalar_t* __restrict__ dy,
    const scalar_t* __restrict__ gate,
    const scalar_t* __restrict__ up,
    scalar_t* __restrict__ d_gate,
    scalar_t* __restrict__ d_up,
    const int64_t n) {
  const int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (idx >= n) {
    return;
  }
  const float dyv = static_cast<float>(dy[idx]);
  const float gv = static_cast<float>(gate[idx]);
  const float uv = static_cast<float>(up[idx]);
  const float s = silu_f32(gv);
  // d_up = dy * silu(gate); d_gate = dy * up * silu'(gate)
  d_up[idx] = static_cast<scalar_t>(dyv * s);
  d_gate[idx] = static_cast<scalar_t>(dyv * uv * silu_grad_f32(gv));
}

template <typename scalar_t>
__global__ void swiglu_packed_forward_kernel(
    const scalar_t* __restrict__ gate_up,
    scalar_t* __restrict__ y,
    const int64_t n,
    const int64_t width) {
  const int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (idx >= n) {
    return;
  }
  const int64_t row = idx / width;
  const int64_t column = idx - row * width;
  const int64_t gate_index = row * (2 * width) + column;
  const float gv = static_cast<float>(gate_up[gate_index]);
  const float uv = static_cast<float>(gate_up[gate_index + width]);
  y[idx] = static_cast<scalar_t>(silu_f32(gv) * uv);
}

template <typename scalar_t>
__global__ void swiglu_packed_backward_kernel(
    const scalar_t* __restrict__ dy,
    const scalar_t* __restrict__ gate_up,
    scalar_t* __restrict__ d_gate,
    scalar_t* __restrict__ d_up,
    const int64_t n,
    const int64_t width) {
  const int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (idx >= n) {
    return;
  }
  const int64_t row = idx / width;
  const int64_t column = idx - row * width;
  const int64_t gate_index = row * (2 * width) + column;
  const float dyv = static_cast<float>(dy[idx]);
  const float gv = static_cast<float>(gate_up[gate_index]);
  const float uv = static_cast<float>(gate_up[gate_index + width]);
  const float s = silu_f32(gv);
  d_up[idx] = static_cast<scalar_t>(dyv * s);
  d_gate[idx] = static_cast<scalar_t>(dyv * uv * silu_grad_f32(gv));
}

static void launch_1d(int64_t n, int& threads, int64_t& blocks) {
  threads = 256;
  blocks = (n + threads - 1) / threads;
  if (blocks == 0) {
    blocks = 1;
  }
}

static void check_cuda_contig(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  // Supported activation dtypes only: fp16 / bf16 / fp32 (reject float64).
  TORCH_CHECK(
      t.scalar_type() == at::kHalf || t.scalar_type() == at::kBFloat16 ||
          t.scalar_type() == at::kFloat,
      name,
      " must be fp16, bf16, or fp32, got ",
      t.scalar_type());
}

static void check_same_device(
    const torch::Tensor& lhs,
    const torch::Tensor& rhs,
    const char* lhs_name,
    const char* rhs_name) {
  TORCH_CHECK(
      lhs.device() == rhs.device(),
      lhs_name,
      " and ",
      rhs_name,
      " must be on the same CUDA device, got ",
      lhs.device(),
      " and ",
      rhs.device());
}

}  // namespace

torch::Tensor silu_forward_cuda(torch::Tensor x) {
  check_cuda_contig(x, "x");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  auto y = torch::empty_like(x);
  const int64_t n = x.numel();
  if (n == 0) {
    return y;
  }
  int threads = 0;
  int64_t blocks = 0;
  launch_1d(n, threads, blocks);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "silu_forward_cuda", [&] {
        silu_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), n);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

torch::Tensor silu_backward_cuda(torch::Tensor dy, torch::Tensor x) {
  check_cuda_contig(dy, "dy");
  check_cuda_contig(x, "x");
  check_same_device(dy, x, "dy", "x");
  TORCH_CHECK(dy.sizes() == x.sizes(), "dy and x must share shape");
  TORCH_CHECK(dy.scalar_type() == x.scalar_type(), "dy and x must share dtype");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  auto dx = torch::empty_like(x);
  const int64_t n = x.numel();
  if (n == 0) {
    return dx;
  }
  int threads = 0;
  int64_t blocks = 0;
  launch_1d(n, threads, blocks);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "silu_backward_cuda", [&] {
        silu_backward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            dy.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(), dx.data_ptr<scalar_t>(), n);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return dx;
}

torch::Tensor swiglu_forward_cuda(torch::Tensor gate, torch::Tensor up) {
  check_cuda_contig(gate, "gate");
  check_cuda_contig(up, "up");
  check_same_device(gate, up, "gate", "up");
  TORCH_CHECK(gate.sizes() == up.sizes(), "gate and up must share shape");
  TORCH_CHECK(gate.scalar_type() == up.scalar_type(), "gate and up must share dtype");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(gate));
  auto y = torch::empty_like(gate);
  const int64_t n = gate.numel();
  if (n == 0) {
    return y;
  }
  int threads = 0;
  int64_t blocks = 0;
  launch_1d(n, threads, blocks);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      gate.scalar_type(),
      "swiglu_forward_cuda",
      [&] {
        swiglu_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            gate.data_ptr<scalar_t>(), up.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), n);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

std::vector<torch::Tensor> swiglu_backward_cuda(
    torch::Tensor dy,
    torch::Tensor gate,
    torch::Tensor up) {
  check_cuda_contig(dy, "dy");
  check_cuda_contig(gate, "gate");
  check_cuda_contig(up, "up");
  check_same_device(dy, gate, "dy", "gate");
  check_same_device(gate, up, "gate", "up");
  TORCH_CHECK(gate.sizes() == up.sizes(), "gate and up must share shape");
  TORCH_CHECK(dy.sizes() == gate.sizes(), "dy and gate must share shape");
  TORCH_CHECK(dy.scalar_type() == gate.scalar_type(), "dy and gate must share dtype");
  TORCH_CHECK(up.scalar_type() == gate.scalar_type(), "up and gate must share dtype");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(gate));
  auto d_gate = torch::empty_like(gate);
  auto d_up = torch::empty_like(up);
  const int64_t n = gate.numel();
  if (n == 0) {
    return {d_gate, d_up};
  }
  int threads = 0;
  int64_t blocks = 0;
  launch_1d(n, threads, blocks);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      gate.scalar_type(),
      "swiglu_backward_cuda",
      [&] {
        swiglu_backward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            dy.data_ptr<scalar_t>(),
            gate.data_ptr<scalar_t>(),
            up.data_ptr<scalar_t>(),
            d_gate.data_ptr<scalar_t>(),
            d_up.data_ptr<scalar_t>(),
            n);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {d_gate, d_up};
}

torch::Tensor swiglu_packed_forward_cuda(torch::Tensor gate_up) {
  check_cuda_contig(gate_up, "gate_up");
  TORCH_CHECK(gate_up.dim() == 2, "gate_up must be [rows, 2 * intermediate]");
  TORCH_CHECK(gate_up.size(1) % 2 == 0, "gate_up width must be even");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(gate_up));
  const int64_t width = gate_up.size(1) / 2;
  auto y = torch::empty({gate_up.size(0), width}, gate_up.options());
  const int64_t n = y.numel();
  if (n == 0) {
    return y;
  }
  int threads = 0;
  int64_t blocks = 0;
  launch_1d(n, threads, blocks);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      gate_up.scalar_type(),
      "swiglu_packed_forward_cuda",
      [&] {
        swiglu_packed_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            gate_up.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), n, width);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

std::vector<torch::Tensor> swiglu_packed_backward_cuda(
    torch::Tensor dy,
    torch::Tensor gate_up) {
  check_cuda_contig(dy, "dy");
  check_cuda_contig(gate_up, "gate_up");
  check_same_device(dy, gate_up, "dy", "gate_up");
  TORCH_CHECK(gate_up.dim() == 2, "gate_up must be [rows, 2 * intermediate]");
  TORCH_CHECK(gate_up.size(1) % 2 == 0, "gate_up width must be even");
  TORCH_CHECK(dy.dim() == 2, "dy must be [rows, intermediate]");
  TORCH_CHECK(
      dy.size(0) == gate_up.size(0) && dy.size(1) * 2 == gate_up.size(1),
      "dy shape must match the packed gate/up halves");
  TORCH_CHECK(dy.scalar_type() == gate_up.scalar_type(), "dy and gate_up must share dtype");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(gate_up));
  auto d_gate = torch::empty_like(dy);
  auto d_up = torch::empty_like(dy);
  const int64_t n = dy.numel();
  if (n == 0) {
    return {d_gate, d_up};
  }
  int threads = 0;
  int64_t blocks = 0;
  launch_1d(n, threads, blocks);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      gate_up.scalar_type(),
      "swiglu_packed_backward_cuda",
      [&] {
        swiglu_packed_backward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            dy.data_ptr<scalar_t>(),
            gate_up.data_ptr<scalar_t>(),
            d_gate.data_ptr<scalar_t>(),
            d_up.data_ptr<scalar_t>(),
            n,
            dy.size(1));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {d_gate, d_up};
}
