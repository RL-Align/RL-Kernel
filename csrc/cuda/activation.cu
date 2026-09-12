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

#if defined(__USE_FAST_MATH__)
#error "P5 clamp_swiglu_weighted requires precise FP32 math; disable --use_fast_math"
#endif

__device__ __forceinline__ float sigmoid_f32_strict(float x) {
  const float denominator = __fadd_rn(1.0f, expf(-x));
  return 1.0f / denominator;
}

__global__ void clamp_swiglu_weighted_forward_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const float* __restrict__ p_s,
    at::BFloat16* __restrict__ h,
    const int64_t n,
    const int64_t width,
    const int64_t input_stride,
    const bool weighted) {
  const int64_t idx =
      blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;

  if (idx >= n) {
    return;
  }

      const int64_t row = idx / width;
      const int64_t column = idx - row * width;
      const int64_t input_offset = row * input_stride + column;

      const float gate_value = gate[input_offset];
      const float up_value = up[input_offset];

  float g = gate_value;
  float u = up_value;

  if (weighted) {
    if (g > 10.0f) {
      g = 10.0f;
    }

    if (u < -10.0f) {
      u = -10.0f;
    } else if (u > 10.0f) {
      u = 10.0f;
    }
  }

  const float sig = sigmoid_f32_strict(g);
  const float silu = __fmul_rn(g, sig);
  const float product = __fmul_rn(silu, u);

  const float h32 =
      weighted
          ? __fmul_rn(product, p_s[row])
          : product;

  h[idx] = static_cast<at::BFloat16>(h32);
}

template <typename scalar_t>
__global__ void clamp_swiglu_weighted_backward_kernel(
    const scalar_t* __restrict__ dh,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const float* __restrict__ p_s,
    float* __restrict__ d_gate,
    float* __restrict__ d_up,
    const int64_t n,
    const int64_t width,
    const int64_t input_stride,
    const bool weighted) {
  const int64_t idx =
      blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;

  if (idx >= n) {
    return;
  }

    const int64_t row = idx / width;
    const int64_t column = idx - row * width;
    const int64_t input_offset = row * input_stride + column;

    const float gate_value = gate[input_offset];
    const float up_value = up[input_offset];

  float g = gate_value;
  float u = up_value;

  if (weighted) {
    if (g > 10.0f) {
      g = 10.0f;
    }

    if (u < -10.0f) {
      u = -10.0f;
    } else if (u > 10.0f) {
      u = 10.0f;
    }
  }

  const float sig = sigmoid_f32_strict(g);
  const float silu = __fmul_rn(g, sig);

  const float dh32 = static_cast<float>(dh[idx]);

  const float weighted_dh =
      weighted
          ? __fmul_rn(dh32, p_s[row])
          : dh32;

  const float one_minus_sig =
      __fadd_rn(1.0f, -sig);

  const float derivative_inner =
      __fadd_rn(
          1.0f,
          __fmul_rn(g, one_minus_sig));

  const float d_silu =
      __fmul_rn(sig, derivative_inner);

  const float gate_mask =
      (!weighted || gate_value < 10.0f)
          ? 1.0f
          : 0.0f;

  const float up_mask =
      (!weighted ||
       (up_value > -10.0f && up_value < 10.0f))
          ? 1.0f
          : 0.0f;

  d_gate[idx] =
      __fmul_rn(
          __fmul_rn(
              __fmul_rn(weighted_dh, u),
              d_silu),
          gate_mask);

  d_up[idx] =
      __fmul_rn(
          __fmul_rn(weighted_dh, silu),
          up_mask);
}

template <typename scalar_t>
__global__ void clamp_swiglu_weighted_dp_s_kernel(
    const scalar_t* __restrict__ dh,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ dp_s,
    const int64_t rows,
    const int64_t width,
    const int64_t input_stride) {
  const int64_t row =
      blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;

  if (row >= rows) {
    return;
  }

  float acc = 0.0f;

  const int64_t input_row_offset = row * input_stride;
  const int64_t output_row_offset = row * width;

#pragma unroll 1
  for (int64_t column = 0; column < width; ++column) {
    const int64_t input_idx = input_row_offset + column;
    const int64_t output_idx = output_row_offset + column;

    float g = gate[input_idx];
    float u = up[input_idx];

    if (g > 10.0f) {
      g = 10.0f;
    }

    if (u < -10.0f) {
      u = -10.0f;
    } else if (u > 10.0f) {
      u = 10.0f;
    }

    const float sig = sigmoid_f32_strict(g);
    const float silu = __fmul_rn(g, sig);

    const float term =
        __fmul_rn(
            __fmul_rn(
                static_cast<float>(dh[output_idx]),
                silu),
            u);

    acc = __fadd_rn(acc, term);
  }

  dp_s[row] = acc;
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

static void check_cuda_contig_fp32(
    const torch::Tensor& t,
    const char* name) {
  TORCH_CHECK(
      t.is_cuda(),
      name,
      " must be a CUDA tensor");

  TORCH_CHECK(
      t.is_contiguous(),
      name,
      " must be contiguous");

  TORCH_CHECK(
      t.scalar_type() == at::kFloat,
      name,
      " must be float32");
}

static void check_same_shape_2d(
    const torch::Tensor& lhs,
    const torch::Tensor& rhs,
    const char* lhs_name,
    const char* rhs_name) {
  TORCH_CHECK(
      lhs.dim() == 2,
      lhs_name,
      " must be 2D [rows, width]");

  TORCH_CHECK(
      rhs.dim() == 2,
      rhs_name,
      " must be 2D [rows, width]");

  TORCH_CHECK(
      lhs.sizes() == rhs.sizes(),
      lhs_name,
      " and ",
      rhs_name,
      " must share shape");
}

static void check_route_weights(
    const torch::optional<torch::Tensor>& p_s,
    const torch::Tensor& reference) {
  if (!p_s.has_value()) {
    return;
  }

  check_cuda_contig_fp32(*p_s, "p_s");
  check_same_device(*p_s, reference, "p_s", "gate");

  TORCH_CHECK(
      p_s->dim() == 1,
      "p_s must be 1D [rows]");

  TORCH_CHECK(
      p_s->size(0) == reference.size(0),
      "p_s must have shape [rows]");
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

std::vector<torch::Tensor> clamp_swiglu_weighted_forward_cuda(
    torch::Tensor gate,
    torch::Tensor up,
    torch::optional<torch::Tensor> p_s) {
  check_cuda_contig_fp32(gate, "gate");
  check_cuda_contig_fp32(up, "up");
  check_same_device(gate, up, "gate", "up");
  check_same_shape_2d(gate, up, "gate", "up");
  check_route_weights(p_s, gate);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(gate));

  auto h =
      torch::empty(
          gate.sizes(),
          gate.options().dtype(torch::kBFloat16));

  const int64_t n = gate.numel();

  if (n == 0) {
    return {h};
  }

  int threads = 0;
  int64_t blocks = 0;

  launch_1d(n, threads, blocks);

  auto stream = at::cuda::getCurrentCUDAStream();

  clamp_swiglu_weighted_forward_kernel
      <<<blocks, threads, 0, stream>>>(
          gate.data_ptr<float>(),
          up.data_ptr<float>(),
          p_s.has_value()
              ? p_s->data_ptr<float>()
              : nullptr,
          h.data_ptr<at::BFloat16>(),
          n,
          gate.size(1),
          gate.size(1),
          p_s.has_value());

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {h};
}

std::vector<torch::Tensor> clamp_swiglu_weighted_backward_cuda(
    torch::Tensor dh,
    torch::Tensor gate,
    torch::Tensor up,
    torch::optional<torch::Tensor> p_s) {
  check_cuda_contig(dh, "dh");
  check_cuda_contig_fp32(gate, "gate");
  check_cuda_contig_fp32(up, "up");

  check_same_device(dh, gate, "dh", "gate");
  check_same_device(gate, up, "gate", "up");

  check_same_shape_2d(gate, up, "gate", "up");
  check_same_shape_2d(gate, dh, "gate", "dh");

  check_route_weights(p_s, gate);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(gate));

  auto d_gate = torch::empty_like(gate);
  auto d_up = torch::empty_like(up);

  auto dp_s =
      p_s.has_value()
          ? torch::zeros(
                {gate.size(0)},
                gate.options())
          : torch::empty(
                {0},
                gate.options());

  const int64_t n = gate.numel();

  if (n == 0) {
    return {d_gate, d_up, dp_s};
  }

  int threads = 0;
  int64_t blocks = 0;

  launch_1d(n, threads, blocks);

  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      dh.scalar_type(),
      "clamp_swiglu_weighted_backward_cuda",
      [&] {
        clamp_swiglu_weighted_backward_kernel<scalar_t>
            <<<blocks, threads, 0, stream>>>(
                dh.data_ptr<scalar_t>(),
                gate.data_ptr<float>(),
                up.data_ptr<float>(),
                p_s.has_value()
                    ? p_s->data_ptr<float>()
                    : nullptr,
                d_gate.data_ptr<float>(),
                d_up.data_ptr<float>(),
                n,
                gate.size(1),
                gate.size(1),
                p_s.has_value());

        if (p_s.has_value()) {
          int dp_threads = 0;
          int64_t dp_blocks = 0;

          launch_1d(
              gate.size(0),
              dp_threads,
              dp_blocks);

          clamp_swiglu_weighted_dp_s_kernel<scalar_t>
              <<<dp_blocks, dp_threads, 0, stream>>>(
                  dh.data_ptr<scalar_t>(),
                  gate.data_ptr<float>(),
                  up.data_ptr<float>(),
                  dp_s.data_ptr<float>(),
                  gate.size(0),
                  gate.size(1),
                  gate.size(1));
        }
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {d_gate, d_up, dp_s};
}

std::vector<torch::Tensor> clamp_swiglu_weighted_packed_forward_cuda(
    torch::Tensor gate_up,
    torch::optional<torch::Tensor> p_s) {
  check_cuda_contig_fp32(gate_up, "gate_up");

  TORCH_CHECK(
      gate_up.dim() == 2,
      "gate_up must be 2D [rows, 2 * width]");

  TORCH_CHECK(
      gate_up.size(1) % 2 == 0,
      "gate_up width must be even");

  check_route_weights(p_s, gate_up);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(gate_up));

  const int64_t rows = gate_up.size(0);
  const int64_t width = gate_up.size(1) / 2;

  auto h = torch::empty(
      {rows, width},
      gate_up.options().dtype(torch::kBFloat16));

  const int64_t n = h.numel();

  if (n == 0) {
    return {h};
  }

  int threads = 0;
  int64_t blocks = 0;
  launch_1d(n, threads, blocks);

  auto stream = at::cuda::getCurrentCUDAStream();

  const float* gate_ptr = gate_up.data_ptr<float>();
  const float* up_ptr = gate_ptr + width;

  clamp_swiglu_weighted_forward_kernel
      <<<blocks, threads, 0, stream>>>(
          gate_ptr,
          up_ptr,
          p_s.has_value()
              ? p_s->data_ptr<float>()
              : nullptr,
          h.data_ptr<at::BFloat16>(),
          n,
          width,
          2 * width,
          p_s.has_value());

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {h};
}

std::vector<torch::Tensor> clamp_swiglu_weighted_packed_backward_cuda(
    torch::Tensor dh,
    torch::Tensor gate_up,
    torch::optional<torch::Tensor> p_s) {
  check_cuda_contig(dh, "dh");
  check_cuda_contig_fp32(gate_up, "gate_up");
  check_same_device(dh, gate_up, "dh", "gate_up");

  TORCH_CHECK(
      gate_up.dim() == 2,
      "gate_up must be 2D [rows, 2 * width]");

  TORCH_CHECK(
      gate_up.size(1) % 2 == 0,
      "gate_up width must be even");

  const int64_t rows = gate_up.size(0);
  const int64_t width = gate_up.size(1) / 2;

  TORCH_CHECK(
      dh.dim() == 2 &&
          dh.size(0) == rows &&
          dh.size(1) == width,
      "dh must have shape [rows, width]");

  check_route_weights(p_s, gate_up);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(gate_up));

  auto d_gate = torch::empty(
      {rows, width},
      gate_up.options());

  auto d_up = torch::empty(
      {rows, width},
      gate_up.options());

  auto dp_s =
      p_s.has_value()
          ? torch::zeros({rows}, gate_up.options())
          : torch::empty({0}, gate_up.options());

  const int64_t n = dh.numel();

  if (n == 0) {
    return {d_gate, d_up, dp_s};
  }

  int threads = 0;
  int64_t blocks = 0;
  launch_1d(n, threads, blocks);

  auto stream = at::cuda::getCurrentCUDAStream();

  const float* gate_ptr = gate_up.data_ptr<float>();
  const float* up_ptr = gate_ptr + width;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      dh.scalar_type(),
      "clamp_swiglu_weighted_packed_backward_cuda",
      [&] {
        clamp_swiglu_weighted_backward_kernel<scalar_t>
            <<<blocks, threads, 0, stream>>>(
                dh.data_ptr<scalar_t>(),
                gate_ptr,
                up_ptr,
                p_s.has_value()
                    ? p_s->data_ptr<float>()
                    : nullptr,
                d_gate.data_ptr<float>(),
                d_up.data_ptr<float>(),
                n,
                width,
                2 * width,
                p_s.has_value());

        if (p_s.has_value()) {
          int dp_threads = 0;
          int64_t dp_blocks = 0;
          launch_1d(rows, dp_threads, dp_blocks);

          clamp_swiglu_weighted_dp_s_kernel<scalar_t>
              <<<dp_blocks, dp_threads, 0, stream>>>(
                  dh.data_ptr<scalar_t>(),
                  gate_ptr,
                  up_ptr,
                  dp_s.data_ptr<float>(),
                  rows,
                  width,
                  2 * width);
        }
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {d_gate, d_up, dp_s};
}