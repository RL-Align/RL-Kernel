// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// P5-2 clamp_swiglu_weighted CUDA kernels.
//
//   h = SiLU(min(gate, 10)) * clamp(up, -10, 10) * p_s
//
// Split out of csrc/cuda/activation.cu: that file holds the generic WS1
// elementwise activations, while this operator belongs to the MoE expert
// contract (route weight p_s, the dp_s reduction, and the packed gate|up
// layout the routed GEMM emits).
//
// All math is FP32 with a single BF16 round on the output, and every step is
// token-local -- no cross-row reduction -- so batch size and padding cannot
// change a row's bytes. dp_s reduces along one row in ascending column order.
//
// The validation helpers below are duplicated from activation.cu rather than
// shared through a header: they are a handful of lines each, live in an
// anonymous namespace, and keeping them local lets the two translation units
// move independently.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace {

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
  TORCH_CHECK(
      t.scalar_type() == at::kFloat || t.scalar_type() == at::kHalf ||
          t.scalar_type() == at::kBFloat16,
      name,
      " must be fp32, fp16, or bf16, got ",
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
      " must be on the same device, got ",
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

}  // namespace

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
