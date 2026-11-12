// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// CUDA RoPE kernel for SM90 (GPT-NeoX / HF rotate-half), matching NativeRoPEOp.
//
// For a row (one [B, H, S] token vector of width D, half = D / 2) at sequence
// index s = row % S and pair index i in [0, half):
//
//   c = cos[s, i]              (fp32, precomputed to match the reference math)
//   sn = sin[s, i] * sin_sign  (sin_sign = +1 forward, -1 backward)
//   out[i]        = x[i] * c - x[i + half] * sn
//   out[i + half] = x[i + half] * c + x[i] * sn
//
// The elementwise rotation is done in fp32 and rounded back to the input dtype.
// Backward reuses the same kernel with sin_sign = -1 (RoPE is an orthogonal
// per-position rotation, so grad_x = grad_out * cos - rotate_half(grad_out) * sin).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

template <typename scalar_t>
__global__ void rope_apply_sm90_kernel(
    const scalar_t* __restrict__ x,   // [n_rows, D]
    const float* __restrict__ cos,    // [S, half]
    const float* __restrict__ sin,    // [S, half]
    scalar_t* __restrict__ out,       // [n_rows, D]
    const int64_t n_rows,
    const int S,
    const int half,
    const float sin_sign) {
  const int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  const int64_t total = n_rows * static_cast<int64_t>(half);
  if (idx >= total) {
    return;
  }

  const int64_t row = idx / half;
  const int i = static_cast<int>(idx % half);
  const int seq = static_cast<int>(row % S);

  const float c = cos[seq * half + i];
  const float sn = sin[seq * half + i] * sin_sign;

  const int64_t base = row * (2LL * half);
  const float x1 = static_cast<float>(x[base + i]);
  const float x2 = static_cast<float>(x[base + i + half]);

  out[base + i] = static_cast<scalar_t>(x1 * c - x2 * sn);
  out[base + i + half] = static_cast<scalar_t>(x2 * c + x1 * sn);
}

}  // namespace

// x: [n_rows, D] contiguous (any float dtype); cos/sin: [S, half] fp32 contiguous.
torch::Tensor rope_apply_sm90(
    torch::Tensor x,
    torch::Tensor cos,
    torch::Tensor sin,
    double sin_sign) {
  TORCH_CHECK(x.is_cuda(), "rope: x must be a CUDA tensor");
  TORCH_CHECK(x.dim() == 2, "rope: x must be 2-D [n_rows, D]");
  TORCH_CHECK(x.is_contiguous(), "rope: x must be contiguous");
  TORCH_CHECK(cos.is_cuda() && sin.is_cuda(), "rope: cos/sin must be CUDA tensors");
  TORCH_CHECK(cos.scalar_type() == torch::kFloat32 && sin.scalar_type() == torch::kFloat32,
              "rope: cos/sin must be fp32");
  TORCH_CHECK(cos.is_contiguous() && sin.is_contiguous(), "rope: cos/sin must be contiguous");

  const int64_t n_rows = x.size(0);
  const int64_t D = x.size(1);
  TORCH_CHECK(D % 2 == 0, "rope: head_dim must be even");
  const int half = static_cast<int>(D / 2);
  const int S = static_cast<int>(cos.size(0));
  TORCH_CHECK(cos.size(1) == half && sin.size(1) == half,
              "rope: cos/sin last dim must equal head_dim/2");
  TORCH_CHECK(S > 0 && n_rows % S == 0,
              "rope: n_rows must be divisible by seq length S");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  auto out = torch::empty_like(x);

  const int64_t total = n_rows * static_cast<int64_t>(half);
  const int threads = 256;
  const int64_t blocks = (total + threads - 1) / threads;
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "rope_apply_sm90", [&] {
        rope_apply_sm90_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            x.data_ptr<scalar_t>(),
            cos.data_ptr<float>(),
            sin.data_ptr<float>(),
            out.data_ptr<scalar_t>(),
            n_rows,
            S,
            half,
            static_cast<float>(sin_sign));
      });
  return out;
}
