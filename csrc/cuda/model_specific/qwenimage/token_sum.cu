// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

bool qwenimage_strict_math_compiled();

namespace {
// Neumaier step: add v to the running sum s and accumulate the rounding error in c.
__device__ __forceinline__ void neumaier_add(float& s, float& c, float v) {
  const float t = __fadd_rn(s, v);
  c = __fadd_rn(c, fabsf(s) >= fabsf(v) ? __fadd_rn(__fsub_rn(s, t), v)
                                         : __fadd_rn(__fsub_rn(v, t), s));
  s = t;
}

// Fixed 32-token tiles, folded level by level. Inside a tile the tokens are added in ascending
// order with Neumaier compensation, and a tile's carried correction joins the next level's
// correction, so small contributions survive next to large cancelling ones. Each
// sample/channel has its own sequence; batching never changes the order.
// The first level reads x only; later levels read the previous level's sums and corrections.
__global__ void token_partials(const float* x, const float* x_corr, float* p, float* p_corr,
                               int64_t seq, int64_t hidden, int64_t tiles) {
  const int64_t h = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t tile = blockIdx.y;
  const int64_t batch = blockIdx.z;
  if (h >= hidden) return;
  float s = 0.0f, c = 0.0f;
#pragma unroll 4
  for (int i = 0; i < 32; ++i) {
    const int64_t row = tile * 32 + i;
    if (row < seq) {
      const int64_t at = (batch * seq + row) * hidden + h;
      neumaier_add(s, c, x[at]);
      if (x_corr != nullptr) c = __fadd_rn(c, x_corr[at]);
    }
  }
  const int64_t out = (batch * tiles + tile) * hidden + h;
  p[out] = s;
  p_corr[out] = c;
}

}

torch::Tensor qwenimage_token_sum(torch::Tensor x) {
  TORCH_CHECK(qwenimage_strict_math_compiled(), "Qwen-Image requires a build without fast math");
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kFloat && x.dim() == 3 && x.is_contiguous(),
              "token_sum requires contiguous CUDA float32 [B,S,H]");
  const auto b = x.size(0), s = x.size(1), h = x.size(2);
  TORCH_CHECK(b > 0 && b <= 65535 && s > 0 && s <= 32 * 65535 && h > 0,
              "token_sum geometry out of bounds");
  const c10::cuda::CUDAGuard guard(x.device());
  auto current = x;
  torch::Tensor correction;
  auto stream = at::cuda::getCurrentCUDAStream();
  int64_t rows = s;
  while (rows > 1) {
    const auto tiles = (rows + 31) / 32;
    auto partials = torch::empty({b, tiles, h}, x.options());
    auto corrections = torch::empty({b, tiles, h}, x.options());
    token_partials<<<dim3((h + 255) / 256, tiles, b), 256, 0, stream>>>(
        current.data_ptr<float>(), correction.defined() ? correction.data_ptr<float>() : nullptr,
        partials.data_ptr<float>(), corrections.data_ptr<float>(), rows, h, tiles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    current = partials;
    correction = corrections;
    rows = tiles;
  }
  // One IEEE add of the final sum and its carried correction.
  if (correction.defined()) current = current + correction;
  return current.view({b, h});
}
