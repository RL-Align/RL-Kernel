// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace {
// Recursive fixed 32-token balanced trees. Each
// sample/channel has its own sequence; batching never changes the tree.
__global__ void token_partials(const float* x, float* p, int64_t seq, int64_t hidden,
                               int64_t tiles) {
  const int64_t h = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t tile = blockIdx.y;
  const int64_t batch = blockIdx.z;
  if (h >= hidden) return;
  float v[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    int64_t row = tile * 32 + i;
    v[i] = row < seq ? x[(batch * seq + row) * hidden + h] : 0.0f;
  }
#pragma unroll
  for (int stride = 1; stride < 32; stride *= 2) {
#pragma unroll
    for (int i = 0; i < 32; i += 2 * stride) v[i] = __fadd_rn(v[i], v[i + stride]);
  }
  p[(batch * tiles + tile) * hidden + h] = v[0];
}

}

torch::Tensor qwenimage_token_sum(torch::Tensor x) {
#ifdef __FAST_MATH__
  TORCH_CHECK(false, "Qwen-Image requires a build without fast math");
#endif
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kFloat && x.dim() == 3 && x.is_contiguous(),
              "token_sum requires contiguous CUDA float32 [B,S,H]");
  const auto b = x.size(0), s = x.size(1), h = x.size(2);
  TORCH_CHECK(b > 0 && b <= 65535 && s > 0 && s <= 32 * 65535 && h > 0,
              "token_sum geometry out of bounds");
  const c10::cuda::CUDAGuard guard(x.device());
  auto current = x;
  auto stream = at::cuda::getCurrentCUDAStream();
  int64_t rows = s;
  while (rows > 1) {
    const auto tiles = (rows + 31) / 32;
    auto partials = torch::empty({b, tiles, h}, x.options());
    token_partials<<<dim3((h + 255) / 256, tiles, b), 256, 0, stream>>>(
        current.data_ptr<float>(), partials.data_ptr<float>(), rows, h, tiles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    current = partials;
    rows = tiles;
  }
  return current.view({b, h});
}
