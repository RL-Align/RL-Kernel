// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>

namespace {

template <typename scalar_t, bool inverse, bool backward>
__global__ void latent_affine_kernel(
    const scalar_t* x, const float* mean, const float* std_inv,
    scalar_t* out, int64_t n, int64_t spatial) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += int64_t(blockDim.x) * gridDim.x) {
    const int c = (i / spatial) % 16;
    const float value = static_cast<float>(x[i]);
    float result;
    if constexpr (backward) {
      result = inverse ? __fdiv_rn(value, std_inv[c]) : __fmul_rn(value, std_inv[c]);
    } else if constexpr (inverse) {
      result = __fadd_rn(__fdiv_rn(value, std_inv[c]), mean[c]);
    } else {
      result = __fmul_rn(__fsub_rn(value, mean[c]), std_inv[c]);
    }
    out[i] = static_cast<scalar_t>(result);
  }
}

void validate_x(const torch::Tensor& x, int64_t block_size) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "x must be contiguous CUDA");
  TORCH_CHECK(x.dim() == 4 || x.dim() == 5, "x must be [B,16,H,W] or [B,16,1,H,W]");
  TORCH_CHECK(x.size(1) == 16, "x must have 16 channels");
  TORCH_CHECK(x.dim() != 5 || x.size(2) == 1, "Phase A supports one latent frame");
  TORCH_CHECK(x.numel() > 0, "latent dimensions must be positive");
  TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kBFloat16,
              "x must be float32 or bfloat16");
  TORCH_CHECK(block_size == 128 || block_size == 256 || block_size == 512,
              "block_size must be 128, 256, or 512");
#if defined(__CUDA_FTZ) && __CUDA_FTZ
  TORCH_CHECK(false, "latent_normalize requires a build without fast-math/FTZ");
#endif
}

void validate_stat(const torch::Tensor& stat, const torch::Tensor& x) {
  TORCH_CHECK(stat.device() == x.device(), "statistics must be on x.device");
  TORCH_CHECK(stat.is_contiguous() && stat.dim() == 1 && stat.size(0) == 16,
              "statistics must be contiguous [16] vectors");
  TORCH_CHECK(stat.scalar_type() == at::kFloat, "statistics must be float32");
  TORCH_CHECK(!stat.requires_grad(), "statistics must be fixed model buffers");
}

template <bool backward>
torch::Tensor launch(const torch::Tensor& x, const torch::Tensor& mean,
                     const torch::Tensor& std_inv, bool inverse, int64_t block_size) {
  validate_x(x, block_size);
  validate_stat(std_inv, x);
  if constexpr (!backward) validate_stat(mean, x);
  const c10::cuda::CUDAGuard guard(x.device());
  auto out = torch::empty_like(x);
  const int64_t n = x.numel();
  const int64_t spatial = x.size(-2) * x.size(-1);
  const int blocks = std::min<int64_t>((n + block_size - 1) / block_size, 4096);
  const float* mu = backward ? nullptr : mean.data_ptr<float>();
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, x.scalar_type(),
                               "latent_normalize", [&] {
    if (inverse) {
      latent_affine_kernel<scalar_t, true, backward><<<blocks, block_size, 0, stream>>>(
          x.data_ptr<scalar_t>(), mu, std_inv.data_ptr<float>(),
          out.data_ptr<scalar_t>(), n, spatial);
    } else {
      latent_affine_kernel<scalar_t, false, backward><<<blocks, block_size, 0, stream>>>(
          x.data_ptr<scalar_t>(), mu, std_inv.data_ptr<float>(),
          out.data_ptr<scalar_t>(), n, spatial);
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
}  // namespace

torch::Tensor latent_normalize_forward_cuda(torch::Tensor x, torch::Tensor mean,
    torch::Tensor std_inv, bool inverse, int64_t block_size) {
  return launch<false>(x, mean, std_inv, inverse, block_size);
}

torch::Tensor latent_normalize_backward_cuda(torch::Tensor grad,
    torch::Tensor std_inv, bool inverse, int64_t block_size) {
  return launch<true>(grad, torch::Tensor(), std_inv, inverse, block_size);
}
