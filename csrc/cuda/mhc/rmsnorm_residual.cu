#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <climits>
#include <vector>

namespace {
using BF = __nv_bfloat16;

__device__ float load(const BF* p, int64_t i) { return __bfloat162float(p[i]); }

__global__ void forward_kernel(const BF* x, const BF* gamma, BF* y,
                               BF* residual, float* saved_r, int d, float eps) {
  int64_t row = blockIdx.x;
  int64_t base = row * d;
  __shared__ float r;

  if (threadIdx.x == 0) {
    float s = 0.0f;
    for (int k = 0; k < d; ++k) {
      float v = load(x, base + k);
      s = __fadd_rn(s, __fmul_rn(v, v));
    }
    float m = __fdiv_rn(s, static_cast<float>(d));
    r = rsqrtf(__fadd_rn(m, eps));
    saved_r[row] = r;
  }

  __syncthreads();

  for (int k = threadIdx.x; k < d; k += blockDim.x) {
    float v = __fmul_rn(__fmul_rn(load(x, base + k), r), load(gamma, k));
    y[base + k] = __float2bfloat16_rn(v);
    residual[base + k] = x[base + k];
  }
}

__global__ void dx_kernel(const BF* dy, const BF* dr, const BF* x,
                          const BF* gamma, const float* saved_r, float* dx,
                          int d) {
  int64_t base = static_cast<int64_t>(blockIdx.x) * d;
  float r = saved_r[blockIdx.x];
  __shared__ float q;

  if (threadIdx.x == 0) {
    float acc = 0.0f;
    for (int k = 0; k < d; ++k) {
      float u = __fmul_rn(load(dy, base + k), load(gamma, k));
      acc = __fadd_rn(acc, __fmul_rn(u, load(x, base + k)));
    }
    q = acc;
  }

  __syncthreads();

  float r3 = __fmul_rn(__fmul_rn(r, r), r);
  for (int k = threadIdx.x; k < d; k += blockDim.x) {
    float u = __fmul_rn(load(dy, base + k), load(gamma, k));
    float rhs = __fdiv_rn(__fmul_rn(__fmul_rn(load(x, base + k), r3), q),
                          static_cast<float>(d));
    float norm = __fsub_rn(__fmul_rn(r, u), rhs);
    dx[base + k] = __fadd_rn(norm, load(dr, base + k));
  }
}

__global__ void dgamma_kernel(const BF* dy, const BF* x, const float* r,
                              float* dg, int t, int d) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= d) return;

  float acc = 0.0f;
  for (int row = 0; row < t; ++row) {
    int64_t i = static_cast<int64_t>(row) * d + k;
    float v = __fmul_rn(__fmul_rn(load(dy, i), load(x, i)), r[row]);
    acc = __fadd_rn(acc, v);
  }
  dg[k] = acc;
}

const BF* input_ptr(const torch::Tensor& x) {
  return reinterpret_cast<const BF*>(x.data_ptr<at::BFloat16>());
}

BF* output_ptr(torch::Tensor& x) {
  return reinterpret_cast<BF*>(x.data_ptr<at::BFloat16>());
}

void check_bf16(const torch::Tensor& v, const torch::Tensor& x,
                const char* name) {
  TORCH_CHECK(v.is_cuda() && v.device() == x.device(), name, ": wrong device");
  TORCH_CHECK(v.scalar_type() == torch::kBFloat16, name, ": expected BF16");
  TORCH_CHECK(v.is_contiguous(), name, ": expected contiguous");
}

void check_inputs(const torch::Tensor& x, const torch::Tensor& gamma) {
  TORCH_CHECK(x.is_cuda(), "x: expected CUDA");
  check_bf16(x, x, "x");
  check_bf16(gamma, x, "gamma");
  TORCH_CHECK(x.dim() == 2 && x.size(0) > 0 && x.size(0) <= INT_MAX,
              "x: expected nonempty [T, D], T <= INT_MAX");
  TORCH_CHECK(x.size(1) == 128 || x.size(1) == 4096,
              "candidate supports D=128/4096");
  TORCH_CHECK(gamma.dim() == 1 && gamma.size(0) == x.size(1),
              "gamma: expected [D]");
}
}  // namespace

std::vector<torch::Tensor> mhc_rmsnorm_residual_forward(torch::Tensor x,
                                                        torch::Tensor gamma,
                                                        double eps) {
  check_inputs(x, gamma);
  TORCH_CHECK(eps == 1e-6, "eps is fixed at 1e-6");
  c10::cuda::CUDAGuard guard(x.device());

  auto y = torch::empty_like(x);
  auto residual = torch::empty_like(x);
  auto r = torch::empty({x.size(0)}, x.options().dtype(torch::kFloat32));
  auto stream = at::cuda::getCurrentCUDAStream();

  forward_kernel<<<static_cast<int>(x.size(0)), 128, 0, stream>>>(
      input_ptr(x), input_ptr(gamma), output_ptr(y), output_ptr(residual),
      r.data_ptr<float>(), static_cast<int>(x.size(1)),
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y, residual, r};
}

std::vector<torch::Tensor> mhc_rmsnorm_residual_backward(torch::Tensor dy,
                                                         torch::Tensor dr,
                                                         torch::Tensor x,
                                                         torch::Tensor gamma,
                                                         torch::Tensor r) {
  check_inputs(x, gamma);
  check_bf16(dy, x, "dy");
  check_bf16(dr, x, "d_residual");
  TORCH_CHECK(dy.sizes() == x.sizes() && dr.sizes() == x.sizes(),
              "gradient shape mismatch");
  TORCH_CHECK(r.is_cuda() && r.device() == x.device() && r.is_contiguous() &&
                  r.scalar_type() == torch::kFloat32 && r.dim() == 1 &&
                  r.size(0) == x.size(0),
              "saved r: expected contiguous FP32 [T]");
  c10::cuda::CUDAGuard guard(x.device());
  auto dx = torch::empty(x.sizes(), x.options().dtype(torch::kFloat32));
  auto dg = torch::empty({x.size(1)}, x.options().dtype(torch::kFloat32));
  int t = static_cast<int>(x.size(0));
  int d = static_cast<int>(x.size(1));
  auto stream = at::cuda::getCurrentCUDAStream();

  dx_kernel<<<t, 128, 0, stream>>>(input_ptr(dy), input_ptr(dr), input_ptr(x),
                                   input_ptr(gamma), r.data_ptr<float>(),
                                   dx.data_ptr<float>(), d);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  dgamma_kernel<<<(d + 127) / 128, 128, 0, stream>>>(
      input_ptr(dy), input_ptr(x), r.data_ptr<float>(), dg.data_ptr<float>(), t,
      d);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dx, dg};
}