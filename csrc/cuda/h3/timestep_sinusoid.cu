// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// MiniMax-H3 sinusoidal timestep features (RFC #420 `timestep_sinusoid_h3`).
//
//   freq[k]   = expf((c * k) * (1 / half))        c = (float)(-ln(max_period))
//   arg[t, k] = t[t] * freq[k]
//   out[t]    = [cosf(arg) | sinf(arg)]           (T, 2 * half) FP32
//
// Every value is produced by the same FP32 operation sequence as diffusers'
// get_timestep_embedding on CUDA (scalar * arange, multiply by the reciprocal
// of the CPU-scalar divisor, exp, multiply, cos/sin), so the output is meant
// to be bitwise equal to that path. Each element depends only on (t, k):
// batch-, position- and repeat-invariant by construction. Build without
// --use_fast_math: expf/sinf/cosf must stay the precise libdevice functions.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cmath>
#include <cstdint>

namespace {

__global__ void h3_timestep_sinusoid_kernel(
    const float* __restrict__ timestep,
    float* __restrict__ out,
    int64_t num_timesteps,
    int64_t half,
    float neg_log_max_period,
    float inv_half) {
  const int64_t total = num_timesteps * half;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t row = idx / half;
    const int64_t k = idx - row * half;
    // __fmul_rn keeps each product a separately rounded FP32 multiply (no FMA
    // contraction), matching the two elementwise torch kernels it replays.
    const float exponent = __fmul_rn(__fmul_rn(neg_log_max_period, static_cast<float>(k)), inv_half);
    const float freq = expf(exponent);
    const float arg = __fmul_rn(timestep[row], freq);
    float* out_row = out + row * 2 * half;
    out_row[k] = cosf(arg);
    out_row[half + k] = sinf(arg);
  }
}

}  // namespace

torch::Tensor h3_timestep_sinusoid_forward(torch::Tensor timestep, int64_t num_channels,
                                           double max_period, bool check_range) {
  TORCH_CHECK(timestep.is_cuda(), "timestep must be a CUDA tensor");
  TORCH_CHECK(timestep.scalar_type() == at::kFloat, "timestep must be float32, got ",
              timestep.scalar_type());
  TORCH_CHECK(timestep.dim() == 1, "timestep must be 1-D (num_timesteps,)");
  TORCH_CHECK(timestep.numel() > 0, "timestep must hold at least one timestep");
  TORCH_CHECK(num_channels > 0 && num_channels % 2 == 0,
              "num_channels must be a positive even number, got ", num_channels);
  TORCH_CHECK(max_period > 0.0, "max_period must be positive");

  const c10::cuda::CUDAGuard device_guard(timestep.device());
  auto t = timestep.contiguous();
  // Check at the native boundary, including direct extension calls. The explicit
  // opt-out is for already-validated inputs and kernel-only profiling.
  if (check_range) {
    TORCH_CHECK_VALUE(((t >= 0) & (t <= 1)).all().item<bool>(),
                      "timestep must be finite and lie in [0, 1]: H3 consumes t = 1 - sigma "
                      "unscaled");
  }
  const int64_t num_timesteps = t.size(0);
  const int64_t half = num_channels / 2;
  auto out = torch::empty({num_timesteps, num_channels}, t.options());

  // torch evaluates `-math.log(max_period) * arange` with the Python double
  // rounded to the FP32 opmath type, and `x / half` as x * (1 / half) in FP32.
  const float neg_log_max_period = static_cast<float>(-std::log(max_period));
  const float inv_half = 1.0f / static_cast<float>(half);

  const int threads = 256;
  const int64_t total = num_timesteps * half;
  const int64_t blocks = std::min<int64_t>((total + threads - 1) / threads, 65535);
  auto stream = at::cuda::getCurrentCUDAStream();
  h3_timestep_sinusoid_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
      t.data_ptr<float>(), out.data_ptr<float>(), num_timesteps, half, neg_log_max_period,
      inv_half);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
