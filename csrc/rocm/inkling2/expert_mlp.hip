// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// Inkling 2 Mini softcapped expert gate/up projection (INK-26), ROCm-native.
// Compiled only for ROCm builds (the CUDA build keeps csrc/cuda/inkling2/expert_mlp.cu);
// exports the same C++ symbols and uses the same transcendental formulation
// (cap * tanhf(x / cap), 1 / (1 + expf(-x))) and FP32 cast points.
//
// Determinism contract:
//   * forward : gate/up = bias + sum_h x[r,h] * W[i,h]   (h ascending, FP32 FMA chain)
//               out = silu(gate_cap*tanh(gate/gate_cap)) * (up_cap*tanh(up/up_cap)), one cast to storage dtype
//               FP32 gate/up pre-activations are saved for backward.
//   * d_gate/d_up are materialised once in FP32 from (dy, gate_pre, up_pre) with the CUDA formula.
//   * d_hidden[r,h] = for i ascending: acc += d_gate*Wg[i,h]; acc += d_up*Wu[i,h]
//   * dW{g,u}[i,h]  = sum over rows r ascending of d_{gate,up}[r,i] * x[r,h]
//   * db{g,u}[i]    = sum over rows r ascending
// Register micro-tiling and LDS staging only change which thread owns an output,
// never its accumulation order, so results are independent of launch shape, batch
// size, and other rows. No float atomics.
//
// Tuned for MI300X (gfx942): wave64, 256-thread blocks, 64x64 output tile, 4x4
// register micro-tile per thread (x2 accumulator sets for gate+up), 16-deep K
// step staged in LDS (<= 3 x 16 x 65 x 4 B ~ 12 KB, so several blocks per CU).
// Sized for H=2304 / I=1152, gate softcap 4, up softcap 25, but shape-generic.

#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <c10/core/DeviceGuard.h>
#include <c10/hip/HIPException.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#include <cstdint>
#include <vector>

namespace {

constexpr int kThreads = 256;  // 4 wave64
constexpr int kTile = 64;
constexpr int kTileK = 16;
constexpr int kMicro = 4;
constexpr int kPad = kTile + 1;
static_assert(kThreads == (kTile / kMicro) * (kTile / kMicro), "thread layout");
static_assert(kThreads % 64 == 0, "block must be whole wavefronts");

__device__ __forceinline__ float softcap_f32(float value, float cap) {
  return cap * tanhf(value / cap);
}

__device__ __forceinline__ float softcap_grad_f32(float value, float cap) {
  const float t = tanhf(value / cap);
  return 1.0f - t * t;
}

__device__ __forceinline__ float silu_f32(float value) {
  const float sigmoid = 1.0f / (1.0f + expf(-value));
  return value * sigmoid;
}

__device__ __forceinline__ float silu_grad_f32(float value) {
  const float sigmoid = 1.0f / (1.0f + expf(-value));
  return sigmoid * (1.0f + value * (1.0f - sigmoid));
}

// out[r,i] = silu(cap(gate)) * cap(up); gate/up accumulate over h in ascending order.
template <typename scalar_t>
__global__ void __launch_bounds__(kThreads) gate_up_forward_kernel(
    const scalar_t* __restrict__ hidden,       // [rows, H]
    const scalar_t* __restrict__ gate_weight,  // [I, H]
    const scalar_t* __restrict__ up_weight,    // [I, H]
    const float* __restrict__ gate_bias,
    const float* __restrict__ up_bias,
    scalar_t* __restrict__ output,
    float* __restrict__ gate_pre,
    float* __restrict__ up_pre,
    int64_t rows, int64_t H, int64_t I, float gate_cap, float up_cap) {
  const int64_t m_base = static_cast<int64_t>(blockIdx.x) * kTile;
  const int64_t n_base = static_cast<int64_t>(blockIdx.y) * kTile;
  __shared__ float as[kTileK][kPad];
  __shared__ float gs[kTileK][kPad];
  __shared__ float us[kTileK][kPad];
  const int tid = threadIdx.x;
  const int tx = tid % (kTile / kMicro);
  const int ty = tid / (kTile / kMicro);

  float g[kMicro][kMicro], u[kMicro][kMicro];
#pragma unroll
  for (int i = 0; i < kMicro; ++i) {
#pragma unroll
    for (int j = 0; j < kMicro; ++j) {
      const int64_t n = n_base + tx * kMicro + j;
      g[i][j] = (gate_bias != nullptr && n < I) ? gate_bias[n] : 0.0f;
      u[i][j] = (up_bias != nullptr && n < I) ? up_bias[n] : 0.0f;
    }
  }

  for (int64_t k0 = 0; k0 < H; k0 += kTileK) {
    for (int idx = tid; idx < kTile * kTileK; idx += kThreads) {
      const int r = idx / kTileK, k = idx % kTileK;  // 16 consecutive k per row
      float va = 0.0f, vg = 0.0f, vu = 0.0f;
      if (k0 + k < H) {
        if (m_base + r < rows) va = static_cast<float>(hidden[(m_base + r) * H + k0 + k]);
        if (n_base + r < I) {
          vg = static_cast<float>(gate_weight[(n_base + r) * H + k0 + k]);
          vu = static_cast<float>(up_weight[(n_base + r) * H + k0 + k]);
        }
      }
      as[k][r] = va;
      gs[k][r] = vg;
      us[k][r] = vu;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < kTileK; ++k) {
      float av[kMicro], gv[kMicro], uv[kMicro];
#pragma unroll
      for (int i = 0; i < kMicro; ++i) av[i] = as[k][ty * kMicro + i];
#pragma unroll
      for (int j = 0; j < kMicro; ++j) {
        gv[j] = gs[k][tx * kMicro + j];
        uv[j] = us[k][tx * kMicro + j];
      }
#pragma unroll
      for (int i = 0; i < kMicro; ++i)
#pragma unroll
        for (int j = 0; j < kMicro; ++j) {
          g[i][j] = fmaf(av[i], gv[j], g[i][j]);
          u[i][j] = fmaf(av[i], uv[j], u[i][j]);
        }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < kMicro; ++i) {
    const int64_t r = m_base + ty * kMicro + i;
    if (r >= rows) continue;
#pragma unroll
    for (int j = 0; j < kMicro; ++j) {
      const int64_t n = n_base + tx * kMicro + j;
      if (n >= I) continue;
      const int64_t index = r * I + n;
      gate_pre[index] = g[i][j];
      up_pre[index] = u[i][j];
      const float gate_capped = softcap_f32(g[i][j], gate_cap);
      const float up_capped = softcap_f32(u[i][j], up_cap);
      output[index] = static_cast<scalar_t>(silu_f32(gate_capped) * up_capped);
    }
  }
}

// d_gate / d_up in FP32 (same expression tree as the CUDA kernel).
template <typename scalar_t>
__global__ void __launch_bounds__(kThreads) gate_up_grad_pre_kernel(
    const scalar_t* __restrict__ dy,
    const float* __restrict__ gate_pre,
    const float* __restrict__ up_pre,
    float* __restrict__ d_gate,
    float* __restrict__ d_up,
    int64_t total, float gate_cap, float up_cap) {
  const int64_t offset = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (offset >= total) return;
  const float gate_capped = softcap_f32(gate_pre[offset], gate_cap);
  const float up_capped = softcap_f32(up_pre[offset], up_cap);
  const float dyv = static_cast<float>(dy[offset]);
  d_gate[offset] = dyv * up_capped * silu_grad_f32(gate_capped) * softcap_grad_f32(gate_pre[offset], gate_cap);
  d_up[offset] = dyv * silu_f32(gate_capped) * softcap_grad_f32(up_pre[offset], up_cap);
}

// d_hidden[r,h] = for i ascending: acc += d_gate[r,i]*Wg[i,h]; acc += d_up[r,i]*Wu[i,h]
template <typename scalar_t>
__global__ void __launch_bounds__(kThreads) gate_up_backward_hidden_kernel(
    const float* __restrict__ d_gate,          // [rows, I]
    const float* __restrict__ d_up,            // [rows, I]
    const scalar_t* __restrict__ gate_weight,  // [I, H]
    const scalar_t* __restrict__ up_weight,    // [I, H]
    scalar_t* __restrict__ d_hidden,           // [rows, H]
    int64_t rows, int64_t H, int64_t I) {
  const int64_t m_base = static_cast<int64_t>(blockIdx.x) * kTile;
  const int64_t n_base = static_cast<int64_t>(blockIdx.y) * kTile;
  __shared__ float ag[kTileK][kPad];
  __shared__ float au[kTileK][kPad];
  __shared__ float gs[kTileK][kPad];
  __shared__ float us[kTileK][kPad];
  const int tid = threadIdx.x;
  const int tx = tid % (kTile / kMicro);
  const int ty = tid / (kTile / kMicro);

  float acc[kMicro][kMicro];
#pragma unroll
  for (int i = 0; i < kMicro; ++i)
#pragma unroll
    for (int j = 0; j < kMicro; ++j) acc[i][j] = 0.0f;

  for (int64_t k0 = 0; k0 < I; k0 += kTileK) {
    for (int idx = tid; idx < kTile * kTileK; idx += kThreads) {
      const int r = idx / kTileK, k = idx % kTileK;
      float vg = 0.0f, vu = 0.0f;
      if (m_base + r < rows && k0 + k < I) {
        vg = d_gate[(m_base + r) * I + k0 + k];
        vu = d_up[(m_base + r) * I + k0 + k];
      }
      ag[k][r] = vg;
      au[k][r] = vu;
    }
    for (int idx = tid; idx < kTile * kTileK; idx += kThreads) {
      const int k = idx / kTile, n = idx % kTile;  // coalesced along h
      float wg = 0.0f, wu = 0.0f;
      if (k0 + k < I && n_base + n < H) {
        wg = static_cast<float>(gate_weight[(k0 + k) * H + n_base + n]);
        wu = static_cast<float>(up_weight[(k0 + k) * H + n_base + n]);
      }
      gs[k][n] = wg;
      us[k][n] = wu;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < kTileK; ++k) {
      float dg[kMicro], du[kMicro], wg[kMicro], wu[kMicro];
#pragma unroll
      for (int i = 0; i < kMicro; ++i) {
        dg[i] = ag[k][ty * kMicro + i];
        du[i] = au[k][ty * kMicro + i];
      }
#pragma unroll
      for (int j = 0; j < kMicro; ++j) {
        wg[j] = gs[k][tx * kMicro + j];
        wu[j] = us[k][tx * kMicro + j];
      }
#pragma unroll
      for (int i = 0; i < kMicro; ++i)
#pragma unroll
        for (int j = 0; j < kMicro; ++j) {
          acc[i][j] = fmaf(dg[i], wg[j], acc[i][j]);
          acc[i][j] = fmaf(du[i], wu[j], acc[i][j]);
        }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < kMicro; ++i) {
    const int64_t r = m_base + ty * kMicro + i;
    if (r >= rows) continue;
#pragma unroll
    for (int j = 0; j < kMicro; ++j) {
      const int64_t h = n_base + tx * kMicro + j;
      if (h < H) d_hidden[r * H + h] = static_cast<scalar_t>(acc[i][j]);
    }
  }
}

// dW[i,h] = sum_r d[r,i] * x[r,h], r ascending; computes both gate and up in one pass over x.
template <typename scalar_t>
__global__ void __launch_bounds__(kThreads) gate_up_backward_weight_kernel(
    const float* __restrict__ d_gate,  // [rows, I]
    const float* __restrict__ d_up,    // [rows, I]
    const scalar_t* __restrict__ hidden,  // [rows, H]
    scalar_t* __restrict__ d_gate_weight,  // [I, H]
    scalar_t* __restrict__ d_up_weight,    // [I, H]
    int64_t rows, int64_t H, int64_t I) {
  const int64_t m_base = static_cast<int64_t>(blockIdx.y) * kTile;  // intermediate index
  const int64_t n_base = static_cast<int64_t>(blockIdx.x) * kTile;  // hidden index
  __shared__ float ag[kTileK][kPad];
  __shared__ float au[kTileK][kPad];
  __shared__ float bs[kTileK][kPad];
  const int tid = threadIdx.x;
  const int tx = tid % (kTile / kMicro);
  const int ty = tid / (kTile / kMicro);

  float cg[kMicro][kMicro], cu[kMicro][kMicro];
#pragma unroll
  for (int i = 0; i < kMicro; ++i)
#pragma unroll
    for (int j = 0; j < kMicro; ++j) cg[i][j] = cu[i][j] = 0.0f;

  for (int64_t k0 = 0; k0 < rows; k0 += kTileK) {
    for (int idx = tid; idx < kTile * kTileK; idx += kThreads) {
      const int k = idx / kTile, col = idx % kTile;  // coalesced along i / h
      float vg = 0.0f, vu = 0.0f, vb = 0.0f;
      if (k0 + k < rows) {
        if (m_base + col < I) {
          vg = d_gate[(k0 + k) * I + m_base + col];
          vu = d_up[(k0 + k) * I + m_base + col];
        }
        if (n_base + col < H) vb = static_cast<float>(hidden[(k0 + k) * H + n_base + col]);
      }
      ag[k][col] = vg;
      au[k][col] = vu;
      bs[k][col] = vb;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < kTileK; ++k) {
      float dg[kMicro], du[kMicro], bv[kMicro];
#pragma unroll
      for (int i = 0; i < kMicro; ++i) {
        dg[i] = ag[k][ty * kMicro + i];
        du[i] = au[k][ty * kMicro + i];
      }
#pragma unroll
      for (int j = 0; j < kMicro; ++j) bv[j] = bs[k][tx * kMicro + j];
#pragma unroll
      for (int i = 0; i < kMicro; ++i)
#pragma unroll
        for (int j = 0; j < kMicro; ++j) {
          cg[i][j] = fmaf(dg[i], bv[j], cg[i][j]);
          cu[i][j] = fmaf(du[i], bv[j], cu[i][j]);
        }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < kMicro; ++i) {
    const int64_t n_i = m_base + ty * kMicro + i;
    if (n_i >= I) continue;
#pragma unroll
    for (int j = 0; j < kMicro; ++j) {
      const int64_t h = n_base + tx * kMicro + j;
      if (h >= H) continue;
      d_gate_weight[n_i * H + h] = static_cast<scalar_t>(cg[i][j]);
      d_up_weight[n_i * H + h] = static_cast<scalar_t>(cu[i][j]);
    }
  }
}

// db[i] = sum_r d[r,i], r ascending; one thread per intermediate column (coalesced across threads).
template <typename scalar_t>
__global__ void __launch_bounds__(kThreads) gate_up_backward_bias_kernel(
    const float* __restrict__ d_gate,
    const float* __restrict__ d_up,
    scalar_t* __restrict__ d_gate_bias,
    scalar_t* __restrict__ d_up_bias,
    int64_t rows, int64_t I) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= I) return;
  float sg = 0.0f, su = 0.0f;
  for (int64_t r = 0; r < rows; ++r) {
    sg += d_gate[r * I + i];
    su += d_up[r * I + i];
  }
  d_gate_bias[i] = static_cast<scalar_t>(sg);
  d_up_bias[i] = static_cast<scalar_t>(su);
}

void check_input(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous(), name, " must be contiguous ROCm");
  TORCH_CHECK(
      tensor.scalar_type() == at::kFloat || tensor.scalar_type() == at::kHalf ||
          tensor.scalar_type() == at::kBFloat16,
      name, " must be float32, float16, or bfloat16");
}

void check_same(const torch::Tensor& lhs, const torch::Tensor& rhs, const char* lhs_name,
                const char* rhs_name) {
  TORCH_CHECK(lhs.device() == rhs.device(), lhs_name, " and ", rhs_name, " must share device");
  TORCH_CHECK(lhs.scalar_type() == rhs.scalar_type(), lhs_name, " and ", rhs_name, " must share dtype");
}

inline unsigned ceil_tiles(int64_t n) { return static_cast<unsigned>((n + kTile - 1) / kTile); }

}  // namespace

std::vector<torch::Tensor> inkling2_expert_gate_up_forward(
    torch::Tensor hidden,
    torch::Tensor gate_weight,
    torch::Tensor up_weight,
    torch::Tensor gate_bias,
    torch::Tensor up_bias,
    double gate_cap,
    double up_cap) {
  check_input(hidden, "hidden");
  check_input(gate_weight, "gate_weight");
  check_input(up_weight, "up_weight");
  check_same(hidden, gate_weight, "hidden", "gate_weight");
  check_same(hidden, up_weight, "hidden", "up_weight");
  TORCH_CHECK(hidden.dim() == 2, "hidden must be [rows, hidden]");
  TORCH_CHECK(gate_weight.dim() == 2 && up_weight.sizes() == gate_weight.sizes(),
              "gate/up weights must be [intermediate, hidden]");
  TORCH_CHECK(hidden.size(1) == gate_weight.size(1), "hidden size mismatch");
  TORCH_CHECK(gate_cap > 0.0 && up_cap > 0.0, "softcaps must be positive");
  TORCH_CHECK(gate_bias.numel() == 0 || gate_bias.dim() == 1, "gate_bias must be [intermediate]");
  TORCH_CHECK(up_bias.numel() == 0 || up_bias.dim() == 1, "up_bias must be [intermediate]");
  if (gate_bias.numel() != 0) {
    check_input(gate_bias, "gate_bias");
    check_same(hidden, gate_bias, "hidden", "gate_bias");
    TORCH_CHECK(gate_bias.size(0) == gate_weight.size(0), "gate_bias size mismatch");
  }
  if (up_bias.numel() != 0) {
    check_input(up_bias, "up_bias");
    check_same(hidden, up_bias, "hidden", "up_bias");
    TORCH_CHECK(up_bias.size(0) == gate_weight.size(0), "up_bias size mismatch");
  }

  const c10::OptionalDeviceGuard device_guard(device_of(hidden));
  const int64_t rows = hidden.size(0), H = hidden.size(1), I = gate_weight.size(0);
  auto output = torch::empty({rows, I}, hidden.options());
  auto gate_pre = torch::empty({rows, I}, hidden.options().dtype(torch::kFloat32));
  auto up_pre = torch::empty_like(gate_pre);
  auto gate_bias_f32 = gate_bias.numel() == 0 ? torch::Tensor() : gate_bias.to(torch::kFloat32);
  auto up_bias_f32 = up_bias.numel() == 0 ? torch::Tensor() : up_bias.to(torch::kFloat32);
  if (rows * I == 0) return {output, gate_pre, up_pre};

  const dim3 grid(ceil_tiles(rows), ceil_tiles(I));
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, hidden.scalar_type(),
                                  "inkling2_expert_gate_up_forward", [&] {
    gate_up_forward_kernel<scalar_t><<<grid, kThreads, 0, stream>>>(
        hidden.data_ptr<scalar_t>(), gate_weight.data_ptr<scalar_t>(), up_weight.data_ptr<scalar_t>(),
        gate_bias_f32.numel() == 0 ? nullptr : gate_bias_f32.data_ptr<float>(),
        up_bias_f32.numel() == 0 ? nullptr : up_bias_f32.data_ptr<float>(),
        output.data_ptr<scalar_t>(), gate_pre.data_ptr<float>(), up_pre.data_ptr<float>(), rows, H, I,
        static_cast<float>(gate_cap), static_cast<float>(up_cap));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, gate_pre, up_pre};
}

std::vector<torch::Tensor> inkling2_expert_gate_up_backward(
    torch::Tensor dy,
    torch::Tensor hidden,
    torch::Tensor gate_weight,
    torch::Tensor up_weight,
    torch::Tensor gate_pre,
    torch::Tensor up_pre,
    double gate_cap,
    double up_cap,
    bool need_gate_bias,
    bool need_up_bias) {
  check_input(dy, "dy");
  check_input(hidden, "hidden");
  check_input(gate_weight, "gate_weight");
  check_input(up_weight, "up_weight");
  TORCH_CHECK(gate_pre.is_cuda() && up_pre.is_cuda() && gate_pre.is_contiguous() && up_pre.is_contiguous(),
              "saved preactivations must be contiguous ROCm");
  check_same(dy, hidden, "dy", "hidden");
  check_same(hidden, gate_weight, "hidden", "gate_weight");
  check_same(hidden, up_weight, "hidden", "up_weight");
  TORCH_CHECK(gate_pre.scalar_type() == at::kFloat && up_pre.scalar_type() == at::kFloat,
              "saved preactivations must be float32");
  TORCH_CHECK(dy.sizes() == gate_pre.sizes() && gate_pre.sizes() == up_pre.sizes(),
              "dy and saved preactivations must share shape");

  const c10::OptionalDeviceGuard device_guard(device_of(hidden));
  const int64_t rows = hidden.size(0), H = hidden.size(1), I = gate_weight.size(0);
  auto d_hidden = torch::empty_like(hidden);
  auto d_gate_weight = torch::empty_like(gate_weight);
  auto d_up_weight = torch::empty_like(up_weight);
  auto d_gate_bias = need_gate_bias ? torch::empty({I}, hidden.options()) : torch::Tensor();
  auto d_up_bias = need_up_bias ? torch::empty({I}, hidden.options()) : torch::Tensor();
  if (hidden.numel() == 0 || gate_weight.numel() == 0) {
    return {d_hidden, d_gate_weight, d_up_weight, d_gate_bias, d_up_bias};
  }
  auto gate_bias_workspace = need_gate_bias ? d_gate_bias : torch::empty({I}, hidden.options());
  auto up_bias_workspace = need_up_bias ? d_up_bias : torch::empty({I}, hidden.options());
  auto d_gate = torch::empty({rows, I}, hidden.options().dtype(torch::kFloat32));
  auto d_up = torch::empty_like(d_gate);

  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t total = rows * I;
  const int pre_blocks = static_cast<int>((total + kThreads - 1) / kThreads);
  const dim3 hidden_grid(ceil_tiles(rows), ceil_tiles(H));
  const dim3 weight_grid(ceil_tiles(H), ceil_tiles(I));
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, hidden.scalar_type(),
                                  "inkling2_expert_gate_up_backward", [&] {
    gate_up_grad_pre_kernel<scalar_t><<<pre_blocks, kThreads, 0, stream>>>(
        dy.data_ptr<scalar_t>(), gate_pre.data_ptr<float>(), up_pre.data_ptr<float>(),
        d_gate.data_ptr<float>(), d_up.data_ptr<float>(), total, static_cast<float>(gate_cap),
        static_cast<float>(up_cap));
    gate_up_backward_hidden_kernel<scalar_t><<<hidden_grid, kThreads, 0, stream>>>(
        d_gate.data_ptr<float>(), d_up.data_ptr<float>(), gate_weight.data_ptr<scalar_t>(),
        up_weight.data_ptr<scalar_t>(), d_hidden.data_ptr<scalar_t>(), rows, H, I);
    gate_up_backward_weight_kernel<scalar_t><<<weight_grid, kThreads, 0, stream>>>(
        d_gate.data_ptr<float>(), d_up.data_ptr<float>(), hidden.data_ptr<scalar_t>(),
        d_gate_weight.data_ptr<scalar_t>(), d_up_weight.data_ptr<scalar_t>(), rows, H, I);
    if (need_gate_bias || need_up_bias) {
      const int bias_blocks = static_cast<int>((I + kThreads - 1) / kThreads);
      gate_up_backward_bias_kernel<scalar_t><<<bias_blocks, kThreads, 0, stream>>>(
          d_gate.data_ptr<float>(), d_up.data_ptr<float>(), gate_bias_workspace.data_ptr<scalar_t>(),
          up_bias_workspace.data_ptr<scalar_t>(), rows, I);
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {d_hidden, d_gate_weight, d_up_weight, d_gate_bias, d_up_bias};
}
