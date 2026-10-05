// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// P2 T06 WS1 reference: ONE-softmax MQA attention with sink-in-denominator.
// Hq=64, Hkv=1, D=512. Sequential d / j / h trees. No Split-KV, no atomics.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <limits>
#include <vector>

namespace {

constexpr int64_t kHq = 64;
constexpr int64_t kD = 512;
constexpr int64_t kMaxGridY = 65535;
constexpr float kScale = 0.044194173824159216f;  // 512 ** -0.5

template <typename scalar_t>
__global__ void qk_kernel(
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K,
    const bool* __restrict__ valid,
    float* __restrict__ scores,
    int64_t T,
    int64_t N,
    float scale) {
  const int j = blockIdx.x;
  const int t = blockIdx.y;
  const int h = threadIdx.x;
  if (j >= N || t >= T || h >= kHq) {
    return;
  }
  const int64_t out_idx = ((int64_t)t * kHq + h) * N + j;
  if (!valid[j]) {
    scores[out_idx] = -INFINITY;
    return;
  }
  const scalar_t* q_ptr = Q + ((int64_t)t * kHq + h) * kD;
  const scalar_t* k_ptr = K + (int64_t)j * kD;
  float acc = 0.0f;
#pragma unroll 8
  for (int d = 0; d < kD; ++d) {
    acc += (float)q_ptr[d] * (float)k_ptr[d];
  }
  scores[out_idx] = scale * acc;
}

__global__ void softmax_sink_kernel(
    float* __restrict__ scores,
    const float* __restrict__ sink,
    const bool* __restrict__ valid,
    float* __restrict__ p_sink,
    float* __restrict__ m_out,
    float* __restrict__ z_out,
    int64_t T,
    int64_t N) {
  const int t = blockIdx.x;
  const int h = threadIdx.x;
  if (t >= T || h >= kHq) {
    return;
  }
  const float sink_v = sink[(int64_t)t * kHq + h];
  float m = sink_v;
  for (int j = 0; j < N; ++j) {
    if (valid[j]) {
      m = fmaxf(m, scores[((int64_t)t * kHq + h) * N + j]);
    }
  }
  const float e_sink = expf(sink_v - m);
  float z = e_sink;
  for (int j = 0; j < N; ++j) {
    const int64_t idx = ((int64_t)t * kHq + h) * N + j;
    if (valid[j]) {
      const float e = expf(scores[idx] - m);
      scores[idx] = e;
      z += e;
    } else {
      scores[idx] = 0.0f;
    }
  }
  for (int j = 0; j < N; ++j) {
    const int64_t idx = ((int64_t)t * kHq + h) * N + j;
    scores[idx] = scores[idx] / z;
  }
  const int64_t oh = (int64_t)t * kHq + h;
  p_sink[oh] = e_sink / z;
  m_out[oh] = m;
  z_out[oh] = z;
}

template <typename in_t, typename out_t>
__global__ void pv_kernel(
    const float* __restrict__ P,
    const in_t* __restrict__ V,
    out_t* __restrict__ O,
    int64_t T,
    int64_t N) {
  const int t = blockIdx.x;
  const int h = blockIdx.y;
  const int d = threadIdx.x;
  if (t >= T || h >= kHq || d >= kD) {
    return;
  }
  float acc = 0.0f;
  for (int j = 0; j < N; ++j) {
    acc += P[((int64_t)t * kHq + h) * N + j] * (float)V[(int64_t)j * kD + d];
  }
  O[((int64_t)t * kHq + h) * kD + d] = (out_t)acc;
}

template <typename scalar_t>
__global__ void dp_kernel(
    const float* __restrict__ dO,
    const scalar_t* __restrict__ V,
    const bool* __restrict__ valid,
    float* __restrict__ dP,
    int64_t T,
    int64_t N) {
  const int j = blockIdx.x;
  const int t = blockIdx.y;
  const int h = threadIdx.x;
  if (j >= N || t >= T || h >= kHq) {
    return;
  }
  const int64_t out_idx = ((int64_t)t * kHq + h) * N + j;
  if (!valid[j]) {
    dP[out_idx] = 0.0f;
    return;
  }
  const float* do_ptr = dO + ((int64_t)t * kHq + h) * kD;
  const scalar_t* v_ptr = V + (int64_t)j * kD;
  float acc = 0.0f;
#pragma unroll 8
  for (int d = 0; d < kD; ++d) {
    acc += do_ptr[d] * (float)v_ptr[d];
  }
  dP[out_idx] = acc;
}

__global__ void softmax_bwd_kernel(
    float* __restrict__ dP,
    const float* __restrict__ P,
    const float* __restrict__ p_sink,
    const bool* __restrict__ valid,
    float* __restrict__ dsink,
    int64_t T,
    int64_t N) {
  const int t = blockIdx.x;
  const int h = threadIdx.x;
  if (t >= T || h >= kHq) {
    return;
  }
  float mu = 0.0f;
  for (int j = 0; j < N; ++j) {
    if (valid[j]) {
      const int64_t idx = ((int64_t)t * kHq + h) * N + j;
      mu += P[idx] * dP[idx];
    }
  }
  dsink[(int64_t)t * kHq + h] = -p_sink[(int64_t)t * kHq + h] * mu;
  for (int j = 0; j < N; ++j) {
    const int64_t idx = ((int64_t)t * kHq + h) * N + j;
    if (valid[j]) {
      dP[idx] = P[idx] * (dP[idx] - mu);
    } else {
      dP[idx] = 0.0f;
    }
  }
}

template <typename scalar_t>
__global__ void dq_kernel(
    const float* __restrict__ dS,
    const scalar_t* __restrict__ K,
    scalar_t* __restrict__ dQ,
    const bool* __restrict__ valid,
    int64_t T,
    int64_t N,
    float scale) {
  const int t = blockIdx.x;
  const int h = blockIdx.y;
  const int d = threadIdx.x;
  if (t >= T || h >= kHq || d >= kD) {
    return;
  }
  float acc = 0.0f;
  for (int j = 0; j < N; ++j) {
    if (valid[j]) {
      acc += dS[((int64_t)t * kHq + h) * N + j] * (float)K[(int64_t)j * kD + d];
    }
  }
  dQ[((int64_t)t * kHq + h) * kD + d] = (scalar_t)(scale * acc);
}

template <typename scalar_t>
__global__ void dk_kernel(
    const float* __restrict__ dS,
    const scalar_t* __restrict__ Q,
    scalar_t* __restrict__ dK,
    const bool* __restrict__ valid,
    int64_t T,
    int64_t N,
    float scale) {
  const int j = blockIdx.x;
  const int d = threadIdx.x;
  if (j >= N || d >= kD) {
    return;
  }
  float acc = 0.0f;
  if (valid[j]) {
    for (int t = 0; t < T; ++t) {
      for (int h = 0; h < kHq; ++h) {
        acc += dS[((int64_t)t * kHq + h) * N + j] * (float)Q[((int64_t)t * kHq + h) * kD + d];
      }
    }
  }
  dK[(int64_t)j * kD + d] = (scalar_t)(scale * acc);
}

template <typename scalar_t>
__global__ void dv_kernel(
    const float* __restrict__ P,
    const float* __restrict__ dO,
    scalar_t* __restrict__ dV,
    const bool* __restrict__ valid,
    int64_t T,
    int64_t N) {
  const int j = blockIdx.x;
  const int d = threadIdx.x;
  if (j >= N || d >= kD) {
    return;
  }
  float acc = 0.0f;
  if (valid[j]) {
    for (int t = 0; t < T; ++t) {
      for (int h = 0; h < kHq; ++h) {
        acc += P[((int64_t)t * kHq + h) * N + j] * dO[((int64_t)t * kHq + h) * kD + d];
      }
    }
  }
  dV[(int64_t)j * kD + d] = (scalar_t)acc;
}

__global__ void reduce_dsink_shared_kernel(const float* dsink_th, float* dsink_h, int64_t T) {
  const int h = threadIdx.x;
  if (h >= kHq) {
    return;
  }
  float acc = 0.0f;
  for (int t = 0; t < T; ++t) {
    acc += dsink_th[(int64_t)t * kHq + h];
  }
  dsink_h[h] = acc;
}

void check_inputs(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "mqa_joint_attention_sink: CUDA tensors required");
  TORCH_CHECK(k.device() == q.device() && v.device() == q.device(),
              "mqa_joint_attention_sink: Q/K/V device mismatch");
  TORCH_CHECK(q.dim() == 3 && q.size(1) == kHq && q.size(2) == kD,
              "mqa_joint_attention_sink: Q must be [T, 64, 512]");
  TORCH_CHECK(k.dim() == 2 && v.dim() == 2 && k.size(1) == kD && v.size(1) == kD && k.size(0) == v.size(0),
              "mqa_joint_attention_sink: K/V must be [N, 512]");
  TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(),
              "mqa_joint_attention_sink: Q/K/V dtype mismatch");
  TORCH_CHECK(
      q.scalar_type() == at::kFloat || q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16,
      "mqa_joint_attention_sink: Q/K/V must be fp32/fp16/bf16");
  TORCH_CHECK(q.size(0) <= kMaxGridY,
              "mqa_joint_attention_sink: T exceeds CUDA grid.y limit 65535");
}

void launch_fwd_into(
    const torch::Tensor& q_c,
    const torch::Tensor& k_c,
    const torch::Tensor& v_c,
    const torch::Tensor& sink_c,
    const torch::Tensor& valid_c,
    double scale,
    bool output_fp32,
    torch::Tensor& out,
    torch::Tensor& scores,
    torch::Tensor& p_sink,
    torch::Tensor& m,
    torch::Tensor& z) {
  const int64_t T = q_c.size(0);
  const int64_t N = k_c.size(0);
  auto stream = at::cuda::getCurrentCUDAStream();
  float* scores_ptr = N == 0 ? nullptr : scores.data_ptr<float>();

  if (N > 0) {
    dim3 qk_grid(N, T);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, q_c.scalar_type(), "mqa_qk", [&] {
          qk_kernel<scalar_t><<<qk_grid, kHq, 0, stream>>>(
              q_c.data_ptr<scalar_t>(),
              k_c.data_ptr<scalar_t>(),
              valid_c.data_ptr<bool>(),
              scores_ptr,
              T,
              N,
              static_cast<float>(scale));
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
  }

  softmax_sink_kernel<<<T, kHq, 0, stream>>>(
      scores_ptr,
      sink_c.data_ptr<float>(),
      N ? valid_c.data_ptr<bool>() : nullptr,
      p_sink.data_ptr<float>(),
      m.data_ptr<float>(),
      z.data_ptr<float>(),
      T,
      N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (N == 0) {
    out.zero_();
  } else {
    dim3 pv_grid(T, kHq);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, q_c.scalar_type(), "mqa_pv", [&] {
          if (output_fp32) {
            pv_kernel<scalar_t, float><<<pv_grid, kD, 0, stream>>>(
                scores_ptr,
                v_c.data_ptr<scalar_t>(),
                out.data_ptr<float>(),
                T,
                N);
          } else {
            pv_kernel<scalar_t, scalar_t><<<pv_grid, kD, 0, stream>>>(
                scores_ptr,
                v_c.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                T,
                N);
          }
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
  }
}

}  // namespace

void mqa_joint_attention_sink_forward_into(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor sink,
    torch::Tensor valid,
    double scale,
    bool output_fp32,
    torch::Tensor out,
    torch::Tensor scores,
    torch::Tensor p_sink,
    torch::Tensor m,
    torch::Tensor z) {
  check_inputs(q, k, v);
  TORCH_CHECK(std::abs(scale - static_cast<double>(kScale)) < 1e-8,
              "mqa_joint_attention_sink: scale must be 512^-0.5");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
              "mqa_joint_attention_sink_forward_into: Q/K/V must be contiguous");
  TORCH_CHECK(sink.is_cuda() && sink.device() == q.device() && sink.dtype() == at::kFloat,
              "mqa_joint_attention_sink_forward_into: sink must be float CUDA");
  TORCH_CHECK(sink.is_contiguous() && sink.sizes() == at::IntArrayRef({q.size(0), kHq}),
              "mqa_joint_attention_sink_forward_into: sink must be contiguous [T,64]");
  TORCH_CHECK(valid.is_cuda() && valid.device() == q.device() && valid.dtype() == at::kBool,
              "mqa_joint_attention_sink_forward_into: valid must be bool CUDA");
  TORCH_CHECK(valid.is_contiguous() && valid.numel() == k.size(0),
              "mqa_joint_attention_sink_forward_into: valid length must equal N");
  const int64_t T = q.size(0);
  const int64_t N = k.size(0);
  TORCH_CHECK(out.is_cuda() && out.device() == q.device(),
              "mqa_joint_attention_sink_forward_into: out device mismatch");
  TORCH_CHECK(out.scalar_type() == (output_fp32 ? at::kFloat : q.scalar_type()),
              "mqa_joint_attention_sink_forward_into: out dtype mismatch");
  TORCH_CHECK(out.is_contiguous() && out.sizes() == at::IntArrayRef({T, kHq, kD}),
              "mqa_joint_attention_sink_forward_into: out must be contiguous [T,64,512]");
  TORCH_CHECK(scores.is_cuda() && scores.device() == q.device(),
              "mqa_joint_attention_sink_forward_into: scores device mismatch");
  TORCH_CHECK(scores.scalar_type() == at::kFloat,
              "mqa_joint_attention_sink_forward_into: scores dtype mismatch");
  TORCH_CHECK(scores.is_contiguous() && scores.sizes() == at::IntArrayRef({T, kHq, N}),
              "mqa_joint_attention_sink_forward_into: scores must be contiguous [T,64,N]");
  TORCH_CHECK(p_sink.is_cuda() && p_sink.device() == q.device(),
              "mqa_joint_attention_sink_forward_into: p_sink device mismatch");
  TORCH_CHECK(p_sink.scalar_type() == at::kFloat,
              "mqa_joint_attention_sink_forward_into: p_sink dtype mismatch");
  TORCH_CHECK(p_sink.is_contiguous() && p_sink.sizes() == at::IntArrayRef({T, kHq}),
              "mqa_joint_attention_sink_forward_into: p_sink must be contiguous [T,64]");
  TORCH_CHECK(m.is_cuda() && m.device() == q.device(),
              "mqa_joint_attention_sink_forward_into: m device mismatch");
  TORCH_CHECK(m.scalar_type() == at::kFloat,
              "mqa_joint_attention_sink_forward_into: m dtype mismatch");
  TORCH_CHECK(m.is_contiguous() && m.sizes() == at::IntArrayRef({T, kHq}),
              "mqa_joint_attention_sink_forward_into: m must be contiguous [T,64]");
  TORCH_CHECK(z.is_cuda() && z.device() == q.device(),
              "mqa_joint_attention_sink_forward_into: z device mismatch");
  TORCH_CHECK(z.scalar_type() == at::kFloat,
              "mqa_joint_attention_sink_forward_into: z dtype mismatch");
  TORCH_CHECK(z.is_contiguous() && z.sizes() == at::IntArrayRef({T, kHq}),
              "mqa_joint_attention_sink_forward_into: z must be contiguous [T,64]");
  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(q));
  launch_fwd_into(q, k, v, sink, valid, scale, output_fp32, out, scores, p_sink, m, z);
}

std::vector<torch::Tensor> mqa_joint_attention_sink_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor sink,
    torch::Tensor valid,
    double scale,
    bool output_fp32) {
  check_inputs(q, k, v);
  TORCH_CHECK(std::abs(scale - static_cast<double>(kScale)) < 1e-8,
              "mqa_joint_attention_sink: scale must be 512^-0.5");
  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(q));
  auto q_c = q.contiguous();
  auto k_c = k.contiguous();
  auto v_c = v.contiguous();
  TORCH_CHECK(sink.is_cuda() && sink.device() == q.device(),
              "mqa_joint_attention_sink: sink must be on Q's CUDA device");
  TORCH_CHECK(valid.is_cuda() && valid.device() == q.device(),
              "mqa_joint_attention_sink: valid must be on Q's CUDA device");
  auto sink_c = sink.contiguous().to(at::kFloat);
  auto valid_c = valid.contiguous();
  TORCH_CHECK(valid_c.dtype() == at::kBool, "valid must be bool");

  const int64_t T = q_c.size(0);
  const int64_t N = k_c.size(0);
  TORCH_CHECK(valid_c.numel() == N, "valid length must equal N");
  if (sink_c.dim() == 1) {
    sink_c = sink_c.unsqueeze(0).expand(T, kHq).contiguous();
  }
  TORCH_CHECK(sink_c.sizes() == at::IntArrayRef({T, kHq}), "sink must be [T,64] or [64]");

  auto scores = torch::empty({T, kHq, std::max(N, (int64_t)0)}, q_c.options().dtype(at::kFloat));
  auto p_sink = torch::empty({T, kHq}, q_c.options().dtype(at::kFloat));
  auto m = torch::empty({T, kHq}, q_c.options().dtype(at::kFloat));
  auto z = torch::empty({T, kHq}, q_c.options().dtype(at::kFloat));
  auto out = output_fp32 ? torch::empty({T, kHq, kD}, q_c.options().dtype(at::kFloat))
                         : torch::empty_like(q_c);
  launch_fwd_into(q_c, k_c, v_c, sink_c, valid_c, scale, output_fp32, out, scores, p_sink, m, z);
  auto p = N == 0 ? torch::empty({T, kHq, 0}, scores.options()) : scores;
  return {out, p, p_sink, m, z};
}

std::vector<torch::Tensor> mqa_joint_attention_sink_backward(
    torch::Tensor dO,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor sink,
    torch::Tensor valid,
    torch::Tensor P,
    torch::Tensor p_sink,
    double scale,
    bool sink_was_shared) {
  check_inputs(q, k, v);
  TORCH_CHECK(std::abs(scale - static_cast<double>(kScale)) < 1e-8,
              "mqa_joint_attention_sink: scale must be 512^-0.5");
  TORCH_CHECK(dO.is_cuda() && dO.device() == q.device(),
              "mqa_joint_attention_sink: dO must be on Q's CUDA device");
  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(q));
  auto dO_c = dO.contiguous().to(at::kFloat);
  auto q_c = q.contiguous();
  auto k_c = k.contiguous();
  auto v_c = v.contiguous();
  auto valid_c = valid.contiguous();
  auto P_c = P.contiguous();
  auto p_sink_c = p_sink.contiguous();
  const int64_t T = q_c.size(0);
  const int64_t N = k_c.size(0);
  auto stream = at::cuda::getCurrentCUDAStream();

  auto dQ = torch::empty_like(q_c);
  auto dK = torch::empty_like(k_c);
  auto dV = torch::empty_like(v_c);
  auto dsink_th = torch::empty({T, kHq}, q_c.options().dtype(at::kFloat));

  if (N == 0) {
    dQ.zero_();
    auto dsink = sink_was_shared ? torch::zeros({kHq}, dsink_th.options()) : torch::zeros({T, kHq}, dsink_th.options());
    return {dQ, dK, dV, dsink};
  }

  auto dP = torch::empty({T, kHq, N}, q_c.options().dtype(at::kFloat));
  dim3 qk_grid(N, T);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, q_c.scalar_type(), "mqa_dp", [&] {
        dp_kernel<scalar_t><<<qk_grid, kHq, 0, stream>>>(
            dO_c.data_ptr<float>(),
            v_c.data_ptr<scalar_t>(),
            valid_c.data_ptr<bool>(),
            dP.data_ptr<float>(),
            T,
            N);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  softmax_bwd_kernel<<<T, kHq, 0, stream>>>(
      dP.data_ptr<float>(),
      P_c.data_ptr<float>(),
      p_sink_c.data_ptr<float>(),
      valid_c.data_ptr<bool>(),
      dsink_th.data_ptr<float>(),
      T,
      N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  dim3 od_grid(T, kHq);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, q_c.scalar_type(), "mqa_dqkv", [&] {
        dq_kernel<scalar_t><<<od_grid, kD, 0, stream>>>(
            dP.data_ptr<float>(),
            k_c.data_ptr<scalar_t>(),
            dQ.data_ptr<scalar_t>(),
            valid_c.data_ptr<bool>(),
            T,
            N,
            static_cast<float>(scale));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        dk_kernel<scalar_t><<<N, kD, 0, stream>>>(
            dP.data_ptr<float>(),
            q_c.data_ptr<scalar_t>(),
            dK.data_ptr<scalar_t>(),
            valid_c.data_ptr<bool>(),
            T,
            N,
            static_cast<float>(scale));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        dv_kernel<scalar_t><<<N, kD, 0, stream>>>(
            P_c.data_ptr<float>(),
            dO_c.data_ptr<float>(),
            dV.data_ptr<scalar_t>(),
            valid_c.data_ptr<bool>(),
            T,
            N);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  torch::Tensor dsink;
  if (sink_was_shared) {
    dsink = torch::empty({kHq}, dsink_th.options());
    reduce_dsink_shared_kernel<<<1, kHq, 0, stream>>>(
        dsink_th.data_ptr<float>(), dsink.data_ptr<float>(), T);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    dsink = dsink_th;
  }
  return {dQ, dK, dV, dsink};
}
