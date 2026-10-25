// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// Deterministic standard-softmax attention (issue #147).
// Forward: QK kernel → masked softmax+LSE kernel → PV kernel.
// All reductions use fixed order; no split-KV or dynamic dispatch.

#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace {

constexpr int64_t kDeterministicAttentionHeadDim = 128;
constexpr int kSoftmaxThreads = 256;

// ---------------------------------------------------------------------------
// QK Kernel: scores[b, hq, q, k] = scale * sum_{d=0}^{D-1} Q[b,hq,q,d]*K[b,kv_head,k,d]
// Grid: (Skv_blocks, Sq_blocks, B * Hq)
// Block: (TILE_K, TILE_Q) threads, each thread computes one score element.
// ---------------------------------------------------------------------------
constexpr int kQKTileQ = 16;
constexpr int kQKTileK = 16;

template <typename scalar_t>
__global__ void qk_kernel(
    const scalar_t* __restrict__ Q,   // [B, Hq, Sq, D]
    const scalar_t* __restrict__ K,   // [B, Hkv, Skv, D]
    float* __restrict__ scores,       // [B, Hq, Sq, Skv]
    int64_t B, int64_t Hq, int64_t Hkv,
    int64_t Sq, int64_t Skv, int64_t D,
    float scale) {

  const int k_idx = blockIdx.x * kQKTileK + threadIdx.x;
  const int q_idx = blockIdx.y * kQKTileQ + threadIdx.y;
  const int bh = blockIdx.z;  // flattened (b * Hq + hq)
  const int b = bh / Hq;
  const int hq = bh % Hq;

  if (q_idx >= Sq || k_idx >= Skv) return;

  const int kv_head = hq / (Hq / Hkv);

  const scalar_t* q_ptr = Q + ((int64_t)b * Hq * Sq * D + (int64_t)hq * Sq * D + (int64_t)q_idx * D);
  const scalar_t* k_ptr = K + ((int64_t)b * Hkv * Skv * D + (int64_t)kv_head * Skv * D + (int64_t)k_idx * D);

  float acc = 0.0f;
  #pragma unroll 8
  for (int64_t d = 0; d < D; ++d) {
    acc += (float)q_ptr[d] * (float)k_ptr[d];
  }

  const int64_t out_idx = (int64_t)b * Hq * Sq * Skv + (int64_t)hq * Sq * Skv + (int64_t)q_idx * Skv + k_idx;
  scores[out_idx] = scale * acc;
}

// H3 fixed-D dot product. Cooperative, contiguous global loads convert each
// BF16 input once per CTA. Padding/transposition keeps shared-memory reads
// conflict-free for the (16 key, 16 query) thread layout. Each score retains
// the original ascending-d, separate-multiply/add FP32 reduction.
template <bool ApplyScale>
__global__ void h3_shared_qk_kernel(
    const at::BFloat16* __restrict__ Q,
    const at::BFloat16* __restrict__ K,
    float* __restrict__ scores,
    int64_t Sq, int64_t Skv, float scale) {
  __shared__ float sq[16][129];
  __shared__ float sk[128][17];
  const int tid = threadIdx.y * 16 + threadIdx.x;
  const int64_t q_start = (int64_t)blockIdx.y * 16;
  const int64_t k_start = (int64_t)blockIdx.x * 16;
  const int64_t bh = blockIdx.z;
  for (int i = tid; i < 16 * 128; i += 256) {
    const int row = i / 128;
    const int d = i % 128;
    sq[row][d] = q_start + row < Sq
        ? (float)Q[(bh * Sq + q_start + row) * 128 + d] : 0.0f;
    sk[d][row] = k_start + row < Skv
        ? (float)K[(bh * Skv + k_start + row) * 128 + d] : 0.0f;
  }
  __syncthreads();
  const int64_t qi = q_start + threadIdx.y;
  const int64_t ki = k_start + threadIdx.x;
  if (qi >= Sq || ki >= Skv) return;
  float acc = 0.0f;
  #pragma unroll 8
  for (int d = 0; d < 128; ++d) {
    acc += sq[threadIdx.y][d] * sk[d][threadIdx.x];
  }
  scores[(bh * Sq + qi) * Skv + ki] = ApplyScale ? scale * acc : acc;
}

// H3 PV/dQ and fused dK/dV reuse staged operands in shared memory.
// Each output retains the native scalar ascending reduction order.
// All threads take every barrier, including tail rows/channels.
template <bool Scale>
__global__ void h3_staged_row_product(
    const float* __restrict__ A, const at::BFloat16* __restrict__ X,
    at::BFloat16* __restrict__ Y, int64_t Rows, int64_t Inner, float scale) {
  __shared__ float a[8][33];
  __shared__ float x[32][33];
  const int tid = threadIdx.y * 32 + threadIdx.x;
  const int64_t bh = blockIdx.z;
  const int64_t row = (int64_t)blockIdx.y * 8 + threadIdx.y;
  const int d = blockIdx.x * 32 + threadIdx.x;
  float acc = 0.0f;
  for (int64_t start = 0; start < Inner; start += 32) {
    const int ar = tid / 32, ak = tid % 32;
    const int64_t global_row = (int64_t)blockIdx.y * 8 + ar;
    a[ar][ak] = global_row < Rows && start + ak < Inner
        ? A[(bh * Rows + global_row) * Inner + start + ak] : 0.0f;
    for (int i = tid; i < 32 * 32; i += 256) {
      const int k = i / 32, c = i % 32;
      x[k][c] = start + k < Inner
          ? (float)X[(bh * Inner + start + k) * 128 + blockIdx.x * 32 + c] : 0.0f;
    }
    __syncthreads();
    // No padded contributions: even signed zero keeps the original operation count.
    const int n = (int)min((int64_t)32, Inner - start);
    #pragma unroll 8
    for (int j = 0; j < n; ++j) {
      acc += a[threadIdx.y][j] * x[j][threadIdx.x];
    }
    __syncthreads();
  }
  if (row < Rows) Y[(bh * Rows + row) * 128 + d] =
      (at::BFloat16)(Scale ? scale * acc : acc);
}

__global__ void h3_staged_dkv(
    const float* __restrict__ DS, const float* __restrict__ P,
    const at::BFloat16* __restrict__ Q, const at::BFloat16* __restrict__ DO,
    float* __restrict__ DK, float* __restrict__ DV, int64_t Sq, int64_t Skv) {
  __shared__ float ds[32][9], p[32][9];
  __shared__ float q[32][33], dout[32][33];
  const int tid = threadIdx.y * 32 + threadIdx.x;
  const int64_t bh = blockIdx.z;
  const int64_t ki = (int64_t)blockIdx.y * 8 + threadIdx.y;
  const int d = blockIdx.x * 32 + threadIdx.x;
  const int64_t index = (bh * Skv + ki) * 128 + d;
  float dk = ki < Skv ? DK[index] : 0.0f;
  float dv = ki < Skv ? DV[index] : 0.0f;
  for (int64_t start = 0; start < Sq; start += 32) {
    const int r = tid / 8, c = tid % 8;
    const int64_t k = (int64_t)blockIdx.y * 8 + c;
    const bool valid = start + r < Sq && k < Skv;
    ds[r][c] = valid ? DS[(bh * Sq + start + r) * Skv + k] : 0.0f;
    p[r][c] = valid ? P[(bh * Sq + start + r) * Skv + k] : 0.0f;
    for (int i = tid; i < 32 * 32; i += 256) {
      const int row = i / 32, dim = i % 32;
      const bool valid_q = start + row < Sq;
      const int64_t offset = (bh * Sq + start + row) * 128 + blockIdx.x * 32 + dim;
      q[row][dim] = valid_q ? (float)Q[offset] : 0.0f;
      dout[row][dim] = valid_q ? (float)DO[offset] : 0.0f;
    }
    __syncthreads();
    const int n = (int)min((int64_t)32, Sq - start);
    #pragma unroll 8
    for (int j = 0; j < n; ++j) {
      dk += ds[j][threadIdx.y] * q[j][threadIdx.x];
      dv += p[j][threadIdx.y] * dout[j][threadIdx.x];
    }
    __syncthreads();
  }
  if (ki < Skv) { DK[index] = dk; DV[index] = dv; }
}

// ---------------------------------------------------------------------------
// Masked Softmax + LSE Kernel
// One CTA per (b, hq, q) row. Fixed 256 threads.
// Applies causal + padding mask, computes max, sum-exp, writes P and LSE.
// ---------------------------------------------------------------------------
__global__ void masked_softmax_lse_kernel(
    float* __restrict__ scores,       // [B, Hq, Sq, Skv] in-place -> P
    float* __restrict__ lse,          // [B, Hq, Sq]
    const bool* __restrict__ pad_mask, // [B, Skv] or nullptr
    int64_t B, int64_t Hq, int64_t Sq, int64_t Skv,
    bool causal) {

  const int row_idx = blockIdx.x;  // flattened (b * Hq * Sq + hq * Sq + q)
  const int b = row_idx / (Hq * Sq);
  const int hq = (row_idx / Sq) % Hq;
  const int q = row_idx % Sq;

  float* row = scores + (int64_t)row_idx * Skv;

  // Causal boundary: key_index <= Skv - Sq + q
  const int64_t causal_limit = causal ? (Skv - Sq + q + 1) : Skv;

  // Phase 1: Apply masks and find max
  __shared__ float smax[kSoftmaxThreads];
  float thread_max = -INFINITY;
  for (int k = threadIdx.x; k < Skv; k += kSoftmaxThreads) {
    bool valid = (k < causal_limit);
    if (valid && pad_mask != nullptr) {
      valid = pad_mask[(int64_t)b * Skv + k];
    }
    if (!valid) {
      row[k] = -INFINITY;
    }
    if (valid) {
      thread_max = fmaxf(thread_max, row[k]);
    }
  }

  // Warp reduction for max
  smax[threadIdx.x] = thread_max;
  __syncthreads();
  // Tree reduction
  for (int stride = kSoftmaxThreads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smax[threadIdx.x] = fmaxf(smax[threadIdx.x], smax[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  float row_max = smax[0];

  // Phase 2: Compute sum of exp(s - max)
  __shared__ float ssum[kSoftmaxThreads];
  float thread_sum = 0.0f;
  for (int k = threadIdx.x; k < Skv; k += kSoftmaxThreads) {
    bool valid = (k < causal_limit);
    if (valid && pad_mask != nullptr) {
      valid = pad_mask[(int64_t)b * Skv + k];
    }
    if (valid) {
      float val = expf(row[k] - row_max);
      row[k] = val;
      thread_sum += val;
    } else {
      row[k] = 0.0f;
    }
  }

  ssum[threadIdx.x] = thread_sum;
  __syncthreads();
  for (int stride = kSoftmaxThreads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      ssum[threadIdx.x] += ssum[threadIdx.x + stride];
    }
    __syncthreads();
  }
  float row_sum = ssum[0];

  // Phase 3: Normalize to get P, compute LSE
  float lse_val;
  if (row_sum == 0.0f) {
    // Fully masked row
    lse_val = -INFINITY;
    for (int k = threadIdx.x; k < Skv; k += kSoftmaxThreads) {
      row[k] = 0.0f;
    }
  } else {
    lse_val = row_max + logf(row_sum);
    for (int k = threadIdx.x; k < Skv; k += kSoftmaxThreads) {
      row[k] /= row_sum;
    }
  }

  if (threadIdx.x == 0) {
    lse[row_idx] = lse_val;
  }
}

// ---------------------------------------------------------------------------
// PV Kernel: out[b, hq, q, d] = sum_{k=0}^{Skv-1} P[b,hq,q,k] * V[b,kv_head,k,d]
// Grid: (D_blocks, Sq_blocks, B * Hq)
// Each thread computes one output element with sequential k accumulation.
// ---------------------------------------------------------------------------
constexpr int kPVTileQ = 16;
constexpr int kPVTileD = 16;

template <typename input_t, typename output_t>
__global__ void pv_kernel(
    const float* __restrict__ P,      // [B, Hq, Sq, Skv]
    const input_t* __restrict__ V,   // [B, Hkv, Skv, D]
    output_t* __restrict__ out,       // [B, Hq, Sq, D]
    int64_t B, int64_t Hq, int64_t Hkv,
    int64_t Sq, int64_t Skv, int64_t D) {

  const int d_idx = blockIdx.x * kPVTileD + threadIdx.x;
  const int q_idx = blockIdx.y * kPVTileQ + threadIdx.y;
  const int bh = blockIdx.z;
  const int b = bh / Hq;
  const int hq = bh % Hq;

  if (q_idx >= Sq || d_idx >= D) return;

  const int kv_head = hq / (Hq / Hkv);

  const float* p_row = P + ((int64_t)b * Hq * Sq * Skv + (int64_t)hq * Sq * Skv + (int64_t)q_idx * Skv);
  const input_t* v_base = V + ((int64_t)b * Hkv * Skv * D + (int64_t)kv_head * Skv * D);

  float acc = 0.0f;
  for (int64_t k = 0; k < Skv; ++k) {
    acc += p_row[k] * (float)v_base[k * D + d_idx];
  }

  const int64_t out_idx = (int64_t)b * Hq * Sq * D + (int64_t)hq * Sq * D + (int64_t)q_idx * D + d_idx;
  out[out_idx] = (output_t)acc;
}

void check_deterministic_attention_inputs(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::optional<torch::Tensor>& key_padding_mask) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(),
              "deterministic_attention: q, k, v must be CUDA tensors");
  TORCH_CHECK(q.device() == k.device() && q.device() == v.device(),
              "deterministic_attention: q, k, v must be on the same device");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
              "deterministic_attention: q/k/v must be 4-D [B, H, S, D]");
  TORCH_CHECK(
      q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16,
      "deterministic_attention: only FP16 and BF16 are supported, got ",
      q.scalar_type());
  TORCH_CHECK(k.scalar_type() == q.scalar_type() && v.scalar_type() == q.scalar_type(),
              "deterministic_attention: q, k, v must share the same dtype");

  const int64_t B = q.size(0);
  const int64_t Hq = q.size(1);
  const int64_t Sq = q.size(2);
  const int64_t D = q.size(3);
  const int64_t Hkv = k.size(1);
  const int64_t Skv = k.size(2);

  TORCH_CHECK(D == kDeterministicAttentionHeadDim,
              "deterministic_attention: head dim D must be ",
              kDeterministicAttentionHeadDim,
              ", got ",
              D);
  TORCH_CHECK(k.size(0) == B && v.size(0) == B,
              "deterministic_attention: batch size mismatch between q/k/v");
  TORCH_CHECK(v.size(1) == Hkv && v.size(2) == Skv && k.size(3) == D && v.size(3) == D,
              "deterministic_attention: k/v shape mismatch");
  TORCH_CHECK(Hq % Hkv == 0,
              "deterministic_attention: Hq (",
              Hq,
              ") must be divisible by Hkv (",
              Hkv,
              ") for GQA");
  TORCH_CHECK(Sq >= 1 && Skv >= 1,
              "deterministic_attention: Sq and Skv must be positive");

  if (key_padding_mask.has_value() && key_padding_mask->defined()) {
    const auto& mask = *key_padding_mask;
    TORCH_CHECK(mask.is_cuda(), "deterministic_attention: key_padding_mask must be CUDA");
    TORCH_CHECK(mask.device() == q.device(),
                "deterministic_attention: key_padding_mask must match q device");
    TORCH_CHECK(mask.scalar_type() == at::kBool,
                "deterministic_attention: key_padding_mask must be bool");
    TORCH_CHECK(mask.dim() == 2 && mask.size(0) == B && mask.size(1) == Skv,
                "deterministic_attention: key_padding_mask must be [B, Skv]");
  }
}

} // namespace

// Returns {out, lse, P}:
//   out: [B, Hq, Sq, D] same dtype as q
//   lse: [B, Hq, Sq] FP32
//   P:   [B, Hq, Sq, Skv] FP32  (softmax probabilities, saved for backward)
std::vector<torch::Tensor> deterministic_attention_forward_impl(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool causal,
    double scale,
    torch::optional<torch::Tensor> key_padding_mask,
    bool output_fp32,
    bool compute_output = true,
    bool h3_shared_qk = false) {
  check_deterministic_attention_inputs(q, k, v, key_padding_mask);

  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(q));

  auto q_contig = q.contiguous();
  auto k_contig = k.contiguous();
  auto v_contig = v.contiguous();
  torch::optional<torch::Tensor> mask_contig;
  if (key_padding_mask.has_value() && key_padding_mask->defined()) {
    mask_contig = key_padding_mask->contiguous();
  }

  const int64_t B = q_contig.size(0);
  const int64_t Hq = q_contig.size(1);
  const int64_t Sq = q_contig.size(2);
  const int64_t D = q_contig.size(3);
  const int64_t Hkv = k_contig.size(1);
  const int64_t Skv = k_contig.size(2);

  auto stream = at::cuda::getCurrentCUDAStream();

  // Allocate scores [B, Hq, Sq, Skv] FP32
  auto scores = torch::empty({B, Hq, Sq, Skv}, q_contig.options().dtype(at::kFloat));

  // --- Launch QK kernel ---
  {
    dim3 block(kQKTileK, kQKTileQ);
    dim3 grid(
        (Skv + kQKTileK - 1) / kQKTileK,
        (Sq + kQKTileQ - 1) / kQKTileQ,
        B * Hq);
    if (h3_shared_qk) {
      h3_shared_qk_kernel<true><<<grid, block, 0, stream>>>(
          q_contig.data_ptr<at::BFloat16>(), k_contig.data_ptr<at::BFloat16>(),
          scores.data_ptr<float>(), Sq, Skv, (float)scale);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::Half, at::ScalarType::BFloat16,
          q_contig.scalar_type(), "qk_kernel", [&] {
            qk_kernel<scalar_t><<<grid, block, 0, stream>>>(
                q_contig.data_ptr<scalar_t>(),
                k_contig.data_ptr<scalar_t>(),
                scores.data_ptr<float>(),
                B, Hq, Hkv, Sq, Skv, D,
                (float)scale);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          });
    }
  }

  // --- Launch Masked Softmax + LSE kernel ---
  auto lse = torch::empty({B, Hq, Sq}, q_contig.options().dtype(at::kFloat));
  {
    const int64_t num_rows = B * Hq * Sq;
    dim3 block(kSoftmaxThreads);
    dim3 grid(num_rows);
    const bool* pad_mask_ptr = nullptr;
    if (mask_contig.has_value()) {
      pad_mask_ptr = mask_contig->data_ptr<bool>();
    }
    masked_softmax_lse_kernel<<<grid, block, 0, stream>>>(
        scores.data_ptr<float>(),
        lse.data_ptr<float>(),
        pad_mask_ptr,
        B, Hq, Sq, Skv, causal);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  if (!compute_output) {
    return {torch::Tensor(), lse, scores};
  }

  // --- Launch PV kernel ---
  auto out = output_fp32
      ? torch::empty(q_contig.sizes(), q_contig.options().dtype(at::kFloat))
      : torch::empty_like(q_contig);
  {
    dim3 block(kPVTileD, kPVTileQ);
    dim3 grid(
        (D + kPVTileD - 1) / kPVTileD,
        (Sq + kPVTileQ - 1) / kPVTileQ,
        B * Hq);
    if (h3_shared_qk && !output_fp32) {
      dim3 h3block(32, 8);
      dim3 h3grid(4, (Sq + 7) / 8, B * Hq);
      h3_staged_row_product<false><<<h3grid, h3block, 0, stream>>>(
          scores.data_ptr<float>(), v_contig.data_ptr<at::BFloat16>(),
          out.data_ptr<at::BFloat16>(), Sq, Skv, (float)scale);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::Half, at::ScalarType::BFloat16,
          q_contig.scalar_type(), "pv_kernel", [&] {
            if (output_fp32) {
              pv_kernel<scalar_t, float><<<grid, block, 0, stream>>>(
                  scores.data_ptr<float>(),
                  v_contig.data_ptr<scalar_t>(),
                  out.data_ptr<float>(),
                  B, Hq, Hkv, Sq, Skv, D);
            } else {
              pv_kernel<scalar_t, scalar_t><<<grid, block, 0, stream>>>(
                  scores.data_ptr<float>(),
                  v_contig.data_ptr<scalar_t>(),
                  out.data_ptr<scalar_t>(),
                  B, Hq, Hkv, Sq, Skv, D);
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          });
    }
  }

  return {out, lse, scores};
}

std::vector<torch::Tensor> deterministic_attention_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal, double scale,
    torch::optional<torch::Tensor> key_padding_mask) {
  return deterministic_attention_forward_impl(
      q, k, v, causal, scale, key_padding_mask, false);
}

std::vector<torch::Tensor> deterministic_attention_forward_fp32(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal, double scale,
    torch::optional<torch::Tensor> key_padding_mask) {
  return deterministic_attention_forward_impl(
      q, k, v, causal, scale, key_padding_mask, true);
}

namespace {
void check_h3_build() {
#if !defined(RL_KERNEL_H3_STRICT) || !RL_KERNEL_H3_STRICT || defined(USE_ROCM)
  TORCH_CHECK(false,
              "H3 requires an audited strict CUDA build; rebuild with RL_KERNEL_H3_STRICT=1");
#endif
}

void check_h3_inputs(const torch::Tensor& q, const torch::Tensor& k,
                     const torch::Tensor& v) {
  check_h3_build();
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
              "H3 requires 4-D head-major Q/K/V");
  TORCH_CHECK(q.size(1) == 56 && k.size(1) == 56 && v.size(1) == 56,
              "H3 requires 56 MHA heads");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16, "H3 requires BF16");
  TORCH_CHECK(q.size(0) > 0 && q.size(0) <= 65535 / 56,
              "H3 batch exceeds CUDA grid bounds");
  // H3's staged PV/dQ/dK/dV kernels use eight rows per grid block. Keep
  // their y-dimension within CUDA's 16-bit grid limit as well as the QK
  // kernel's sixteen-row limit.
  TORCH_CHECK(q.size(2) <= 65535 * 8 && k.size(2) <= 65535 * 8,
              "H3 sequence exceeds CUDA grid bounds");
  TORCH_CHECK(q.size(0) * 56 * q.size(2) <= 2147483647LL,
              "H3 query chunk exceeds softmax row index bounds");
  check_deterministic_attention_inputs(q, k, v, torch::nullopt);
  TORCH_CHECK(at::cuda::getDeviceProperties(q.get_device())->major >= 8,
              "H3 BF16 CUDA profile requires SM80 or newer");
}
} // namespace

std::map<std::string, std::string> h3_attention_build_info() {
#if defined(RL_KERNEL_H3_STRICT) && RL_KERNEL_H3_STRICT && !defined(USE_ROCM)
  return {{"profile", "h3-cuda-strict-v1"}, {"fmad", "false"}, {"ftz", "false"},
          {"prec_div", "true"}, {"prec_sqrt", "true"}, {"fast_math", "false"},
          {"cuda_compiler", std::to_string(__CUDACC_VER_MAJOR__) + "." +
                            std::to_string(__CUDACC_VER_MINOR__) + "." +
                            std::to_string(__CUDACC_VER_BUILD__)}};
#else
  return {{"profile", "unqualified"}};
#endif
}

std::vector<torch::Tensor> h3_attention_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v) {
  check_h3_inputs(q, k, v);
  return deterministic_attention_forward_impl(q, k, v, false, 1.0 / std::sqrt(128.0),
                                               torch::nullopt, false, true, true);
}

// Recompute only probabilities during H3 backward; do not execute an unused PV.
torch::Tensor h3_attention_probabilities(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
  check_h3_inputs(q, k, v);
  return deterministic_attention_forward_impl(q, k, v, false, 1.0 / std::sqrt(128.0),
                                               torch::nullopt, false, false, true)[2];
}

// ===========================================================================
// BACKWARD
// ===========================================================================
namespace {

// ---------------------------------------------------------------------------
// dP kernel: dP[b,hq,q,k] = sum_{d=0}^{D-1} dO[b,hq,q,d] * V[b,kv_head,k,d]
// Grid: (Skv_blocks, Sq_blocks, B*Hq)
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void dp_kernel(
    const scalar_t* __restrict__ dO,  // [B, Hq, Sq, D]
    const scalar_t* __restrict__ V,   // [B, Hkv, Skv, D]
    float* __restrict__ dP,           // [B, Hq, Sq, Skv]
    int64_t B, int64_t Hq, int64_t Hkv,
    int64_t Sq, int64_t Skv, int64_t D) {

  const int k_idx = blockIdx.x * kQKTileK + threadIdx.x;
  const int q_idx = blockIdx.y * kQKTileQ + threadIdx.y;
  const int bh = blockIdx.z;
  const int b = bh / Hq;
  const int hq = bh % Hq;

  if (q_idx >= Sq || k_idx >= Skv) return;

  const int kv_head = hq / (Hq / Hkv);

  const scalar_t* do_ptr = dO + ((int64_t)b * Hq * Sq * D + (int64_t)hq * Sq * D + (int64_t)q_idx * D);
  const scalar_t* v_ptr = V + ((int64_t)b * Hkv * Skv * D + (int64_t)kv_head * Skv * D + (int64_t)k_idx * D);

  float acc = 0.0f;
  #pragma unroll 8
  for (int64_t d = 0; d < D; ++d) {
    acc += (float)do_ptr[d] * (float)v_ptr[d];
  }

  const int64_t out_idx = (int64_t)b * Hq * Sq * Skv + (int64_t)hq * Sq * Skv + (int64_t)q_idx * Skv + k_idx;
  dP[out_idx] = acc;
}

// ---------------------------------------------------------------------------
// Softmax backward kernel: one CTA per (b, hq, q) row
// delta[row] = sum_k(dP[row,k] * P[row,k])
// dS[row,k] = P[row,k] * (dP[row,k] - delta)
// Writes dS in-place over the dP buffer.
// ---------------------------------------------------------------------------
__global__ void softmax_backward_kernel(
    float* __restrict__ dP_dS,        // [B, Hq, Sq, Skv] - input dP, output dS
    const float* __restrict__ P,      // [B, Hq, Sq, Skv]
    int64_t Skv) {

  const int row_idx = blockIdx.x;
  float* ds_row = dP_dS + (int64_t)row_idx * Skv;
  const float* p_row = P + (int64_t)row_idx * Skv;

  // Compute delta = sum_k(dP * P)
  __shared__ float sdelta[kSoftmaxThreads];
  float thread_delta = 0.0f;
  for (int k = threadIdx.x; k < Skv; k += kSoftmaxThreads) {
    thread_delta += ds_row[k] * p_row[k];
  }
  sdelta[threadIdx.x] = thread_delta;
  __syncthreads();
  for (int stride = kSoftmaxThreads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sdelta[threadIdx.x] += sdelta[threadIdx.x + stride];
    }
    __syncthreads();
  }
  float delta = sdelta[0];

  // dS = P * (dP - delta)
  for (int k = threadIdx.x; k < Skv; k += kSoftmaxThreads) {
    ds_row[k] = p_row[k] * (ds_row[k] - delta);
  }
}

// ---------------------------------------------------------------------------
// dQ kernel: dQ[b,hq,q,d] = scale * sum_{k=0}^{Skv-1} dS[b,hq,q,k] * K[b,kv_head,k,d]
// Grid: (D_blocks, Sq_blocks, B*Hq)
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void dq_kernel(
    const float* __restrict__ dS,     // [B, Hq, Sq, Skv]
    const scalar_t* __restrict__ K,   // [B, Hkv, Skv, D]
    scalar_t* __restrict__ dQ,        // [B, Hq, Sq, D]
    int64_t B, int64_t Hq, int64_t Hkv,
    int64_t Sq, int64_t Skv, int64_t D,
    float scale) {

  const int d_idx = blockIdx.x * kPVTileD + threadIdx.x;
  const int q_idx = blockIdx.y * kPVTileQ + threadIdx.y;
  const int bh = blockIdx.z;
  const int b = bh / Hq;
  const int hq = bh % Hq;

  if (q_idx >= Sq || d_idx >= D) return;

  const int kv_head = hq / (Hq / Hkv);

  const float* ds_row = dS + ((int64_t)b * Hq * Sq * Skv + (int64_t)hq * Sq * Skv + (int64_t)q_idx * Skv);
  const scalar_t* k_base = K + ((int64_t)b * Hkv * Skv * D + (int64_t)kv_head * Skv * D);

  float acc = 0.0f;
  for (int64_t k = 0; k < Skv; ++k) {
    acc += ds_row[k] * (float)k_base[k * D + d_idx];
  }

  const int64_t out_idx = (int64_t)b * Hq * Sq * D + (int64_t)hq * Sq * D + (int64_t)q_idx * D + d_idx;
  dQ[out_idx] = (scalar_t)(scale * acc);
}

// ---------------------------------------------------------------------------
// dK kernel: dK[b,hkv,k,d] = scale * sum_{local=0..g-1} sum_{q=0..Sq-1} dS[b,hq,q,k]*Q[b,hq,q,d]
// Grid: (D_blocks, Skv_blocks, B*Hkv)
// Each thread: single writer for one dK element (§4.1 fixed order).
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void dk_kernel(
    const float* __restrict__ dS,     // [B, Hq, Sq, Skv]
    const scalar_t* __restrict__ Q,   // [B, Hq, Sq, D]
    scalar_t* __restrict__ dK,        // [B, Hkv, Skv, D]
    int64_t B, int64_t Hq, int64_t Hkv,
    int64_t Sq, int64_t Skv, int64_t D,
    float scale) {

  const int d_idx = blockIdx.x * kPVTileD + threadIdx.x;
  const int k_idx = blockIdx.y * kPVTileQ + threadIdx.y;
  const int b_hkv = blockIdx.z;
  const int b = b_hkv / Hkv;
  const int hkv = b_hkv % Hkv;

  if (k_idx >= Skv || d_idx >= D) return;

  const int64_t g = Hq / Hkv;

  float acc = 0.0f;
  for (int64_t local = 0; local < g; ++local) {
    int64_t hq = hkv * g + local;
    for (int64_t qi = 0; qi < Sq; ++qi) {
      float ds_val = dS[(int64_t)b * Hq * Sq * Skv + hq * Sq * Skv + qi * Skv + k_idx];
      float q_val = (float)Q[(int64_t)b * Hq * Sq * D + hq * Sq * D + qi * D + d_idx];
      acc += ds_val * q_val;
    }
  }

  const int64_t out_idx = (int64_t)b * Hkv * Skv * D + (int64_t)hkv * Skv * D + (int64_t)k_idx * D + d_idx;
  dK[out_idx] = (scalar_t)(scale * acc);
}

// ---------------------------------------------------------------------------
// dV kernel: dV[b,hkv,k,d] = sum_{local=0..g-1} sum_{q=0..Sq-1} P[b,hq,q,k]*dO[b,hq,q,d]
// Grid: (D_blocks, Skv_blocks, B*Hkv)
// Each thread: single writer for one dV element (§4.1 fixed order).
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void dv_kernel(
    const float* __restrict__ P,      // [B, Hq, Sq, Skv]
    const scalar_t* __restrict__ dO,  // [B, Hq, Sq, D]
    scalar_t* __restrict__ dV,        // [B, Hkv, Skv, D]
    int64_t B, int64_t Hq, int64_t Hkv,
    int64_t Sq, int64_t Skv, int64_t D) {

  const int d_idx = blockIdx.x * kPVTileD + threadIdx.x;
  const int k_idx = blockIdx.y * kPVTileQ + threadIdx.y;
  const int b_hkv = blockIdx.z;
  const int b = b_hkv / Hkv;
  const int hkv = b_hkv % Hkv;

  if (k_idx >= Skv || d_idx >= D) return;

  const int64_t g = Hq / Hkv;

  float acc = 0.0f;
  for (int64_t local = 0; local < g; ++local) {
    int64_t hq = hkv * g + local;
    for (int64_t qi = 0; qi < Sq; ++qi) {
      float p_val = P[(int64_t)b * Hq * Sq * Skv + hq * Sq * Skv + qi * Skv + k_idx];
      float do_val = (float)dO[(int64_t)b * Hq * Sq * D + hq * Sq * D + qi * D + d_idx];
      acc += p_val * do_val;
    }
  }

  const int64_t out_idx = (int64_t)b * Hkv * Skv * D + (int64_t)hkv * Skv * D + (int64_t)k_idx * D + d_idx;
  dV[out_idx] = (scalar_t)acc;
}

} // namespace (backward kernels)

// Returns {dQ, dK, dV}
std::vector<torch::Tensor> deterministic_attention_backward(
    torch::Tensor grad_output,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor P,       // saved from forward [B, Hq, Sq, Skv] FP32
    bool causal,
    double scale,
    torch::optional<torch::Tensor> key_padding_mask) {

  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(q));

  auto dO = grad_output.contiguous();
  auto q_c = q.contiguous();
  auto k_c = k.contiguous();
  auto v_c = v.contiguous();
  auto P_c = P.contiguous();

  const int64_t B = q_c.size(0);
  const int64_t Hq = q_c.size(1);
  const int64_t Sq = q_c.size(2);
  const int64_t D = q_c.size(3);
  const int64_t Hkv = k_c.size(1);
  const int64_t Skv = k_c.size(2);

  auto stream = at::cuda::getCurrentCUDAStream();

  // dP = dO @ V^T  [B, Hq, Sq, Skv]
  auto dP = torch::empty({B, Hq, Sq, Skv}, q_c.options().dtype(at::kFloat));
  {
    dim3 block(kQKTileK, kQKTileQ);
    dim3 grid(
        (Skv + kQKTileK - 1) / kQKTileK,
        (Sq + kQKTileQ - 1) / kQKTileQ,
        B * Hq);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q_c.scalar_type(), "dp_kernel", [&] {
          dp_kernel<scalar_t><<<grid, block, 0, stream>>>(
              dO.data_ptr<scalar_t>(),
              v_c.data_ptr<scalar_t>(),
              dP.data_ptr<float>(),
              B, Hq, Hkv, Sq, Skv, D);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
  }

  // Softmax backward: dS = P * (dP - delta), writes in-place over dP
  {
    const int64_t num_rows = B * Hq * Sq;
    softmax_backward_kernel<<<num_rows, kSoftmaxThreads, 0, stream>>>(
        dP.data_ptr<float>(),
        P_c.data_ptr<float>(),
        Skv);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  // dP buffer now contains dS

  // dQ = scale * dS @ K
  auto dQ = torch::empty_like(q_c);
  {
    dim3 block(kPVTileD, kPVTileQ);
    dim3 grid(
        (D + kPVTileD - 1) / kPVTileD,
        (Sq + kPVTileQ - 1) / kPVTileQ,
        B * Hq);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q_c.scalar_type(), "dq_kernel", [&] {
          dq_kernel<scalar_t><<<grid, block, 0, stream>>>(
              dP.data_ptr<float>(),
              k_c.data_ptr<scalar_t>(),
              dQ.data_ptr<scalar_t>(),
              B, Hq, Hkv, Sq, Skv, D,
              (float)scale);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
  }

  // dK = scale * dS^T @ Q  (per kv_head, accumulate over query heads in group)
  auto dK = torch::empty_like(k_c);
  {
    dim3 block(kPVTileD, kPVTileQ);
    dim3 grid(
        (D + kPVTileD - 1) / kPVTileD,
        (Skv + kPVTileQ - 1) / kPVTileQ,
        B * Hkv);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q_c.scalar_type(), "dk_kernel", [&] {
          dk_kernel<scalar_t><<<grid, block, 0, stream>>>(
              dP.data_ptr<float>(),
              q_c.data_ptr<scalar_t>(),
              dK.data_ptr<scalar_t>(),
              B, Hq, Hkv, Sq, Skv, D,
              (float)scale);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
  }

  // dV = P^T @ dO  (per kv_head, accumulate over query heads in group)
  auto dV = torch::empty_like(v_c);
  {
    dim3 block(kPVTileD, kPVTileQ);
    dim3 grid(
        (D + kPVTileD - 1) / kPVTileD,
        (Skv + kPVTileQ - 1) / kPVTileQ,
        B * Hkv);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q_c.scalar_type(), "dv_kernel", [&] {
          dv_kernel<scalar_t><<<grid, block, 0, stream>>>(
              P_c.data_ptr<float>(),
              dO.data_ptr<scalar_t>(),
              dV.data_ptr<scalar_t>(),
              B, Hq, Hkv, Sq, Skv, D);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
  }

  return {dQ, dK, dV};
}

// H3 backward continues FP32 accumulators across query chunks.
std::vector<torch::Tensor> h3_attention_backward_accumulate(
    torch::Tensor grad_output,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor P,       // recomputed [B, 56, Sq, Skv] FP32
    torch::Tensor dK, torch::Tensor dV) {
  check_h3_inputs(q, k, v);
  TORCH_CHECK(grad_output.sizes() == q.sizes() && grad_output.device() == q.device()
              && grad_output.scalar_type() == q.scalar_type(), "invalid H3 grad_output");
  TORCH_CHECK(P.dim() == 4 && P.size(0) == q.size(0) && P.size(1) == 56
              && P.size(2) == q.size(2) && P.size(3) == k.size(2)
              && P.device() == q.device() && P.scalar_type() == at::kFloat, "invalid H3 P");
  for (const auto& acc : {dK, dV}) {
    TORCH_CHECK(acc.device() == q.device() && acc.scalar_type() == at::kFloat
                && acc.sizes() == k.sizes() && acc.is_contiguous(), "invalid H3 accumulator");
  }

  at::assert_no_overlap(dK, dV);
  for (const auto& input : {q, k, v, P, grad_output}) {
    at::assert_no_overlap(dK, input);
    at::assert_no_overlap(dV, input);
  }

  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(q));

  auto dO = grad_output.contiguous();
  auto q_c = q.contiguous();
  auto k_c = k.contiguous();
  auto v_c = v.contiguous();
  auto P_c = P.contiguous();

  const int64_t B = q_c.size(0);
  const int64_t Hq = q_c.size(1);
  const int64_t Sq = q_c.size(2);
  const int64_t Skv = k_c.size(2);

  auto stream = at::cuda::getCurrentCUDAStream();

  // dP = dO @ V^T  [B, Hq, Sq, Skv]
  auto dP = torch::empty({B, Hq, Sq, Skv}, q_c.options().dtype(at::kFloat));
  {
    dim3 block(kQKTileK, kQKTileQ);
    dim3 grid(
        (Skv + kQKTileK - 1) / kQKTileK,
        (Sq + kQKTileQ - 1) / kQKTileQ,
        B * Hq);
    h3_shared_qk_kernel<false><<<grid, block, 0, stream>>>(
        dO.data_ptr<at::BFloat16>(), v_c.data_ptr<at::BFloat16>(),
        dP.data_ptr<float>(), Sq, Skv, 1.0f);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  // Softmax backward: dS = P * (dP - delta), writes in-place over dP
  {
    const int64_t num_rows = B * Hq * Sq;
    softmax_backward_kernel<<<num_rows, kSoftmaxThreads, 0, stream>>>(
        dP.data_ptr<float>(),
        P_c.data_ptr<float>(),
        Skv);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  // dP buffer now contains dS

  auto dQ = torch::empty_like(q_c);
  {
    dim3 block(32, 8);
    dim3 grid(4, (Sq + 7) / 8, B * Hq);
    h3_staged_row_product<true><<<grid, block, 0, stream>>>(
        dP.data_ptr<float>(), k_c.data_ptr<at::BFloat16>(),
        dQ.data_ptr<at::BFloat16>(), Sq, Skv, (float)(1.0 / std::sqrt(128.0)));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  {
    dim3 block(32, 8);
    dim3 grid(4, (Skv + 7) / 8, B * Hq);
    h3_staged_dkv<<<grid, block, 0, stream>>>(
        dP.data_ptr<float>(), P_c.data_ptr<float>(),
        q_c.data_ptr<at::BFloat16>(), dO.data_ptr<at::BFloat16>(),
        dK.data_ptr<float>(), dV.data_ptr<float>(), Sq, Skv);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  return {dQ, dK, dV};
}

#if defined(USE_ROCM)
namespace {

template <typename scalar_t>
__global__ void deterministic_rope_kernel(
    const scalar_t* __restrict__ x,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    scalar_t* __restrict__ out,
    int64_t n_rows,
    int table_rows,
    int half,
    float sin_sign) {
  const int64_t index = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  const int64_t count = n_rows * static_cast<int64_t>(half);
  if (index >= count) {
    return;
  }

  const int64_t row = index / half;
  const int pair = static_cast<int>(index % half);
  const int table_row = static_cast<int>(row % table_rows);
  const float c = cos[table_row * half + pair];
  const float s = sin[table_row * half + pair] * sin_sign;
  const int64_t base = row * (2LL * half);
  const float low = static_cast<float>(x[base + pair]);
  const float high = static_cast<float>(x[base + pair + half]);

  out[base + pair] = static_cast<scalar_t>(low * c - high * s);
  out[base + pair + half] = static_cast<scalar_t>(high * c + low * s);
}

template <typename scalar_t>
__global__ void deterministic_rope_token_major_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ positions,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    scalar_t* __restrict__ out,
    int64_t tokens,
    int64_t input_token_stride,
    int heads,
    int head_dim,
    int half,
    int table_rows,
    float sin_sign) {
  const int64_t index = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  const int64_t values_per_token = static_cast<int64_t>(heads) * half;
  const int64_t count = tokens * values_per_token;
  if (index >= count) {
    return;
  }

  const int64_t token = index / values_per_token;
  const int64_t head_pair = index % values_per_token;
  const int head = static_cast<int>(head_pair / half);
  const int pair = static_cast<int>(head_pair % half);
  const int64_t position = positions[token];
  if (position < 0 || position >= table_rows) {
    return;
  }
  const float c = cos[position * half + pair];
  const float s = sin[position * half + pair] * sin_sign;
  const int64_t input_base = token * input_token_stride + head * head_dim;
  const int64_t output_base = (token * heads + head) * static_cast<int64_t>(head_dim);
  const float low = static_cast<float>(x[input_base + pair]);
  const float high = static_cast<float>(x[input_base + pair + half]);

  out[output_base + pair] = static_cast<scalar_t>(low * c - high * s);
  out[output_base + pair + half] = static_cast<scalar_t>(high * c + low * s);
}

}  // namespace

torch::Tensor deterministic_rope_apply_rocm(
    torch::Tensor x,
    torch::Tensor cos,
    torch::Tensor sin,
    double sin_sign) {
  TORCH_CHECK(x.is_cuda(), "ROCm RoPE: x must be a GPU tensor");
  TORCH_CHECK(x.dim() == 2 && x.is_contiguous(),
              "ROCm RoPE: x must be contiguous [rows, head_dim]");
  TORCH_CHECK(cos.is_cuda() && sin.is_cuda(),
              "ROCm RoPE: cos and sin must be GPU tensors");
  TORCH_CHECK(cos.scalar_type() == torch::kFloat32 &&
                  sin.scalar_type() == torch::kFloat32,
              "ROCm RoPE: cos and sin must be FP32");
  TORCH_CHECK(cos.is_contiguous() && sin.is_contiguous(),
              "ROCm RoPE: cos and sin must be contiguous");
  TORCH_CHECK(cos.dim() == 2 && sin.sizes() == cos.sizes(),
              "ROCm RoPE: cos and sin must have shape [table_rows, head_dim/2]");
  TORCH_CHECK(x.size(1) % 2 == 0, "ROCm RoPE: head_dim must be even");
  TORCH_CHECK(cos.size(0) > 0 && cos.size(1) == x.size(1) / 2,
              "ROCm RoPE: invalid cos/sin table shape");
  TORCH_CHECK(x.size(0) % cos.size(0) == 0,
              "ROCm RoPE: row count must be divisible by the position table size");

  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  auto out = torch::empty_like(x);
  const int64_t n_rows = x.size(0);
  const int half = static_cast<int>(x.size(1) / 2);
  const int table_rows = static_cast<int>(cos.size(0));
  const int64_t count = n_rows * static_cast<int64_t>(half);
  constexpr int threads = 256;
  const int64_t blocks = (count + threads - 1) / threads;
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x.scalar_type(),
      "deterministic_rope_apply_rocm",
      [&] {
        deterministic_rope_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            x.data_ptr<scalar_t>(),
            cos.data_ptr<float>(),
            sin.data_ptr<float>(),
            out.data_ptr<scalar_t>(),
            n_rows,
            table_rows,
            half,
            static_cast<float>(sin_sign));
      });
  return out;
}

torch::Tensor deterministic_rope_apply_token_major_rocm(
    torch::Tensor x,
    torch::Tensor positions,
    torch::Tensor cos,
    torch::Tensor sin,
    int64_t head_dim,
    double sin_sign) {
  TORCH_CHECK(x.is_cuda(), "ROCm token-major RoPE: x must be a GPU tensor");
  TORCH_CHECK(x.dim() == 2 && x.stride(1) == 1,
              "ROCm token-major RoPE: x must be [tokens, heads * head_dim] with unit inner stride");
  TORCH_CHECK(head_dim > 0 && head_dim % 2 == 0 && x.size(1) % head_dim == 0,
              "ROCm token-major RoPE: invalid head dimension");
  TORCH_CHECK(positions.is_cuda() && positions.scalar_type() == torch::kInt64 &&
                  positions.dim() == 1 && positions.size(0) == x.size(0) &&
                  positions.is_contiguous(),
              "ROCm token-major RoPE: positions must be contiguous GPU int64 [tokens]");
  TORCH_CHECK(cos.is_cuda() && sin.is_cuda() && cos.scalar_type() == torch::kFloat32 &&
                  sin.scalar_type() == torch::kFloat32 && cos.is_contiguous() &&
                  sin.is_contiguous() && cos.dim() == 2 && sin.sizes() == cos.sizes(),
              "ROCm token-major RoPE: cos/sin must be contiguous GPU FP32 tables");
  TORCH_CHECK(cos.size(0) > 0 && cos.size(1) == head_dim / 2,
              "ROCm token-major RoPE: invalid cos/sin table shape");
  TORCH_CHECK(x.get_device() == positions.get_device() &&
                  x.get_device() == cos.get_device() && x.get_device() == sin.get_device(),
              "ROCm token-major RoPE: all tensors must share one device");

  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  auto out = torch::empty({x.size(0), x.size(1)}, x.options());
  const int64_t tokens = x.size(0);
  const int heads = static_cast<int>(x.size(1) / head_dim);
  const int half = static_cast<int>(head_dim / 2);
  const int table_rows = static_cast<int>(cos.size(0));
  const int64_t count = tokens * static_cast<int64_t>(heads) * half;
  constexpr int threads = 256;
  const int64_t blocks = (count + threads - 1) / threads;
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x.scalar_type(),
      "deterministic_rope_apply_token_major_rocm",
      [&] {
        hipLaunchKernelGGL(
            (deterministic_rope_token_major_kernel<scalar_t>),
            dim3(blocks),
            dim3(threads),
            0,
            stream,
            x.data_ptr<scalar_t>(),
            positions.data_ptr<int64_t>(),
            cos.data_ptr<float>(),
            sin.data_ptr<float>(),
            out.data_ptr<scalar_t>(),
            tokens,
            x.stride(0),
            heads,
            static_cast<int>(head_dim),
            half,
            table_rows,
            static_cast<float>(sin_sign));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
#endif
