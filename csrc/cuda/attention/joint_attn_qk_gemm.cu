// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// Qwen-Image WS1 joint attention QK GEMM (issue #386 row `joint_attn_qk_gemm`).
// scores = single_cast((Q K^T) * scale) over the non-causal [text, image] joint
// sequence. Contract `joint-attn-qk-gemm-tree-v1`; the authoritative definition
// lives in rl_engine/reference/attention/joint_attn_qk_gemm.py and
// docs/operators/joint-attn-qk-gemm.md.
//
// Arithmetic discipline, matching the fp32 CPU gold byte for byte:
//   * Every multiply-into-add site is an explicit __fmaf_rn, so the result does
//     not depend on compiler FP contraction. No `acc += a * b`.
//   * Forward: each score element reduces over the head dim D alone, in 32-wide
//     ascending-d leaf chains started from +0.0f, combined by a mid-split tree.
//     For a power-of-two leaf count the mid-split tree is identical to a
//     level-by-level halving combine, which is what the template below emits.
//     Because the tree depends only on D, bits are independent of batch size,
//     batch position, prompt padding, launch geometry and tiling.
//   * scale is applied once in fp32 after the complete tree, then a single RNE
//     cast at the output boundary.
//   * Backward: the sequence-dim reductions are NOT closed in-kernel. Each
//     kernel emits one 32-wide leaf partial in fp32 and the host combines them
//     with the very same mid-split tree code the reference uses, so the merge
//     order is shared by construction rather than by reimplementation. This
//     also keeps the general (non-power-of-two, short-tail) tree exact, which an
//     in-kernel halving loop could not do.
//   * No Split-K, no Stream-K, no atomics, no TF32, no fast math, no autotuning.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

constexpr int kLeafWidth = 32;
constexpr int kMaxLeaves = 32;  // head dims up to kLeafWidth * kMaxLeaves = 1024
constexpr int kFwdTileKeys = 32;
constexpr int kFwdTileQueries = 8;
constexpr int kBwdThreads = 128;

// ---------------------------------------------------------------------------
// Forward: one thread per score element, full D tree in registers.
// Grid: (ceil(Sk / kFwdTileKeys), ceil(Sq / kFwdTileQueries), B * H)
// ---------------------------------------------------------------------------
template <typename scalar_t, typename out_t, int LEAF_COUNT>
__global__ void joint_qk_gemm_forward_kernel(
    const scalar_t* __restrict__ Q,  // [B, H, Sq, D]
    const scalar_t* __restrict__ K,  // [B, H, Sk, D]
    out_t* __restrict__ scores,      // [B, H, Sq, Sk]
    int64_t H,
    int64_t Sq,
    int64_t Sk,
    float scale) {
  constexpr int64_t D = static_cast<int64_t>(LEAF_COUNT) * kLeafWidth;

  const int64_t key = static_cast<int64_t>(blockIdx.x) * kFwdTileKeys + threadIdx.x;
  const int64_t query = static_cast<int64_t>(blockIdx.y) * kFwdTileQueries + threadIdx.y;
  if (query >= Sq || key >= Sk) {
    return;
  }

  const int64_t bh = blockIdx.z;
  const int64_t b = bh / H;
  const int64_t h = bh % H;

  const scalar_t* __restrict__ q_row = Q + (((b * H) + h) * Sq + query) * D;
  const scalar_t* __restrict__ k_row = K + (((b * H) + h) * Sk + key) * D;

  // 32-wide ascending-d leaf chains, each started from +0.0f.
  float part[LEAF_COUNT];
#pragma unroll
  for (int leaf = 0; leaf < LEAF_COUNT; ++leaf) {
    float acc = 0.0f;
    const int d0 = leaf * kLeafWidth;
#pragma unroll
    for (int t = 0; t < kLeafWidth; ++t) {
      acc = __fmaf_rn(static_cast<float>(q_row[d0 + t]), static_cast<float>(k_row[d0 + t]), acc);
    }
    part[leaf] = acc;
  }

  // Mid-split combine; for a power-of-two LEAF_COUNT this halving loop emits
  // exactly T(l, r) = T(l, m) + T(m, r).
#pragma unroll
  for (int width = LEAF_COUNT; width > 1; width >>= 1) {
#pragma unroll
    for (int i = 0; i < width / 2; ++i) {
      part[i] = part[2 * i] + part[2 * i + 1];
    }
  }

  const int64_t out_index = (((b * H) + h) * Sq + query) * Sk + key;
  scores[out_index] = static_cast<out_t>(part[0] * scale);
}

// ---------------------------------------------------------------------------
// Backward dQ leaf: partial[b, h, i, d] = sum_{j in [j0, j1)} G[b,h,i,j] * K[b,h,j,d]
// One thread per (d); grid (ceil(D / kBwdThreads), Sq, B * H).
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void joint_qk_gemm_dq_leaf_kernel(
    const float* __restrict__ G,     // [B, H, Sq, Sk] fp32
    const scalar_t* __restrict__ K,  // [B, H, Sk, D]
    float* __restrict__ partial,     // [B, H, Sq, D] fp32
    int64_t H,
    int64_t Sq,
    int64_t Sk,
    int64_t D,
    int64_t j0,
    int64_t j1) {
  const int64_t d = static_cast<int64_t>(blockIdx.x) * kBwdThreads + threadIdx.x;
  if (d >= D) {
    return;
  }
  const int64_t query = blockIdx.y;
  const int64_t bh = blockIdx.z;
  const int64_t b = bh / H;
  const int64_t h = bh % H;

  const float* __restrict__ g_row = G + (((b * H) + h) * Sq + query) * Sk;
  const scalar_t* __restrict__ k_base = K + ((b * H) + h) * Sk * D;

  float acc = 0.0f;
  for (int64_t j = j0; j < j1; ++j) {
    acc = __fmaf_rn(g_row[j], static_cast<float>(k_base[j * D + d]), acc);
  }
  partial[(((b * H) + h) * Sq + query) * D + d] = acc;
}

// ---------------------------------------------------------------------------
// Backward dK leaf: partial[b, h, j, d] = sum_{i in [i0, i1)} G[b,h,i,j] * Q[b,h,i,d]
// One thread per (d); grid (ceil(D / kBwdThreads), Sk, B * H).
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void joint_qk_gemm_dk_leaf_kernel(
    const float* __restrict__ G,     // [B, H, Sq, Sk] fp32
    const scalar_t* __restrict__ Q,  // [B, H, Sq, D]
    float* __restrict__ partial,     // [B, H, Sk, D] fp32
    int64_t H,
    int64_t Sq,
    int64_t Sk,
    int64_t D,
    int64_t i0,
    int64_t i1) {
  const int64_t d = static_cast<int64_t>(blockIdx.x) * kBwdThreads + threadIdx.x;
  if (d >= D) {
    return;
  }
  const int64_t key = blockIdx.y;
  const int64_t bh = blockIdx.z;
  const int64_t b = bh / H;
  const int64_t h = bh % H;

  const float* __restrict__ g_base = G + ((b * H) + h) * Sq * Sk;
  const scalar_t* __restrict__ q_base = Q + ((b * H) + h) * Sq * D;

  float acc = 0.0f;
  for (int64_t i = i0; i < i1; ++i) {
    acc = __fmaf_rn(g_base[i * Sk + key], static_cast<float>(q_base[i * D + d]), acc);
  }
  partial[(((b * H) + h) * Sk + key) * D + d] = acc;
}

void check_qk_pair(const at::Tensor& q, const at::Tensor& k, const char* fn) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda(), fn, ": q and k must be CUDA tensors");
  TORCH_CHECK(q.device() == k.device(), fn, ": q and k must share a device");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4, fn, ": q and k must be [B, H, S, D]");
  TORCH_CHECK(q.scalar_type() == k.scalar_type(), fn, ": q and k dtypes must match");
  TORCH_CHECK(
      q.scalar_type() == at::kFloat || q.scalar_type() == at::kBFloat16,
      fn,
      ": dtype is outside the WS1 contract for this operator (fp32 or bf16 only)");
  TORCH_CHECK(q.size(0) == k.size(0), fn, ": batch mismatch");
  TORCH_CHECK(q.size(1) == k.size(1), fn, ": head count mismatch (no GQA for joint attention)");
  TORCH_CHECK(q.size(3) == k.size(3), fn, ": head dim mismatch");
}

}  // namespace

// Forward. `fp32_out` selects the fp32 gold-comparison output; otherwise the
// single RNE cast lands on the input dtype.
torch::Tensor joint_attn_qk_gemm_forward_cuda(
    torch::Tensor q,
    torch::Tensor k,
    double scale,
    bool fp32_out) {
  check_qk_pair(q, k, "joint_attn_qk_gemm_forward");
  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(q));

  auto q_contig = q.contiguous();
  auto k_contig = k.contiguous();

  const int64_t B = q_contig.size(0);
  const int64_t H = q_contig.size(1);
  const int64_t Sq = q_contig.size(2);
  const int64_t Sk = k_contig.size(2);
  const int64_t D = q_contig.size(3);

  TORCH_CHECK(Sq >= 1 && Sk >= 1, "joint_attn_qk_gemm_forward: empty sequence");
  TORCH_CHECK(
      D % kLeafWidth == 0,
      "joint_attn_qk_gemm_forward: head dim ",
      D,
      " must be a multiple of the ",
      kLeafWidth,
      "-wide leaf; other head dims fall back to the reference");
  const int64_t leaf_count = D / kLeafWidth;
  TORCH_CHECK(
      leaf_count <= kMaxLeaves && (leaf_count & (leaf_count - 1)) == 0,
      "joint_attn_qk_gemm_forward: head dim ",
      D,
      " needs a power-of-two leaf count no greater than ",
      kMaxLeaves,
      "; other geometries fall back to the reference so the general mid-split "
      "tree stays exact");

  auto out_options = q_contig.options().dtype(fp32_out ? at::kFloat : q_contig.scalar_type());
  auto scores = torch::empty({B, H, Sq, Sk}, out_options);

  const dim3 block(kFwdTileKeys, kFwdTileQueries);
  const dim3 grid(
      static_cast<unsigned>((Sk + kFwdTileKeys - 1) / kFwdTileKeys),
      static_cast<unsigned>((Sq + kFwdTileQueries - 1) / kFwdTileQueries),
      static_cast<unsigned>(B * H));
  auto stream = at::cuda::getCurrentCUDAStream();
  const float scale_f = static_cast<float>(scale);

#define RLK_LAUNCH_FWD(OUT_T, LC)                                                  \
  joint_qk_gemm_forward_kernel<scalar_t, OUT_T, LC><<<grid, block, 0, stream>>>(    \
      q_contig.data_ptr<scalar_t>(),                                               \
      k_contig.data_ptr<scalar_t>(),                                               \
      scores.data_ptr<OUT_T>(),                                                    \
      H,                                                                           \
      Sq,                                                                          \
      Sk,                                                                          \
      scale_f)

#define RLK_DISPATCH_LEAVES(OUT_T)                                                 \
  switch (leaf_count) {                                                            \
    case 1:                                                                        \
      RLK_LAUNCH_FWD(OUT_T, 1);                                                     \
      break;                                                                       \
    case 2:                                                                        \
      RLK_LAUNCH_FWD(OUT_T, 2);                                                     \
      break;                                                                       \
    case 4:                                                                        \
      RLK_LAUNCH_FWD(OUT_T, 4);                                                     \
      break;                                                                       \
    case 8:                                                                        \
      RLK_LAUNCH_FWD(OUT_T, 8);                                                     \
      break;                                                                       \
    case 16:                                                                       \
      RLK_LAUNCH_FWD(OUT_T, 16);                                                    \
      break;                                                                       \
    case 32:                                                                       \
      RLK_LAUNCH_FWD(OUT_T, 32);                                                    \
      break;                                                                       \
    default:                                                                       \
      TORCH_CHECK(false, "joint_attn_qk_gemm_forward: unsupported leaf count");    \
  }

  AT_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, q_contig.scalar_type(), "joint_qk_gemm_forward", [&] {
        if (fp32_out) {
          RLK_DISPATCH_LEAVES(float);
        } else {
          RLK_DISPATCH_LEAVES(scalar_t);
        }
      });

#undef RLK_DISPATCH_LEAVES
#undef RLK_LAUNCH_FWD

  return scores;
}

// One dQ leaf partial over keys [j0, j1), returned in fp32 for the host tree.
torch::Tensor joint_attn_qk_gemm_dq_leaf_cuda(
    torch::Tensor grad_scores_fp32,
    torch::Tensor k,
    int64_t j0,
    int64_t j1) {
  TORCH_CHECK(grad_scores_fp32.is_cuda() && k.is_cuda(), "dq_leaf: tensors must be CUDA");
  TORCH_CHECK(
      grad_scores_fp32.scalar_type() == at::kFloat, "dq_leaf: grad must already be fp32");
  TORCH_CHECK(grad_scores_fp32.dim() == 4 && k.dim() == 4, "dq_leaf: expected 4-D tensors");
  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(k));

  auto g = grad_scores_fp32.contiguous();
  auto k_contig = k.contiguous();
  const int64_t B = g.size(0);
  const int64_t H = g.size(1);
  const int64_t Sq = g.size(2);
  const int64_t Sk = g.size(3);
  const int64_t D = k_contig.size(3);
  TORCH_CHECK(k_contig.size(2) == Sk, "dq_leaf: grad key extent does not match k");
  TORCH_CHECK(0 <= j0 && j0 < j1 && j1 <= Sk, "dq_leaf: leaf range out of bounds");

  auto partial = torch::empty({B, H, Sq, D}, k_contig.options().dtype(at::kFloat));
  const dim3 block(kBwdThreads);
  const dim3 grid(
      static_cast<unsigned>((D + kBwdThreads - 1) / kBwdThreads),
      static_cast<unsigned>(Sq),
      static_cast<unsigned>(B * H));
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, k_contig.scalar_type(), "joint_qk_gemm_dq_leaf", [&] {
        joint_qk_gemm_dq_leaf_kernel<scalar_t><<<grid, block, 0, stream>>>(
            g.data_ptr<float>(),
            k_contig.data_ptr<scalar_t>(),
            partial.data_ptr<float>(),
            H,
            Sq,
            Sk,
            D,
            j0,
            j1);
      });
  return partial;
}

// One dK leaf partial over queries [i0, i1), returned in fp32 for the host tree.
torch::Tensor joint_attn_qk_gemm_dk_leaf_cuda(
    torch::Tensor grad_scores_fp32,
    torch::Tensor q,
    int64_t i0,
    int64_t i1) {
  TORCH_CHECK(grad_scores_fp32.is_cuda() && q.is_cuda(), "dk_leaf: tensors must be CUDA");
  TORCH_CHECK(
      grad_scores_fp32.scalar_type() == at::kFloat, "dk_leaf: grad must already be fp32");
  TORCH_CHECK(grad_scores_fp32.dim() == 4 && q.dim() == 4, "dk_leaf: expected 4-D tensors");
  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(q));

  auto g = grad_scores_fp32.contiguous();
  auto q_contig = q.contiguous();
  const int64_t B = g.size(0);
  const int64_t H = g.size(1);
  const int64_t Sq = g.size(2);
  const int64_t Sk = g.size(3);
  const int64_t D = q_contig.size(3);
  TORCH_CHECK(q_contig.size(2) == Sq, "dk_leaf: grad query extent does not match q");
  TORCH_CHECK(0 <= i0 && i0 < i1 && i1 <= Sq, "dk_leaf: leaf range out of bounds");

  auto partial = torch::empty({B, H, Sk, D}, q_contig.options().dtype(at::kFloat));
  const dim3 block(kBwdThreads);
  const dim3 grid(
      static_cast<unsigned>((D + kBwdThreads - 1) / kBwdThreads),
      static_cast<unsigned>(Sk),
      static_cast<unsigned>(B * H));
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, q_contig.scalar_type(), "joint_qk_gemm_dk_leaf", [&] {
        joint_qk_gemm_dk_leaf_kernel<scalar_t><<<grid, block, 0, stream>>>(
            g.data_ptr<float>(),
            q_contig.data_ptr<scalar_t>(),
            partial.data_ptr<float>(),
            H,
            Sq,
            Sk,
            D,
            i0,
            i1);
      });
  return partial;
}
