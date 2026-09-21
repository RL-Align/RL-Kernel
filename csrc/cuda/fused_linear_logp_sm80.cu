// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// SM80 (Ampere, e.g. A100) native fused linear_logp -- STAGE-2 PoC.
//
// Forward-only, BF16, single GPU. Computes, for each token row n:
//
//   logit[n, v] = hidden[n, :] @ weight[v, :]^T        (weight is [V, D])
//   logp[n]      = logit[n, target[n]] - logsumexp_v(logit[n, v])
//
// The full [N, V] logit tensor is NEVER materialized: each CTA owns BM token
// rows and streams over the whole vocabulary in BN-wide tiles. For every tile
// it runs a tiny shared-memory WMMA GEMM (m16n16k16, BF16 inputs, FP32
// accumulators) and folds the result into a per-row online-softmax state
// (running max / sum-exp) while separately remembering the raw target logit.
// The four warps of a CTA split the BN vocab columns; their partial states are
// merged once at the end: logp = target_logit - (max + log sumexp).
//
// Stage-2 design choices:
//   * nvcuda::wmma API, no hand-written mma.sync PTX;
//   * cp.async global-to-shared copies with a two-stage K pipeline;
//   * BM=16 rows per CTA, one accumulator fragment per K step;
//   * no bias / temperature / TP / padding-vocab masking (V must be a
//     multiple of BN);
//   * forward only; no backward kernel.
//
// Validated shape: D=4096, V=128256, BF16, N in [128, 4096]. The kernel itself
// only requires D % 16 == 0 and V % 64 == 0.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <mma.h>

namespace kernel_align {
namespace sm80_poc {

// Stage-2 tuning point. Keep these compile-time constants so nvcc can fully
// unroll the fragment grid; benchmarked variants change only these three.
constexpr int BM = 32;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int WARPS = 8;
constexpr int THREADS = WARPS * 32;
constexpr int STAGES = 2;
constexpr int WM = BM / 16;
constexpr int WN = BN / 8;
constexpr int FRAGMENTS = WM * WN;
constexpr int FRAGS_PER_WARP = (FRAGMENTS + WARPS - 1) / WARPS;
static_assert(BM % 16 == 0 && BN % 16 == 0 && BK % 16 == 0);

__device__ __forceinline__ void cp_async_16(void* smem_ptr,
                                            const void* global_ptr,
                                            int valid_bytes = 16) {
  const unsigned smem_addr = static_cast<unsigned>(
      __cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
               :: "r"(smem_addr), "l"(global_ptr), "r"(valid_bytes));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_group 0;\n" ::);
}

__device__ __forceinline__ void ldmatrix_x4(
    unsigned (&dst)[4], const __nv_bfloat16* ptr) {
  const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(ptr));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
      : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2_trans(
    unsigned (&dst)[2], const __nv_bfloat16* ptr) {
  const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(ptr));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
      : "=r"(dst[0]), "=r"(dst[1]) : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2(
    unsigned (&dst)[2], const __nv_bfloat16* ptr) {
  const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(ptr));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
      : "=r"(dst[0]), "=r"(dst[1]) : "r"(addr));
}

__device__ __forceinline__ void mma_m16n8k16(float (&d)[4],
                                              const unsigned (&a)[4],
                                              const unsigned (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
        "r"(b[0]), "r"(b[1]));
}

// Per-warp, per-row state, held in shared memory:
//   state 0: running max (m)
//   state 1: sum exp(x - m)
//   state 2: raw target logit seen in this warp's tiles (-inf if none yet)
__global__ void fused_linear_logp_sm80_kernel(
    const __nv_bfloat16* __restrict__ hidden,  // [N, D], row major, BF16
    const __nv_bfloat16* __restrict__ weight,  // [V, D], row major, BF16
    const int* __restrict__ target,            // [N]
    float* __restrict__ out_logp,              // [N], FP32
    int N, int D, int V) {
  const int row_base = blockIdx.x * BM;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int num_rows = min(BM, N - row_base);

  __shared__ __nv_bfloat16 sH[STAGES][BM * BK];
  __shared__ __nv_bfloat16 sW[STAGES][BN * BK];
  __shared__ float sC[WARPS][16 * 8];
  // One online-softmax partial per 8-column vocab fragment and token row.
  __shared__ float sPart[WN][BM * 3];

  // m = -inf, sumexp = 0, target_logit = -inf for every (warp, row) triple.
  // sPart is [WN][BM*3]; flatten with the correct row stride (BM*3).
  for (int idx = threadIdx.x; idx < WN * BM * 3; idx += THREADS) {
    const int w = idx / (BM * 3);
    const int j = idx % (BM * 3);  // within a warp: row*3 + {0:m,1:sum,2:tgt}
    sPart[w][j] = (j % 3 == 1) ? 0.0f : -CUDART_INF_F;
  }
  __syncthreads();

  float acc[FRAGS_PER_WARP][4];

  auto prefetch_k = [&](int stage, int v0, int k0) {
    // Every transfer is 16 aligned bytes (8 bf16 values).  cp.async's
    // src-size operand zero-fills rows outside N without a divergent fallback.
    constexpr int ELEMS_PER_COPY = 8;
    constexpr int H_COPIES = BM * BK / ELEMS_PER_COPY;
    constexpr int W_COPIES = BN * BK / ELEMS_PER_COPY;
    for (int copy = threadIdx.x; copy < H_COPIES; copy += THREADS) {
      const int elem = copy * ELEMS_PER_COPY;
      const int r = elem / BK;
      const int k = elem % BK;
      const int gr = row_base + r;
      const auto* src = hidden + static_cast<long long>(gr) * D + k0 + k;
      cp_async_16(sH[stage] + elem, src, gr < N ? 16 : 0);
    }
    for (int copy = threadIdx.x; copy < W_COPIES; copy += THREADS) {
      const int elem = copy * ELEMS_PER_COPY;
      const int v = elem / BK;
      const int k = elem % BK;
      const auto* src = weight + static_cast<long long>(v0 + v) * D + k0 + k;
      cp_async_16(sW[stage] + elem, src);
    }
    cp_async_commit();
  };

  for (int v0 = 0; v0 < V; v0 += BN) {
    #pragma unroll
    for (int q = 0; q < FRAGS_PER_WARP; ++q)
      #pragma unroll
      for (int i = 0; i < 4; ++i) acc[q][i] = 0.0f;

    prefetch_k(0, v0, 0);
    for (int k0 = 0, step = 0; k0 < D; k0 += BK, ++step) {
      const int stage = step & 1;
      cp_async_wait_all();
      __syncthreads();

      if (k0 + BK < D) prefetch_k(stage ^ 1, v0, k0 + BK);

      #pragma unroll
      for (int kk = 0; kk < BK; kk += 16) {
        #pragma unroll
        for (int q = 0; q < FRAGS_PER_WARP; ++q) {
          const int frag = warp + q * WARPS;
          if (frag < FRAGMENTS) {
          const int mfrag = frag / WN;
          const int nfrag = frag % WN;
          const int group = lane >> 2;
          const int tid = lane & 3;
          unsigned a[4], b[2];
          const int a_row = (lane & 7) + ((lane >> 3) & 1) * 8;
          const int a_col = (lane >> 4) * 8;
          ldmatrix_x4(a, sH[stage] + mfrag * 16 * BK +
                         a_row * BK + kk + a_col);
          // sW is physically [N,K] row-major, i.e. the logical [K,N]
          // operand is already column-major. Therefore B uses non-transposed
          // ldmatrix; .trans is for a logical row-major [K,N] shared tile.
          const int b_row = lane & 7;
          const int b_col = ((lane >> 3) & 1) * 8;
          ldmatrix_x2(b, sW[stage] + nfrag * 8 * BK +
                         b_row * BK + kk + b_col);
          mma_m16n8k16(acc[q], a, b);
          }
        }
      }
    }

    #pragma unroll
    for (int q = 0; q < FRAGS_PER_WARP; ++q) {
      const int frag = warp + q * WARPS;
      if (frag >= FRAGMENTS) continue;
      const int mfrag = frag / WN;
      const int nfrag = frag % WN;
      // mma.m16n8 accumulator ownership: d0/d1 are row lane/4, d2/d3
      // are row lane/4+8; adjacent register pairs own adjacent columns.
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int rr = (lane >> 2) + (i >> 1) * 8;
        const int cc = (lane & 3) * 2 + (i & 1);
        sC[warp][rr * 8 + cc] = acc[q][i];
      }
      __syncwarp();

      // Fold this fragment into its vocab-partition online state.
      for (int lr = 0; lr < 16; ++lr) {
        const int r = mfrag * 16 + lr;
        if (r >= num_rows) continue;
        const int global_row = row_base + r;
        const int target_v = target[global_row];
        const int vocab = v0 + nfrag * 8 + lane;

        float x = -CUDART_INF_F;
        if (lane < 8 && vocab < V) x = sC[warp][lr * 8 + lane];

        float tile_max = x;
        for (int off = 16; off >= 1; off >>= 1)
          tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, off));

        float tile_sum = (lane < 8) ? expf(x - tile_max) : 0.0f;
        for (int off = 16; off >= 1; off >>= 1)
          tile_sum += __shfl_xor_sync(0xffffffffu, tile_sum, off);

        float tile_tgt = (lane < 8 && vocab == target_v) ? x : -CUDART_INF_F;
        for (int off = 16; off >= 1; off >>= 1)
          tile_tgt = fmaxf(tile_tgt, __shfl_xor_sync(0xffffffffu, tile_tgt, off));

        if (lane == 0) {
          float* state = sPart[nfrag] + r * 3;
          const float m_old = state[0];
          const float m_new = fmaxf(m_old, tile_max);
          const float corr_old = expf(m_old - m_new);
          const float corr_new = expf(tile_max - m_new);
          state[0] = m_new;
          state[1] = state[1] * corr_old + tile_sum * corr_new;
          state[2] = fmaxf(state[2], tile_tgt);
        }
      }
    }
    // All warps must finish touching sC / sPart before the next tile's loads.
    __syncthreads();
  }

  // Merge the four warp-local vocab partials for each row into the global
  // running max and sum-exp, giving LSE = m + log(s); the raw target logit is
  // simply the one non-(-inf) value across warps. logp = target_logit - LSE.
  for (int r = threadIdx.x; r < num_rows; r += THREADS) {
    float m = -CUDART_INF_F;
    float s = 0.0f;
    float target_logit = -CUDART_INF_F;
    for (int w = 0; w < WN; ++w) {
      const float* state = sPart[w] + r * 3;
      const float mw = state[0];
      const float sw = state[1];
      const float tw = state[2];
      const float mn = fmaxf(m, mw);
      s = s * expf(m - mn) + sw * expf(mw - mn);
      target_logit = fmaxf(target_logit, tw);
      m = mn;
    }
    out_logp[row_base + r] = target_logit - (m + logf(s));
  }
}

torch::Tensor fused_linear_logp_sm80_forward(torch::Tensor hidden,
                                             torch::Tensor weight,
                                             torch::Tensor target) {
  TORCH_CHECK(hidden.is_cuda() && weight.is_cuda(),
              "fused_linear_logp_sm80: hidden and weight must be CUDA tensors");
  TORCH_CHECK(weight.device() == hidden.device(),
              "fused_linear_logp_sm80: weight must be on the same device as hidden");
  TORCH_CHECK(hidden.scalar_type() == at::kBFloat16,
              "fused_linear_logp_sm80: hidden must be bfloat16");
  TORCH_CHECK(weight.scalar_type() == at::kBFloat16,
              "fused_linear_logp_sm80: weight must be bfloat16");
  TORCH_CHECK(hidden.dim() == 2 && weight.dim() == 2,
              "fused_linear_logp_sm80: hidden/weight must be 2-D");
  TORCH_CHECK(hidden.is_contiguous() && weight.is_contiguous(),
              "fused_linear_logp_sm80: inputs must be contiguous");

  const int N = static_cast<int>(hidden.size(0));
  const int D = static_cast<int>(hidden.size(1));
  const int V = static_cast<int>(weight.size(0));
  TORCH_CHECK(weight.size(1) == D,
              "fused_linear_logp_sm80: hidden/weight hidden-dim mismatch");
  TORCH_CHECK(target.numel() == N,
              "fused_linear_logp_sm80: target must have one id per token");
  TORCH_CHECK(D % BK == 0, "fused_linear_logp_sm80: D must be a multiple of ", BK);
  TORCH_CHECK(V % BN == 0, "fused_linear_logp_sm80: V must be a multiple of ", BN);
  TORCH_CHECK(N > 0, "fused_linear_logp_sm80: N must be positive");

  c10::cuda::CUDAGuard device_guard(hidden.device());
  auto target_i = target.to(at::kInt).contiguous();
  auto out_logp = torch::empty({N}, hidden.options().dtype(at::kFloat));

  const int blocks = (N + BM - 1) / BM;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  fused_linear_logp_sm80_kernel<<<blocks, THREADS, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(hidden.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
      target_i.data_ptr<int>(), out_logp.data_ptr<float>(), N, D, V);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return out_logp;
}

}  // namespace sm80_poc
}  // namespace kernel_align

// Global-scope entry point referenced by the pybind declarations in csrc/ops.cpp.
torch::Tensor fused_linear_logp_sm80_forward(torch::Tensor hidden,
                                             torch::Tensor weight,
                                             torch::Tensor target) {
  return kernel_align::sm80_poc::fused_linear_logp_sm80_forward(
      std::move(hidden), std::move(weight), std::move(target));
}
