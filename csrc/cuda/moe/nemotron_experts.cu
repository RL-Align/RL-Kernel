// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// csrc/cuda/moe/nemotron_experts.cu
//
// Nemotron Nano grouped expert MLP + fixed-order weighted combine (SM90).
//
// The grouped GEMM arithmetic is the det_gemm contract, copied deliberately:
// `csrc/cuda/gemm/det_gemm_kernel.cu` owns `det_gemm_sm90_kernel` and
// `docs/operators/det-gemm.md` documents it.  Do not edit one without the
// other: `tests/backends/cuda/test_nemotron_experts_cuda.py` pins the up
// projection bitwise against `_C.det_gemm_fwd_rhs_transposed`.
//   * BK = K_TREE_LEAF = 32 (one TMA tile == one FP32 leaf),
//   * each leaf is two m16n8k16 mma.sync calls into an FP32 accumulator and
//     rounds once to BF16,
//   * leaves merge with the mid-split tree (`mid_tree_merge_count`), BF16 nodes,
//   * fixed ascending K order, one CTA per output tile, no split-K, no atomics.
//
// Differences from det_gemm, all in the mapping rather than in the arithmetic:
//   * segments: grid.y = cdiv(rows, BM) + nseg; each CTA reads `offsets` in
//     shared memory and resolves its (expert, row tile) itself, so the host
//     never reads `offsets` (CUDA Graph capture forbids the sync);
//   * stacked weights: `w_up [E,F,H]` and `w_down [E,H,F]` are one tensor map
//     over [E*N,K]; expert e owns rows [e*N, (e+1)*N);
//   * five operand modes (MODE_*) cover grouped forward, grouped backward and
//     weight gradients with `ldmatrix.trans` on the transposed operand;
//   * relu-square is applied to bf16 fragments, which is bitwise identical to
//     bf16(relu(fp32(x))**2) because a bf16 product is exact in FP32.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdint>

#if defined(RL_KERNEL_ENABLE_SM90)
#include "../gemm/det_gemm_tma.cuh"
#endif

namespace {

using nv_bf16 = __nv_bfloat16;

__host__ __device__ constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

// Must match det_gemm's SM90 BK so a leaf is the same 32 K values.
constexpr int K_TREE_LEAF = 32;

template <typename output_t>
__device__ __forceinline__ output_t cast_output(float value);

template <>
__device__ __forceinline__ nv_bf16 cast_output<nv_bf16>(float value) {
  return __float2bfloat16(value);
}

template <>
__device__ __forceinline__ float cast_output<float>(float value) {
  return value;
}

// Fixed FP32 midpoint tree over reduction leaves (see det_gemm_kernel.cu).
__host__ __device__ __forceinline__ int mid_tree_merge_count(int leaf, int n) {
  int lo = 0, hi = n, count = 0;
  while (hi - lo > 1) {
    const int mid = lo + (hi - lo) / 2;
    if (leaf < mid) {
      hi = mid;
      count = 0;
    } else {
      lo = mid;
      ++count;
    }
  }
  return count;
}

// ---------------------------------------------------------------------------
// Elementwise stages: permutation inverse, fixed-order combine, exit cast,
// route gradient and the FP32 weight-gradient dot products.
// ---------------------------------------------------------------------------

__global__ void nemotron_experts_invert_kernel(int64_t* __restrict__ inverse,
                                               const int64_t* __restrict__ permutation,
                                               int routes) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  // `permutation` is a bijection (router contract), so this scatter is race-free.
  if (index < routes) inverse[permutation[index]] = index;
}

// routed[t, j] = sum over slots ascending of weights[t, s] * Y[route_row(t, s), j].
// One thread owns four FP32 columns of one token, so the slot order and the FP32
// accumulation order are independent of the batch and of thread scheduling.
// The operator's only exit cast, fused with the combine: four FP32 columns per
// thread gather the six slots in ascending order, add the shared expert and round
// once, so no `routed` buffer is ever materialized.
__global__ void nemotron_experts_combine_kernel(nv_bf16* __restrict__ out,
                                                const float* __restrict__ down,
                                                const int64_t* __restrict__ inverse,
                                                const float* __restrict__ weights,
                                                const float* __restrict__ shared_down,
                                                int width4) {
  const int token = blockIdx.x;
  const int segment = blockIdx.y * blockDim.x + threadIdx.x;
  if (segment >= width4) return;
  const int64_t slot_base = (int64_t)token * 6;
  float4 total = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
  for (int slot = 0; slot < 6; ++slot) {
    const float weight = weights[slot_base + slot];
    const int64_t row = inverse[slot_base + slot];
    const float4 value =
        reinterpret_cast<const float4*>(down + row * (int64_t)width4 * 4)[segment];
    total.x += weight * value.x;
    total.y += weight * value.y;
    total.z += weight * value.z;
    total.w += weight * value.w;
  }
  if (shared_down != nullptr) {
    const float4 extra =
        reinterpret_cast<const float4*>(shared_down + (int64_t)token * width4 * 4)[segment];
    total.x += extra.x;
    total.y += extra.y;
    total.z += extra.z;
    total.w += extra.w;
  }
  nv_bf16* target = out + (int64_t)token * width4 * 4 + (int64_t)segment * 4;
  target[0] = __float2bfloat16(total.x);
  target[1] = __float2bfloat16(total.y);
  target[2] = __float2bfloat16(total.z);
  target[3] = __float2bfloat16(total.w);
}

// dY[packed row] = bf16(weights[t, slot] * dOut[t]): one route weight per row of
// the packed payload, so the cast happens once per element.
__global__ void nemotron_experts_dy_kernel(nv_bf16* __restrict__ dy,
                                           const nv_bf16* __restrict__ grad_out,
                                           const int64_t* __restrict__ permutation,
                                           const float* __restrict__ weights,
                                           int width8) {
  const int row = blockIdx.x;
  const int segment = blockIdx.y * blockDim.x + threadIdx.x;
  if (segment >= width8) return;
  const int64_t original = permutation[row];
  const float weight = weights[original];
  const nv_bf16* source = grad_out + (original / 6) * (int64_t)width8 * 8 + (int64_t)segment * 8;
  nv_bf16* target = dy + (int64_t)row * width8 * 8 + (int64_t)segment * 8;
#pragma unroll
  for (int index = 0; index < 8; ++index) {
    target[index] = __float2bfloat16(weight * __bfloat162float(source[index]));
  }
}

// One thread per (route, 32-wide leaf): sequential FP32 FMA inside the leaf, so
// the leaf value never depends on the block or warp shape.
__global__ void nemotron_experts_dweights_leaf_kernel(float* __restrict__ leaves,
                                                      const float* __restrict__ down,
                                                      const nv_bf16* __restrict__ grad_out,
                                                      const int64_t* __restrict__ inverse,
                                                      int routes, int leaf_count, int width) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int route = index / leaf_count;
  if (route >= routes) return;
  const int leaf = index % leaf_count;
  const int token = route / 6;
  const float* down_row = down + inverse[route] * (int64_t)width;
  const nv_bf16* grad_row = grad_out + (int64_t)token * width;
  const int lo = leaf * K_TREE_LEAF;
  const int hi = lo + K_TREE_LEAF < width ? lo + K_TREE_LEAF : width;
  float acc = 0.0f;
  for (int column = lo; column < hi; ++column)
    acc = fmaf(down_row[column], __bfloat162float(grad_row[column]), acc);
  leaves[(int64_t)route * leaf_count + leaf] = acc;
}

// Fixed FP32 midpoint tree over the leaf values; one thread per route.
__global__ void nemotron_experts_dweights_merge_kernel(float* __restrict__ dweights,
                                                       const float* __restrict__ leaves,
                                                       int leaf_count, int routes) {
  const int route = blockIdx.x * blockDim.x + threadIdx.x;
  if (route >= routes) return;
  const float* values = leaves + (int64_t)route * leaf_count;
  float stack[16];
  int depth = 0;
  float value = 0.0f;
  for (int leaf = 0; leaf < leaf_count; ++leaf) {
    value = values[leaf];
    int merge = mid_tree_merge_count(leaf, leaf_count);
    while (merge-- > 0) value = stack[--depth] + value;
    if (leaf + 1 < leaf_count) {
      stack[depth] = value;
      ++depth;
    }
  }
  dweights[route] = value;
}

}  // namespace

#if defined(RL_KERNEL_ENABLE_SM90)

namespace {

constexpr int BM = 128, BN = 64, BK = 32;
static_assert(BK == K_TREE_LEAF, "grouped tile width must match the K-tree leaf");
constexpr int WARPS = 4;
constexpr int WG_THREADS = WARPS * 32;
constexpr int STAGES = 2;
constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16;
constexpr int WARP_M = BM / WARPS;
constexpr int M_TILES = WARP_M / MMA_M;
constexpr int N_TILES = BN / MMA_N;
constexpr int K_TILES = BK / MMA_K;
constexpr int KK_GROUPS = BK / 32;
constexpr int TREE_DEPTH = 12;

// Operand modes.  A/B tile layouts and the ldmatrix flavor are the only things
// that change; the leaf and tree arithmetic is shared.
enum {
  MODE_UP = 0,        // A=[rows,K] row-major, B=stacked [E*N,K]; no activation
  MODE_DOWN = 1,      // same layout, relu-square applied to the A fragments
  MODE_DX = 2,        // A=[rows,K], B=stacked [E*K,N] (transposed weight operand)
  MODE_WGRAD = 3,     // A=A^T [rows,M], B=[rows,N]; weight gradient
  MODE_WGRAD_R2 = 4,  // same, relu-square applied to the B fragments
};

__device__ __forceinline__ void ldmatrix_x4(uint32_t regs[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
               : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t regs[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
               : "r"(addr));
}

__device__ __forceinline__ void mma_m16n8k16(const uint32_t A[4], const uint32_t B[2], float D[4]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
               : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
               : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
                 "f"(D[0]), "f"(D[1]), "f"(D[2]), "f"(D[3]));
}

// bf16(relu(x)**2) on two lanes at a time.  A bf16 product is exact in FP32, so
// this equals rounding the FP32 square once.
__device__ __forceinline__ void relu_square_inplace(uint32_t& reg) {
  __nv_bfloat162 value = *reinterpret_cast<__nv_bfloat162*>(&reg);
  const __nv_bfloat162 zero = __floats2bfloat162_rn(0.0f, 0.0f);
  value = __hmax2(value, zero);
  value = __hmul2(value, value);
  reg = *reinterpret_cast<uint32_t*>(&value);
}

template <typename output_t, int MODE, bool EPI_RELU2_GRAD>
__global__ void nemotron_experts_gemm(const __grid_constant__ CUtensorMap a_tmap,
                                      const __grid_constant__ CUtensorMap b_tmap,
                                      output_t* __restrict__ C,
                                      const nv_bf16* __restrict__ aux,
                                      const int64_t* __restrict__ offsets,
                                      int nseg, int M, int N, int K, int rows) {
  constexpr bool WGRAD = MODE >= MODE_WGRAD;
  constexpr bool TRANSPOSE_A = WGRAD;
  constexpr bool TRANSPOSE_B = MODE == MODE_DX || WGRAD;
  constexpr bool ACT_A = MODE == MODE_DOWN;
  constexpr bool ACT_B = MODE == MODE_WGRAD_R2;

  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int col_base = blockIdx.x * BN;

  // Resolve this CTA's segment.  Only thread 0 walks the offsets; the host never
  // reads them, so CUDA Graph capture stays legal.
  __shared__ int s_expert, s_tile;
  __shared__ int64_t s_lo, s_hi;
  if (tid == 0) {
    int expert = -1;
    int64_t lo = 0, hi = 0;
    int tile = 0;
    if (offsets == nullptr) {
      // Single segment [0, rows): the shared expert reuses this kernel verbatim.
      const int tiles = cdiv(WGRAD ? M : rows, BM);
      if (blockIdx.y < (int64_t)tiles && rows > 0) {
        expert = 0;
        lo = 0;
        hi = rows;
        tile = (int)blockIdx.y * BM;
      }
    } else if constexpr (WGRAD) {
      // Every expert owns cdiv(M,BM) output tiles, including empty segments:
      // those write zeros instead of requiring a memset of the whole tensor.
      const int tiles = cdiv(M, BM);
      if (blockIdx.y < (int64_t)tiles * nseg) {
        expert = blockIdx.y / tiles;
        tile = (blockIdx.y % tiles) * BM;
        lo = offsets[expert];
        hi = offsets[expert + 1];
      }
    } else {
      int remaining = blockIdx.y;
      for (int e = 0; e < nseg; ++e) {
        const int64_t begin = offsets[e], end = offsets[e + 1];
        const int tiles = (int)((end - begin + BM - 1) / BM);
        if (remaining < tiles) {
          expert = e;
          tile = remaining * BM;
          lo = begin;
          hi = end;
          break;
        }
        remaining -= tiles;
      }
    }
    s_expert = expert;
    s_tile = tile;
    s_lo = lo;
    s_hi = hi;
  }
  __syncthreads();
  if (s_expert < 0) return;  // Padding CTA above the real tile count.

  const int expert = s_expert;
  const int64_t seg_lo = s_lo, seg_hi = s_hi;
  // Row start of the A operand (grouped modes) or of the output tile (dW).
  const int64_t row_base = WGRAD ? (int64_t)s_tile : seg_lo + s_tile;
  const int out_row_base = WGRAD ? s_tile : (int)row_base;
  const int64_t out_offset = WGRAD ? (int64_t)expert * M : 0;
  const int tile_rows = M - s_tile;
  const int64_t segment_rows = seg_hi - row_base;
  const int local_rows = WGRAD ? (tile_rows < BM ? tile_rows : BM)
                               : (int)(segment_rows < (int64_t)BM ? segment_rows : (int64_t)BM);
  // Reduction depth: a fixed feature width, or the segment length for dW.
  const int kd = WGRAD ? (int)((seg_hi - seg_lo + BK - 1) / BK) : (K / BK);
  const int k_tail = (WGRAD && kd > 0) ? (int)(seg_hi - seg_lo - (int64_t)(kd - 1) * BK) : 0;

  extern __shared__ __align__(1024) char nemotron_smem[];
  nv_bf16* sA = reinterpret_cast<nv_bf16*>(nemotron_smem);
  nv_bf16* sB = reinterpret_cast<nv_bf16*>(sA + STAGES * BM * BK);
  int* mbar_base = reinterpret_cast<int*>(sB + STAGES * BN * BK);

  const uint32_t sA_base = static_cast<uint32_t>(__cvta_generic_to_shared(sA));
  const uint32_t sB_base = static_cast<uint32_t>(__cvta_generic_to_shared(sB));
  uint32_t mbar[STAGES];
#pragma unroll
  for (int s = 0; s < STAGES; ++s)
    mbar[s] = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_base + 2 * s));

  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < STAGES; ++s) det_gemm::mbar_init(mbar[s], 1);
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  __syncthreads();

  const uint32_t tile_bytes = (BM * BK + BN * BK) * sizeof(nv_bf16);

  auto issue_load = [&](int k) {
    const int buf = k % STAGES;
    int a_x, a_y, b_x, b_y;
    if constexpr (TRANSPOSE_A) {
      a_x = s_tile;  // column of the transposed payload == output row
      a_y = (int)(seg_lo + (int64_t)k * BK);
    } else {
      a_x = k * BK;
      a_y = (int)row_base;
    }
    if constexpr (MODE == MODE_DX) {
      b_x = col_base;
      b_y = (int)((int64_t)expert * K + (int64_t)k * BK);
    } else if constexpr (WGRAD) {
      b_x = col_base;
      b_y = (int)(seg_lo + (int64_t)k * BK);
    } else {
      b_x = k * BK;
      b_y = expert * N + col_base;
    }
    det_gemm::tma_2d_g2s(sA_base + buf * BM * BK * sizeof(nv_bf16), &a_tmap, a_x, a_y, mbar[buf]);
    det_gemm::tma_2d_g2s(sB_base + buf * BN * BK * sizeof(nv_bf16), &b_tmap, b_x, b_y, mbar[buf]);
    det_gemm::mbar_arrive_expect_tx(mbar[buf], tile_bytes);
  };

  int phase[STAGES];
#pragma unroll
  for (int s = 0; s < STAGES; ++s) phase[s] = 0;

  float tile_acc[M_TILES][N_TILES][4];
  __nv_bfloat162 tree_v[M_TILES][N_TILES][2];
  __nv_bfloat162 tree_stk[TREE_DEPTH][M_TILES][N_TILES][2];
  int sp = 0;

  // An empty expert has kd == 0, so the loop below never assigns tree_v. The
  // weight-gradient grid deliberately covers every expert, so the tile is then
  // stored as exact zeros instead of uninitialized stack.
  const __nv_bfloat162 tree_zero = __floats2bfloat162_rn(0.0f, 0.0f);
#pragma unroll
  for (int mi = 0; mi < M_TILES; ++mi)
#pragma unroll
    for (int n = 0; n < N_TILES; ++n)
#pragma unroll
      for (int i = 0; i < 2; ++i) tree_v[mi][n][i] = tree_zero;

  if (tid == 0)
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s)
      if (s < kd) issue_load(s);

  for (int k = 0; k < kd; ++k) {  // fixed ascending tile order, NO split-K
    const int buf = k % STAGES;
    if (tid == 0 && k + (STAGES - 1) < kd) issue_load(k + (STAGES - 1));
    if (tid == 0) det_gemm::mbar_wait(mbar[buf], phase[buf]);
    phase[buf] ^= 1;
    __syncthreads();

    const uint32_t sA_buf = sA_base + buf * BM * BK * sizeof(nv_bf16);
    const uint32_t sB_buf = sB_base + buf * BN * BK * sizeof(nv_bf16);

    if constexpr (WGRAD) {
      // The last reduction tile can read rows past this expert's segment; those
      // rows belong to the next expert and must not join the reduction.
      if (k + 1 == kd && k_tail < BK) {
        nv_bf16* a_stage = sA + buf * BM * BK;
        nv_bf16* b_stage = sB + buf * BN * BK;
        for (int index = tid; index < (BK - k_tail) * BM; index += WG_THREADS)
          a_stage[(int64_t)(k_tail + index / BM) * BM + index % BM] = __float2bfloat16(0.0f);
        for (int index = tid; index < (BK - k_tail) * BN; index += WG_THREADS)
          b_stage[(int64_t)(k_tail + index / BN) * BN + index % BN] = __float2bfloat16(0.0f);
        __syncthreads();
      }
    }

#pragma unroll
    for (int mi = 0; mi < M_TILES; ++mi)
#pragma unroll
      for (int n = 0; n < N_TILES; ++n)
        tile_acc[mi][n][0] = tile_acc[mi][n][1] = tile_acc[mi][n][2] = tile_acc[mi][n][3] = 0.0f;

    uint32_t A[M_TILES][K_TILES][4];
#pragma unroll
    for (int mi = 0; mi < M_TILES; ++mi) {
#pragma unroll
      for (int kt = 0; kt < K_TILES; ++kt) {
        if constexpr (TRANSPOSE_A) {
          // Source tile is [BK,BM] (k-major): ldmatrix.trans builds the
          // row-major A fragment.
          const int k_row = kt * MMA_K + (lane / 16) * 8 + (lane % 8);
          const int m_col = warp * WARP_M + mi * MMA_M + ((lane / 8) % 2) * 8;
          ldmatrix_x4_trans(A[mi][kt], sA_buf + (k_row * BM + m_col) * sizeof(nv_bf16));
        } else {
          const int row0 = warp * WARP_M + mi * MMA_M + (lane % 16);
          ldmatrix_x4(A[mi][kt],
                      sA_buf + (row0 * BK + (lane / 16) * 8 + kt * MMA_K) * sizeof(nv_bf16));
        }
        if constexpr (ACT_A) {
#pragma unroll
          for (int reg = 0; reg < 4; ++reg) relu_square_inplace(A[mi][kt][reg]);
        }
      }
    }

#pragma unroll
    for (int n = 0; n < N_TILES; ++n) {
#pragma unroll
      for (int kk = 0; kk < KK_GROUPS; ++kk) {
        uint32_t b4[4];
        if constexpr (TRANSPOSE_B) {
          // Source tile is [BK,BN] (k-major).
          const int k_row = kk * 32 + lane;
          ldmatrix_x4_trans(b4, sB_buf + (k_row * BN + n * MMA_N) * sizeof(nv_bf16));
        } else {
          ldmatrix_x4(b4, sB_buf + ((n * MMA_N + (lane % 8)) * BK + (lane / 8) * 8 + kk * 32) *
                                       sizeof(nv_bf16));
        }
        if constexpr (ACT_B) {
#pragma unroll
          for (int reg = 0; reg < 4; ++reg) relu_square_inplace(b4[reg]);
        }
        const uint32_t B0[2] = {b4[0], b4[1]};
        const uint32_t B1[2] = {b4[2], b4[3]};
#pragma unroll
        for (int mi = 0; mi < M_TILES; ++mi) {
          mma_m16n8k16(A[mi][2 * kk + 0], B0, tile_acc[mi][n]);
          mma_m16n8k16(A[mi][2 * kk + 1], B1, tile_acc[mi][n]);
        }
      }
    }
    __syncthreads();

#pragma unroll
    for (int mi = 0; mi < M_TILES; ++mi)
#pragma unroll
      for (int n = 0; n < N_TILES; ++n)
#pragma unroll
        for (int i = 0; i < 2; ++i)
          tree_v[mi][n][i] =
              __floats2bfloat162_rn(tile_acc[mi][n][2 * i + 0], tile_acc[mi][n][2 * i + 1]);

    const int merge_count = mid_tree_merge_count(k, kd);
    for (int merge = 0; merge < merge_count; ++merge) {
#pragma unroll
      for (int mi = 0; mi < M_TILES; ++mi)
#pragma unroll
        for (int n = 0; n < N_TILES; ++n)
#pragma unroll
          for (int i = 0; i < 2; ++i)
            tree_v[mi][n][i] = __hadd2(tree_stk[sp - 1][mi][n][i], tree_v[mi][n][i]);
      --sp;
    }
    if (k + 1 < kd) {
#pragma unroll
      for (int mi = 0; mi < M_TILES; ++mi)
#pragma unroll
        for (int n = 0; n < N_TILES; ++n)
#pragma unroll
          for (int i = 0; i < 2; ++i) tree_stk[sp][mi][n][i] = tree_v[mi][n][i];
      ++sp;
    }
  }

  auto store_value = [&](int row_local, int col, float value) {
    const int64_t index = (out_offset + out_row_base + row_local) * (int64_t)N + col;
    if constexpr (EPI_RELU2_GRAD) {
      // relu-square derivative: dU = dZ * 2 * relu(U), one rounding.
      const float up = __bfloat162float(aux[index]);
      value = value * (2.0f * (up > 0.0f ? up : 0.0f));
    }
    C[index] = cast_output<output_t>(value);
  };

#pragma unroll
  for (int mi = 0; mi < M_TILES; ++mi) {
    const int row = warp * WARP_M + mi * MMA_M + lane / 4;
#pragma unroll
    for (int n = 0; n < N_TILES; ++n) {
      const int col = col_base + n * MMA_N + (lane % 4) * 2;
      if (col + 1 < N) {
        if (row < local_rows) {
          store_value(row, col + 0, __low2float(tree_v[mi][n][0]));
          store_value(row, col + 1, __high2float(tree_v[mi][n][0]));
        }
        if (row + 8 < local_rows) {
          store_value(row + 8, col + 0, __low2float(tree_v[mi][n][1]));
          store_value(row + 8, col + 1, __high2float(tree_v[mi][n][1]));
        }
      }
    }
  }
}

template <typename output_t, int MODE, bool EPI_RELU2_GRAD>
void launch_grouped(const nv_bf16* a_ptr, uint64_t a_height, uint64_t a_width,
                    const nv_bf16* b_ptr, uint64_t b_height, uint64_t b_width,
                    output_t* c_ptr, const nv_bf16* aux_ptr,
                    const int64_t* offsets, int nseg, int M, int N, int K, int rows,
                    cudaStream_t stream) {
  CUtensorMap a_tmap, b_tmap;
  if constexpr (MODE >= MODE_WGRAD) {
    // A is the transposed payload: [rows, M] addressed in (k, m) tiles.
    det_gemm::init_tmap_noswizzle(&a_tmap, a_ptr, a_height, a_width, BK, BM);
    det_gemm::init_tmap_noswizzle(&b_tmap, b_ptr, b_height, b_width, BK, BN);
  } else {
    det_gemm::init_tmap_noswizzle(&a_tmap, a_ptr, a_height, a_width, BM, BK);
    if constexpr (MODE == MODE_DX)
      det_gemm::init_tmap_noswizzle(&b_tmap, b_ptr, b_height, b_width, BK, BN);
    else
      det_gemm::init_tmap_noswizzle(&b_tmap, b_ptr, b_height, b_width, BN, BK);
  }

  // Grouped tiles: ceil(n_e/BM) sums to at most `rows`, so the tighter of the
  // additive bound and `rows` avoids launching padding CTAs at decode batches
  // (T=1 has 6 routes, not 129 candidate tiles).
  const int64_t additive = (int64_t)cdiv(rows, BM) + nseg;
  const int64_t grid_y = MODE >= MODE_WGRAD
                             ? (int64_t)cdiv(M, BM) * nseg
                             : (additive < rows ? additive : (int64_t)rows);
  if (grid_y <= 0) return;
  const int smem = STAGES * (BM * BK + BN * BK) * sizeof(nv_bf16) + STAGES * 8;
  if (smem > 48 * 1024)
    cudaFuncSetAttribute(nemotron_experts_gemm<output_t, MODE, EPI_RELU2_GRAD>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  dim3 grid(cdiv(N, BN), (unsigned)grid_y);
  nemotron_experts_gemm<output_t, MODE, EPI_RELU2_GRAD>
      <<<grid, WG_THREADS, smem, stream>>>(a_tmap, b_tmap, c_ptr, aux_ptr, offsets, nseg, M, N, K,
                                           rows);
}

}  // namespace

#endif  // RL_KERNEL_ENABLE_SM90

bool nemotron_experts_compiled() {
#if defined(RL_KERNEL_ENABLE_SM90)
  return true;
#else
  return false;
#endif
}

namespace {

const nv_bf16* bf16_ptr(const torch::Tensor& tensor) {
  return reinterpret_cast<const nv_bf16*>(tensor.data_ptr<at::BFloat16>());
}
nv_bf16* bf16_mut(torch::Tensor& tensor) {
  return reinterpret_cast<nv_bf16*>(tensor.data_ptr<at::BFloat16>());
}

// cuTensorMapEncodeTiled requires a 16-byte aligned global base; a contiguous
// view can miss that even when its values are right.
torch::Tensor aligned16(const torch::Tensor& tensor) {
  if (reinterpret_cast<std::uintptr_t>(tensor.data_ptr()) % 16 == 0) return tensor;
  return tensor.clone();
}

void check_bf16(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16, name, " must be bfloat16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_integer(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.scalar_type() == torch::kInt64, name, " must be int64");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>
nemotron_experts_forward(torch::Tensor packed, c10::optional<torch::Tensor> x,
                         torch::Tensor weights, torch::Tensor permutation,
                         torch::Tensor offsets, torch::Tensor w_up, torch::Tensor w_down,
                         c10::optional<torch::Tensor> w_sh1,
                         c10::optional<torch::Tensor> w_sh2) {
#if !defined(RL_KERNEL_ENABLE_SM90)
  TORCH_CHECK(false, "nemotron experts require an SM90 build (KERNEL_ALIGN_DET_GEMM_SM90=1)");
#else
  check_bf16(packed, "packed");
  check_bf16(w_up, "w_up");
  check_bf16(w_down, "w_down");
  check_integer(permutation, "permutation");
  check_integer(offsets, "offsets");
  TORCH_CHECK(weights.is_cuda() && weights.is_contiguous() &&
                  weights.scalar_type() == torch::kFloat32,
              "weights must be contiguous float32");
  TORCH_CHECK(packed.dim() == 2 && packed.size(0) % 6 == 0, "packed must be [6*tokens, hidden]");
  TORCH_CHECK(w_up.dim() == 3 && w_down.dim() == 3, "stacked weights must be rank three");
  const int rows = packed.size(0);
  const int width = packed.size(1);
  const int intermediate = w_up.size(1);
  const int experts = w_up.size(0);
  const int tokens = rows / 6;
  // 64 is the TMA inner-box requirement.  A width that is not a multiple of BN
  // lets a B tile cross into the next expert's stacked rows, which only feeds
  // output columns the store masks drop, so the schedule may exceed the width.
  TORCH_CHECK(width % 64 == 0 && intermediate % 64 == 0, "hidden widths must be 64 aligned");
  TORCH_CHECK(width % 4 == 0, "hidden width must be vector aligned");
  TORCH_CHECK(w_up.size(2) == width && w_down.size(1) == width && w_down.size(2) == intermediate,
              "stacked weight geometry must match the payload");
  TORCH_CHECK(w_down.size(0) == experts, "up and down weights must cover the same experts");
  TORCH_CHECK(offsets.numel() == experts + 1, "offsets must hold experts+1 prefix sums");
  TORCH_CHECK(permutation.numel() == rows, "permutation must cover every packed row");
  TORCH_CHECK(weights.numel() == rows, "weights must hold one value per route");
  TORCH_CHECK(w_sh1.has_value() == w_sh2.has_value(), "shared expert weights come as a pair");

  const bool shared = w_sh1.has_value();
  torch::Tensor shared_x, shared_w1, shared_w2, shared_up, shared_down;
  int shared_width = 0;
  if (shared) {
    shared_w1 = aligned16(*w_sh1);
    shared_w2 = aligned16(*w_sh2);
    check_bf16(shared_w1, "w_sh1");
    check_bf16(shared_w2, "w_sh2");
    TORCH_CHECK(x.has_value(), "shared expert requires the token payload x");
    shared_x = aligned16(*x);
    check_bf16(shared_x, "x");
    TORCH_CHECK(shared_w1.dim() == 2 && shared_w2.dim() == 2, "shared weights must be rank two");
    TORCH_CHECK(shared_w1.size(1) == width && shared_w2.size(0) == width,
                "shared weight hidden widths must agree");
    TORCH_CHECK(shared_x.sizes() == torch::IntArrayRef({tokens, width}),
                "x must be [tokens, hidden]");
    shared_width = shared_w1.size(0);
    TORCH_CHECK(shared_w2.size(1) == shared_width, "shared intermediate widths must agree");
    TORCH_CHECK(shared_width % BN == 0, "shared intermediate width must be 64 aligned");
  }

  auto options = packed.options();
  auto float_options = options.dtype(torch::kFloat32);
  auto out = torch::empty({tokens, width}, options);
  auto up = torch::empty({rows, intermediate}, options);
  auto down = torch::empty({rows, width}, float_options);
  auto inverse = torch::empty({rows}, options.dtype(torch::kInt64));
  if (tokens == 0)
    return {out, up, down, inverse, c10::optional<torch::Tensor>(),
            c10::optional<torch::Tensor>()};

  auto packed_data_t = aligned16(packed);
  auto up_weight_t = aligned16(w_up);
  auto down_weight_t = aligned16(w_down);
  const nv_bf16* packed_data = bf16_ptr(packed_data_t);
  const nv_bf16* up_weight = bf16_ptr(up_weight_t);
  const nv_bf16* down_weight = bf16_ptr(down_weight_t);
  auto stream = at::cuda::getCurrentCUDAStream();

  nemotron_experts_invert_kernel<<<cdiv(rows, 256), 256, 0, stream>>>(
      inverse.data_ptr<int64_t>(), permutation.data_ptr<int64_t>(), rows);

  launch_grouped<nv_bf16, MODE_UP, false>(
      packed_data, rows, width, up_weight, (uint64_t)experts * intermediate, width, bf16_mut(up),
      nullptr, offsets.data_ptr<int64_t>(), experts, rows, intermediate, width, rows, stream);
  launch_grouped<float, MODE_DOWN, false>(
      bf16_ptr(up), rows, intermediate, down_weight, (uint64_t)experts * width, intermediate,
      down.data_ptr<float>(), nullptr, offsets.data_ptr<int64_t>(), experts, rows, width,
      intermediate, rows, stream);

  if (shared) {
    shared_up = torch::empty({tokens, shared_width}, options);
    shared_down = torch::empty({tokens, width}, float_options);
    launch_grouped<nv_bf16, MODE_UP, false>(
        bf16_ptr(shared_x), tokens, width, bf16_ptr(shared_w1), shared_width, width,
        bf16_mut(shared_up), nullptr, nullptr, 1, tokens, shared_width, width, tokens, stream);
    launch_grouped<float, MODE_DOWN, false>(
        bf16_ptr(shared_up), tokens, shared_width, bf16_ptr(shared_w2), width, shared_width,
        shared_down.data_ptr<float>(), nullptr, nullptr, 1, tokens, width, shared_width, tokens,
        stream);
  }

  // Combine, shared add and the single exit cast in one pass over the tokens.
  const int width4 = width / 4;
  nemotron_experts_combine_kernel<<<dim3(tokens, cdiv(width4, 256)), 256, 0, stream>>>(
      bf16_mut(out), down.data_ptr<float>(), inverse.data_ptr<int64_t>(),
      weights.data_ptr<float>(), shared ? shared_down.data_ptr<float>() : nullptr, width4);

  if (!shared)
    return {out, up, down, inverse, c10::optional<torch::Tensor>(),
            c10::optional<torch::Tensor>()};
  return {out, up, down, inverse, shared_up, shared_down};
#endif
}

std::vector<torch::Tensor> nemotron_experts_backward(
    torch::Tensor grad_out, torch::Tensor packed, c10::optional<torch::Tensor> x,
    torch::Tensor weights, torch::Tensor permutation, torch::Tensor offsets, torch::Tensor up,
    torch::Tensor down, torch::Tensor inverse, torch::Tensor w_up, torch::Tensor w_down,
    c10::optional<torch::Tensor> w_sh1, c10::optional<torch::Tensor> w_sh2,
    c10::optional<torch::Tensor> shared_up, c10::optional<torch::Tensor> shared_down,
    bool need_packed, bool need_x, bool need_weights, bool need_weight_grad,
    bool need_shared_weight_grad) {
#if !defined(RL_KERNEL_ENABLE_SM90)
  TORCH_CHECK(false, "nemotron experts require an SM90 build (KERNEL_ALIGN_DET_GEMM_SM90=1)");
#else
  check_bf16(grad_out, "grad_out");
  check_bf16(packed, "packed");
  check_bf16(up, "up");
  check_integer(permutation, "permutation");
  check_integer(offsets, "offsets");
  const int rows = packed.size(0);
  const int width = packed.size(1);
  const int intermediate = w_up.size(1);
  const int experts = w_up.size(0);
  const int tokens = rows / 6;
  const bool shared = w_sh1.has_value();
  TORCH_CHECK(grad_out.sizes() == torch::IntArrayRef({tokens, width}),
              "grad_out must be [tokens, hidden]");
  TORCH_CHECK(down.scalar_type() == torch::kFloat32 && down.is_contiguous() && down.is_cuda(),
              "saved down must be contiguous float32");

  auto options = packed.options();
  auto float_options = options.dtype(torch::kFloat32);
  torch::Tensor empty;
  torch::Tensor d_packed, d_x, d_weights, d_w_up, d_w_down, d_w_sh1, d_w_sh2;
  if (tokens == 0)
    return {empty, empty, torch::zeros({tokens, 6}, float_options), empty, empty, empty, empty};
  if (!need_packed && !need_x && !need_weights && !need_weight_grad &&
      !(shared && need_shared_weight_grad))
    return {empty, empty, empty, empty, empty, empty, empty};

  auto packed_data_t = aligned16(packed);
  auto up_weight_t = aligned16(w_up);
  auto down_weight_t = aligned16(w_down);
  const nv_bf16* packed_data = bf16_ptr(packed_data_t);
  const nv_bf16* up_weight = bf16_ptr(up_weight_t);
  const nv_bf16* down_weight = bf16_ptr(down_weight_t);
  const int64_t* route_offsets = offsets.data_ptr<int64_t>();
  auto stream = at::cuda::getCurrentCUDAStream();
  const int width8 = width / 8;

  auto dy = torch::empty({rows, width}, options);
  nemotron_experts_dy_kernel<<<dim3(rows, cdiv(width8, 256)), 256, 0, stream>>>(
      bf16_mut(dy), bf16_ptr(grad_out), permutation.data_ptr<int64_t>(),
      weights.data_ptr<float>(), width8);

  if (need_weights) {
    d_weights = torch::empty({tokens, 6}, float_options);
    const int leaf_count = cdiv(width, K_TREE_LEAF);
    auto leaves = torch::empty({rows, leaf_count}, float_options);
    const int64_t total = (int64_t)rows * leaf_count;
    nemotron_experts_dweights_leaf_kernel<<<cdiv((int)total, 256), 256, 0, stream>>>(
        leaves.data_ptr<float>(), down.data_ptr<float>(), bf16_ptr(grad_out),
        inverse.data_ptr<int64_t>(), rows, leaf_count, width);
    nemotron_experts_dweights_merge_kernel<<<cdiv(rows, 256), 256, 0, stream>>>(
        d_weights.data_ptr<float>(), leaves.data_ptr<float>(), leaf_count, rows);
  }

  // dU = dZ * 2*relu(U) is required by both dPacked and the up/down weight grads.
  torch::Tensor du;
  if (need_packed || need_weight_grad) {
    du = torch::empty({rows, intermediate}, options);
    launch_grouped<nv_bf16, MODE_DX, true>(
        bf16_ptr(dy), rows, width, down_weight, (uint64_t)experts * width, intermediate,
        bf16_mut(du), bf16_ptr(up), route_offsets, experts, rows, intermediate, width, rows,
        stream);
  }
  if (need_packed) {
    d_packed = torch::empty({rows, width}, options);
    launch_grouped<nv_bf16, MODE_DX, false>(
        bf16_ptr(du), rows, intermediate, up_weight, (uint64_t)experts * intermediate, width,
        bf16_mut(d_packed), nullptr, route_offsets, experts, rows, width, intermediate, rows,
        stream);
  }
  if (need_weight_grad) {
    d_w_up = torch::empty_like(w_up);
    d_w_down = torch::empty_like(w_down);
    // dW_up[e] = dU_e^T @ packed_e, dW_down[e] = dY_e^T @ relu(U_e).
    launch_grouped<nv_bf16, MODE_WGRAD, false>(
        bf16_ptr(du), rows, intermediate, packed_data, rows, width, bf16_mut(d_w_up), nullptr,
        route_offsets, experts, intermediate, width, 0, rows, stream);
    launch_grouped<nv_bf16, MODE_WGRAD_R2, false>(
        bf16_ptr(dy), rows, width, bf16_ptr(up), rows, intermediate, bf16_mut(d_w_down), nullptr,
        route_offsets, experts, width, intermediate, 0, rows, stream);
  }

  if (shared) {
    auto shared_w1 = aligned16(*w_sh1);
    auto shared_w2 = aligned16(*w_sh2);
    const int shared_width = shared_w1.size(0);
    const nv_bf16* shared_grad = bf16_ptr(grad_out);
    const nv_bf16* saved_shared_up = bf16_ptr(*shared_up);
    torch::Tensor d_shared_up;
    if (need_x || need_shared_weight_grad) {
      d_shared_up = torch::empty({tokens, shared_width}, options);
      launch_grouped<nv_bf16, MODE_DX, true>(
          shared_grad, tokens, width, bf16_ptr(shared_w2), width, shared_width,
          bf16_mut(d_shared_up), saved_shared_up, nullptr, 1, tokens, shared_width, width, tokens,
          stream);
    }
    if (need_x) {
      d_x = torch::empty({tokens, width}, options);
      launch_grouped<nv_bf16, MODE_DX, false>(
          bf16_ptr(d_shared_up), tokens, shared_width, bf16_ptr(shared_w1), shared_width, width,
          bf16_mut(d_x), nullptr, nullptr, 1, tokens, width, shared_width, tokens, stream);
    }
    if (need_shared_weight_grad) {
      auto shared_x_t = aligned16(*x);
      d_w_sh1 = torch::empty({1, shared_width, width}, options);
      d_w_sh2 = torch::empty({1, width, shared_width}, options);
      launch_grouped<nv_bf16, MODE_WGRAD, false>(
          bf16_ptr(d_shared_up), tokens, shared_width, bf16_ptr(shared_x_t), tokens, width,
          bf16_mut(d_w_sh1), nullptr, nullptr, 1, shared_width, width, 0, tokens, stream);
      launch_grouped<nv_bf16, MODE_WGRAD_R2, false>(
          shared_grad, tokens, width, saved_shared_up, tokens, shared_width,
          bf16_mut(d_w_sh2), nullptr, nullptr, 1, width, shared_width, 0, tokens, stream);
    }
  }

  return {d_packed, d_x, d_weights, d_w_up, d_w_down, d_w_sh1, d_w_sh2};
#endif
}
