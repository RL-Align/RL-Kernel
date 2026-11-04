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
//   * BK = K_TREE_LEAF = 32 (one FP32 leaf),
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
//   * four operand modes (MODE_*) cover grouped forward, grouped backward and
//     weight gradients; the operand that is stored transposed is read either
//     with `ldmatrix.trans` (v1) or as an MN-major wgmma operand (v2);
//   * the relu-square of the down projection is materialized by the up
//     projection's epilogue as `Z = bf16(relu(U)**2)`, applied to bf16 values
//     which is bitwise identical to the `__hmax2`/`__hmul2` fragment path and
//     removes a per-tile operand transform from the down and dW_down stages.
//
// Two instruction paths satisfy the same arithmetic contract.  Both are local
// to this file: the wgmma path needs det_gemm's descriptor, wgmma and swizzled
// TMA machinery, so it carries its own copy of those primitives rather than
// sharing them.
//   * v1 `nemotron_experts_gemm`: 128x64 tiles, `mma.sync` + `ldmatrix`, one
//     32-wide K tile per TMA stage, the shape fallback;
//   * v2 `nemotron_experts_gemm_v2`: 64x128 tiles, 128B TMA swizzle and
//     `wgmma.mma_async` reading both operands from shared-memory descriptors,
//     one producer warp plus a four-warp consumer warpgroup, bitwise identical
//     to v1.
// `RL_KERNEL_NEMOTRON_EXPERTS_BACKEND=sm90_v1` forces the fallback for A/B
// measurement; `auto` (the default) picks v2 whenever the shape admits it.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>

#if defined(RL_KERNEL_ENABLE_SM90)
#include "../gemm/det_gemm_tma.cuh"
#endif

namespace {

using nv_bf16 = __nv_bfloat16;

__host__ __device__ constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

// Must match det_gemm's SM90 BK so a leaf is the same 32 K values.
constexpr int K_TREE_LEAF = 32;
// Merge-tree stack slots. The binary-counter merge holds one slot per level, so
// `TREE_DEPTH` covers `kd <= 1 << TREE_DEPTH` leaves; 14 covers the operator's
// whole domain (a segment can hold up to 6 * MAX_TOKENS = 393216 packed rows,
// i.e. kd = 12288) and `run_grouped` raises instead of overflowing if a caller
// ever exceeds it. det_gemm's own kernel keeps 12 because it caps K at 32768.
constexpr int TREE_DEPTH = 14;

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

// Operand modes.  The A/B tile layouts are the only things that change; the
// leaf and tree arithmetic is shared by v1 and v2.
//
//   MODE      A operand                     B operand
//   UP        [rows,K]   K contiguous       [E*F, H]   K contiguous
//   DOWN      [rows,F]   K contiguous       [E*H, F]   K contiguous
//   DX        [rows,K]   K contiguous       [E*K, N]   N contiguous
//   WGRAD     [rows,M]   M contiguous       [rows,N]   N contiguous
//
// K contiguous is a K-major (Major::K) wgmma operand, read through
// `ldmatrix.trans` in v2's predecessors; M/N contiguous is Major::MN.
enum {
  MODE_UP = 0,    // grouped up projection, also the shared expert's
  MODE_DOWN = 1,  // grouped down projection, FP32 output for the combine
  MODE_DX = 2,    // dU = dY @ w_down^T, and dPacked = dU @ w_up^T
  MODE_WGRAD = 3,  // dW_up = dU^T @ packed, dW_down = dY^T @ Z
};

// Which CTA owns which (expert, row tile).  The host never reads `offsets`, so
// CUDA Graph capture stays legal.  Shared by both instruction paths.
struct Segment {
  int expert;
  int tile;
  int64_t lo;
  int64_t hi;
};

template <bool WGRAD, int BM_>
__device__ __forceinline__ Segment resolve_segment(const int64_t* offsets, int nseg, int M,
                                                   int rows) {
  Segment out{-1, 0, 0, 0};
  if (offsets == nullptr) {
    // Single segment [0, rows): the shared expert reuses this kernel verbatim.
    const int tiles = cdiv(WGRAD ? M : rows, BM_);
    if (blockIdx.y < (int64_t)tiles && rows > 0) {
      out.expert = 0;
      out.hi = rows;
      out.tile = (int)blockIdx.y * BM_;
    }
  } else if constexpr (WGRAD) {
    // Every expert owns cdiv(M,BM) output tiles, including empty segments:
    // those write zeros instead of requiring a memset of the whole tensor.
    const int tiles = cdiv(M, BM_);
    if (blockIdx.y < (int64_t)tiles * nseg) {
      out.expert = blockIdx.y / tiles;
      out.tile = (blockIdx.y % tiles) * BM_;
      out.lo = offsets[out.expert];
      out.hi = offsets[out.expert + 1];
    }
  } else {
    int remaining = blockIdx.y;
    for (int e = 0; e < nseg; ++e) {
      const int64_t begin = offsets[e], end = offsets[e + 1];
      const int tiles = (int)((end - begin + BM_ - 1) / BM_);
      if (remaining < tiles) {
        out.expert = e;
        out.tile = remaining * BM_;
        out.lo = begin;
        out.hi = end;
        break;
      }
      remaining -= tiles;
    }
  }
  return out;
}

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

// bf16(relu(x)**2) on a lane pair.  A bf16 product is exact in FP32, so this
// equals rounding the FP32 square once, which is why the up projection can
// materialize the down projection's input without changing a bit.
__device__ __forceinline__ __nv_bfloat162 relu_square(__nv_bfloat162 value) {
  const __nv_bfloat162 zero = __floats2bfloat162_rn(0.0f, 0.0f);
  const __nv_bfloat162 clamped = __hmax2(value, zero);
  return __hmul2(clamped, clamped);
}

// One output column pair.  Every mode writes BF16 pairs; MODE_UP additionally
// materializes Z = bf16(relu(U)**2), the down projection's operand, in two packed
// instructions that the down stage used to spend on every one of its operand
// tiles.  `index` is the flat position of the pair's first element.
template <typename output_t, int MODE, bool EPI_RELU2_GRAD>
__device__ __forceinline__ void store_pair(output_t* __restrict__ C, nv_bf16* __restrict__ z,
                                           const nv_bf16* __restrict__ aux, int64_t index,
                                           float v0, float v1) {
  if constexpr (EPI_RELU2_GRAD) {
    // relu-square derivative: dU = dZ * 2 * relu(U), one rounding.
    const __nv_bfloat162 saved = *reinterpret_cast<const __nv_bfloat162*>(aux + index);
    const float lo = __low2float(saved), hi = __high2float(saved);
    v0 = v0 * (2.0f * (lo > 0.0f ? lo : 0.0f));
    v1 = v1 * (2.0f * (hi > 0.0f ? hi : 0.0f));
  }
  if constexpr (std::is_same<output_t, nv_bf16>::value) {
    const __nv_bfloat162 pair = __floats2bfloat162_rn(v0, v1);
    *reinterpret_cast<__nv_bfloat162*>(C + index) = pair;
    if constexpr (MODE == MODE_UP)
      *reinterpret_cast<__nv_bfloat162*>(z + index) = relu_square(pair);
  } else {
    C[index + 0] = cast_output<output_t>(v0);
    C[index + 1] = cast_output<output_t>(v1);
  }
}

template <typename output_t, int MODE, bool EPI_RELU2_GRAD>
__global__ void nemotron_experts_gemm(const __grid_constant__ CUtensorMap a_tmap,
                                      const __grid_constant__ CUtensorMap b_tmap,
                                      output_t* __restrict__ C,
                                      nv_bf16* __restrict__ z,
                                      const nv_bf16* __restrict__ aux,
                                      const int64_t* __restrict__ offsets,
                                      int nseg, int M, int N, int K, int rows) {
  constexpr bool WGRAD = MODE == MODE_WGRAD;
  constexpr bool TRANSPOSE_A = WGRAD;
  constexpr bool TRANSPOSE_B = MODE == MODE_DX || WGRAD;

  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int col_base = blockIdx.x * BN;

  __shared__ Segment s_seg;
  if (tid == 0) s_seg = resolve_segment<WGRAD, BM>(offsets, nseg, M, rows);
  __syncthreads();
  if (s_seg.expert < 0) return;  // Padding CTA above the real tile count.

  const int expert = s_seg.expert;
  const int64_t seg_lo = s_seg.lo, seg_hi = s_seg.hi;
  // Row start of the A operand (grouped modes) or of the output tile (dW).
  const int64_t row_base = WGRAD ? (int64_t)s_seg.tile : seg_lo + s_seg.tile;
  const int out_row_base = WGRAD ? s_seg.tile : (int)row_base;
  const int64_t out_offset = WGRAD ? (int64_t)expert * M : 0;
  const int tile_rows = M - s_seg.tile;
  const int64_t segment_rows = seg_hi - row_base;
  const int local_rows = WGRAD ? (tile_rows < BM ? tile_rows : BM)
                               : (int)(segment_rows < (int64_t)BM ? segment_rows : (int64_t)BM);
  const int s_tile = s_seg.tile;
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

  // One column pair per (row, two adjacent columns).
  auto emit = [&](int row_local, int col, float v0, float v1) {
    const int64_t index = (out_offset + out_row_base + row_local) * (int64_t)N + col;
    store_pair<output_t, MODE, EPI_RELU2_GRAD>(C, z, aux, index, v0, v1);
  };

#pragma unroll
  for (int mi = 0; mi < M_TILES; ++mi) {
    const int row = warp * WARP_M + mi * MMA_M + lane / 4;
#pragma unroll
    for (int n = 0; n < N_TILES; ++n) {
      const int col = col_base + n * MMA_N + (lane % 4) * 2;
      if (col + 1 < N) {
        if (row < local_rows)
          emit(row, col, __low2float(tree_v[mi][n][0]), __high2float(tree_v[mi][n][0]));
        if (row + 8 < local_rows)
          emit(row + 8, col, __low2float(tree_v[mi][n][1]), __high2float(tree_v[mi][n][1]));
      }
    }
  }
}

template <typename output_t, int MODE, bool EPI_RELU2_GRAD>
void launch_grouped_v1(const nv_bf16* a_ptr, uint64_t a_height, uint64_t a_width,
                       const nv_bf16* b_ptr, uint64_t b_height, uint64_t b_width,
                       output_t* c_ptr, nv_bf16* z_ptr, const nv_bf16* aux_ptr,
                       const int64_t* offsets, int nseg, int M, int N, int K, int rows,
                       cudaStream_t stream) {
  CUtensorMap a_tmap, b_tmap;
  if constexpr (MODE == MODE_WGRAD) {
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
  const int64_t grid_y = MODE == MODE_WGRAD
                             ? (int64_t)cdiv(M, BM) * nseg
                             : (additive < rows ? additive : (int64_t)rows);
  if (grid_y <= 0) return;
  const int smem = STAGES * (BM * BK + BN * BK) * sizeof(nv_bf16) + STAGES * 8;
  if (smem > 48 * 1024)
    cudaFuncSetAttribute(nemotron_experts_gemm<output_t, MODE, EPI_RELU2_GRAD>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  dim3 grid(cdiv(N, BN), (unsigned)grid_y);
  nemotron_experts_gemm<output_t, MODE, EPI_RELU2_GRAD><<<grid, WG_THREADS, smem, stream>>>(
      a_tmap, b_tmap, c_ptr, z_ptr, aux_ptr, offsets, nseg, M, N, K, rows);
}

// ---------------------------------------------------------------------------
// Version 2: the wgmma instruction path, applied per expert segment.
//
// The state machine is a warp-specialised wgmma pipeline: TMA with a 128-byte
// swizzle, one producer warp, a four-warp consumer warpgroup and three stages,
// so the K loop holds no __syncthreads.  It is self-contained - the descriptor
// encoding, the wgmma wrappers, the swizzled tensor map and the extra barrier
// primitives below are this file's own.  Grouped additions:
//   * the CTA resolves its own (expert, row tile) exactly as v1 does;
//   * either operand may be K-major or MN-major (`Major::K` / `Major::MN`), which
//     selects the descriptor start offsets and the instruction's imm-trans flag;
//   * a weight gradient's last reduction tile is zero-padded in shared memory,
//     because a segment length is not a multiple of the 32-wide leaf.
// ---------------------------------------------------------------------------

constexpr int V2_BM = 64, V2_BN = 128;
constexpr int V2_KT = 64;  // one 128-byte inner box == two leaves
constexpr int V2_LEAVES = V2_KT / K_TREE_LEAF;
constexpr int V2_STAGES = 3;
constexpr int V2_WARPS = 4;                      // consumer warpgroup
constexpr int V2_THREADS = (V2_WARPS + 1) * 32;  // plus the producer warp
constexpr int V2_GROUPS = V2_BN / 8;             // wgmma column groups of eight
constexpr int V2_ACC = 4 * V2_GROUPS;            // FP32 accumulators per thread
constexpr int V2_A_ELEMS = V2_BM * V2_KT;
constexpr int V2_B_ELEMS = V2_BN * V2_KT;
// MN-major blocks are 64 wide, so B's second block sits one 8 KB block further on.
constexpr int V2_BLOCK_ELEMS = 64 * V2_KT;
constexpr uint32_t V2_TILE_BYTES = (uint32_t)(V2_A_ELEMS + V2_B_ELEMS) * sizeof(nv_bf16);
constexpr uint32_t V2_MN_LEADING = V2_BLOCK_ELEMS * sizeof(nv_bf16) / 16;

__device__ __forceinline__ uint32_t v2_smem_addr(const void* pointer) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

// wgmma matrix descriptor: 14-bit start address in 16-byte units, leading and
// stride byte offsets, 128-byte swizzle in the top field.  For both operand
// majors the core matrices are eight 128-byte rows, so the stride offset is
// 1024 bytes; only the leading offset differs - it is unused for a K-major
// operand and the 8 KB MN-block step for a two-block MN-major B.
__device__ __forceinline__ uint64_t v2_desc(uint32_t smem_base, uint32_t byte_offset,
                                            uint32_t leading) {
  constexpr uint32_t kStride = 8 * 128 / 16;
  const uint32_t address = smem_base + byte_offset;
  return (static_cast<uint64_t>(address >> 4) & 0x3FFFu) |
         (static_cast<uint64_t>(leading) << 16) | (static_cast<uint64_t>(kStride) << 32) |
         (uint64_t{1} << 62);
}

#define NE_V2_ACC_OPERANDS                                                     \
  "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15"                     \
  ",%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31"           \
  ",%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47"           \
  ",%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}"

#define NE_V2_ACC_CONSTRAINTS                                                     \
  "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),          \
  "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),        \
  "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]),    \
  "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),    \
  "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]),    \
  "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]),    \
  "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]),    \
  "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),    \
  "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]),    \
  "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]),    \
  "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])

// kAccumulate = false passes scale-d = 0, which overwrites the accumulator
// instead of accumulating into it, so the first k16 of a leaf needs no clear.
#define NE_V2_WGMMA(NAME, TRANS_A, TRANS_B)                                            \
  template <bool kAccumulate>                                                          \
  __device__ __forceinline__ void NAME(float (&d)[V2_ACC], uint64_t a, uint64_t b) {    \
    const uint32_t accumulate = kAccumulate ? 1u : 0u;                                 \
    asm volatile(                                                                      \
        "{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"                                    \
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 " NE_V2_ACC_OPERANDS     \
        ", %64, %65, p, 1, 1, " #TRANS_A ", " #TRANS_B ";\n}\n"                         \
        : NE_V2_ACC_CONSTRAINTS                                                        \
        : "l"(a), "l"(b), "r"(accumulate));                                             \
  }

NE_V2_WGMMA(wgmma_kk, 0, 0)    // A K-major, B K-major
NE_V2_WGMMA(wgmma_kmn, 0, 1)   // A K-major, B MN-major
NE_V2_WGMMA(wgmma_mnmn, 1, 1)  // both operands MN-major

__device__ __forceinline__ void wgmma_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;");
}
__device__ __forceinline__ void wgmma_wait_all() {
  asm volatile("wgmma.wait_group.sync.aligned 0;");
}
// Leaves at most one accumulator group in flight, so the previous leaf can be
// rounded and merged while the current one is still on the tensor pipe.
__device__ __forceinline__ void wgmma_wait_one() {
  asm volatile("wgmma.wait_group.sync.aligned 1;");
}
// Generic-to-async proxy fence: the shared-memory writes below must be visible
// to wgmma, which reads through the async proxy.
__device__ __forceinline__ void v2_fence_view_async_shared() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
// Plain barrier arrival: a consumer warp releasing its stage.
__device__ __forceinline__ void v2_mbar_arrive(uint32_t addr) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(addr) : "memory");
}
// Named barrier over the consumer warpgroup only (the producer warp never joins).
__device__ __forceinline__ void v2_consumer_barrier() {
  asm volatile("barrier.sync 1, 128;" ::: "memory");
}

// One leaf of two k16 steps; `scale` picks whether the first step overwrites.
template <bool A_MN, bool B_MN, bool kAccumulate>
__device__ __forceinline__ void wgmma_step(float (&d)[V2_ACC], uint32_t a_stage,
                                           uint32_t b_stage, uint32_t a_off, uint32_t b_off) {
  const uint64_t a = v2_desc(a_stage, a_off, 1);
  const uint64_t b = v2_desc(b_stage, b_off, B_MN ? V2_MN_LEADING : 1);
  if constexpr (!A_MN && !B_MN)
    wgmma_kk<kAccumulate>(d, a, b);
  else if constexpr (!A_MN && B_MN)
    wgmma_kmn<kAccumulate>(d, a, b);
  else
    wgmma_mnmn<kAccumulate>(d, a, b);
}

// Zeroes the K rows of one shared-memory operand block that lie past the end of
// the expert's segment, so they cannot join the weight-gradient reduction.
__device__ __forceinline__ void v2_zero_rows(nv_bf16* tile, int first_row, int row_width,
                                             int lane_offset, int stride) {
  const int total = (V2_KT - first_row) * row_width;
  for (int index = lane_offset; index < total; index += stride)
    tile[(int64_t)(first_row + index / row_width) * row_width + index % row_width] =
        __float2bfloat16(0.0f);
}

template <typename output_t, int MODE, bool EPI_RELU2_GRAD>
__global__ void nemotron_experts_gemm_v2(const __grid_constant__ CUtensorMap a_tmap,
                                         const __grid_constant__ CUtensorMap b_tmap,
                                         output_t* __restrict__ C, nv_bf16* __restrict__ z,
                                         const nv_bf16* __restrict__ aux,
                                         const int64_t* __restrict__ offsets,
                                         int nseg, int M, int N, int K, int rows) {
  constexpr bool WGRAD = MODE == MODE_WGRAD;
  constexpr bool A_MN = WGRAD;                        // A read in (k, m) tiles
  constexpr bool B_MN = MODE == MODE_DX || WGRAD;      // B read in (k, n) tiles
  constexpr uint32_t kStepA = A_MN ? 128u : 2u;        // bytes per K position
  constexpr uint32_t kStepB = B_MN ? 128u : 2u;

  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int col_base = blockIdx.x * V2_BN;

  __shared__ Segment s_seg;
  if (tid == 0) s_seg = resolve_segment<WGRAD, V2_BM>(offsets, nseg, M, rows);
  __syncthreads();
  if (s_seg.expert < 0) return;  // Padding CTA above the real tile count.

  const int expert = s_seg.expert;
  const int64_t seg_lo = s_seg.lo, seg_hi = s_seg.hi;
  const int64_t row_base = WGRAD ? (int64_t)s_seg.tile : seg_lo + s_seg.tile;
  const int out_row_base = WGRAD ? s_seg.tile : (int)row_base;
  const int64_t out_offset = WGRAD ? (int64_t)expert * M : 0;
  const int tile_rows = M - s_seg.tile;
  const int64_t segment_rows = seg_hi - row_base;
  const int local_rows = WGRAD ? (tile_rows < V2_BM ? tile_rows : V2_BM)
                               : (int)(segment_rows < (int64_t)V2_BM ? segment_rows
                                                                     : (int64_t)V2_BM);
  const int64_t seg_len = seg_hi - seg_lo;
  const int kd = WGRAD ? (int)((seg_len + K_TREE_LEAF - 1) / K_TREE_LEAF) : (K / K_TREE_LEAF);
  // Segment rows the last leaf must not see.
  const int k_pad = WGRAD ? (int)((int64_t)kd * K_TREE_LEAF - seg_len) : 0;
  const int chunks = cdiv(kd, V2_LEAVES);
  // B's row coordinate: the expert's stacked [K, N] slice, or the segment rows.
  const int64_t b_row_base = MODE == MODE_DX ? (int64_t)expert * K : seg_lo;

  extern __shared__ __align__(1024) char nemotron_smem[];
  nv_bf16* sA = reinterpret_cast<nv_bf16*>(nemotron_smem);
  nv_bf16* sB = sA + V2_STAGES * V2_A_ELEMS;
  int64_t* barriers = reinterpret_cast<int64_t*>(sB + V2_STAGES * V2_B_ELEMS);
  uint32_t full[V2_STAGES], empty[V2_STAGES];
#pragma unroll
  for (int s = 0; s < V2_STAGES; ++s) {
    full[s] = v2_smem_addr(barriers + 2 * s);
    empty[s] = v2_smem_addr(barriers + 2 * s + 1);
  }
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < V2_STAGES; ++s) {
      det_gemm::mbar_init(full[s], 1);
      det_gemm::mbar_init(empty[s], V2_WARPS);
    }
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  __syncthreads();

  if (warp == V2_WARPS) {  // ---- producer warp ----
    if (lane == 0) {
      for (int chunk = 0; chunk < chunks; ++chunk) {
        const int buf = chunk % V2_STAGES;
        if (chunk >= V2_STAGES) det_gemm::mbar_wait(empty[buf], ((chunk / V2_STAGES) + 1) & 1);
        const uint32_t a_dst = v2_smem_addr(sA) + buf * V2_A_ELEMS * sizeof(nv_bf16);
        const uint32_t b_dst = v2_smem_addr(sB) + buf * V2_B_ELEMS * sizeof(nv_bf16);
        const int koff = chunk * V2_KT;
        if constexpr (A_MN) {
          det_gemm::tma_2d_g2s(a_dst, &a_tmap, s_seg.tile, (int)(seg_lo + koff), full[buf]);
        } else {
          det_gemm::tma_2d_g2s(a_dst, &a_tmap, koff, (int)row_base, full[buf]);
        }
        if constexpr (B_MN) {
          det_gemm::tma_2d_g2s(b_dst, &b_tmap, col_base, (int)(b_row_base + koff), full[buf]);
          det_gemm::tma_2d_g2s(b_dst + V2_BLOCK_ELEMS * sizeof(nv_bf16), &b_tmap, col_base + 64,
                               (int)(b_row_base + koff), full[buf]);
        } else {
          det_gemm::tma_2d_g2s(b_dst, &b_tmap, koff, expert * N + col_base, full[buf]);
        }
        det_gemm::mbar_arrive_expect_tx(full[buf], V2_TILE_BYTES);
      }
    }
    return;
  }

  // ---- consumer warpgroup ----
  float acc_a[V2_ACC];
  float acc_b[V2_ACC];
  __nv_bfloat162 tree_v[V2_ACC / 2];
  __nv_bfloat162 tree_stk[TREE_DEPTH][V2_ACC / 2];
  int sp = 0;
  const __nv_bfloat162 tree_zero = __floats2bfloat162_rn(0.0f, 0.0f);
  // An empty expert has kd == 0, so the loop below never assigns tree_v; the
  // weight-gradient grid deliberately covers every expert, so such a tile is
  // stored as exact zeros instead of uninitialized registers.
#pragma unroll
  for (int i = 0; i < V2_ACC / 2; ++i) tree_v[i] = tree_zero;

  for (int chunk = 0; chunk < chunks; ++chunk) {
    const int buf = chunk % V2_STAGES;
    const int leaf_base = chunk * V2_LEAVES;
    const int leaves_here = (chunk + 1 == chunks) ? (kd - leaf_base) : V2_LEAVES;
    det_gemm::mbar_wait(full[buf], (chunk / V2_STAGES) & 1);  // every consumer thread waits

    const uint32_t a_stage = v2_smem_addr(sA) + buf * V2_A_ELEMS * sizeof(nv_bf16);
    const uint32_t b_stage = v2_smem_addr(sB) + buf * V2_B_ELEMS * sizeof(nv_bf16);

    if constexpr (WGRAD) {
      if (k_pad > 0 && chunk + 1 == chunks) {
        // The last chunk's K rows past the segment end hold the next expert's
        // rows; zero them so they cannot join the reduction.
        const int first = (int)(seg_len - (int64_t)chunk * V2_KT);
        nv_bf16* b_tile = sB + buf * V2_B_ELEMS;
        v2_zero_rows(sA + buf * V2_A_ELEMS, first, V2_BM, tid, V2_WARPS * 32);
        v2_zero_rows(b_tile, first, 64, tid, V2_WARPS * 32);
        v2_zero_rows(b_tile + V2_BLOCK_ELEMS, first, 64, tid, V2_WARPS * 32);
        v2_fence_view_async_shared();
        v2_consumer_barrier();
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
      }
    }

    // Two static accumulators, used as a two-leaf group: while leaf 0 is rounded
    // and merged the tensor pipe already executes leaf 1, so no drain sits in
    // the K loop.  A runtime-indexed array would push both into local memory.
    auto issue_leaf = [&](float (&dst)[V2_ACC], int leaf) {
      const uint32_t a_off = leaf * K_TREE_LEAF * kStepA;
      const uint32_t b_off = leaf * K_TREE_LEAF * kStepB;
      // Two k16 steps make one 32-wide FP32 leaf, exactly like two mma.sync in v1.
      wgmma_step<A_MN, B_MN, false>(dst, a_stage, b_stage, a_off, b_off);
      wgmma_step<A_MN, B_MN, true>(dst, a_stage, b_stage, a_off + 16 * kStepA,
                                   b_off + 16 * kStepB);
      wgmma_commit();
    };

    auto finish_leaf = [&](float (&src)[V2_ACC], int index) {
      // Leaf value: one BF16 rounding per output element, as v1.
#pragma unroll
      for (int i = 0; i < V2_ACC / 2; ++i)
        tree_v[i] = __floats2bfloat162_rn(src[2 * i + 0], src[2 * i + 1]);
      const int merges = mid_tree_merge_count(index, kd);
      for (int merge = 0; merge < merges; ++merge) {
#pragma unroll
        for (int i = 0; i < V2_ACC / 2; ++i) tree_v[i] = __hadd2(tree_stk[sp - 1][i], tree_v[i]);
        --sp;
      }
      if (index + 1 < kd) {
#pragma unroll
        for (int i = 0; i < V2_ACC / 2; ++i) tree_stk[sp][i] = tree_v[i];
        ++sp;
      }
    };

    if (leaves_here == V2_LEAVES) {
      issue_leaf(acc_a, 0);
      issue_leaf(acc_b, 1);
      wgmma_wait_one();  // leaf 0 done, leaf 1 still on the tensor pipe
      finish_leaf(acc_a, leaf_base + 0);
      wgmma_wait_all();
      finish_leaf(acc_b, leaf_base + 1);
    } else {
      issue_leaf(acc_a, 0);
      wgmma_wait_all();
      finish_leaf(acc_a, leaf_base + 0);
    }
    if (lane == 0) v2_mbar_arrive(empty[buf]);  // this warp is done with the stage
  }

  // ---- epilogue: the wgmma accumulator matches v1's tree_v layout ----
  const int row0 = warp * 16 + lane / 4;
  auto emit = [&](int row_local, int col, float v0, float v1) {
    const int64_t index = (out_offset + out_row_base + row_local) * (int64_t)N + col;
    store_pair<output_t, MODE, EPI_RELU2_GRAD>(C, z, aux, index, v0, v1);
  };
#pragma unroll
  for (int g = 0; g < V2_GROUPS; ++g) {
    const int col = col_base + g * 8 + 2 * (lane % 4);
    if (col + 1 < N) {
      if (row0 < local_rows)
        emit(row0, col, __low2float(tree_v[2 * g + 0]), __high2float(tree_v[2 * g + 0]));
      if (row0 + 8 < local_rows)
        emit(row0 + 8, col, __low2float(tree_v[2 * g + 1]), __high2float(tree_v[2 * g + 1]));
    }
  }
}

// wgmma reads its operands through a 128-byte swizzle, so the v2 tensor maps
// differ from the v1 ones only in that field.  The inner box is one swizzle row,
// i.e. at most 64 BF16 elements.
void nemotron_init_tmap_swizzle128(CUtensorMap* tmap, const nv_bf16* gmem, uint64_t height,
                                   uint64_t width, uint32_t box_h, uint32_t box_w) {
  TORCH_CHECK(box_w * sizeof(nv_bf16) <= 128,
              "nemotron experts: swizzle128 inner box must fit 128 bytes");
  uint64_t size[2] = {width, height};
  uint64_t stride[1] = {width * sizeof(nv_bf16)};
  uint32_t box[2] = {box_w, box_h};
  uint32_t estride[2] = {1, 1};
  const CUresult res = cuTensorMapEncodeTiled(
      tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<nv_bf16*>(gmem), size, stride, box,
      estride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(res == CUDA_SUCCESS, "nemotron experts: cuTensorMapEncodeTiled (swizzle128) failed");
}

template <typename output_t, int MODE, bool EPI_RELU2_GRAD>
void launch_grouped_v2(const nv_bf16* a_ptr, uint64_t a_height, uint64_t a_width,
                       const nv_bf16* b_ptr, uint64_t b_height, uint64_t b_width,
                       output_t* c_ptr, nv_bf16* z_ptr, const nv_bf16* aux_ptr,
                       const int64_t* offsets, int nseg, int M, int N, int K, int rows,
                       cudaStream_t stream) {
  constexpr bool WGRAD = MODE == MODE_WGRAD;
  CUtensorMap a_tmap, b_tmap;
  if constexpr (WGRAD) {
    // A is the transposed payload: [rows, M] addressed in (k, m) tiles.
    nemotron_init_tmap_swizzle128(&a_tmap, a_ptr, a_height, a_width, V2_KT, V2_BM);
    nemotron_init_tmap_swizzle128(&b_tmap, b_ptr, b_height, b_width, V2_KT, 64);
  } else {
    nemotron_init_tmap_swizzle128(&a_tmap, a_ptr, a_height, a_width, V2_BM, V2_KT);
    if constexpr (MODE == MODE_DX)
      nemotron_init_tmap_swizzle128(&b_tmap, b_ptr, b_height, b_width, V2_KT, 64);
    else
      nemotron_init_tmap_swizzle128(&b_tmap, b_ptr, b_height, b_width, V2_BN, V2_KT);
  }

  const int64_t additive = (int64_t)cdiv(rows, V2_BM) + nseg;
  const int64_t grid_y = WGRAD ? (int64_t)cdiv(M, V2_BM) * nseg
                               : (additive < rows ? additive : (int64_t)rows);
  if (grid_y <= 0) return;
  const int smem =
      V2_STAGES * (V2_A_ELEMS + V2_B_ELEMS) * sizeof(nv_bf16) + V2_STAGES * 16;
  static bool attributed = false;
  if (!attributed) {
    cudaFuncSetAttribute(nemotron_experts_gemm_v2<output_t, MODE, EPI_RELU2_GRAD>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    attributed = true;
  }
  dim3 grid(cdiv(N, V2_BN), (unsigned)grid_y);
  nemotron_experts_gemm_v2<output_t, MODE, EPI_RELU2_GRAD><<<grid, V2_THREADS, smem, stream>>>(
      a_tmap, b_tmap, c_ptr, z_ptr, aux_ptr, offsets, nseg, M, N, K, rows);
}

// `RL_KERNEL_NEMOTRON_EXPERTS_BACKEND`: `auto` (default) runs the wgmma path,
// `sm90_v1` pins the mma.sync path so the two can be measured against each
// other.  Both satisfy the same contract and are bitwise identical.  The
// variable is read per launch rather than cached, so a test or benchmark can
// compare the two paths inside one process.
bool experts_prefer_v2() {
  const char* value = std::getenv("RL_KERNEL_NEMOTRON_EXPERTS_BACKEND");
  if (value == nullptr) return true;
  const std::string text(value);
  if (text.empty() || text == "auto" || text == "sm90_v2" || text == "v2") return true;
  TORCH_CHECK(text == "sm90_v1" || text == "v1",
              "RL_KERNEL_NEMOTRON_EXPERTS_BACKEND must be auto, sm90_v1 or sm90_v2");
  return false;
}

template <typename output_t, int MODE, bool EPI_RELU2_GRAD>
void run_grouped(const nv_bf16* a_ptr, uint64_t a_height, uint64_t a_width,
                 const nv_bf16* b_ptr, uint64_t b_height, uint64_t b_width, output_t* c_ptr,
                 nv_bf16* z_ptr, const nv_bf16* aux_ptr, const int64_t* offsets, int nseg, int M,
                 int N, int K, int rows, cudaStream_t stream) {
  // A weight gradient reconciles one leaf per 32 rows of an expert segment, and
  // the merge stack holds one slot per level: fail closed rather than write past
  // `tree_stk` when a segment is longer than the tree can cover. The check uses
  // the row count, not `offsets`, so the host still never reads the routing table.
  TORCH_CHECK(cdiv(rows, K_TREE_LEAF) <= (1 << TREE_DEPTH),
              "nemotron experts: reduction depth exceeds the merge tree (",
              cdiv(rows, K_TREE_LEAF), " > ", (1 << TREE_DEPTH), " leaves)");
  // v2 handles every geometry the operator accepts (its weight-gradient K is
  // padded); the width check below only guards a hand-built call.
  const bool v2_shape = N % 64 == 0 && M > 0 && (MODE == MODE_WGRAD || K % K_TREE_LEAF == 0);
  if (experts_prefer_v2() && v2_shape) {
    launch_grouped_v2<output_t, MODE, EPI_RELU2_GRAD>(a_ptr, a_height, a_width, b_ptr, b_height,
                                                      b_width, c_ptr, z_ptr, aux_ptr, offsets, nseg,
                                                      M, N, K, rows, stream);
    return;
  }
  launch_grouped_v1<output_t, MODE, EPI_RELU2_GRAD>(a_ptr, a_height, a_width, b_ptr, b_height,
                                                    b_width, c_ptr, z_ptr, aux_ptr, offsets, nseg,
                                                    M, N, K, rows, stream);
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

// Which instruction path the next launch takes: the env-selected preference,
// before the per-shape gate. Provenance for route reporting and for the test
// that compares the two paths; "none" means the operator is not compiled.
const char* nemotron_experts_preferred_path() {
#if !defined(RL_KERNEL_ENABLE_SM90)
  return "none";
#else
  return experts_prefer_v2() ? "sm90_v2" : "sm90_v1";
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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           c10::optional<torch::Tensor>, c10::optional<torch::Tensor>,
           c10::optional<torch::Tensor>>
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
  torch::Tensor shared_x, shared_w1, shared_w2, shared_up, shared_z, shared_down;
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
  auto z = torch::empty({rows, intermediate}, options);
  auto down = torch::empty({rows, width}, float_options);
  auto inverse = torch::empty({rows}, options.dtype(torch::kInt64));
  const c10::optional<torch::Tensor> none;
  if (tokens == 0) return {out, up, z, down, inverse, none, none, none};

  auto packed_data_t = aligned16(packed);
  auto up_weight_t = aligned16(w_up);
  auto down_weight_t = aligned16(w_down);
  const nv_bf16* packed_data = bf16_ptr(packed_data_t);
  const nv_bf16* up_weight = bf16_ptr(up_weight_t);
  const nv_bf16* down_weight = bf16_ptr(down_weight_t);
  auto stream = at::cuda::getCurrentCUDAStream();

  nemotron_experts_invert_kernel<<<cdiv(rows, 256), 256, 0, stream>>>(
      inverse.data_ptr<int64_t>(), permutation.data_ptr<int64_t>(), rows);

  run_grouped<nv_bf16, MODE_UP, false>(
      packed_data, rows, width, up_weight, (uint64_t)experts * intermediate, width, bf16_mut(up),
      bf16_mut(z), nullptr, offsets.data_ptr<int64_t>(), experts, rows, intermediate, width, rows,
      stream);
  run_grouped<float, MODE_DOWN, false>(
      bf16_ptr(z), rows, intermediate, down_weight, (uint64_t)experts * width, intermediate,
      down.data_ptr<float>(), nullptr, nullptr, offsets.data_ptr<int64_t>(), experts, rows, width,
      intermediate, rows, stream);

  if (shared) {
    shared_up = torch::empty({tokens, shared_width}, options);
    shared_z = torch::empty({tokens, shared_width}, options);
    shared_down = torch::empty({tokens, width}, float_options);
    run_grouped<nv_bf16, MODE_UP, false>(
        bf16_ptr(shared_x), tokens, width, bf16_ptr(shared_w1), shared_width, width,
        bf16_mut(shared_up), bf16_mut(shared_z), nullptr, nullptr, 1, tokens, shared_width, width,
        tokens, stream);
    run_grouped<float, MODE_DOWN, false>(
        bf16_ptr(shared_z), tokens, shared_width, bf16_ptr(shared_w2), width, shared_width,
        shared_down.data_ptr<float>(), nullptr, nullptr, nullptr, 1, tokens, width, shared_width,
        tokens, stream);
  }

  // Combine, shared add and the single exit cast in one pass over the tokens.
  const int width4 = width / 4;
  nemotron_experts_combine_kernel<<<dim3(tokens, cdiv(width4, 256)), 256, 0, stream>>>(
      bf16_mut(out), down.data_ptr<float>(), inverse.data_ptr<int64_t>(),
      weights.data_ptr<float>(), shared ? shared_down.data_ptr<float>() : nullptr, width4);

  if (!shared) return {out, up, z, down, inverse, none, none, none};
  return {out, up, z, down, inverse, shared_up, shared_z, shared_down};
#endif
}

std::vector<torch::Tensor> nemotron_experts_backward(
    torch::Tensor grad_out, torch::Tensor packed, c10::optional<torch::Tensor> x,
    torch::Tensor weights, torch::Tensor permutation, torch::Tensor offsets, torch::Tensor up,
    torch::Tensor z, torch::Tensor down, torch::Tensor inverse, torch::Tensor w_up,
    torch::Tensor w_down, c10::optional<torch::Tensor> w_sh1, c10::optional<torch::Tensor> w_sh2,
    c10::optional<torch::Tensor> shared_up, c10::optional<torch::Tensor> shared_z,
    c10::optional<torch::Tensor> shared_down, bool need_packed, bool need_x, bool need_weights,
    bool need_weight_grad, bool need_shared_weight_grad) {
#if !defined(RL_KERNEL_ENABLE_SM90)
  TORCH_CHECK(false, "nemotron experts require an SM90 build (KERNEL_ALIGN_DET_GEMM_SM90=1)");
#else
  check_bf16(grad_out, "grad_out");
  check_bf16(packed, "packed");
  check_bf16(up, "up");
  check_bf16(z, "z");
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
    run_grouped<nv_bf16, MODE_DX, true>(
        bf16_ptr(dy), rows, width, down_weight, (uint64_t)experts * width, intermediate,
        bf16_mut(du), nullptr, bf16_ptr(up), route_offsets, experts, rows, intermediate, width,
        rows, stream);
  }
  if (need_packed) {
    d_packed = torch::empty({rows, width}, options);
    run_grouped<nv_bf16, MODE_DX, false>(
        bf16_ptr(du), rows, intermediate, up_weight, (uint64_t)experts * intermediate, width,
        bf16_mut(d_packed), nullptr, nullptr, route_offsets, experts, rows, width, intermediate,
        rows, stream);
  }
  if (need_weight_grad) {
    d_w_up = torch::empty_like(w_up);
    d_w_down = torch::empty_like(w_down);
    // dW_up[e] = dU_e^T @ packed_e, dW_down[e] = dY_e^T @ Z_e.
    run_grouped<nv_bf16, MODE_WGRAD, false>(
        bf16_ptr(du), rows, intermediate, packed_data, rows, width, bf16_mut(d_w_up), nullptr,
        nullptr, route_offsets, experts, intermediate, width, 0, rows, stream);
    run_grouped<nv_bf16, MODE_WGRAD, false>(
        bf16_ptr(dy), rows, width, bf16_ptr(z), rows, intermediate, bf16_mut(d_w_down), nullptr,
        nullptr, route_offsets, experts, width, intermediate, 0, rows, stream);
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
      run_grouped<nv_bf16, MODE_DX, true>(
          shared_grad, tokens, width, bf16_ptr(shared_w2), width, shared_width,
          bf16_mut(d_shared_up), nullptr, saved_shared_up, nullptr, 1, tokens, shared_width,
          width, tokens, stream);
    }
    if (need_x) {
      d_x = torch::empty({tokens, width}, options);
      run_grouped<nv_bf16, MODE_DX, false>(
          bf16_ptr(d_shared_up), tokens, shared_width, bf16_ptr(shared_w1), shared_width, width,
          bf16_mut(d_x), nullptr, nullptr, nullptr, 1, tokens, width, shared_width, tokens,
          stream);
    }
    if (need_shared_weight_grad) {
      auto shared_x_t = aligned16(*x);
      // `w_sh1`/`w_sh2` are rank two, so their gradients must be rank two as
      // well: the WGRAD kernel writes `[M, N]` from offset zero, so a 2-D
      // allocation holds exactly the same bytes.
      d_w_sh1 = torch::empty({shared_width, width}, options);
      d_w_sh2 = torch::empty({width, shared_width}, options);
      run_grouped<nv_bf16, MODE_WGRAD, false>(
          bf16_ptr(d_shared_up), tokens, shared_width, bf16_ptr(shared_x_t), tokens, width,
          bf16_mut(d_w_sh1), nullptr, nullptr, nullptr, 1, shared_width, width, 0, tokens,
          stream);
      run_grouped<nv_bf16, MODE_WGRAD, false>(
          shared_grad, tokens, width, bf16_ptr(*shared_z), tokens, shared_width,
          bf16_mut(d_w_sh2), nullptr, nullptr, nullptr, 1, width, shared_width, 0, tokens,
          stream);
    }
  }

  return {d_packed, d_x, d_weights, d_w_up, d_w_down, d_w_sh1, d_w_sh2};
#endif
}
