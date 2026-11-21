// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
// csrc/cuda/gemm/det_gemm_kernel.cu
//
// WS1 Batch-invariant deterministic GEMM (hand-written, no CUTLASS).
//
//   SM90 path : TMA load + mma.sync (m16n8k16), FP32 accum, single-CTA-per-tile,
//               fixed K order, NO split-K -> batch-invariant.
//   Fallback  : naive FP32 scalar kernel (also the correctness ground truth).
//
// Both: BF16 in / FP32 accum / BF16 store / no TF32 / no split-K.
// K is reduced with a mid-split tree. A contiguous half-K GEMM is one child,
// so simulated TP=2 (a+b) matches TP=1. TP=8 left-fold does not.
// Leaves stay FP32 (naive: 32-wide MAC; SM90: one BK).
//   fwd: C = A @ B   |   dA = dC @ B^T   |   dB = A^T @ dC
// Backward reuses the forward kernel on transposed operands.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cstdint>

#if defined(RL_KERNEL_ENABLE_SM90)
#include "det_gemm_tma.cuh"
#endif

namespace {

using nv_bf16 = __nv_bfloat16;

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

__host__ __device__ constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

enum class RhsLayout {
  // Physical RHS is the logical [K,N] matrix.
  kKN,
  // Physical RHS is [N,K] and represents the transpose of the logical RHS.
  kNK,
};

enum class OutputLayout {
  // Store the logical [M,N] result in its usual contiguous layout.
  kMN,
  // Store logical (m,n) at contiguous [N,M](n,m).
  kNM,
};

template <bool TRANSPOSE_OUTPUT>
__device__ __forceinline__ int64_t output_offset(int row, int col, int M, int N) {
  if constexpr (TRANSPOSE_OUTPUT)
    return static_cast<int64_t>(col) * M + row;
  return static_cast<int64_t>(row) * N + col;
}

// Leaf width of the K tree; the SM90 path reuses it as its BK tile width.
// Midpoint splitting keeps leaves in [16, 32], so the naive tree only matches
// SM90's 32-wide tiles bit-for-bit when K / 2^n stays a multiple of 32
// (K=4096 does, K=14336 leaves 28-wide). The two paths are not required to
// agree exactly; each stays batch-invariant on its own.
constexpr int K_TREE_LEAF = 32;
// The tree only splits while a range is wider than one leaf, so a path has at
// most ceil(log2(K / K_TREE_LEAF)) + 1 nodes. K is an int, which bounds any
// K at 27 levels; the stack below holds one pending right sibling per level,
// so 28 slots cover every reachable K. At 12 bytes per frame this stays far
// below the 1 KB default device stack that the previous recursion overflowed.
constexpr int K_TREE_MAX_DEPTH = 28;

__device__ __forceinline__ nv_bf16 bf16_add(nv_bf16 a, nv_bf16 b) {
  return __float2bfloat16(__bfloat162float(a) + __bfloat162float(b));
}

struct KTreePending {
  int mid;
  int hi;
  uint8_t stage;  // 0: left child running, 1: right child running
  nv_bf16 left;   // completed left subtree, read once the right one returns
};

// Same midpoint tree, leaf ranges and rounding points as the recursive form,
// but with a fixed-size explicit stack: deep K (e.g. the vocab-sized K of an
// lm_head backward) would otherwise overflow the default device stack. The
// current [lo, hi) range stays in registers; the stack only holds pending
// right siblings, so the leaf loop and the descent stay off local memory.
template <bool FIXED_LEAVES>
__device__ nv_bf16 k_tree_naive(const nv_bf16* __restrict__ a_row,
                                const nv_bf16* __restrict__ b_col, int stride_bk, int K) {
  const nv_bf16 zero = __float2bfloat16(0.0f);
  KTreePending stack[K_TREE_MAX_DEPTH];
  int sp = 0;
  int lo = 0;
  int hi = K;
  nv_bf16 value = zero;

  while (true) {
    if (hi - lo > K_TREE_LEAF) {
      const int length = hi - lo;
      // Muse-Glimmer splits the list of 32-wide leaves, including its final
      // partial leaf. Keep the historical element midpoint for other callers.
      const int leaves = 1 + (length - 1) / K_TREE_LEAF;
      const int mid = lo + (FIXED_LEAVES ? (leaves / 2) * K_TREE_LEAF : length / 2);
      stack[sp] = KTreePending{mid, hi, 0, zero};
      ++sp;
      hi = mid;
      continue;
    }

    float acc = 0.0f;
    const nv_bf16* a = a_row + lo;
    const nv_bf16* b = b_col + static_cast<int64_t>(lo) * stride_bk;
    for (int k = lo; k < hi; ++k, ++a, b += stride_bk)
      acc += __bfloat162float(*a) * __bfloat162float(*b);
    value = __float2bfloat16(acc);

    while (true) {
      if (sp == 0) return value;
      KTreePending& pending = stack[sp - 1];
      if (pending.stage == 0) {
        pending.stage = 1;
        pending.left = value;
        lo = pending.mid;
        hi = pending.hi;
        break;
      }
      value = bf16_add(pending.left, value);
      --sp;
    }
  }
}

// Naive FP32 scalar kernel (fallback + ground truth). Batch-invariant by
// construction: one thread = one output element, mid-split K tree.
constexpr int NAIVE_TILE = 16;

template <typename output_t, bool TRANSPOSE_OUTPUT, bool FIXED_LEAVES>
__global__ void det_gemm_naive(const nv_bf16* __restrict__ A,
                               const nv_bf16* __restrict__ B,
                               output_t* __restrict__ C,
                               int M, int N, int K, bool transpose_rhs) {
  const int row = blockIdx.y * NAIVE_TILE + threadIdx.y;
  const int col = blockIdx.x * NAIVE_TILE + threadIdx.x;
  if (row >= M || col >= N) return;
  // 64-bit row/column bases: an operand with M*K (or K*N) above INT_MAX would
  // otherwise wrap the linear index and read out of bounds.
  const nv_bf16* a_row = A + static_cast<int64_t>(row) * K;
  const nv_bf16* b_col = B + (transpose_rhs ? static_cast<int64_t>(col) * K : col);
  C[output_offset<TRANSPOSE_OUTPUT>(row, col, M, N)] = cast_output<output_t>(
      __bfloat162float(k_tree_naive<FIXED_LEAVES>(a_row, b_col, transpose_rhs ? 1 : N, K)));
}

// Weight gradients in the FFN are an outer-product GEMM when the token batch
// is small: dW = dY^T @ X.  The regular scalar fallback assigns one 16x16
// block to each output tile.  For Qwen3's [out,in] projections that creates
// roughly 200k blocks at M=1/8 and repeatedly reloads the same token rows.
// This path keeps the exact scalar accumulation order (tokens are visited in
// ascending order and the result is rounded once to BF16), but stages one
// token tile and computes a 128x128 output tile with 256 threads.  It also
// consumes X in its native [tokens,in] layout, so no X.T.contiguous() helper
// allocation is needed.
constexpr int SMALL_K_TILE = K_TREE_LEAF;
constexpr int SMALL_M_TILE = 128;
constexpr int SMALL_N_TILE = 128;
constexpr int SMALL_THREADS = 256;

template <typename output_t>
__global__ void det_gemm_db_small_k(const nv_bf16* __restrict__ X,
                                    const nv_bf16* __restrict__ dY,
                                    output_t* __restrict__ dW,
                                    int tokens,
                                    int in_features,
                                    int out_features) {
  // Give this dynamic shared-memory symbol a kernel-specific name.  CUDA
  // 12.4 diagnoses same-TU extern __shared__ declarations with different
  // element types as incompatible, even though they belong to different
  // kernels.
  extern __shared__ __align__(1024) nv_bf16 small_k_smem[];
  nv_bf16* sX = small_k_smem;
  nv_bf16* sY = sX + SMALL_K_TILE * SMALL_N_TILE;

  const int tid = threadIdx.x;
  const int in_base = blockIdx.x * SMALL_N_TILE;
  const int out_base = blockIdx.y * SMALL_M_TILE;

  // The shared tile is padded to 32 tokens. Zero padding makes the launch
  // shape independent of the token count while the loop below still visits
  // exactly the original [0,tokens) reduction range.
  for (int index = tid; index < SMALL_K_TILE * SMALL_N_TILE; index += blockDim.x) {
    const int token = index / SMALL_N_TILE;
    const int feature = index % SMALL_N_TILE;
    const int global_feature = in_base + feature;
    sX[index] = (token < tokens && global_feature < in_features)
                    ? X[static_cast<int64_t>(token) * in_features + global_feature]
                    : __float2bfloat16(0.0f);
  }
  for (int index = tid; index < SMALL_K_TILE * SMALL_M_TILE; index += blockDim.x) {
    const int token = index / SMALL_M_TILE;
    const int output = index % SMALL_M_TILE;
    const int global_output = out_base + output;
    sY[index] = (token < tokens && global_output < out_features)
                    ? dY[static_cast<int64_t>(token) * out_features + global_output]
                    : __float2bfloat16(0.0f);
  }
  __syncthreads();

  // dW is physically [out_features,in_features]. Each thread computes eight
  // elements; neighboring threads therefore issue contiguous stores.
  for (int index = tid; index < SMALL_M_TILE * SMALL_N_TILE; index += blockDim.x) {
    const int output = index / SMALL_N_TILE;
    const int feature = index % SMALL_N_TILE;
    const int global_output = out_base + output;
    const int global_feature = in_base + feature;
    if (global_output >= out_features || global_feature >= in_features) continue;

    float acc = 0.0f;
    for (int token = 0; token < tokens; ++token)
      acc += __bfloat162float(sX[token * SMALL_N_TILE + feature]) *
             __bfloat162float(sY[token * SMALL_M_TILE + output]);
    dW[static_cast<int64_t>(global_output) * in_features + global_feature] = cast_output<output_t>(acc);
  }
}

template <typename output_t>
void launch_db_small_k(const nv_bf16* X,
                       const nv_bf16* dY,
                       output_t* dW,
                       int tokens,
                       int in_features,
                       int out_features,
                       cudaStream_t stream) {
  dim3 block(SMALL_THREADS);
  dim3 grid(cdiv(in_features, SMALL_N_TILE), cdiv(out_features, SMALL_M_TILE));
  constexpr int smem_elements = SMALL_K_TILE * (SMALL_N_TILE + SMALL_M_TILE);
  det_gemm_db_small_k<output_t><<<grid, block, smem_elements * sizeof(nv_bf16), stream>>>(
      X, dY, dW, tokens, in_features, out_features);
}

template <typename output_t, bool TRANSPOSE_OUTPUT, bool FIXED_LEAVES = false>
void launch_naive(const nv_bf16* A, const nv_bf16* B, output_t* C,
                  int M, int N, int K, cudaStream_t stream, bool transpose_rhs = false) {
  dim3 block(NAIVE_TILE, NAIVE_TILE);
  dim3 grid(cdiv(N, NAIVE_TILE), cdiv(M, NAIVE_TILE));
  det_gemm_naive<output_t, TRANSPOSE_OUTPUT, FIXED_LEAVES><<<grid, block, 0, stream>>>(
      A, B, C, M, N, K, transpose_rhs);
}

// The 96-token training weight gradient has exactly three full leaves.
// Each warp reads contiguous input features and writes dW in its native
// [out,in] layout. Reuse the same inputs across a 4x4 per-thread tile.
__global__ void det_gemm_fixed_wgrad_96(const nv_bf16* __restrict__ x,
                                        const nv_bf16* __restrict__ dy,
                                        nv_bf16* __restrict__ dw, int in, int out) {
  const int col = blockIdx.x * 128 + threadIdx.x;
  const int row = blockIdx.y * 32 + threadIdx.y;
  nv_bf16 left[4][4], middle[4][4];
#pragma unroll
  for (int leaf = 0; leaf < 3; ++leaf) {
    float acc[4][4] = {};
#pragma unroll
    for (int k = 0; k < K_TREE_LEAF; ++k) {
      const int token = leaf * K_TREE_LEAF + k;
      float inputs[4];
#pragma unroll
      for (int c = 0; c < 4; ++c)
        inputs[c] = __bfloat162float(x[static_cast<int64_t>(token) * in + col + 32 * c]);
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const float grad = __bfloat162float(dy[static_cast<int64_t>(token) * out + row + 8 * r]);
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[r][c] = __fmaf_rn(inputs[c], grad, acc[r][c]);
      }
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const nv_bf16 node = __float2bfloat16_rn(acc[r][c]);
        if (leaf == 0) left[r][c] = node;
        else if (leaf == 1) middle[r][c] = node;
        else dw[static_cast<int64_t>(row + 8 * r) * in + col + 32 * c] =
            bf16_add(left[r][c], bf16_add(middle[r][c], node));
      }
    }
  }
}

// At a fixed depth of the midpoint tree, each subtree contains one or two
// leaves. A thread computes that subtree in order. The remaining levels are
// a balanced tree across threads; all intermediate values stay on chip.
__host__ __device__ constexpr int fixed_tree_width(int leaves) {
  int width = 1;
  while (width <= leaves / 2) width *= 2;
  return width;
}

struct alignas(16) Bf16Vector {
  nv_bf16 values[8];
};

template <int WIDTH>
__device__ __forceinline__ float fixed_warp_tree(float value) {
#pragma unroll
  for (int step = 1; step < WIDTH; step *= 2) {
    const float right = __shfl_down_sync(0xffffffff, value, step);
    value = __bfloat162float(__float2bfloat16_rn(value + right));
  }
  return value;
}

template <int K, int ROWS>
__global__ void det_gemm_fixed_projection(const nv_bf16* __restrict__ x,
                                          const nv_bf16* __restrict__ weight,
                                          nv_bf16* __restrict__ output, int M, int N) {
  constexpr int leaves = K / K_TREE_LEAF;
  constexpr int width = fixed_tree_width(leaves);
  constexpr int warps = width / 32;
  const int tid = threadIdx.x;
  const int col = blockIdx.x;
  const int row = blockIdx.y * ROWS;
  int start = 0, count = leaves;
#pragma unroll
  for (int bit = width / 2; bit; bit /= 2) {
    const int left = count / 2;
    if (tid & bit) start += left;
    count = (tid & bit) ? count - left : left;
  }

  float values[ROWS] = {};
  for (int i = 0; i < count; ++i) {
    float acc[ROWS] = {};
#pragma unroll
    for (int offset = 0; offset < K_TREE_LEAF; offset += 8) {
      const int k = (start + i) * K_TREE_LEAF + offset;
      const auto w = *reinterpret_cast<const Bf16Vector*>(weight + static_cast<int64_t>(col) * K + k);
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        const int src = min(row + r, M - 1);
        const auto a = *reinterpret_cast<const Bf16Vector*>(x + static_cast<int64_t>(src) * K + k);
#pragma unroll
        for (int j = 0; j < 8; ++j)
          acc[r] = __fmaf_rn(__bfloat162float(a.values[j]), __bfloat162float(w.values[j]), acc[r]);
      }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const float leaf = __bfloat162float(__float2bfloat16_rn(acc[r]));
      values[r] = i == 0 ? leaf : __bfloat162float(__float2bfloat16_rn(values[r] + leaf));
    }
  }

  __shared__ float warp_roots[ROWS][warps];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    const float root = fixed_warp_tree<32>(values[r]);
    if (tid % 32 == 0) warp_roots[r][tid / 32] = root;
  }
  __syncthreads();
  if (tid < 32) {
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const float root = fixed_warp_tree<warps>(warp_roots[r][tid % warps]);
      if (tid == 0 && row + r < M)
        output[static_cast<int64_t>(row + r) * N + col] = __float2bfloat16_rn(root);
    }
  }
}

template <int K>
void launch_fixed_projection(const nv_bf16* x, const nv_bf16* weight, nv_bf16* output,
                             int M, int N, cudaStream_t stream) {
  constexpr int threads = fixed_tree_width(K / K_TREE_LEAF);
  constexpr int rows = threads > 128 ? 4 : 8;
  if (M == 1)
    det_gemm_fixed_projection<K, 1><<<N, threads, 0, stream>>>(x, weight, output, M, N);
  else
    det_gemm_fixed_projection<K, rows><<<dim3(N, cdiv(M, rows)), threads, 0, stream>>>(
        x, weight, output, M, N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#if defined(RL_KERNEL_ENABLE_SM90)
// SM90 path: TMA load + mma.sync. C[M,N] = A[M,K] @ B[K,N].
// Each CTA owns one [BM,BN] output tile, walks full K in fixed order (no
// split-K). A tile [BM,BK] row-major; B operand col-major [n,k] supplied by
// passing B^T ([N,K] row-major) so the B smem tile is [BN,BK] (row=n,col=k),
// matching the validated logp ldmatrix addressing.
constexpr int BM = 128, BN = 64, BK = 32;
static_assert(BK == K_TREE_LEAF, "SM90 tile width must match the naive K-tree leaf");
constexpr int WARPS = 4;
constexpr int WG_THREADS = WARPS * 32;  // 128
constexpr int STAGES = 2;

constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16;
constexpr int WARP_M = BM / WARPS;        // 32
constexpr int M_TILES = WARP_M / MMA_M;   // 2
constexpr int N_TILES = BN / MMA_N;       // 8
constexpr int K_TILES = BK / MMA_K;       // 2
constexpr int KK_GROUPS = BK / 32;        // 1
// Online mid-tree reduction needs ceil(log2(K / BK)) live levels.  The
// training contract caps a rank's GEMM K at 32768, so ten levels cover every
// configured shape while avoiding six never-addressed per-thread stack slots.
constexpr int TREE_DEPTH = 10;

__device__ __forceinline__ int mid_tree_merge_count(int leaf, int n) {
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

__device__ __forceinline__ void ldmatrix_x4(uint32_t regs[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
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

template <typename output_t, bool TRANSPOSE_OUTPUT>
__global__ void det_gemm_sm90_kernel(const __grid_constant__ CUtensorMap a_tmap,
                                     const __grid_constant__ CUtensorMap bt_tmap,
                                     output_t* __restrict__ C,
                                     int M, int N, int K) {
  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int row_base = blockIdx.y * BM;
  const int col_base = blockIdx.x * BN;
  const int kd = K / BK;

  extern __shared__ __align__(1024) char sm90_smem[];
  nv_bf16* sA = reinterpret_cast<nv_bf16*>(sm90_smem);
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
    const int koff = k * BK;
    det_gemm::tma_2d_g2s(sA_base + buf * BM * BK * sizeof(nv_bf16), &a_tmap, koff, row_base, mbar[buf]);
    det_gemm::tma_2d_g2s(sB_base + buf * BN * BK * sizeof(nv_bf16), &bt_tmap, koff, col_base, mbar[buf]);
    det_gemm::mbar_arrive_expect_tx(mbar[buf], tile_bytes);
  };

  int phase[STAGES];
#pragma unroll
  for (int s = 0; s < STAGES; ++s) phase[s] = 0;

  float tile_acc[M_TILES][N_TILES][4];
  __nv_bfloat162 tree_v[M_TILES][N_TILES][2];
  __nv_bfloat162 tree_stk[TREE_DEPTH][M_TILES][N_TILES][2];
  int sp = 0;

  if (tid == 0)
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s)
      if (s < kd) issue_load(s);

  for (int k = 0; k < kd; ++k) {       // fixed ascending tile order, NO split-K
    const int buf = k % STAGES;
    if (tid == 0 && k + (STAGES - 1) < kd) issue_load(k + (STAGES - 1));
    if (tid == 0) det_gemm::mbar_wait(mbar[buf], phase[buf]);
    phase[buf] ^= 1;
    __syncthreads();

    const uint32_t sA_buf = sA_base + buf * BM * BK * sizeof(nv_bf16);
    const uint32_t sB_buf = sB_base + buf * BN * BK * sizeof(nv_bf16);

#pragma unroll
    for (int mi = 0; mi < M_TILES; ++mi)
#pragma unroll
      for (int n = 0; n < N_TILES; ++n)
        tile_acc[mi][n][0] = tile_acc[mi][n][1] = tile_acc[mi][n][2] = tile_acc[mi][n][3] = 0.0f;

    uint32_t A[M_TILES][K_TILES][4];
#pragma unroll
    for (int mi = 0; mi < M_TILES; ++mi) {
      const int row0 = warp * WARP_M + mi * MMA_M + (lane % 16);
#pragma unroll
      for (int kt = 0; kt < K_TILES; ++kt) {
        const uint32_t a_addr =
            sA_buf + (row0 * BK + (lane / 16) * 8 + kt * MMA_K) * sizeof(nv_bf16);
        ldmatrix_x4(A[mi][kt], a_addr);
      }
    }

#pragma unroll
    for (int n = 0; n < N_TILES; ++n) {
#pragma unroll
      for (int kk = 0; kk < KK_GROUPS; ++kk) {
        uint32_t b4[4];
        const uint32_t b_addr =
            sB_buf + ((n * MMA_N + (lane % 8)) * BK + (lane / 8) * 8 + kk * 32) * sizeof(nv_bf16);
        ldmatrix_x4(b4, b_addr);
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
          tree_v[mi][n][i] = __floats2bfloat162_rn(tile_acc[mi][n][2 * i + 0],
                                                    tile_acc[mi][n][2 * i + 1]);

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

#pragma unroll
  for (int mi = 0; mi < M_TILES; ++mi) {
    const int row = row_base + warp * WARP_M + mi * MMA_M + lane / 4;
#pragma unroll
    for (int n = 0; n < N_TILES; ++n) {
      const int col = col_base + n * MMA_N + (lane % 4) * 2;
      if (row < M && col + 1 < N) {
        C[output_offset<TRANSPOSE_OUTPUT>(row, col + 0, M, N)] =
            cast_output<output_t>(__low2float(tree_v[mi][n][0]));
        C[output_offset<TRANSPOSE_OUTPUT>(row, col + 1, M, N)] =
            cast_output<output_t>(__high2float(tree_v[mi][n][0]));
      }
      if (row + 8 < M && col + 1 < N) {
        C[output_offset<TRANSPOSE_OUTPUT>(row + 8, col + 0, M, N)] =
            cast_output<output_t>(__low2float(tree_v[mi][n][1]));
        C[output_offset<TRANSPOSE_OUTPUT>(row + 8, col + 1, M, N)] =
            cast_output<output_t>(__high2float(tree_v[mi][n][1]));
      }
    }
  }
}

template <typename output_t, bool TRANSPOSE_OUTPUT>
bool launch_sm90(const nv_bf16* A, const nv_bf16* Bt, output_t* C,
                 int M, int N, int K, cudaStream_t stream) {
  if (M % BM != 0 || N % BN != 0 || K % BK != 0) return false;  // fall back
  if (K / BK > (1 << TREE_DEPTH)) return false;

  CUtensorMap a_tmap, bt_tmap;
  det_gemm::init_tmap_noswizzle(&a_tmap, A, M, K, BM, BK);
  det_gemm::init_tmap_noswizzle(&bt_tmap, Bt, N, K, BN, BK);

  const int smem = STAGES * (BM * BK + BN * BK) * sizeof(nv_bf16) + STAGES * 8;
  if (smem > 48 * 1024)
    cudaFuncSetAttribute(det_gemm_sm90_kernel<output_t, TRANSPOSE_OUTPUT>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

  dim3 grid(cdiv(N, BN), cdiv(M, BM));
  det_gemm_sm90_kernel<output_t, TRANSPOSE_OUTPUT>
      <<<grid, WG_THREADS, smem, stream>>>(a_tmap, bt_tmap, C, M, N, K);
  return true;
}
#endif  // RL_KERNEL_ENABLE_SM90

int sm_major() {
  int dev = 0; cudaGetDevice(&dev);
  cudaDeviceProp p{}; cudaGetDeviceProperties(&p, dev);
  return p.major;
}
inline const nv_bf16* bf16(const torch::Tensor& t) {
  return reinterpret_cast<const nv_bf16*>(t.data_ptr<at::BFloat16>());
}
inline nv_bf16* bf16o(torch::Tensor& t) {
  return reinterpret_cast<nv_bf16*>(t.data_ptr<at::BFloat16>());
}
void check_in(const torch::Tensor& t, const char* n) {
  TORCH_CHECK(t.is_cuda(), n, " must be CUDA");
  TORCH_CHECK(t.scalar_type() == torch::kBFloat16, n, " must be bf16");
}

void check_fixed_leaf_math() {
#if defined(RL_KERNEL_USE_FAST_MATH)
  TORCH_CHECK(false, "fixed-leaf GEMM requires a build with KERNEL_ALIGN_USE_FAST_MATH=0");
#endif
}

torch::Tensor gemm_dispatch(const torch::Tensor& a, const torch::Tensor& rhs,
                            RhsLayout rhs_layout = RhsLayout::kKN,
                            OutputLayout output_layout = OutputLayout::kMN,
                            bool fixed_leaf_tree = false, bool output_fp32 = false) {
  const int M = a.size(0), K = a.size(1);
  const int N = rhs_layout == RhsLayout::kKN ? rhs.size(1) : rhs.size(0);
  const bool transpose_output = output_layout == OutputLayout::kNM;
  auto options = a.options().dtype(output_fp32 ? torch::kFloat32 : torch::kBFloat16);
  auto c = transpose_output ? torch::empty({N, M}, options) : torch::empty({M, N}, options);
  auto stream = at::cuda::getCurrentCUDAStream();

  // Explicit scalar arithmetic also on Hopper: MMA sums within a leaf in
  // hardware order, which can change the rounded BF16 leaf.
  if (fixed_leaf_tree) {
    check_fixed_leaf_math();
    const bool transpose_rhs = rhs_layout == RhsLayout::kNK;
    if (transpose_rhs && !transpose_output && M > 0 && N > 0 && M <= 4 * 65535 &&
        (reinterpret_cast<std::uintptr_t>(a.data_ptr()) % 16 == 0) &&
        (reinterpret_cast<std::uintptr_t>(rhs.data_ptr()) % 16 == 0)) {
      switch (K) {
        case 4096:
          launch_fixed_projection<4096>(bf16(a), bf16(rhs), bf16o(c), M, N, stream);
          return c;
        case 6656:
          launch_fixed_projection<6656>(bf16(a), bf16(rhs), bf16o(c), M, N, stream);
          return c;
        case 19968:
          launch_fixed_projection<19968>(bf16(a), bf16(rhs), bf16o(c), M, N, stream);
          return c;
      }
    }
    if (transpose_output)
      launch_naive<nv_bf16, true, true>(
          bf16(a), bf16(rhs), bf16o(c), M, N, K, stream, transpose_rhs);
    else
      launch_naive<nv_bf16, false, true>(
          bf16(a), bf16(rhs), bf16o(c), M, N, K, stream, transpose_rhs);
    return c;
  }

#if defined(RL_KERNEL_ENABLE_SM90)
  // Tensor-core path requires N,K tile-aligned. M is padded up to a multiple of
  // BM so that EVERY M (including M=1 and non-aligned M) takes the SAME kernel.
  // Selecting a different kernel based on M would itself break batch-invariance
  // because M is the batch dimension.
  if (sm_major() >= 9 && K % BK == 0 && N % BN == 0) {
    const int Mp = cdiv(M, BM) * BM;
    torch::Tensor a_use = a;
    if (Mp != M) {
      a_use = torch::zeros({Mp, K}, a.options());
      a_use.narrow(0, 0, M).copy_(a);
    } else if (reinterpret_cast<std::uintptr_t>(a_use.data_ptr()) % 16 != 0) {
      // cuTensorMapEncodeTiled requires a 16-byte-aligned global base. A
      // contiguous offset view is not guaranteed to satisfy that contract.
      a_use = a.clone();
    }
    torch::Tensor c_use = c;
    if (Mp != M)
      c_use = transpose_output ? torch::empty({N, Mp}, options)
                               : torch::empty({Mp, N}, options);

    // TMA consumes the physical [N,K] representation.  The explicit kNK
    // contract lets callers provide that layout directly without a round-trip
    // transpose through logical [K,N].
    torch::Tensor bt = rhs_layout == RhsLayout::kNK ? rhs : rhs.t().contiguous();
    if (reinterpret_cast<std::uintptr_t>(bt.data_ptr()) % 16 != 0)
      bt = bt.clone();
    bool launched = false;
    if (output_fp32) {
      launched = transpose_output
          ? launch_sm90<float, true>(
                bf16(a_use), bf16(bt), c_use.data_ptr<float>(), Mp, N, K, stream)
          : launch_sm90<float, false>(
                bf16(a_use), bf16(bt), c_use.data_ptr<float>(), Mp, N, K, stream);
    } else {
      launched = transpose_output
          ? launch_sm90<nv_bf16, true>(bf16(a_use), bf16(bt), bf16o(c_use), Mp, N, K, stream)
          : launch_sm90<nv_bf16, false>(bf16(a_use), bf16(bt), bf16o(c_use), Mp, N, K, stream);
    }
    if (launched) {
      if (Mp != M)
        c.copy_(c_use.narrow(transpose_output ? 1 : 0, 0, M));
      return c;
    }
  }
#endif

  // The scalar fallback keeps its original logical [K,N] operand traversal.
  // Materialize only for the new physical-[N,K] contract.
  torch::Tensor b = rhs_layout == RhsLayout::kKN ? rhs : rhs.t().contiguous();
  if (output_fp32) {
    if (transpose_output)
      launch_naive<float, true>(bf16(a), bf16(b), c.data_ptr<float>(), M, N, K, stream);
    else
      launch_naive<float, false>(bf16(a), bf16(b), c.data_ptr<float>(), M, N, K, stream);
  } else {
    if (transpose_output)
      launch_naive<nv_bf16, true>(bf16(a), bf16(b), bf16o(c), M, N, K, stream);
    else
      launch_naive<nv_bf16, false>(bf16(a), bf16(b), bf16o(c), M, N, K, stream);
  }
  return c;
}

}  // anonymous namespace

bool det_gemm_sm90_compiled() {
#if defined(RL_KERNEL_ENABLE_SM90)
  return true;
#else
  return false;
#endif
}

torch::Tensor det_gemm_fwd(torch::Tensor a, torch::Tensor b, bool fixed_leaf_tree) {
  check_in(a, "A"); check_in(b, "B");
  a = a.contiguous(); b = b.contiguous();
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "det_gemm_fwd: expect 2D [M,K]@[K,N]");
  TORCH_CHECK(b.size(0) == a.size(1), "det_gemm_fwd: K mismatch");
  return gemm_dispatch(a, b, RhsLayout::kKN, OutputLayout::kMN, fixed_leaf_tree);
}

torch::Tensor det_gemm_fwd_rhs_transposed(torch::Tensor a, torch::Tensor bt, bool fixed_leaf_tree) {
  check_in(a, "A"); check_in(bt, "Bt");
  a = a.contiguous(); bt = bt.contiguous();
  TORCH_CHECK(a.dim() == 2 && bt.dim() == 2,
              "det_gemm_fwd_rhs_transposed: expect A[M,K] and Bt[N,K]");
  TORCH_CHECK(bt.size(1) == a.size(1), "det_gemm_fwd_rhs_transposed: K mismatch");
  return gemm_dispatch(a, bt, RhsLayout::kNK, OutputLayout::kMN, fixed_leaf_tree);
}

torch::Tensor det_gemm_fwd_fp32(torch::Tensor a, torch::Tensor b) {
  check_in(a, "A"); check_in(b, "B");
  a = a.contiguous(); b = b.contiguous();
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "det_gemm_fwd_fp32: expect 2D [M,K]@[K,N]");
  TORCH_CHECK(b.size(0) == a.size(1), "det_gemm_fwd_fp32: K mismatch");
  // Keep the FP32 running sum; only the final store is BF16.
  return gemm_dispatch(a, b);
}

torch::Tensor det_gemm_da(torch::Tensor dc, torch::Tensor b) {
  check_in(dc, "dC"); check_in(b, "B");
  dc = dc.contiguous(); b = b.contiguous();
  TORCH_CHECK(dc.dim() == 2 && b.dim() == 2,
              "det_gemm_da: expect dC[M,N] and B[K,N]");
  TORCH_CHECK(b.size(1) == dc.size(1), "det_gemm_da: N mismatch");
  // dA = dC @ B^T.  B already is the physical [K,N] transpose of that
  // logical RHS, which is exactly the SM90 TMA layout.
  return gemm_dispatch(dc, b, RhsLayout::kNK);
}

torch::Tensor det_gemm_db(torch::Tensor a, torch::Tensor dc) {
  check_in(a, "A"); check_in(dc, "dC");
  dc = dc.contiguous();
  TORCH_CHECK(a.dim() == 2 && dc.dim() == 2,
              "det_gemm_db: expect A[M,K] and dC[M,N]");
  auto at = a.t().contiguous();
  TORCH_CHECK(dc.size(0) == at.size(1), "det_gemm_db: M mismatch");
  return gemm_dispatch(at, dc);
}

torch::Tensor det_gemm_db_transposed(torch::Tensor a, torch::Tensor dc, bool fixed_leaf_tree) {
  check_in(a, "A"); check_in(dc, "dC");
  TORCH_CHECK(a.dim() == 2 && dc.dim() == 2,
              "det_gemm_db_transposed: expect A[M,K] and dC[M,N]");
  const int tokens = a.size(0);
  const int in_features = a.size(1);
  const int out_features = dc.size(1);
  TORCH_CHECK(dc.size(0) == tokens, "det_gemm_db_transposed: M mismatch");
  if (fixed_leaf_tree) check_fixed_leaf_math();

  if (fixed_leaf_tree && tokens == 3 * K_TREE_LEAF && in_features > 0 && out_features > 0 &&
      in_features % 128 == 0 && out_features % 32 == 0 && out_features / 32 <= 65535) {
    a = a.contiguous();
    dc = dc.contiguous();
    auto output = torch::empty({out_features, in_features}, a.options());
    det_gemm_fixed_wgrad_96<<<dim3(in_features / 128, out_features / 32), dim3(32, 8), 0,
                              at::cuda::getCurrentCUDAStream()>>>(
        bf16(a), bf16(dc), bf16o(output), in_features, out_features);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
  }

  // The normal SM90 path expects A^T to be physically contiguous. For short
  // token batches, materializing that transpose and launching the 16x16
  // scalar fallback dominates the actual outer-product work. The tiled path
  // reads A directly and preserves the same ascending-token reduction order.
  if (tokens > 0 && tokens < SMALL_K_TILE) {
    a = a.contiguous();
    dc = dc.contiguous();
    auto output = torch::empty({out_features, in_features}, a.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    launch_db_small_k<nv_bf16>(
        bf16(a), bf16(dc), bf16o(output), tokens, in_features, out_features, stream);
    return output;
  }

  dc = dc.contiguous();
  auto at = a.t().contiguous();
  // Preserve the exact A^T @ dC MMA/tree evaluation and change only the final
  // address mapping so the canonical [N,K] weight-gradient is born contiguous.
  return gemm_dispatch(at, dc, RhsLayout::kKN, OutputLayout::kNM, fixed_leaf_tree);
}
