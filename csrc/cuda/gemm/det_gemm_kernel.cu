// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
// csrc/cuda/gemm/det_gemm_kernel.cu
//
// WS1 batch-invariant deterministic GEMM.
//
//   SM90 path : TMA load + mma.sync (m16n8k16), FP32 accum, single-CTA-per-tile,
//               fixed K order, NO split-K -> batch-invariant.
//   Fallback  : naive FP32 scalar kernel (also the correctness ground truth).
//
// Both: BF16 in / FP32 accum / BF16 store / no TF32 / no split-K.
// K is reduced with a mid-split tree. A contiguous half-K GEMM is one child,
// so simulated TP=2 (a+b) matches TP=1. TP=8 left-fold does not.
// Leaves stay FP32 (naive: 32-wide MAC; SM90: one BK).
// Muse-Glimmer opts into a tree over consecutive 32-wide leaves. Its tiled
// scalar kernels preserve each individual FMA and BF16 rounding point.
//   fwd: C = A @ B   |   dA = dC @ B^T   |   dB = A^T @ dC
// Backward reuses the forward kernel on transposed operands.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdint>

#if defined(__CUDACC__) && !defined(__HIPCC__) && __has_include(<cute/tensor.hpp>)
#include <cute/tensor.hpp>
#define RL_KERNEL_FIXED_TILE_CUTE 1
#endif

#if defined(RL_KERNEL_ENABLE_SM90)
#include "det_gemm_tma.cuh"
#endif

namespace {

using nv_bf16 = __nv_bfloat16;

inline const nv_bf16* bf16(const torch::Tensor& t) {
  return reinterpret_cast<const nv_bf16*>(t.data_ptr<at::BFloat16>());
}
inline nv_bf16* bf16o(torch::Tensor& t) {
  return reinterpret_cast<nv_bf16*>(t.data_ptr<at::BFloat16>());
}

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
      acc = __fmaf_rn(__bfloat162float(*a), __bfloat162float(*b), acc);
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
      acc = __fmaf_rn(__bfloat162float(sX[token * SMALL_N_TILE + feature]),
                     __bfloat162float(sY[token * SMALL_M_TILE + output]), acc);
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
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
    const float right = __shfl_down(value, step, 32);
#else
    const float right = __shfl_down_sync(0xffffffffu, value, step, 32);
#endif
    value = __bfloat162float(__float2bfloat16(value + right));
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
  constexpr bool pack_leaves = ROWS == 1 && leaves != width;
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
  // In decode, adjacent threads read consecutive leaves of one weight row.
  // Re-index their rounded values on chip to recover the tree's child pairs.
  __shared__ float leaf_values[pack_leaves ? leaves : 1];
  const int first_leaf = pack_leaves ? tid : start;
  const int end_leaf = pack_leaves ? leaves : start + count;
  for (int i = first_leaf; i < end_leaf; i += pack_leaves ? width : 1) {
    float acc[ROWS] = {};
#pragma unroll
    for (int offset = 0; offset < K_TREE_LEAF; offset += 8) {
      const int k = i * K_TREE_LEAF + offset;
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
      const float leaf = __bfloat162float(__float2bfloat16(acc[r]));
      if constexpr (pack_leaves)
        leaf_values[i] = leaf;
      else
        values[r] = i == first_leaf ? leaf : __bfloat162float(__float2bfloat16(values[r] + leaf));
    }
  }

  if constexpr (pack_leaves) {
    __syncthreads();
    values[0] = leaf_values[start];
    if (count == 2)
      values[0] = __bfloat162float(__float2bfloat16(values[0] + leaf_values[start + 1]));
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
        output[static_cast<int64_t>(row + r) * N + col] = __float2bfloat16(root);
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

// Each block evaluates one canonical subtree for a 32x64 output tile. The
// model's large K dimensions have a multiple of 16 leaves, so the first
// four midpoint splits give 16 equal subtrees. K=256 fits in one subtree.
constexpr int FIXED_GROUPS = 16;
constexpr int TILE_ROWS = 32, TILE_COLS = 64, TILE_THREADS = 128;
constexpr int WGRAD_TILE_ROWS = 64;

struct LeafMerge {
  uint8_t depth;
  uint8_t joins;
};

__host__ __device__ constexpr LeafMerge leaf_merge(int leaf, int leaves) {
  LeafMerge merge{};
  while (leaves > 1) {
    const int half = leaves / 2;
    ++merge.depth;
    if (leaf < half) {
      leaves = half;
      merge.joins = 0;
    } else {
      leaf -= half;
      leaves -= half;
      ++merge.joins;
    }
  }
  return merge;
}

constexpr int fixed_tree_depth(int leaves) {
  return leaves <= 1 ? 0 : 1 + fixed_tree_depth((leaves + 1) / 2);
}

// Each warp reads the same precomputed midpoint-tree step for the current leaf.
template <int LEAVES>
struct FixedLeafSchedule {
  LeafMerge steps[LEAVES]{};
  constexpr FixedLeafSchedule() {
    for (int leaf = 0; leaf < LEAVES; ++leaf) steps[leaf] = leaf_merge(leaf, LEAVES);
  }
};

template <int LEAVES>
__device__ __constant__ FixedLeafSchedule<LEAVES> fixed_leaf_schedule;

struct alignas(16) FloatQuad {
  float values[4];
};
struct alignas(8) Bf16Quad {
  nv_bf16 values[4];
};

__device__ __forceinline__ Bf16Vector load_fixed_vector(const nv_bf16* p, bool aligned) {
  Bf16Vector result;
  if (aligned) {
#if defined(RL_KERNEL_FIXED_TILE_CUTE)
    auto src = cute::make_tensor(cute::make_gmem_ptr(p), cute::make_layout(cute::_8{}));
    auto dst = cute::make_tensor(cute::make_rmem_ptr(result.values), cute::make_layout(cute::_8{}));
    cute::copy(cute::recast<cute::uint128_t>(src), cute::recast<cute::uint128_t>(dst));
#else
    result = *reinterpret_cast<const Bf16Vector*>(p);
#endif
  } else {
#pragma unroll
    for (int i = 0; i < 8; ++i) result.values[i] = p[i];
  }
  return result;
}

// Each four-feature register fragment is contiguous in shared memory. When
// transposing a [feature,K] input, distribute the four K packets over banks.
template <int WIDTH, bool INNER_K>
__device__ __forceinline__ int fixed_quad_index(int feature, int k) {
  if constexpr (INNER_K) feature ^= (k / 8) * 8;
  return k * (WIDTH / 4) + feature / 4;
}

template <int WIDTH, int THREADS, bool INNER_K>
struct FixedOperandTile {
  static constexpr int vectors = WIDTH * K_TREE_LEAF / 8;
  Bf16Vector packets[cdiv(vectors, THREADS)];

  __device__ static int feature(int i) { return INNER_K ? i / 4 : i % (WIDTH / 8) * 8; }
  __device__ static int k_offset(int i) { return INNER_K ? i % 4 * 8 : i / (WIDTH / 8); }

  __device__ void load(const nv_bf16* input, int stride, int origin, int k0, int extent) {
    const bool aligned = (reinterpret_cast<std::uintptr_t>(input) % alignof(Bf16Vector)) == 0;
    // Both physical layouts have an eight-element-aligned row stride.
#pragma unroll
    for (int slot = 0; slot < cdiv(vectors, THREADS); ++slot) {
      const int i = threadIdx.x + slot * THREADS;
      if (i >= vectors) continue;
      const int src = min(origin + feature(i), extent - (INNER_K ? 1 : 8));
      const int k = k0 + k_offset(i);
      const int64_t offset = INNER_K ? static_cast<int64_t>(src) * stride + k
                                     : static_cast<int64_t>(k) * stride + src;
      packets[slot] = load_fixed_vector(input + offset, aligned);
    }
  }

  template <bool PARTIAL_K, bool STRIDED>
  __device__ void load_wgrad(const nv_bf16* input, int origin, int k0, int extent, int leaf_len,
                             int64_t token_stride, int64_t feature_stride) {
    static_assert(!INNER_K);
    if constexpr (!STRIDED) {
      token_stride = extent;
      feature_stride = 1;
    }
    const bool aligned = (reinterpret_cast<std::uintptr_t>(input) % alignof(Bf16Vector)) == 0 &&
                         token_stride % 8 == 0;
#pragma unroll
    for (int slot = 0; slot < cdiv(vectors, THREADS); ++slot) {
      const int i = threadIdx.x + slot * THREADS;
      if (i >= vectors) continue;
      const int k = k_offset(i);
      if constexpr (PARTIAL_K) {
        if (k >= leaf_len) {
#pragma unroll
          for (int j = 0; j < 8; ++j) packets[slot].values[j] = __float2bfloat16(0.0f);
          continue;
        }
      }
      const auto* base = input + (k0 + k) * token_stride + (origin + feature(i)) * feature_stride;
      if (feature_stride == 1) {
        packets[slot] = load_fixed_vector(base, aligned);
      } else {
#pragma unroll
        for (int j = 0; j < 8; ++j) packets[slot].values[j] = base[j * feature_stride];
      }
    }
  }

  __device__ void store(FloatQuad* tile) const {
#pragma unroll
    for (int slot = 0; slot < cdiv(vectors, THREADS); ++slot) {
      const int i = threadIdx.x + slot * THREADS;
      if (i >= vectors) continue;
      const int x = feature(i), k = k_offset(i);
      if constexpr (INNER_K) {
#pragma unroll
        for (int j = 0; j < 8; ++j)
          tile[fixed_quad_index<WIDTH, true>(x, k + j)].values[x % 4] =
              __bfloat162float(packets[slot].values[j]);
      } else {
#pragma unroll
        for (int j = 0; j < 8; j += 4) {
          FloatQuad v;
#pragma unroll
          for (int c = 0; c < 4; ++c) v.values[c] = __bfloat162float(packets[slot].values[j + c]);
          tile[fixed_quad_index<WIDTH, false>(x + j, k)] = v;
        }
      }
    }
  }
};

template <int K, int GROUPS, bool TRANSPOSE_RHS, bool WGRAD_96 = false>
__global__ void det_gemm_fixed_tiles(const nv_bf16* __restrict__ a,
                                     const nv_bf16* __restrict__ b,
                                     nv_bf16* __restrict__ partial, int M, int N) {
  static_assert(K % (GROUPS * K_TREE_LEAF) == 0);
  static_assert(!WGRAD_96 || (K == 96 && GROUPS == 1 && !TRANSPOSE_RHS));
  constexpr int leaves = K / (GROUPS * K_TREE_LEAF);
  constexpr int rows = WGRAD_96 ? WGRAD_TILE_ROWS : TILE_ROWS, cols = TILE_COLS;
  constexpr int per_row = WGRAD_96 ? 8 : 4, per_col = 4;
  constexpr int threads = (rows / per_row) * (cols / per_col);
  static_assert(threads == TILE_THREADS);
  // The native-weight forward can load the next leaf while the current
  // leaf's independent accumulators consume the other shared-memory buffer.
  constexpr bool prefetch = TRANSPOSE_RHS && K > 256;
  constexpr int stages = prefetch ? 2 : 1;
  __shared__ FloatQuad shared_a[stages][rows * K_TREE_LEAF / 4];
  __shared__ FloatQuad shared_b[stages][cols * K_TREE_LEAF / 4];
  __shared__ Bf16Quad pending[fixed_tree_depth(leaves)][per_row][threads];
  const int row0 = blockIdx.y * rows, col0 = blockIdx.x * cols;
  const int row_quad = threadIdx.x / (cols / per_col) * 4;
  const int col = threadIdx.x % (cols / per_col) * per_col;
  FixedOperandTile<rows, threads, !WGRAD_96> a_tile;
  FixedOperandTile<cols, threads, TRANSPOSE_RHS> b_tile;
  const int group_k = blockIdx.z * leaves * K_TREE_LEAF;
  if constexpr (prefetch) {
    a_tile.load(a, K, row0, group_k, M);
    b_tile.load(b, K, col0, group_k, N);
  }

#pragma unroll 1
  for (int leaf = 0; leaf < leaves; ++leaf) {
    const int k0 = group_k + leaf * K_TREE_LEAF;
    auto* lhs = shared_a[leaf % stages];
    auto* rhs = shared_b[leaf % stages];
    if constexpr (!prefetch) a_tile.load(a, WGRAD_96 ? M : K, row0, k0, M);
    a_tile.store(lhs);
    if constexpr (!prefetch) b_tile.load(b, TRANSPOSE_RHS ? K : N, col0, k0, N);
    b_tile.store(rhs);
    __syncthreads();
    if constexpr (prefetch) {
      if (leaf + 1 < leaves) {
        a_tile.load(a, K, row0, k0 + K_TREE_LEAF, M);
        b_tile.load(b, K, col0, k0 + K_TREE_LEAF, N);
      }
    }

    float acc[per_row][per_col] = {};
#pragma unroll
    for (int k = 0; k < K_TREE_LEAF; ++k) {
      const auto w = rhs[fixed_quad_index<cols, TRANSPOSE_RHS>(col, k)];
#pragma unroll
      for (int r = 0; r < per_row; r += 4) {
        const auto x = lhs[fixed_quad_index<rows, !WGRAD_96>(row_quad + r * (rows / per_row), k)];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
          for (int j = 0; j < per_col; ++j)
            acc[r + i][j] = WGRAD_96 ? __fmaf_rn(w.values[j], x.values[i], acc[r + i][j])
                                     : __fmaf_rn(x.values[i], w.values[j], acc[r + i][j]);
        }
      }
    }
    if constexpr (!prefetch) __syncthreads();

    // Consecutive right turns close pending left/right pairs, deepest first.
    // A thread owns its stack slots and writes four adjacent output features.
    const LeafMerge merge = fixed_leaf_schedule<leaves>.steps[leaf];
#pragma unroll
    for (int r = 0; r < per_row; ++r) {
      Bf16Quad value;
#pragma unroll
      for (int j = 0; j < per_col; ++j) value.values[j] = __float2bfloat16(acc[r][j]);
      for (int i = 0; i < merge.joins; ++i) {
        const auto left = pending[merge.depth - 1 - i][r][threadIdx.x];
#pragma unroll
        for (int j = 0; j < per_col; ++j) value.values[j] = bf16_add(left.values[j], value.values[j]);
      }
      if (leaf + 1 < leaves) {
        pending[merge.depth - merge.joins - 1][r][threadIdx.x] = value;
      } else {
        const int row = row0 + row_quad + r / 4 * 4 * (rows / per_row) + r % 4;
        if (row < M) {
          auto* dst = partial + (static_cast<int64_t>(blockIdx.z) * M + row) * N + col0 + col;
          *reinterpret_cast<Bf16Quad*>(dst) = value;
        }
      }
    }
  }
}

template <int GROUPS>
__device__ __forceinline__ nv_bf16 fixed_group_root(const nv_bf16* p, int stride) {
  if constexpr (GROUPS == 1) {
    return *p;
  } else {
    return bf16_add(fixed_group_root<GROUPS / 2>(p, stride),
                    fixed_group_root<GROUPS / 2>(p + (GROUPS / 2) * stride, stride));
  }
}

__global__ void det_gemm_fixed_tile_roots(const nv_bf16* partial, nv_bf16* out, int elements) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < elements) out[i] = fixed_group_root<FIXED_GROUPS>(partial + i, elements);
}

template <int K>
void launch_fixed_tiles(const torch::Tensor& a, const torch::Tensor& b, torch::Tensor& c,
                        int M, int N, bool transpose_rhs, cudaStream_t stream) {
  constexpr int groups = K == 256 ? 1 : FIXED_GROUPS;
  constexpr int max_grid_rows = 65535 * TILE_ROWS;
  int chunk_m = min(M, max_grid_rows);
  torch::Tensor partial;
  if constexpr (groups > 1) {
    constexpr int max_elements = (64 << 20) / (groups * sizeof(nv_bf16));
    int max_chunk_m = min(max_grid_rows, max_elements / N);
    if (max_chunk_m < 1) {
      for (int64_t row = 0; row < M; row += 65535 * NAIVE_TILE) {
        const int rows = std::min<int64_t>(M - row, 65535 * NAIVE_TILE);
        launch_naive<nv_bf16, false, true>(bf16(a) + row * K, bf16(b), bf16o(c) + row * N,
                                          rows, N, K, stream, transpose_rhs);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
      return;
    }
    if (max_chunk_m >= TILE_ROWS) max_chunk_m = max_chunk_m / TILE_ROWS * TILE_ROWS;
    chunk_m = min(M, max_chunk_m);
    partial = torch::empty({groups, chunk_m, N}, c.options());
  }

  for (int64_t row = 0; row < M; row += chunk_m) {
    const int rows = std::min<int64_t>(chunk_m, M - row);
    const auto* lhs = bf16(a) + row * K;
    auto* output = bf16o(c) + row * N;
    auto* dst = groups == 1 ? output : bf16o(partial);
    const dim3 grid(N / TILE_COLS, cdiv(rows, TILE_ROWS), groups);
    if (transpose_rhs)
      det_gemm_fixed_tiles<K, groups, true><<<grid, TILE_THREADS, 0, stream>>>(
          lhs, bf16(b), dst, rows, N);
    else
      det_gemm_fixed_tiles<K, groups, false><<<grid, TILE_THREADS, 0, stream>>>(
          lhs, bf16(b), dst, rows, N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if constexpr (groups > 1) {
      const int elements = rows * N;
      det_gemm_fixed_tile_roots<<<cdiv(elements, 128), 128, 0, stream>>>(
          dst, output, elements);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }
}

constexpr int WGRAD_PER_ROW = 4, WGRAD_PER_COL = 4;
constexpr int WGRAD_OPERAND_SMEM = (TILE_ROWS + TILE_COLS) * K_TREE_LEAF * sizeof(float);

template <bool FULL_LEAF>
__device__ __forceinline__ void fixed_wgrad_leaf(
    const FloatQuad* dy, const FloatQuad* x, int row, int col, int leaf_len,
    float (&acc)[WGRAD_PER_ROW][WGRAD_PER_COL]) {
#pragma unroll (FULL_LEAF ? K_TREE_LEAF : 1)
  for (int k = 0; k < (FULL_LEAF ? K_TREE_LEAF : leaf_len); ++k) {
    const auto lhs = dy[fixed_quad_index<TILE_ROWS, false>(row, k)];
    const auto rhs = x[fixed_quad_index<TILE_COLS, false>(col, k)];
#pragma unroll
    for (int r = 0; r < WGRAD_PER_ROW; ++r) {
#pragma unroll
      for (int j = 0; j < WGRAD_PER_COL; ++j)
        acc[r][j] = __fmaf_rn(rhs.values[j], lhs.values[r], acc[r][j]);
    }
  }
}

template <bool STRIDED>
__global__ void det_gemm_fixed_wgrad(const nv_bf16* __restrict__ dy,
                                     const nv_bf16* __restrict__ x,
                                     nv_bf16* __restrict__ dw, int tokens, int leaves,
                                     int out_features, int in_features, int64_t dy_token_stride,
                                     int64_t dy_feature_stride, int64_t x_token_stride,
                                     int64_t x_feature_stride) {
  extern __shared__ Bf16Quad wgrad_pending[][WGRAD_PER_ROW][TILE_THREADS];
  __shared__ FloatQuad shared_dy[TILE_ROWS * K_TREE_LEAF / 4];
  __shared__ FloatQuad shared_x[TILE_COLS * K_TREE_LEAF / 4];
  const int row0 = blockIdx.y * TILE_ROWS, col0 = blockIdx.x * TILE_COLS;
  const int row = threadIdx.x / (TILE_COLS / WGRAD_PER_COL) * WGRAD_PER_ROW;
  const int col = threadIdx.x % (TILE_COLS / WGRAD_PER_COL) * WGRAD_PER_COL;
  FixedOperandTile<TILE_ROWS, TILE_THREADS, false> dy_tile;
  FixedOperandTile<TILE_COLS, TILE_THREADS, false> x_tile;

  for (int leaf = 0; leaf < leaves; ++leaf) {
    const int k0 = leaf * K_TREE_LEAF;
    const int leaf_len = min(tokens - k0, K_TREE_LEAF);
    const bool full_leaf = leaf_len == K_TREE_LEAF;
    if (full_leaf) {
      dy_tile.load_wgrad<false, STRIDED>(dy, row0, k0, out_features, leaf_len,
                                         dy_token_stride, dy_feature_stride);
      x_tile.load_wgrad<false, STRIDED>(x, col0, k0, in_features, leaf_len,
                                        x_token_stride, x_feature_stride);
    } else {
      dy_tile.load_wgrad<true, STRIDED>(dy, row0, k0, out_features, leaf_len,
                                        dy_token_stride, dy_feature_stride);
      x_tile.load_wgrad<true, STRIDED>(x, col0, k0, in_features, leaf_len,
                                       x_token_stride, x_feature_stride);
    }
    dy_tile.store(shared_dy);
    x_tile.store(shared_x);
    __syncthreads();

    float acc[WGRAD_PER_ROW][WGRAD_PER_COL] = {};
    if (full_leaf) {
      fixed_wgrad_leaf<true>(shared_dy, shared_x, row, col, leaf_len, acc);
    } else {
      // Padded zero FMAs could change a negative zero from the last valid term.
      fixed_wgrad_leaf<false>(shared_dy, shared_x, row, col, leaf_len, acc);
    }
    __syncthreads();

    const LeafMerge merge = leaf_merge(leaf, leaves);
#pragma unroll
    for (int r = 0; r < WGRAD_PER_ROW; ++r) {
      Bf16Quad value;
#pragma unroll
      for (int j = 0; j < WGRAD_PER_COL; ++j) value.values[j] = __float2bfloat16(acc[r][j]);
      for (int i = 0; i < merge.joins; ++i) {
        const auto left = wgrad_pending[merge.depth - 1 - i][r][threadIdx.x];
#pragma unroll
        for (int j = 0; j < WGRAD_PER_COL; ++j) value.values[j] = bf16_add(left.values[j], value.values[j]);
      }
      if (leaf + 1 < leaves) {
        wgrad_pending[merge.depth - merge.joins - 1][r][threadIdx.x] = value;
      } else {
        auto* dst = dw + static_cast<int64_t>(row0 + row + r) * in_features + col0 + col;
        *reinterpret_cast<Bf16Quad*>(dst) = value;
      }
    }
  }
}

template <bool STRIDED>
void launch_fixed_wgrad(const torch::Tensor& x, const torch::Tensor& dy, torch::Tensor& dw,
                        int tokens, int leaves, int stack_bytes, int optin_bytes) {
  if (optin_bytes != 0)
    C10_CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(det_gemm_fixed_wgrad<STRIDED>),
                   cudaFuncAttributeMaxDynamicSharedMemorySize, optin_bytes));
  const int out_features = dy.size(1), in_features = x.size(1);
  const dim3 grid(in_features / TILE_COLS, out_features / TILE_ROWS);
  det_gemm_fixed_wgrad<STRIDED><<<grid, TILE_THREADS, stack_bytes, at::cuda::getCurrentCUDAStream()>>>(
      bf16(dy), bf16(x), bf16o(dw), tokens, leaves, out_features, in_features,
      dy.stride(0), dy.stride(1), x.stride(0), x.stride(1));
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
    // Thin forward batches still use the vectorized row kernel below.
    if (!transpose_output && M >= (transpose_rhs ? 16 : 1) && N > 0 && N % TILE_COLS == 0) {
      switch (K) {
        case 256:
          launch_fixed_tiles<256>(a, rhs, c, M, N, transpose_rhs, stream);
          return c;
        case 4096:
          launch_fixed_tiles<4096>(a, rhs, c, M, N, transpose_rhs, stream);
          return c;
        case 6656:
          launch_fixed_tiles<6656>(a, rhs, c, M, N, transpose_rhs, stream);
          return c;
        case 19968:
          launch_fixed_tiles<19968>(a, rhs, c, M, N, transpose_rhs, stream);
          return c;
      }
    }
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

  if (fixed_leaf_tree && tokens >= K_TREE_LEAF && in_features > 0 && out_features > 0 &&
      in_features % TILE_COLS == 0 && out_features % TILE_ROWS == 0 &&
      out_features / TILE_ROWS <= 65535) {
    if (tokens == 3 * K_TREE_LEAF) {
      a = a.contiguous();
      dc = dc.contiguous();
      auto output = torch::empty({out_features, in_features}, a.options());
      det_gemm_fixed_tiles<96, 1, false, true>
          <<<dim3(in_features / TILE_COLS, cdiv(out_features, WGRAD_TILE_ROWS)), TILE_THREADS, 0,
             at::cuda::getCurrentCUDAStream()>>>(bf16(dc), bf16(a), bf16o(output), out_features, in_features);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return output;
    }

    const int leaves = 1 + (tokens - 1) / K_TREE_LEAF;
    const int stack_bytes = fixed_tree_depth(leaves) * WGRAD_PER_ROW * TILE_THREADS * sizeof(Bf16Quad);
    const int total_smem = WGRAD_OPERAND_SMEM + stack_bytes;
    const int default_smem = 48 * 1024;
    const int smem_limit = total_smem <= default_smem
                               ? default_smem
                               : at::cuda::getCurrentDeviceProperties()->sharedMemPerBlockOptin;
    if (total_smem <= smem_limit) {
      // Set a device-wide limit independent of this launch's stack depth.
      const int optin_bytes = total_smem > default_smem ? smem_limit - WGRAD_OPERAND_SMEM : 0;
      auto output = torch::empty({out_features, in_features}, a.options());
      if (a.is_contiguous() && dc.is_contiguous())
        launch_fixed_wgrad<false>(a, dc, output, tokens, leaves, stack_bytes, optin_bytes);
      else
        launch_fixed_wgrad<true>(a, dc, output, tokens, leaves, stack_bytes, optin_bytes);
      return output;
    }
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
