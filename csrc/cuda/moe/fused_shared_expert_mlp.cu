// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// Fused shared-expert fc1 + SwiGLU (DSv4 MoE, P5-5 performance path).
//
//   h = BF16( SiLU(gate) * up ),   [gate | up] = x @ w_fc1^T
//
// One kernel instead of two: the FP32 z [T, 2F] intermediate never reaches
// global memory. At T=2048, F=2048 that removes a 33 MB write and a 33 MB read
// per call.
//
// Numeric profile ``p5-det-gemm-v1`` -- the same profile the unfused
// det_gemm-composed path already declares, and bit-identical to it:
//
//   * The K reduction is det_gemm's mid-split tree with a 32-wide leaf. Each
//     leaf accumulates in FP32 through mma.sync and rounds once to BF16; the
//     tree merges in BF16. A contiguous half-K shard is therefore exactly one
//     child of the tree, which is what keeps the GEMM equivalent under a
//     power-of-two K split (det_gemm's TP-equivalence design).
//   * The SwiGLU epilogue reproduces p5_swiglu_shared_forward exactly:
//     sig = 1/(1+expf(-g)), silu = __fmul_rn(g, sig), h = BF16(silu * u).
//     No clamp and no route weight -- that is the routed expert's variant
//     (P5-2), not the shared one (start-kit decision D6).
//
// So this kernel's output equals det_gemm_fwd_rhs_transposed(x, w_fc1).float()
// followed by p5_swiglu_shared_forward, byte for byte, which the tests assert.
//
// A note on what the tree does and does not buy here. Under the standard
// DSv4 shared-expert layout fc1 is column-parallel: every rank holds the full
// K = hidden and owns a slice of the 2F output columns, so fc1's K is not
// split and the tree is not what makes fc1 TP-safe (holding matching gate/up
// columns on one rank is). The tree matters for fc2, whose K = ffn is split by
// the row-parallel layout. It is used here so this kernel stays numerically
// identical to the unfused det path, and so the GEMM half remains composable
// if a future layout does shard K -- note that such a layout could not keep
// the activation fused, because SiLU(g1+g2)*(u1+u2) != SiLU(g1)*u1 + SiLU(g2)*u2.
//
// Batch invariance: one CTA owns one [BM, 32] output tile and walks the whole
// K in a fixed ascending order with no split-K and no atomics; tile constants
// are compile-time and never chosen from the token count. Rows past T are
// zero-filled by TMA and masked at the store, so a row's bytes do not depend
// on how many rows are in flight.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>

#if defined(RL_KERNEL_ENABLE_SM90)
#include <cuda.h>
#include <cudaTypedefs.h>
#include "../gemm/det_gemm_tma.cuh"
#endif

// The shared-expert sigmoid must bit-match torch.sigmoid / the strict CUDA
// core, which rules out the approximate expf and the flush-to-zero division
// that --use_fast_math substitutes.
#if defined(__USE_FAST_MATH__)
#error "fused_shared_expert_mlp requires precise FP32 math; disable --use_fast_math"
#endif

namespace {

using nv_bf16 = __nv_bfloat16;

// Must match det_gemm's K_TREE_LEAF: the two paths are required to agree.
constexpr int K_TREE_LEAF = 32;

__host__ __device__ constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

// h = BF16(SiLU(g) * u), FP32 math, one round. Identical instruction sequence
// to p5_swiglu_shared_forward in csrc/cuda/moe/shared_expert_mlp.cu.
__device__ __forceinline__ nv_bf16 swiglu_shared(float g, float u) {
  const float sig = 1.0f / (1.0f + expf(-g));
  const float silu = __fmul_rn(g, sig);
  return __float2bfloat16(__fmul_rn(silu, u));
}

// ----------------------------------------------------------- scalar path ---
// Fallback for shapes the tensor-core tile cannot cover. Same mid-split tree
// with the same 32-wide FP32 leaf, so it emits the same bytes as the SM90
// path wherever both are legal.
__device__ nv_bf16 k_tree_scalar(const nv_bf16* __restrict__ x_row,
                                 const nv_bf16* __restrict__ w_row,
                                 int lo,
                                 int hi) {
  if (hi - lo <= K_TREE_LEAF) {
    float acc = 0.0f;
    for (int k = lo; k < hi; ++k) {
      acc += __bfloat162float(x_row[k]) * __bfloat162float(w_row[k]);
    }
    return __float2bfloat16(acc);
  }
  const int mid = lo + (hi - lo) / 2;
  const nv_bf16 a = k_tree_scalar(x_row, w_row, lo, mid);
  const nv_bf16 b = k_tree_scalar(x_row, w_row, mid, hi);
  return __float2bfloat16(__bfloat162float(a) + __bfloat162float(b));
}

constexpr int SCALAR_TILE = 16;

__global__ void fused_shared_expert_scalar(const nv_bf16* __restrict__ x,
                                           const nv_bf16* __restrict__ w_fc1,
                                           nv_bf16* __restrict__ h,
                                           int T,
                                           int H,
                                           int F) {
  const int row = blockIdx.y * SCALAR_TILE + threadIdx.y;
  const int col = blockIdx.x * SCALAR_TILE + threadIdx.x;
  if (row >= T || col >= F) return;
  const nv_bf16* x_row = x + static_cast<int64_t>(row) * H;
  const nv_bf16* gate_row = w_fc1 + static_cast<int64_t>(col) * H;
  const nv_bf16* up_row = w_fc1 + static_cast<int64_t>(F + col) * H;
  const float g = __bfloat162float(k_tree_scalar(x_row, gate_row, 0, H));
  const float u = __bfloat162float(k_tree_scalar(x_row, up_row, 0, H));
  h[static_cast<int64_t>(row) * F + col] = swiglu_shared(g, u);
}

#if defined(RL_KERNEL_ENABLE_SM90)

// --------------------------------------------------------------- SM90 ------
// One CTA owns [BM tokens x 32 h-columns]. The B tile carries 64 weight rows
// laid out as [gate 0-31 | up 0-31] of this h-column tile, so the mma.sync
// accumulator fragment puts h-column c (slice n) and its partner up column
// c + 32 (slice n + 4) in the same thread: the SwiGLU pair never crosses a
// lane and the epilogue needs no shuffle.
constexpr int BM = 128, BN = 64, BK = 32;
static_assert(BK == K_TREE_LEAF, "SM90 tile width must match the K-tree leaf");
constexpr int H_COLS = BN / 2;  // 32 h-columns per CTA
constexpr int WARPS = 4;
constexpr int WG_THREADS = WARPS * 32;  // 128
constexpr int STAGES = 2;

constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16;
constexpr int WARP_M = BM / WARPS;       // 32
constexpr int M_TILES = WARP_M / MMA_M;  // 2
constexpr int N_TILES = BN / MMA_N;      // 8 -> slices 0..3 gate, 4..7 up
constexpr int K_TILES = BK / MMA_K;      // 2
constexpr int TREE_DEPTH = 16;

// Number of tree merges to run after leaf ``leaf`` of ``n``: how many times in
// a row it lands in the right half of a mid-split. Identical to det_gemm's.
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

__global__ void __launch_bounds__(WG_THREADS, 1)
fused_shared_expert_sm90(const __grid_constant__ CUtensorMap x_tmap,
                         const __grid_constant__ CUtensorMap w_tmap,
                         nv_bf16* __restrict__ h,
                         int T,
                         int H,
                         int F) {
  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int row_base = blockIdx.y * BM;
  const int col_base = blockIdx.x * H_COLS;  // first h-column of this tile
  const int kd = H / BK;

  extern __shared__ __align__(1024) char smem[];
  nv_bf16* sA = reinterpret_cast<nv_bf16*>(smem);
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
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  const uint32_t tile_bytes = (BM * BK + BN * BK) * sizeof(nv_bf16);
  constexpr int kHalfB = H_COLS * BK * sizeof(nv_bf16);  // bytes of one 32-row half

  // Gate rows [col_base, +32) and up rows [F + col_base, +32) are not adjacent
  // in w_fc1, so the B tile arrives as two 32-row boxes.
  auto issue_load = [&](int k) {
    const int buf = k % STAGES;
    const int koff = k * BK;
    det_gemm::tma_2d_g2s(sA_base + buf * BM * BK * sizeof(nv_bf16), &x_tmap, koff, row_base,
                         mbar[buf]);
    const uint32_t b_dst = sB_base + buf * BN * BK * sizeof(nv_bf16);
    det_gemm::tma_2d_g2s(b_dst, &w_tmap, koff, col_base, mbar[buf]);
    det_gemm::tma_2d_g2s(b_dst + kHalfB, &w_tmap, koff, F + col_base, mbar[buf]);
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

  for (int k = 0; k < kd; ++k) {  // fixed ascending leaf order, NO split-K
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
        ldmatrix_x4(A[mi][kt],
                    sA_buf + (row0 * BK + (lane / 16) * 8 + kt * MMA_K) * sizeof(nv_bf16));
      }
    }

#pragma unroll
    for (int n = 0; n < N_TILES; ++n) {
      uint32_t b4[4];
      ldmatrix_x4(b4, sB_buf + ((n * MMA_N + (lane % 8)) * BK + (lane / 8) * 8) * sizeof(nv_bf16));
      const uint32_t B0[2] = {b4[0], b4[1]};
      const uint32_t B1[2] = {b4[2], b4[3]};
#pragma unroll
      for (int mi = 0; mi < M_TILES; ++mi) {
        mma_m16n8k16(A[mi][0], B0, tile_acc[mi][n]);
        mma_m16n8k16(A[mi][1], B1, tile_acc[mi][n]);
      }
    }
    __syncthreads();

    // Leaf complete: one BF16 round, then merge up the mid-split tree.
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

  // Epilogue: slice n holds gate column (n * 8 + (lane % 4) * 2) and slice
  // n + 4 holds the matching up column, both in this thread. i = 0 is the
  // fragment's first row, i = 1 the row eight below it.
#pragma unroll
  for (int mi = 0; mi < M_TILES; ++mi) {
    const int row = row_base + warp * WARP_M + mi * MMA_M + lane / 4;
#pragma unroll
    for (int n = 0; n < N_TILES / 2; ++n) {
      const int col = col_base + n * MMA_N + (lane % 4) * 2;
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int r = row + i * 8;
        if (r >= T) continue;
        const __nv_bfloat162 gate = tree_v[mi][n][i];
        const __nv_bfloat162 up = tree_v[mi][n + N_TILES / 2][i];
        const nv_bf16 h0 = swiglu_shared(__low2float(gate), __low2float(up));
        const nv_bf16 h1 = swiglu_shared(__high2float(gate), __high2float(up));
        *reinterpret_cast<__nv_bfloat162*>(h + static_cast<int64_t>(r) * F + col) =
            __halves2bfloat162(h0, h1);
      }
    }
  }
}

inline void init_tmap_bf16(CUtensorMap* tmap,
                           const nv_bf16* gmem,
                           uint64_t rows,
                           uint64_t cols,
                           uint32_t box_rows,
                           uint32_t box_cols) {
  uint64_t size[2] = {cols, rows};
  uint64_t stride[1] = {cols * sizeof(nv_bf16)};
  uint32_t box[2] = {box_cols, box_rows};
  uint32_t estride[2] = {1, 1};
  const CUresult res = cuTensorMapEncodeTiled(
      tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<nv_bf16*>(gmem), size, stride, box,
      estride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(res == CUDA_SUCCESS, "fused_shared_expert_mlp: cuTensorMapEncodeTiled failed (", res,
              ")");
}

int sm_major_of(const torch::Tensor& t) {
  return at::cuda::getDeviceProperties(t.get_device())->major;
}

#endif  // RL_KERNEL_ENABLE_SM90

void check_bf16_2d(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.dim() == 2, name, " must be 2-D");
  TORCH_CHECK(t.scalar_type() == at::kBFloat16, name, " must be BF16, got ", t.scalar_type());
}

}  // namespace

// Fused shared-expert fc1 + SwiGLU: x [T, H] BF16, w_fc1 [2F, H] BF16 -> h [T, F] BF16.
torch::Tensor fused_shared_expert_fc1_swiglu(torch::Tensor x, torch::Tensor w_fc1) {
  check_bf16_2d(x, "x");
  check_bf16_2d(w_fc1, "w_fc1");
  TORCH_CHECK(x.device() == w_fc1.device(), "x and w_fc1 must be on the same device");
  const int64_t T = x.size(0);
  const int64_t H = x.size(1);
  TORCH_CHECK(w_fc1.size(1) == H, "w_fc1 K mismatch: x has H=", H, ", w_fc1 has ", w_fc1.size(1));
  TORCH_CHECK(w_fc1.size(0) % 2 == 0, "w_fc1 must have an even number of rows (gate | up)");
  const int64_t F = w_fc1.size(0) / 2;
  TORCH_CHECK(H > 0 && H % K_TREE_LEAF == 0, "H must be a positive multiple of ", K_TREE_LEAF,
              ", got ", H);
  TORCH_CHECK(T < (1LL << 31) && F < (1LL << 31), "T and F must fit in int32");

  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  auto h = torch::empty({T, F}, x.options());
  if (T == 0 || F == 0) return h;

  auto stream = at::cuda::getCurrentCUDAStream();
  const auto* x_ptr = reinterpret_cast<const nv_bf16*>(x.data_ptr<at::BFloat16>());
  const auto* w_ptr = reinterpret_cast<const nv_bf16*>(w_fc1.data_ptr<at::BFloat16>());
  auto* h_ptr = reinterpret_cast<nv_bf16*>(h.data_ptr<at::BFloat16>());

#if defined(RL_KERNEL_ENABLE_SM90)
  // The tensor-core tile needs 32 h-columns per CTA. M is handled by TMA's
  // zero fill plus a masked store, so any T takes the same kernel -- selecting
  // on T would itself break batch invariance.
  if (sm_major_of(x) >= 9 && F % H_COLS == 0) {
    CUtensorMap x_tmap, w_tmap;
    init_tmap_bf16(&x_tmap, x_ptr, static_cast<uint64_t>(T), static_cast<uint64_t>(H), BM, BK);
    init_tmap_bf16(&w_tmap, w_ptr, static_cast<uint64_t>(2 * F), static_cast<uint64_t>(H), H_COLS,
                   BK);
    const int smem = STAGES * (BM * BK + BN * BK) * sizeof(nv_bf16) + STAGES * 8;
    static bool attr_set = false;
    if (!attr_set) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(fused_shared_expert_sm90,
                                          cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
      attr_set = true;
    }
    dim3 grid(cdiv(static_cast<int>(F), H_COLS), cdiv(static_cast<int>(T), BM));
    fused_shared_expert_sm90<<<grid, WG_THREADS, smem, stream>>>(
        x_tmap, w_tmap, h_ptr, static_cast<int>(T), static_cast<int>(H), static_cast<int>(F));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return h;
  }
#endif

  dim3 block(SCALAR_TILE, SCALAR_TILE);
  dim3 grid(cdiv(static_cast<int>(F), SCALAR_TILE), cdiv(static_cast<int>(T), SCALAR_TILE));
  fused_shared_expert_scalar<<<grid, block, 0, stream>>>(
      x_ptr, w_ptr, h_ptr, static_cast<int>(T), static_cast<int>(H), static_cast<int>(F));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return h;
}
