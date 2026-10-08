// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// MLP down projection on the portable fp32 CUDA-core path: order
// ``mlp-down-gemm-tree``.
//
// This file is the general NVIDIA path of the row: what runs when the Hopper
// TMA + wgmma kernel (``mlp_down_gemm_sm90.cu``, order
// ``mlp-down-gemm-mma``) cannot serve the contraction, and on every
// pre-Hopper device. It implements the reduction *tree* of the independent fp32
// CPU reference in ``rl_engine/kernels/ops/pytorch/linear/mlp_down_gemm.py``
// literally -- ``tree_gemm``, ``left_fold_weight_gradient`` and
// ``left_fold_bias_gradient`` are the definition -- so every output byte is
// equal to ``mlp_down_gemm_reference_forward`` /
// ``mlp_down_gemm_reference_backward``. A tree cannot use tensor cores, so the
// price is fp32 CUDA-core throughput (tens of TFLOP/s) instead of the Hopper
// path's ~460 TFLOP/s; the bytes, not the speed, are the point.
//
// The order, frozen:
//
//   * the reduction length R splits into 32-wide leaves (the last may be
//     short, the missing k being ``+0.0``); each leaf is an ascending-k chain
//     from ``+0.0`` in which every multiply-into-add is one correctly-rounded
//     fp32 FMA (``__fmaf_rn``);
//   * leaves combine through the mid-split tree ``T(l, r) = T(l, m) + T(m, r)``
//     with ``m = l + (r - l) / 2``, every combine a single correctly-rounded
//     fp32 add, so the association depends only on R;
//   * bias is added once, in fp32, after the complete tree;
//   * exactly one fp32 -> bf16 round-to-nearest-even cast happens, at the store;
//   * ``dx`` is the same tree with the operands swapped (the reduction runs
//     over N, so its leaf count is ``ceil(N / 32)``);
//   * ``dW`` and ``db`` are the reference's ascending-row fp32 left folds, one
//     ``__fmaf_rn`` per row (``db`` is therefore identical under both
//     orders, which is why the Hopper path shares this entry point).
//
// Shape of the tree kernel. One CTA owns a TILE_M x TILE_N output tile and
// each thread a TM x TN register block. The k dimension is walked a leaf at a
// time through a two-deep smem ring (both operands staged red-major, so the
// compute-side reads are the ones the mma kernel uses, just in fp32), and each
// leaf's 32-step chain runs entirely in registers -- exactly the reference's
// leaf -- after which the leaf is merged into a *level* array.
//
// The level array is what makes the mid-split tree exact without a data
// structure: for leaf i the reference recursion gives a depth ``d(i)`` (the
// depth of the leaf node) and a merge count ``m(i)`` (how many ancestors close
// when this leaf completes). With ``lvl[j]`` holding the pending partial of the
// left child of the node at depth j, the leaf is evaluated as
//
//   cur = leaf;  for j = d-1 down to d-m: cur = lvl[j] + cur;  lvl[d-m-1] = cur
//
// (the store is skipped when the root closes), which is a depth-first replay of
// ``T(l, r)`` for the leaves in ascending order -- bit-for-bit the reference's
// association. The live state is one partial per level, i.e. the tree *depth*
// (<= 12 here) rather than the leaf count, and the kernel is instantiated per
// needed depth so every ``lvl`` index is a compile-time constant.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <optional>

namespace {

using nv_bf16 = __nv_bfloat16;

constexpr int kLeafWidth = 32;  // LEAF_WIDTH in the fp32 reference

// The row's CUDA backend is gated to sm_80 and up (the wrapper applies the same
// gate before dispatching): fail closed instead of running a configuration the
// row does not validate.
bool mlp_tree_device_ok() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) {
        return false;
    }
    int major = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) {
        return false;
    }
    return major >= 8;
}

constexpr const char* kMlpTreeNeedsSm80 =
    "mlp_down_gemm's CUDA backend requires a compute capability >= 8.0 (sm80+) device; "
    "on older GPUs use the PyTorch reference backend";

// ---------------------------------------------------------------------------
// tile geometry
// ---------------------------------------------------------------------------
// One CTA owns TILE_M x TILE_N outputs; each thread owns TM x TN of them, the
// TN being contiguous so each k step is a single 16-byte smem read. 512 threads
// over a 64 x 64 tile, one leaf (32 k) staged per pipeline stage.
constexpr int TILE_M = 64;
constexpr int TILE_N = 64;
constexpr int TM = 2;
constexpr int TN = 4;
constexpr int N_OUT = TM * TN;           // outputs per thread
constexpr int TCOLS = TILE_N / TN;       // 16 threads per output row
constexpr int TROWS = TILE_M / TM;       // 32 thread rows
constexpr int THREADS = TROWS * TCOLS;   // 512
constexpr int BK = kLeafWidth;           // one leaf per stage
constexpr int STAGES = 2;
// Staged tile shape: both operands are red-major, i.e. sa[red][out].
// LD_A = TILE_M + 1: consecutive red land on consecutive banks, so the staged
// stores and the compute-side scalar reads are conflict free.
// LD_B = TILE_N + 4: 16-byte aligned rows (the TN-wide read is one LDS.128).
// TCOLS == 16 puts the whole warp's column blocks in one contiguous 256-byte
// run, so that LDS.128 is conflict free. Measured (H100 PCIe, forward,
// K = 12288, N = 3072, S = 4096): TM/TN 2x4 and 4x4 run at the same ~11 TFLOP/s,
// while 4x8 or 8x4 -- which need TCOLS == 8, i.e. a 32-byte column stride, or
// only 4 warps per CTA -- fall to ~4.8 TFLOP/s.
constexpr int LD_A = TILE_M + 1;
constexpr int LD_B = TILE_N + 4;
constexpr int MAX_LEAVES = 4096;  // 12 tree levels; longer reductions fail closed

// ---------------------------------------------------------------------------
// the frozen mid-split tree schedule
// ---------------------------------------------------------------------------
// Depth of leaf ``leaf`` in the reference recursion and the number of merges
// that close when it completes. This is the reference's
// ``merge(lo, hi)`` walk with ``mid = lo + (hi - lo) / 2``; the historical
// kernel in git history computes the same merge count, and the pair is checked
// against ``tree_gemm`` for every leaf count the suite uses.
__host__ __device__ __forceinline__ void tree_leaf_schedule(int leaf, int leaves, int& depth,
                                                            int& merges) {
    int lo = 0, hi = leaves, d = 0, m = 0;
    while (hi - lo > 1) {
        const int mid = lo + (hi - lo) / 2;
        ++d;
        if (leaf < mid) {
            hi = mid;
            m = 0;
        } else {
            lo = mid;
            ++m;
        }
    }
    depth = d;
    merges = m;
}

// ceil(log2(leaves)), i.e. the deepest leaf node; 1 for a single leaf (which
// needs no level slot at all, but keeps the template instantiations simple).
__host__ __device__ __forceinline__ int tree_levels(int leaves) {
    int d = 0;
    while ((1 << d) < leaves) {
        ++d;
    }
    return d < 1 ? 1 : d;
}

__host__ __device__ __forceinline__ int leaves_of(long reduction) {
    return static_cast<int>((reduction + kLeafWidth - 1) / kLeafWidth);
}

// One leaf's chain: ``cur`` starts from +0.0 and every multiply-into-add is one
// correctly-rounded fp32 FMA, over the leaf's k values in ascending order --
// exactly the reference's leaf. ``KEND`` is the compile-time chain length when
// the leaf is full (a fully unrolled, software-pipelined body) and 0 when the
// tail is short, where the bound is the runtime ``klen``. A short tail leaf
// simply stops: it does not add padding zeros, which would normalise a -0.0
// accumulator the reference keeps.
template <int KEND>
__device__ __forceinline__ void tree_leaf_chain(const float* __restrict__ sa_c,
                                                const float* __restrict__ sb_c, int klen,
                                                int m_thr, int n_thr, float (&cur)[N_OUT]) {
    const int end = (KEND > 0) ? KEND : klen;
#pragma unroll
    for (int r = 0; r < N_OUT; ++r) {
        cur[r] = 0.0f;
    }
#pragma unroll
    for (int kk = 0; kk < end; ++kk) {
        const float* const arow = sa_c + kk * LD_A + m_thr;
        const float* const brow = sb_c + kk * LD_B + n_thr;
        float av[TM];
        float bv[TN];
#pragma unroll
        for (int mi = 0; mi < TM; ++mi) {
            av[mi] = arow[mi];
        }
#pragma unroll
        for (int q = 0; q < TN / 4; ++q) {
            const float4 b4 = *reinterpret_cast<const float4*>(brow + 4 * q);
            bv[4 * q + 0] = b4.x;
            bv[4 * q + 1] = b4.y;
            bv[4 * q + 2] = b4.z;
            bv[4 * q + 3] = b4.w;
        }
#pragma unroll
        for (int mi = 0; mi < TM; ++mi) {
#pragma unroll
            for (int rn = 0; rn < TN; ++rn) {
                cur[mi * TN + rn] = __fmaf_rn(av[mi], bv[rn], cur[mi * TN + rn]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// the tree kernel: out = single_bf16_cast(tree(A @ B^T) + bias)
// ---------------------------------------------------------------------------
// A is [M, R] with R contiguous (a_stride = R). B is either [N, R] with R
// contiguous (B_OUT_CONTIG == false, the forward's weight [N, K]) or [R, N] with
// N contiguous (B_OUT_CONTIG == true, dx's weight [N, K] read as [red][out]).
// LVL is the templated tree depth (``lvl`` must be constant-indexed to stay in
// registers).
template <int LVL>
__global__ void __launch_bounds__(THREADS) mlp_tree_kernel(
    const float* __restrict__ gA, const float* __restrict__ gB, const float* __restrict__ bias,
    nv_bf16* __restrict__ out, int M, int N, long R, int leaves, long a_stride, long b_stride,
    int b_out_contig) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    float* const smem = reinterpret_cast<float*>(smem_raw);
    float* const sa = smem;                       // [STAGES][BK][LD_A]
    float* const sb = sa + STAGES * BK * LD_A;    // [STAGES][BK][LD_B]
    // (depth << 4) | merges per leaf, built once by the block.
    __shared__ unsigned char s_sched[MAX_LEAVES];

    const int tid = threadIdx.x;
    const int trow = tid / TCOLS;
    const int tcol = tid % TCOLS;
    const int m_thr = trow * TM;
    const int n_thr = tcol * TN;
    const int m_base = blockIdx.y * TILE_M;
    const int n_base = blockIdx.x * TILE_N;

    for (int leaf = tid; leaf < leaves; leaf += THREADS) {
        int d = 0, m = 0;
        tree_leaf_schedule(leaf, leaves, d, m);
        s_sched[leaf] = static_cast<unsigned char>((d << 4) | m);
    }

    // Stage one leaf of both operands into ring slot ``slot``.
    auto stage = [&](int leaf, int slot) {
        const long k0 = static_cast<long>(leaf) * BK;
        float* const sa_s = sa + slot * (BK * LD_A);
        float* const sb_s = sb + slot * (BK * LD_B);
        // A: [out][red], red contiguous.
        for (int i = tid; i < TILE_M * BK; i += THREADS) {
            const int rr = i / BK, cc = i % BK;
            const long gout = m_base + rr, gred = k0 + cc;
            float v = 0.0f;
            if (gout < M && gred < R) {
                v = gA[gout * a_stride + gred];
            }
            sa_s[cc * LD_A + rr] = v;
        }
        if (b_out_contig) {
            // B: [red][out], out contiguous; staged contiguous in out.
            for (int i = tid; i < BK * TILE_N; i += THREADS) {
                const int rr = i / TILE_N, cc = i % TILE_N;
                const long gred = k0 + rr, gout = n_base + cc;
                float v = 0.0f;
                if (gred < R && gout < N) {
                    v = gB[gred * b_stride + gout];
                }
                sb_s[rr * LD_B + cc] = v;
            }
        } else {
            // B: [out][red], red contiguous; staged red-major (a transposed
            // write, conflict free with LD_B % 32 == 4).
            for (int i = tid; i < TILE_N * BK; i += THREADS) {
                const int rr = i / BK, cc = i % BK;
                const long gout = n_base + rr, gred = k0 + cc;
                float v = 0.0f;
                if (gout < N && gred < R) {
                    v = gB[gout * b_stride + gred];
                }
                sb_s[cc * LD_B + rr] = v;
            }
        }
    };

    // lvl[j]: pending partial of the left child of the node at depth j; cur:
    // the leaf chain and then the partial being merged upward.
    float lvl[LVL][N_OUT];
    float cur[N_OUT];
#pragma unroll
    for (int j = 0; j < LVL; ++j) {
#pragma unroll
        for (int r = 0; r < N_OUT; ++r) {
            lvl[j][r] = 0.0f;
        }
    }
#pragma unroll
    for (int r = 0; r < N_OUT; ++r) {
        cur[r] = 0.0f;
    }

    if (leaves > 0) {
        stage(0, 0);
    }
    __syncthreads();

    for (int leaf = 0; leaf < leaves; ++leaf) {
        const int slot = leaf & 1;
        if (leaf + 1 < leaves) {
            stage(leaf + 1, slot ^ 1);
        }

        const float* const sa_c = sa + slot * (BK * LD_A);
        const float* const sb_c = sb + slot * (BK * LD_B);
        const long k0 = static_cast<long>(leaf) * BK;
        const int klen = static_cast<int>(R - k0 < BK ? R - k0 : BK);

        // Every leaf but a short tail runs the fully unrolled chain.
        if (klen == BK) {
            tree_leaf_chain<BK>(sa_c, sb_c, klen, m_thr, n_thr, cur);
        } else {
            tree_leaf_chain<0>(sa_c, sb_c, klen, m_thr, n_thr, cur);
        }

        // Replay the tree: merge at depths d-1 .. d-m (descending, so the adds
        // are the reference's T(l, m) + T(m, r) order), then publish the
        // completed node at depth d-m-1. All of it is CTA-uniform.
        const int sched = s_sched[leaf];
        const int depth = sched >> 4;
        const int merges = sched & 0xf;
#pragma unroll
        for (int j = LVL - 1; j >= 0; --j) {
            if (j >= depth) {
                continue;
            }
            if (j >= depth - merges) {
#pragma unroll
                for (int r = 0; r < N_OUT; ++r) {
                    cur[r] = lvl[j][r] + cur[r];
                }
                continue;
            }
#pragma unroll
            for (int r = 0; r < N_OUT; ++r) {
                lvl[j][r] = cur[r];
            }
            break;
        }
        __syncthreads();
    }

    // Epilogue: bias once in fp32 after the complete tree, then the single RNE
    // bf16 cast at the store.
#pragma unroll
    for (int mi = 0; mi < TM; ++mi) {
        const long row = m_base + m_thr + mi;
#pragma unroll
        for (int rn = 0; rn < TN; ++rn) {
            const long col = n_base + n_thr + rn;
            if (row < M && col < N) {
                float v = cur[mi * TN + rn];
                if (bias != nullptr) {
                    v = v + bias[col];
                }
                out[row * N + col] = __float2bfloat16(v);
            }
        }
    }
}

template <int LVL>
void launch_tree(const float* a, const float* b, const float* bias, nv_bf16* out, int M, int N,
                 long R, int leaves, long a_stride, long b_stride, int b_out_contig,
                 cudaStream_t stream) {
    if (M == 0 || N == 0) {
        return;
    }
    const dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    const int smem_bytes = STAGES * (BK * LD_A + BK * LD_B) * static_cast<int>(sizeof(float));
    mlp_tree_kernel<LVL><<<grid, THREADS, smem_bytes, stream>>>(a, b, bias, out, M, N, R, leaves,
                                                                a_stride, b_stride, b_out_contig);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Dispatch on the tree depth the reduction needs: ``lvl`` is a register array,
// so its size has to be a template constant (the unused levels would otherwise
// occupy registers for every shape).
void launch_tree_dispatch(const float* a, const float* b, const float* bias, nv_bf16* out, int M,
                          int N, long R, long a_stride, long b_stride, int b_out_contig,
                          cudaStream_t stream) {
    const int leaves = leaves_of(R);
    TORCH_CHECK(leaves <= MAX_LEAVES, "mlp_down_gemm: reduction length ", R,
                " exceeds the frozen tree depth (", MAX_LEAVES, " leaves)");
    switch (tree_levels(leaves)) {
        case 1:
            launch_tree<1>(a, b, bias, out, M, N, R, leaves, a_stride, b_stride, b_out_contig,
                           stream);
            break;
        case 2:
            launch_tree<2>(a, b, bias, out, M, N, R, leaves, a_stride, b_stride, b_out_contig,
                           stream);
            break;
        case 3:
            launch_tree<3>(a, b, bias, out, M, N, R, leaves, a_stride, b_stride, b_out_contig,
                           stream);
            break;
        case 4:
            launch_tree<4>(a, b, bias, out, M, N, R, leaves, a_stride, b_stride, b_out_contig,
                           stream);
            break;
        case 5:
            launch_tree<5>(a, b, bias, out, M, N, R, leaves, a_stride, b_stride, b_out_contig,
                           stream);
            break;
        case 6:
            launch_tree<6>(a, b, bias, out, M, N, R, leaves, a_stride, b_stride, b_out_contig,
                           stream);
            break;
        case 7:
            launch_tree<7>(a, b, bias, out, M, N, R, leaves, a_stride, b_stride, b_out_contig,
                           stream);
            break;
        case 8:
            launch_tree<8>(a, b, bias, out, M, N, R, leaves, a_stride, b_stride, b_out_contig,
                           stream);
            break;
        case 9:
            launch_tree<9>(a, b, bias, out, M, N, R, leaves, a_stride, b_stride, b_out_contig,
                           stream);
            break;
        case 10:
            launch_tree<10>(a, b, bias, out, M, N, R, leaves, a_stride, b_stride, b_out_contig,
                            stream);
            break;
        case 11:
            launch_tree<11>(a, b, bias, out, M, N, R, leaves, a_stride, b_stride, b_out_contig,
                            stream);
            break;
        default:
            launch_tree<12>(a, b, bias, out, M, N, R, leaves, a_stride, b_stride, b_out_contig,
                            stream);
            break;
    }
}

// ---------------------------------------------------------------------------
// dW: the reference's ascending-row fp32 left fold
// ---------------------------------------------------------------------------
// dw[n, k] = fold over rows s ascending of fma(grad[s, n], x[s, k], acc).
// One CTA owns a [DW_TN, DW_TK] tile of dW and each thread a DW_N x DW_K
// register block; the grad/x rows of a chunk are staged in smem so every global
// element is read once per tile. The row order -- the whole reduction -- is the
// staging chunk order, and it is strictly ascending.
constexpr int DW_TN = 64;
constexpr int DW_TK = 64;
constexpr int DW_SC = 32;  // rows staged per round
constexpr int DW_N = 4;
constexpr int DW_K = 4;
constexpr int DW_THREADS = (DW_TN / DW_N) * (DW_TK / DW_K);  // 256

__global__ void mlp_down_dw_left_fold_kernel(const float* __restrict__ grad,
                                             const float* __restrict__ x,
                                             nv_bf16* __restrict__ dw, long rows, long out_dim,
                                             long in_dim) {
    __shared__ float sg[DW_SC][DW_TN + 1];
    __shared__ float sx[DW_SC][DW_TK + 1];

    const int tid = threadIdx.x;
    const int tn = (tid % (DW_TK / DW_K)) * DW_K;
    const int tn2 = (tid / (DW_TK / DW_K)) * DW_N;
    const long n_base = static_cast<long>(blockIdx.x) * DW_TN;
    const long k_base = static_cast<long>(blockIdx.y) * DW_TK;

    float acc[DW_N][DW_K];
#pragma unroll
    for (int i = 0; i < DW_N; ++i) {
#pragma unroll
        for (int j = 0; j < DW_K; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for (long s0 = 0; s0 < rows; s0 += DW_SC) {
        __syncthreads();
        for (int i = tid; i < DW_SC * DW_TN; i += DW_THREADS) {
            const int sr = i / DW_TN;
            const int nn = i % DW_TN;
            const long gs = s0 + sr;
            const long gn = n_base + nn;
            sg[sr][nn] =
                (gs < rows && gn < out_dim) ? grad[gs * out_dim + gn] : 0.0f;
        }
        for (int i = tid; i < DW_SC * DW_TK; i += DW_THREADS) {
            const int sr = i / DW_TK;
            const int kk = i % DW_TK;
            const long gs = s0 + sr;
            const long gk = k_base + kk;
            sx[sr][kk] = (gs < rows && gk < in_dim) ? x[gs * in_dim + gk] : 0.0f;
        }
        __syncthreads();

        const long s_end = (s0 + DW_SC < rows) ? DW_SC : (rows - s0);
        for (long sr = 0; sr < s_end; ++sr) {  // ascending rows: the left fold
            float gv[DW_N];
            float xv[DW_K];
#pragma unroll
            for (int i = 0; i < DW_N; ++i) {
                gv[i] = sg[sr][tn2 + i];
            }
#pragma unroll
            for (int j = 0; j < DW_K; ++j) {
                xv[j] = sx[sr][tn + j];
            }
#pragma unroll
            for (int i = 0; i < DW_N; ++i) {
#pragma unroll
                for (int j = 0; j < DW_K; ++j) {
                    acc[i][j] = __fmaf_rn(gv[i], xv[j], acc[i][j]);
                }
            }
        }
    }

#pragma unroll
    for (int i = 0; i < DW_N; ++i) {
        const long gn = n_base + tn2 + i;
#pragma unroll
        for (int j = 0; j < DW_K; ++j) {
            const long gk = k_base + tn + j;
            if (gn < out_dim && gk < in_dim) {
                dw[gn * in_dim + gk] = __float2bfloat16(acc[i][j]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// db: the ascending-row fp32 left fold of the gradient, over columns
// ---------------------------------------------------------------------------
// Shared with the Hopper path, which is why this entry point is not part of
// either tree/mma split: db[n] = fold over rows ascending of grad[row][n], one
// correctly-rounded fp32 add per row (the reference's
// ``left_fold_bias_gradient``), so it is byte-equal under both orders.
//
// One thread per column would launch 3072 threads and walk a 12 KB-strided
// column each, which is latency-bound; instead the block stages row tiles
// through shared memory with coalesced 16-byte reads and feeds the per-column
// chains from smem. The arithmetic (single running fp32 accumulator per column,
// ascending row order) is untouched. bf16 -> fp32 is exact, so reading the bf16
// gradient directly is bit-identical to converting first, at half the bytes.
constexpr int DB_CT = 64;     // output columns per block
constexpr int DB_TR = 256;    // rows staged per tile
constexpr int DB_THREADS = 256;

__global__ void __launch_bounds__(DB_THREADS, 2)
mlp_down_db_fold_wide_kernel(const nv_bf16* __restrict__ grad, float* __restrict__ db, int rows,
                             int cols) {
    extern __shared__ nv_bf16 db_tile[];
    const int n0 = blockIdx.x * DB_CT;
    const int tid = threadIdx.x;
    const int ncols = (cols - n0 < DB_CT) ? (cols - n0) : DB_CT;
    if (ncols <= 0) {
        return;
    }
    const int ntiles = (rows + DB_TR - 1) / DB_TR;
    // A 16-byte vector read needs the row start to be 8 elements aligned.
    const bool vec_ok = (cols % 8 == 0) && (n0 % 8 == 0);

    auto load = [&](int t, nv_bf16* buf) {
        const int r0 = t * DB_TR;
        const int nvalid = (rows - r0 < DB_TR) ? (rows - r0) : DB_TR;
#pragma unroll
        for (int k = 0; k < DB_TR * DB_CT / 8 / DB_THREADS; ++k) {
            const int flat = (k * DB_THREADS + tid) * 8;
            const int c = flat % DB_CT;
            const int r = flat / DB_CT;
            nv_bf16* dst = buf + flat;
            if (vec_ok && r < nvalid && c + 8 <= ncols) {
                *reinterpret_cast<uint4*>(dst) =
                    *reinterpret_cast<const uint4*>(&grad[(long)(r0 + r) * cols + n0 + c]);
            } else {
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    dst[e] = (r < nvalid && c + e < ncols)
                                 ? grad[(long)(r0 + r) * cols + n0 + c + e]
                                 : __float2bfloat16(0.0f);
                }
            }
        }
    };

    float acc = 0.0f;
    const bool owner = tid < ncols;
    if (ntiles > 0) {
        load(0, db_tile);
    }
    __syncthreads();
    for (int t = 0; t < ntiles; ++t) {
        // Double buffered: the next tile streams in while this one folds.
        nv_bf16* cur = db_tile + (t & 1) * (DB_TR * DB_CT);
        if (t + 1 < ntiles) {
            load(t + 1, db_tile + ((t & 1) ^ 1) * (DB_TR * DB_CT));
        }
        if (owner) {
            const int nr = (rows - t * DB_TR < DB_TR) ? (rows - t * DB_TR) : DB_TR;
            const nv_bf16* col = cur + tid;
            int r = 0;
            // Ascending, one fp32 add per row, single chain: the frozen rule.
            for (; r + 8 <= nr; r += 8) {
                float v[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    v[i] = __bfloat162float(col[(r + i) * DB_CT]);
                }
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    acc = acc + v[i];
                }
            }
            for (; r < nr; ++r) {
                acc = acc + __bfloat162float(col[r * DB_CT]);
            }
        }
        __syncthreads();
    }
    if (owner) {
        db[n0 + tid] = acc;
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// entry points (names and signatures unchanged across the order split)
// ---------------------------------------------------------------------------

torch::Tensor mlp_down_gemm_cuda_forward(torch::Tensor x, torch::Tensor weight,
                                        std::optional<torch::Tensor> bias) {
    const c10::cuda::CUDAGuard guard(x.device());
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "x and weight must be CUDA tensors");
    TORCH_CHECK(mlp_tree_device_ok(), kMlpTreeNeedsSm80);
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && weight.scalar_type() == at::kBFloat16,
                "mlp-down-gemm-tree requires bf16 operands");
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 2, "x and weight must be 2-D");
    TORCH_CHECK(x.size(1) == weight.size(1), "x K must match weight K");
    const int64_t S = x.size(0);
    const int64_t N = weight.size(0);
    const int64_t K = x.size(1);
    auto out = torch::empty({S, N}, x.options());
    if (out.numel() == 0) {
        return out;
    }
    // The tree is fp32 arithmetic; the bf16 operands convert exactly.
    auto xf = x.to(at::kFloat).contiguous();
    auto wf = weight.to(at::kFloat).contiguous();
    const float* bias_ptr = nullptr;
    torch::Tensor bias_f;
    if (bias.has_value()) {
        TORCH_CHECK(bias->numel() == N, "bias must have N elements");
        TORCH_CHECK(bias->device() == x.device(), "bias must live on x's device");
        bias_f = bias->to(at::kFloat).contiguous();
        bias_ptr = bias_f.data_ptr<float>();
    }
    launch_tree_dispatch(xf.data_ptr<float>(), wf.data_ptr<float>(), bias_ptr,
                         reinterpret_cast<nv_bf16*>(out.data_ptr<at::BFloat16>()),
                         static_cast<int>(S), static_cast<int>(N), K, K, K, /*b_out_contig=*/0,
                         at::cuda::getCurrentCUDAStream());
    return out;
}

torch::Tensor mlp_down_gemm_cuda_dx(torch::Tensor g, torch::Tensor weight) {
    const c10::cuda::CUDAGuard guard(g.device());
    TORCH_CHECK(g.is_cuda() && weight.is_cuda(), "g and weight must be CUDA tensors");
    TORCH_CHECK(mlp_tree_device_ok(), kMlpTreeNeedsSm80);
    TORCH_CHECK(g.scalar_type() == at::kBFloat16 && weight.scalar_type() == at::kBFloat16,
                "mlp-down-gemm-tree requires bf16 operands");
    TORCH_CHECK(g.dim() == 2 && weight.dim() == 2, "g and weight must be 2-D");
    TORCH_CHECK(g.size(1) == weight.size(0), "g N must match weight N");
    const int64_t S = g.size(0);
    const int64_t N = g.size(1);  // reduction length for dx
    const int64_t K = weight.size(1);
    auto out = torch::empty({S, K}, g.options());
    if (out.numel() == 0) {
        return out;
    }
    // dx is the same tree with the operands swapped: A = grad [S, N] (reduction
    // contiguous), B = weight [N, K] read as [red][out] (out contiguous).
    auto gf = g.to(at::kFloat).contiguous();
    auto wf = weight.to(at::kFloat).contiguous();
    launch_tree_dispatch(gf.data_ptr<float>(), wf.data_ptr<float>(), nullptr,
                         reinterpret_cast<nv_bf16*>(out.data_ptr<at::BFloat16>()),
                         static_cast<int>(S), static_cast<int>(K), N, N, K, /*b_out_contig=*/1,
                         at::cuda::getCurrentCUDAStream());
    return out;
}

torch::Tensor mlp_down_gemm_cuda_dw(torch::Tensor g, torch::Tensor x) {
    const c10::cuda::CUDAGuard guard(g.device());
    TORCH_CHECK(g.is_cuda() && x.is_cuda(), "g and x must be CUDA tensors");
    TORCH_CHECK(mlp_tree_device_ok(), kMlpTreeNeedsSm80);
    TORCH_CHECK(g.scalar_type() == at::kBFloat16 && x.scalar_type() == at::kBFloat16,
                "mlp-down-gemm-tree requires bf16 operands");
    TORCH_CHECK(g.dim() == 2 && x.dim() == 2, "g and x must be 2-D");
    TORCH_CHECK(g.size(0) == x.size(0), "g and x must share the row dim");
    const int64_t rows = g.size(0);
    const int64_t out_dim = g.size(1);
    const int64_t in_dim = x.size(1);
    auto dw = torch::empty({out_dim, in_dim}, g.options());
    const long total = static_cast<long>(out_dim) * in_dim;
    if (total == 0 || rows == 0) {
        if (total != 0) {
            dw.zero_();
        }
        return dw;
    }
    auto gf = g.to(at::kFloat).contiguous();
    auto xf = x.to(at::kFloat).contiguous();
    const dim3 grid((out_dim + DW_TN - 1) / DW_TN, (in_dim + DW_TK - 1) / DW_TK);
    mlp_down_dw_left_fold_kernel<<<grid, DW_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        gf.data_ptr<float>(), xf.data_ptr<float>(),
        reinterpret_cast<nv_bf16*>(dw.data_ptr<at::BFloat16>()), rows, out_dim, in_dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return dw;
}

// Bias gradient: ascending-row left fold over the bf16 gradient, fp32 out
// (identical under both orders).
torch::Tensor mlp_down_gemm_cuda_db(torch::Tensor grad) {
    const c10::cuda::CUDAGuard guard(grad.device());
    TORCH_CHECK(grad.is_cuda(), "grad must be a CUDA tensor");
    TORCH_CHECK(grad.scalar_type() == at::kBFloat16, "mlp_down_gemm_cuda_db requires bf16 operands");
    const int64_t cols = grad.size(-1);
    const int64_t rows = grad.numel() / cols;
    auto db = torch::empty({cols}, grad.options().dtype(at::kFloat));
    if (db.numel() == 0 || rows == 0) {
        if (db.numel() != 0) {
            db.zero_();
        }
        return db;
    }
    auto gc = grad.contiguous();
    const int blocks = static_cast<int>((cols + DB_CT - 1) / DB_CT);
    const int smem = 2 * DB_TR * DB_CT * static_cast<int>(sizeof(nv_bf16));
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(mlp_down_db_fold_wide_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        configured = true;
    }
    mlp_down_db_fold_wide_kernel<<<blocks, DB_THREADS, smem, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const nv_bf16*>(gc.data_ptr<at::BFloat16>()), db.data_ptr<float>(),
        static_cast<int>(rows), static_cast<int>(cols));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return db;
}
