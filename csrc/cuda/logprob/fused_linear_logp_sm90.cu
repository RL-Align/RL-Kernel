// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// Hopper (SM90) fused linear log-prob:
//     logp[n] = log_softmax(hidden[n] @ W^T + b)[target[n]]

#include "../utils/tma_utils.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <mma.h>
#include <string>
#include <torch/extension.h>

namespace {

namespace wmma = nvcuda::wmma;

constexpr int BM = 256;       // tokens per CTA
constexpr int BN = 64;        // vocab per tile
constexpr int BK = 32;        // hidden-dim slice streamed per TMA load
constexpr int WARPS = 4;      // one warpgroup
constexpr int WG_THREADS = WARPS * 32; // 128
constexpr int STAGES = 2;     // double-buffering

// ---------------------------------------------------------------------------
// Bitwise reduction contract (bit-relevant constants; changes require a
// contract-version bump):
//   * VOCAB tile width .......... BN  (64)   - which columns share a tile
//   * K slab width .............. BK  (32)   - GEMM accumulation order
//   * vocab split count ......... N_SPLIT_CONTRACT (64), a function of V only
//   * in-tile sum ............... adjacent-pairwise balanced binary tree over
//                                 BN registers (one add per level)
//   * cross-tile / cross-split .. ascending-index sequential scalar chains
//   * row max ................... successive fmaxf (order-free)
//   * transcendentals ........... expf / logf (IEEE), never __expf
//   * padding lanes ............. -inf to max, exp() -> exact 0 to sum
//   * temperature ............... scales stats and selected logit in the
//                                 epilogue, never the stored logits
//   * final clamp ............... logp = min(zt - lse, 0)
// Launch axes that only move work along M (row blocks, warps, stages) are
// bit-neutral and may be tuned freely.
// ---------------------------------------------------------------------------
constexpr int N_SPLIT_CONTRACT = 64;

constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;

constexpr int WARP_M = BM / WARPS;       // logit rows per warp
constexpr int M_TILES = WARP_M / MMA_M;  // MMA m-tiles each warp owns
constexpr int N_TILES = BN / MMA_N;      // 8 n-tiles per warp
constexpr int K_TILES = BK / MMA_K;      // MMA k-steps per TMA tile
constexpr int KK_GROUPS = BK / 32;       // 32-wide ldmatrix.x4 groups (2 k-steps each)

constexpr int STREAM_BWD_THREADS = 256;
constexpr int STREAM_BWD_VECS = 4;
constexpr int STREAM_BWD_MAX_D = 4096;
constexpr int STREAM_BWD_MAX_SLOTS = STREAM_BWD_MAX_D / STREAM_BWD_THREADS;

constexpr int LOGITS_WMMA_M = 16;
constexpr int LOGITS_WMMA_N = 16;
constexpr int LOGITS_WMMA_K = 8;
constexpr int LOGITS_WMMA_WARPS = 4;
constexpr int LOGITS_WMMA_THREADS = LOGITS_WMMA_WARPS * 32;

constexpr int GRAD_W_WMMA_M = 16;
constexpr int GRAD_W_WMMA_N = 16;
constexpr int GRAD_W_WMMA_K = 16;
constexpr int GRAD_W_WMMA_WARPS = 8;
constexpr int GRAD_W_WMMA_THREADS = GRAD_W_WMMA_WARPS * 32;

static_assert(WARP_M % MMA_M == 0, "rows per warp must be a multiple of MMA_M");
static_assert(BK % 32 == 0, "BK must be a multiple of 32 (ldmatrix.x4 spans 32 cols)");

inline void init_tensor_map_noswizzle(CUtensorMap *tmap, const nv_bfloat16 *gmem,
                                      uint64_t gmem_height, uint64_t gmem_width,
                                      uint32_t box_height, uint32_t box_width);

// Tensor-core helpers (Ampere/Hopper warp-level MMA). Same layout as
// prefix_shared_attention.cu, validated on this repo's Hopper GPUs.
__device__ __forceinline__ void ldmatrix_x4(uint32_t regs[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
                 : "r"(addr));
}

// D[m16,n8] += A[m16,k16] * B[n8,k16]   (A row-major, B col-major; fp32 accum)
__device__ __forceinline__ void mma_m16n8k16(const uint32_t A[4], const uint32_t B[2],
                                             float D[4]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
                 : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
                 : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
                   "f"(D[0]), "f"(D[1]), "f"(D[2]), "f"(D[3]));
}

__global__ void fused_linear_logp_sm90_kernel(const __grid_constant__ CUtensorMap h_tmap,
                                              const __grid_constant__ CUtensorMap w_tmap,
                                              const int *__restrict__ target,
                                              const float *__restrict__ bias, // may be null
                                              const float *__restrict__ temperature, // may be null
                                              float *__restrict__ logits_out,  // may be null
                                              float *__restrict__ part_max,   // [n_split, N]
                                              float *__restrict__ part_sum,   // [n_split, N]
                                              float *__restrict__ part_zt,    // [n_split, N]
                                              int N, int D, int V, int n_split,
                                              int vocab_start_index, int real_vocab_end) {
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int row_block = blockIdx.x;
    const int split = blockIdx.y;
    const int row_base = row_block * BM;
    const int num_rows = min(BM, N - row_base);
    const int kd = D / BK; // D is validated to be a multiple of BK on the host

    // This CTA owns a contiguous slice of the vocab tiles (split-V): partitioning
    // the V loop across blockIdx.y fills the GPU when N/BM alone is too few CTAs.
    const int total_vtiles = (V + BN - 1) / BN;
    const int vtiles_per_split = (total_vtiles + n_split - 1) / n_split;
    const int vt_begin = split * vtiles_per_split;
    const int vt_end = min(vt_begin + vtiles_per_split, total_vtiles);

    extern __shared__ __align__(1024) char smem[];
    nv_bfloat16 *sH = reinterpret_cast<nv_bfloat16 *>(smem);
    nv_bfloat16 *sW = reinterpret_cast<nv_bfloat16 *>(sH + STAGES * BM * BK);
    float *sLogits = reinterpret_cast<float *>(sW + STAGES * BN * BK);
    float *sMax = sLogits + BM * BN;
    float *sSum = sMax + BM;
    float *sZt = sSum + BM;
    int *mbar_base = reinterpret_cast<int *>(sZt + BM); // STAGES mbarriers (8B each)

    const uint64_t sH_base_tma = __cvta_generic_to_shared(sH);
    const uint64_t sW_base_tma = __cvta_generic_to_shared(sW);
    const uint32_t sH_base = static_cast<uint32_t>(sH_base_tma);
    const uint32_t sW_base = static_cast<uint32_t>(sW_base_tma);
    // mbarrier PTX expects the 64-bit shared address, while ldmatrix below uses
    // the narrowed 32-bit shared address form.
    uint64_t mbar[STAGES];
#pragma unroll
    for (int s = 0; s < STAGES; ++s)
        mbar[s] = __cvta_generic_to_shared(mbar_base + 2 * s);

    for (int r = tid; r < num_rows; r += WG_THREADS) {
        sMax[r] = -CUDART_INF_F;
        sSum[r] = 0.0f;
        sZt[r] = 0.0f;
    }
    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; ++s)
            mbarrier_init(mbar[s], 1);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();

    const uint32_t tile_bytes = (BM * BK + BN * BK) * sizeof(nv_bfloat16);

    // Issue the TMA for D-slice k of vocab tile vt into buffer (k % STAGES).
    auto issue_load = [&](int k, int col_base) {
        const int buf = k % STAGES;
        const int k_off = k * BK;
        mbarrier_arrive_expect_tx(mbar[buf], tile_bytes);
        tma_2d_g2s(sH_base_tma + buf * BM * BK * sizeof(nv_bfloat16), &h_tmap, k_off, row_base,
                   mbar[buf]);
        tma_2d_g2s(sW_base_tma + buf * BN * BK * sizeof(nv_bfloat16), &w_tmap, k_off, col_base,
                   mbar[buf]);
    };

    int phase[STAGES];
#pragma unroll
    for (int s = 0; s < STAGES; ++s)
        phase[s] = 0;

    for (int vt = vt_begin; vt < vt_end; ++vt) {
        const int col_base = vt * BN;

        // Per-warp accumulators: this warp's M_TILES*16 rows x N_TILES n-tiles.
        float acc[M_TILES][N_TILES][4];
#pragma unroll
        for (int mi = 0; mi < M_TILES; ++mi)
#pragma unroll
            for (int n = 0; n < N_TILES; ++n)
                acc[mi][n][0] = acc[mi][n][1] = acc[mi][n][2] = acc[mi][n][3] = 0.0f;

        // Double Buffering: TMA loads in flight so the
        // next H/W slices stream in while the current one feeds tensor-core MMAs.
        if (tid == 0) {
#pragma unroll
            for (int s = 0; s < STAGES - 1; ++s)
                if (s < kd)
                    issue_load(s, col_base);
        }
        for (int k = 0; k < kd; ++k) {
            const int buf = k % STAGES;
            if (tid == 0 && k + (STAGES - 1) < kd)
                issue_load(k + (STAGES - 1), col_base); // overlaps with the MMAs below
            mbarrier_wait(mbar[buf], phase[buf]);
            phase[buf] ^= 1;
            __syncthreads();

            const uint32_t sH_buf = sH_base + buf * BM * BK * sizeof(nv_bfloat16);
            const uint32_t sW_buf = sW_base + buf * BN * BK * sizeof(nv_bfloat16);

            // Load A (this warp's M_TILES*16 rows) for every MMA k-step.
            uint32_t A[M_TILES][K_TILES][4];
#pragma unroll
            for (int mi = 0; mi < M_TILES; ++mi) {
                const int row0 = warp * WARP_M + mi * MMA_M + (lane % 16);
#pragma unroll
                for (int kt = 0; kt < K_TILES; ++kt) {
                    const uint32_t a_addr =
                        sH_buf + (row0 * BK + (lane / 16) * 8 + kt * MMA_K) * sizeof(nv_bfloat16);
                    ldmatrix_x4(A[mi][kt], a_addr);
                }
            }

            // Load B (all n-tiles, shared across m-tiles) and contract.
#pragma unroll
            for (int n = 0; n < N_TILES; ++n) {
#pragma unroll
                for (int kk = 0; kk < KK_GROUPS; ++kk) {
                    uint32_t b4[4];
                    const uint32_t b_addr =
                        sW_buf + ((n * MMA_N + (lane % 8)) * BK + (lane / 8) * 8 + kk * 32) *
                                     sizeof(nv_bfloat16);
                    ldmatrix_x4(b4, b_addr);
                    const uint32_t B0[2] = {b4[0], b4[1]};
                    const uint32_t B1[2] = {b4[2], b4[3]};
#pragma unroll
                    for (int mi = 0; mi < M_TILES; ++mi) {
                        mma_m16n8k16(A[mi][2 * kk + 0], B0, acc[mi][n]);
                        mma_m16n8k16(A[mi][2 * kk + 1], B1, acc[mi][n]);
                    }
                }
            }
            __syncthreads();
        }

#pragma unroll
        for (int mi = 0; mi < M_TILES; ++mi) {
            const int row = warp * WARP_M + mi * MMA_M + lane / 4;
#pragma unroll
            for (int n = 0; n < N_TILES; ++n) {
                const int col = n * MMA_N + (lane % 4) * 2;
                sLogits[row * BN + col + 0] = acc[mi][n][0];
                sLogits[row * BN + col + 1] = acc[mi][n][1];
                sLogits[(row + 8) * BN + col + 0] = acc[mi][n][2];
                sLogits[(row + 8) * BN + col + 1] = acc[mi][n][3];
            }
        }
        __syncthreads();

        // Online softmax under the frozen reduction contract: threads stride
        // over rows; each folds this tile's BN columns with an in-register
        // adjacent-pairwise tree, then merges tiles with the pinned ascending
        // sequential chain. Temperature scales the stats and the selected
        // logit; the stored logits (when requested) stay unscaled.
        for (int r = tid; r < num_rows; r += WG_THREADS) {
            const int tgt = target[row_base + r];
            const float inv_t =
                (temperature != nullptr) ? 1.0f / temperature[row_base + r] : 1.0f;
            float x[BN];
#pragma unroll
            for (int c = 0; c < BN; ++c) {
                const int col = col_base + c;
                float val = -CUDART_INF_F;  // padded lane: -inf max, exp() -> 0
                if (col < V && col < real_vocab_end) {
                    val = sLogits[r * BN + c];
                    if (bias != nullptr)
                        val += bias[col];
                    if (logits_out != nullptr)
                        logits_out[(size_t)(row_base + r) * V + col] = val;
                } else if (logits_out != nullptr && col < V) {
                    logits_out[(size_t)(row_base + r) * V + col] = -CUDART_INF_F;
                }
                if (temperature != nullptr)
                    val *= inv_t;
                if (vocab_start_index + col == tgt)
                    sZt[r] = val;
                x[c] = val;
            }
            float tmax = -CUDART_INF_F;
#pragma unroll
            for (int c = 0; c < BN; ++c) {
                tmax = fmaxf(tmax, x[c]);
            }
            float tsum = 0.0f;
            if (tmax != -CUDART_INF_F) {  // all-padded tile contributes exact 0
#pragma unroll
                for (int c = 0; c < BN; ++c)
                    x[c] = expf(x[c] - tmax);
#pragma unroll
                for (int stride = BN / 2; stride >= 1; stride >>= 1)
#pragma unroll
                    for (int c = 0; c < stride; ++c)
                        x[c] += x[c + stride];
                tsum = x[0];
            }
            const float old_max = sMax[r];
            const float new_max = fmaxf(old_max, tmax);
            if (new_max != -CUDART_INF_F) {  // avoid (-inf) - (-inf) = NaN
                sSum[r] = sSum[r] * expf(old_max - new_max) + tsum * expf(tmax - new_max);
                sMax[r] = new_max;
            }
        }
        __syncthreads();
    }

    // Emit this split's partial online-softmax state; a combine pass merges the
    // per-split (max, sum, target-logit) into the final logp/lse.
    for (int r = tid; r < num_rows; r += WG_THREADS) {
        const int idx = split * N + row_base + r;
        part_max[idx] = sMax[r];
        part_sum[idx] = sSum[r];
        part_zt[idx] = sZt[r];
    }
}

__global__ void fused_linear_logp_logits_tile_bf16_mma_kernel(
    const __grid_constant__ CUtensorMap h_tmap,
    const __grid_constant__ CUtensorMap w_tmap,
    float *__restrict__ logits,
    nv_bfloat16 *__restrict__ dlogits_bf16,
    const int *__restrict__ target,
    const float *__restrict__ grad_logp,
    const float *__restrict__ lse,
    int N,
    int D,
    int V,
    int64_t logits_stride0,
    int64_t dlogits_stride0,
    int vocab_start_index,
    bool write_dlogits) {
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int row_block = blockIdx.x;
    const int vt = blockIdx.y;
    const int row_base = row_block * BM;
    const int col_base = vt * BN;
    const int num_rows = min(BM, N - row_base);
    const int kd = D / BK;

    extern __shared__ __align__(1024) char smem[];
    nv_bfloat16 *sH = reinterpret_cast<nv_bfloat16 *>(smem);
    nv_bfloat16 *sW = reinterpret_cast<nv_bfloat16 *>(sH + STAGES * BM * BK);
    float *sLogits = reinterpret_cast<float *>(sW + STAGES * BN * BK);
    int *mbar_base = reinterpret_cast<int *>(sLogits + BM * BN);

    const uint64_t sH_base_tma = __cvta_generic_to_shared(sH);
    const uint64_t sW_base_tma = __cvta_generic_to_shared(sW);
    const uint32_t sH_base = static_cast<uint32_t>(sH_base_tma);
    const uint32_t sW_base = static_cast<uint32_t>(sW_base_tma);
    uint64_t mbar[STAGES];
#pragma unroll
    for (int s = 0; s < STAGES; ++s)
        mbar[s] = __cvta_generic_to_shared(mbar_base + 2 * s);

    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; ++s)
            mbarrier_init(mbar[s], 1);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();

    const uint32_t tile_bytes = (BM * BK + BN * BK) * sizeof(nv_bfloat16);
    auto issue_load = [&](int k) {
        const int buf = k % STAGES;
        const int k_off = k * BK;
        mbarrier_arrive_expect_tx(mbar[buf], tile_bytes);
        tma_2d_g2s(sH_base_tma + buf * BM * BK * sizeof(nv_bfloat16), &h_tmap, k_off, row_base,
                   mbar[buf]);
        tma_2d_g2s(sW_base_tma + buf * BN * BK * sizeof(nv_bfloat16), &w_tmap, k_off, col_base,
                   mbar[buf]);
    };

    int phase[STAGES];
#pragma unroll
    for (int s = 0; s < STAGES; ++s)
        phase[s] = 0;

    float acc[M_TILES][N_TILES][4];
#pragma unroll
    for (int mi = 0; mi < M_TILES; ++mi)
#pragma unroll
        for (int n = 0; n < N_TILES; ++n)
            acc[mi][n][0] = acc[mi][n][1] = acc[mi][n][2] = acc[mi][n][3] = 0.0f;

    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < STAGES - 1; ++s)
            if (s < kd)
                issue_load(s);
    }
    for (int k = 0; k < kd; ++k) {
        const int buf = k % STAGES;
        if (tid == 0 && k + (STAGES - 1) < kd)
            issue_load(k + (STAGES - 1));
        mbarrier_wait(mbar[buf], phase[buf]);
        phase[buf] ^= 1;
        __syncthreads();

        const uint32_t sH_buf = sH_base + buf * BM * BK * sizeof(nv_bfloat16);
        const uint32_t sW_buf = sW_base + buf * BN * BK * sizeof(nv_bfloat16);

        uint32_t A[M_TILES][K_TILES][4];
#pragma unroll
        for (int mi = 0; mi < M_TILES; ++mi) {
            const int row0 = warp * WARP_M + mi * MMA_M + (lane % 16);
#pragma unroll
            for (int kt = 0; kt < K_TILES; ++kt) {
                const uint32_t a_addr =
                    sH_buf + (row0 * BK + (lane / 16) * 8 + kt * MMA_K) * sizeof(nv_bfloat16);
                ldmatrix_x4(A[mi][kt], a_addr);
            }
        }

#pragma unroll
        for (int n = 0; n < N_TILES; ++n) {
#pragma unroll
            for (int kk = 0; kk < KK_GROUPS; ++kk) {
                uint32_t b4[4];
                const uint32_t b_addr =
                    sW_buf + ((n * MMA_N + (lane % 8)) * BK + (lane / 8) * 8 + kk * 32) *
                                 sizeof(nv_bfloat16);
                ldmatrix_x4(b4, b_addr);
                const uint32_t B0[2] = {b4[0], b4[1]};
                const uint32_t B1[2] = {b4[2], b4[3]};
#pragma unroll
                for (int mi = 0; mi < M_TILES; ++mi) {
                    mma_m16n8k16(A[mi][2 * kk + 0], B0, acc[mi][n]);
                    mma_m16n8k16(A[mi][2 * kk + 1], B1, acc[mi][n]);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int mi = 0; mi < M_TILES; ++mi) {
        const int row = warp * WARP_M + mi * MMA_M + lane / 4;
#pragma unroll
        for (int n = 0; n < N_TILES; ++n) {
            const int col = n * MMA_N + (lane % 4) * 2;
            sLogits[row * BN + col + 0] = acc[mi][n][0];
            sLogits[row * BN + col + 1] = acc[mi][n][1];
            sLogits[(row + 8) * BN + col + 0] = acc[mi][n][2];
            sLogits[(row + 8) * BN + col + 1] = acc[mi][n][3];
        }
    }
    __syncthreads();

    for (int idx = tid; idx < num_rows * BN; idx += WG_THREADS) {
        const int r = idx / BN;
        const int c = idx - r * BN;
        const int col = col_base + c;
        const int global_row = row_base + r;
        if (col < V) {
            const float logit = sLogits[r * BN + c];
            if (write_dlogits) {
                const int global_col = vocab_start_index + col;
                float dz = -__expf(logit - lse[global_row]);
                if (target[global_row] == global_col)
                    dz += 1.0f;
                dlogits_bf16[(static_cast<int64_t>(global_row) * dlogits_stride0) + col] =
                    __float2bfloat16(dz * grad_logp[global_row]);
            } else {
                logits[(static_cast<int64_t>(global_row) * logits_stride0) + col] = logit;
            }
        }
    }
}

// Merge per-split partials: M = max_s m_s, S = sum_s s_s*exp(m_s - M),
// zt = sum_s zt_s (exactly one split holds the target column), then
// logp = min(zt - (M + log S), 0). One thread per token row.
// The split fold is the pinned ascending-index sequential chain; the split
// count itself is a contract constant (function of V only, never of batch).
__global__ void fused_linear_logp_sm90_combine_kernel(const float *__restrict__ part_max,
                                                      const float *__restrict__ part_sum,
                                                      const float *__restrict__ part_zt,
                                                      float *__restrict__ out_value,
                                                      float *__restrict__ out_lse, int N,
                                                      int n_split,
                                                      bool return_target_logit) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= N)
        return;

    float M = -CUDART_INF_F;
    for (int s = 0; s < n_split; ++s)
        M = fmaxf(M, part_max[s * N + r]);

    float S = 0.0f;
    float zt = 0.0f;
    for (int s = 0; s < n_split; ++s) {  // ascending-index sequential chain
        const int idx = s * N + r;
        S += part_sum[idx] * expf(part_max[idx] - M);
        zt += part_zt[idx];
    }
    const float lse = M + logf(S);
    out_value[r] = return_target_logit ? zt : fminf(zt - lse, 0.0f);
    out_lse[r] = lse;
}

__global__ void fused_linear_logp_backward_dlogits_kernel(float *__restrict__ logits,
                                                          const int *__restrict__ target,
                                                          const float *__restrict__ grad_logp,
                                                          const float *__restrict__ lse, int N,
                                                          int V, int vocab_start_index,
                                                          bool logits_are_probs) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(N) * V;
    if (idx >= total)
        return;

    const int row = static_cast<int>(idx / V);
    const int local_col = static_cast<int>(idx - static_cast<int64_t>(row) * V);
    const int global_col = vocab_start_index + local_col;
    float dz = logits_are_probs ? -logits[idx] : -__expf(logits[idx] - lse[row]);
    if (target[row] == global_col)
        dz += 1.0f;
    logits[idx] = dz * grad_logp[row];
}

__global__ void fused_linear_logp_backward_dlogits_row_kernel(float *__restrict__ logits,
                                                              const int *__restrict__ target,
                                                              const float *__restrict__ grad_logp,
                                                              const float *__restrict__ lse,
                                                              int N,
                                                              int V,
                                                              int vocab_start_index,
                                                              bool logits_are_probs) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= N)
        return;

    __shared__ float row_lse;
    __shared__ float row_grad;
    __shared__ int target_local;
    if (tid == 0) {
        row_lse = lse[row];
        row_grad = grad_logp[row];
        target_local = target[row] - vocab_start_index;
    }
    __syncthreads();

    const int64_t row_offset = static_cast<int64_t>(row) * V;
    for (int col = tid; col < V; col += blockDim.x) {
        const int64_t offset = row_offset + col;
        float dz = logits_are_probs ? -logits[offset] : -__expf(logits[offset] - row_lse);
        if (col == target_local)
            dz += 1.0f;
        logits[offset] = dz * row_grad;
    }
}

__global__ void fused_linear_logp_backward_dlogits_bf16_kernel(
    const float *__restrict__ logits_or_probs,
    nv_bfloat16 *__restrict__ dlogits_bf16,
    const int *__restrict__ target,
    const float *__restrict__ grad_logp,
    const float *__restrict__ lse,
    int N,
    int V,
    int64_t logits_stride0,
    int64_t dlogits_stride0,
    int vocab_start_index,
    bool logits_are_probs) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(N) * V;
    if (idx >= total)
        return;

    const int row = static_cast<int>(idx / V);
    const int local_col = static_cast<int>(idx - static_cast<int64_t>(row) * V);
    const int global_col = vocab_start_index + local_col;
    const int64_t logits_idx = static_cast<int64_t>(row) * logits_stride0 + local_col;
    const int64_t dlogits_idx = static_cast<int64_t>(row) * dlogits_stride0 + local_col;
    float dz = logits_are_probs ? -logits_or_probs[logits_idx]
                                : -__expf(logits_or_probs[logits_idx] - lse[row]);
    if (target[row] == global_col)
        dz += 1.0f;
    dlogits_bf16[dlogits_idx] = __float2bfloat16(dz * grad_logp[row]);
}

__global__ void fused_linear_logp_backward_dlogits_bf16_row_kernel(
    const float *__restrict__ logits_or_probs,
    nv_bfloat16 *__restrict__ dlogits_bf16,
    const int *__restrict__ target,
    const float *__restrict__ grad_logp,
    const float *__restrict__ lse,
    int N,
    int V,
    int64_t logits_stride0,
    int64_t dlogits_stride0,
    int vocab_start_index,
    bool logits_are_probs) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= N)
        return;

    __shared__ float row_lse;
    __shared__ float row_grad;
    __shared__ int target_local;
    if (tid == 0) {
        row_lse = lse[row];
        row_grad = grad_logp[row];
        target_local = target[row] - vocab_start_index;
    }
    __syncthreads();

    const int64_t logits_row_offset = static_cast<int64_t>(row) * logits_stride0;
    const int64_t dlogits_row_offset = static_cast<int64_t>(row) * dlogits_stride0;
    for (int col = tid; col < V; col += blockDim.x) {
        const int64_t logits_idx = logits_row_offset + col;
        const int64_t dlogits_idx = dlogits_row_offset + col;
        float dz = logits_are_probs ? -logits_or_probs[logits_idx]
                                    : -__expf(logits_or_probs[logits_idx] - row_lse);
        if (col == target_local)
            dz += 1.0f;
        dlogits_bf16[dlogits_idx] = __float2bfloat16(dz * row_grad);
    }
}

__global__ void linear_logp_probs_bf16_forward_kernel(
    const nv_bfloat16 *__restrict__ logits,
    const int *__restrict__ target,
    float *__restrict__ out_logp,
    float *__restrict__ out_target_logit,
    float *__restrict__ out_lse,
    nv_bfloat16 *__restrict__ probs,
    int N,
    int V,
    int64_t logits_stride0,
    int64_t probs_stride0,
    int vocab_start_index) {
    constexpr int THREADS = 256;
    __shared__ float reduce[THREADS];
    __shared__ float row_max_shared;
    __shared__ float row_sum_shared;
    __shared__ float target_logit_shared;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= N)
        return;

    const int tgt = target[row] - vocab_start_index;
    float local_max = -CUDART_INF_F;
    float local_target = 0.0f;
    for (int col = tid; col < V; col += blockDim.x) {
        const float val =
            __bfloat162float(logits[static_cast<int64_t>(row) * logits_stride0 + col]);
        local_max = fmaxf(local_max, val);
        if (col == tgt)
            local_target = val;
    }
    reduce[tid] = local_max;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset)
            reduce[tid] = fmaxf(reduce[tid], reduce[tid + offset]);
        __syncthreads();
    }
    if (tid == 0) {
        row_max_shared = reduce[0];
        target_logit_shared = local_target;
    }
    __syncthreads();
    if (tid != 0 && tgt >= 0 && tgt < V && (tgt % blockDim.x) == tid) {
        target_logit_shared =
            __bfloat162float(logits[static_cast<int64_t>(row) * logits_stride0 + tgt]);
    }
    __syncthreads();

    const float row_max = row_max_shared;
    // A top-p replay mask can leave an entire TP vocab shard empty for a
    // particular row.  Treat that shard as the additive logsumexp identity
    // (LSE=-inf, selected-logit=0) instead of evaluating -inf - -inf below.
    // At least one TP rank still owns a finite nucleus entry for every valid
    // sampled row, so the global rank-ordered merge remains well-defined.
    const bool empty_row = row_max == -CUDART_INF_F;
    float local_sum = 0.0f;
    for (int col = tid; col < V; col += blockDim.x) {
        const float val =
            __bfloat162float(logits[static_cast<int64_t>(row) * logits_stride0 + col]);
        local_sum += empty_row ? 0.0f : __expf(val - row_max);
    }
    reduce[tid] = local_sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset)
            reduce[tid] += reduce[tid + offset];
        __syncthreads();
    }
    if (tid == 0)
        row_sum_shared = reduce[0];
    __syncthreads();

    if (probs != nullptr) {
        const float inv_sum = empty_row ? 0.0f : 1.0f / row_sum_shared;
        for (int col = tid; col < V; col += blockDim.x) {
            const float val =
                __bfloat162float(logits[static_cast<int64_t>(row) * logits_stride0 + col]);
            probs[static_cast<int64_t>(row) * probs_stride0 + col] =
                __float2bfloat16(empty_row ? 0.0f : __expf(val - row_max) * inv_sum);
        }
    }
    if (tid == 0) {
        const float lse = empty_row ? -CUDART_INF_F : row_max + logf(row_sum_shared);
        const float target_logit = empty_row ? 0.0f : target_logit_shared;
        if (out_logp != nullptr)
            out_logp[row] = empty_row ? -CUDART_INF_F : target_logit - lse;
        if (out_target_logit != nullptr)
            out_target_logit[row] = target_logit;
        if (out_lse != nullptr)
            out_lse[row] = lse;
    }
}

// Top-p replay normally contains only a handful of tokens, but materializing a
// [N, V_local] boolean mask and scanning the complete vocabulary on every
// decode step makes that sparse case unnecessarily expensive.  This kernel
// consumes the compact replay set directly.  The ids are sorted in shared
// memory and assigned to the same col % 256 lanes as the dense kernel above,
// so each lane visits its finite logits in the same ascending-column order and
// the final 256-lane reduction remains bitwise identical to the dense masked
// path.
__global__ void linear_logp_top_p_local_bf16_forward_kernel(
    const nv_bfloat16 *__restrict__ logits,
    const int *__restrict__ target,
    const int *__restrict__ replay_ids,
    const float *__restrict__ replay_logprobs,
    const float *__restrict__ temperature,
    float *__restrict__ out_target_logit,
    float *__restrict__ out_lse,
    int N,
    int V,
    int K,
    int64_t logits_stride0,
    int64_t replay_ids_stride0,
    int64_t replay_logprobs_stride0,
    int vocab_start_index) {
    constexpr int THREADS = 256;
    constexpr int SORT_SIZE = 64;
    __shared__ int sorted_cols[SORT_SIZE];
    __shared__ float reduce[THREADS];
    __shared__ float row_max_shared;
    __shared__ float row_sum_shared;
    __shared__ float target_logit_shared;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= N)
        return;

    if (tid < SORT_SIZE) {
        int col = INT_MAX;
        if (tid < K) {
            const int64_t replay_offset =
                static_cast<int64_t>(row) * replay_ids_stride0 + tid;
            const int64_t value_offset =
                static_cast<int64_t>(row) * replay_logprobs_stride0 + tid;
            const int global_id = replay_ids[replay_offset];
            if (isfinite(replay_logprobs[value_offset]) &&
                global_id >= vocab_start_index &&
                global_id < vocab_start_index + V) {
                col = global_id - vocab_start_index;
            }
        }
        sorted_cols[tid] = col;
    }
    __syncthreads();

    // Fixed-size bitonic sort. Invalid entries are INT_MAX and naturally move
    // to the end. Duplicate ids (the sampled token is also present in top-k)
    // become adjacent and are consumed once below, matching boolean-mask
    // semantics.
    for (int size = 2; size <= SORT_SIZE; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            if (tid < SORT_SIZE) {
                const int peer = tid ^ stride;
                if (peer > tid) {
                    const int a = sorted_cols[tid];
                    const int b = sorted_cols[peer];
                    const bool ascending = (tid & size) == 0;
                    if ((ascending && a > b) || (!ascending && a < b)) {
                        sorted_cols[tid] = b;
                        sorted_cols[peer] = a;
                    }
                }
            }
            __syncthreads();
        }
    }

    const int tgt = target[row] - vocab_start_index;
    const float temp = temperature[row];
    float local_max = -CUDART_INF_F;
    float local_target = 0.0f;
    for (int index = 0; index < SORT_SIZE; ++index) {
        const int col = sorted_cols[index];
        if (col == INT_MAX)
            break;
        if (index > 0 && col == sorted_cols[index - 1])
            continue;
        if ((col & (THREADS - 1)) != tid)
            continue;
        const float raw = __bfloat162float(
            logits[static_cast<int64_t>(row) * logits_stride0 + col]);
        const float val = __bfloat162float(__float2bfloat16(raw / temp));
        local_max = fmaxf(local_max, val);
        if (col == tgt)
            local_target = val;
    }
    reduce[tid] = local_max;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset)
            reduce[tid] = fmaxf(reduce[tid], reduce[tid + offset]);
        __syncthreads();
    }
    if (tid == 0) {
        row_max_shared = reduce[0];
        target_logit_shared = 0.0f;
    }
    __syncthreads();
    if (local_target != 0.0f ||
        (tgt >= 0 && tgt < V && (tgt & (THREADS - 1)) == tid)) {
        // Only the target-owning lane writes. The second condition preserves
        // a legitimate target logit of exactly zero.
        for (int index = 0; index < SORT_SIZE; ++index) {
            const int col = sorted_cols[index];
            if (col == INT_MAX)
                break;
            if (col == tgt) {
                target_logit_shared = local_target;
                break;
            }
        }
    }
    __syncthreads();

    const float row_max = row_max_shared;
    const bool empty_row = row_max == -CUDART_INF_F;
    float local_sum = 0.0f;
    for (int index = 0; index < SORT_SIZE; ++index) {
        const int col = sorted_cols[index];
        if (col == INT_MAX)
            break;
        if (index > 0 && col == sorted_cols[index - 1])
            continue;
        if ((col & (THREADS - 1)) != tid)
            continue;
        const float raw = __bfloat162float(
            logits[static_cast<int64_t>(row) * logits_stride0 + col]);
        const float val = __bfloat162float(__float2bfloat16(raw / temp));
        local_sum += empty_row ? 0.0f : __expf(val - row_max);
    }
    reduce[tid] = local_sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset)
            reduce[tid] += reduce[tid + offset];
        __syncthreads();
    }
    if (tid == 0) {
        row_sum_shared = reduce[0];
        out_lse[row] = empty_row ? -CUDART_INF_F : row_max + logf(row_sum_shared);
        out_target_logit[row] = empty_row ? 0.0f : target_logit_shared;
    }
}

__global__ void linear_logp_probs_bf16_to_dlogits_kernel(nv_bfloat16 *__restrict__ probs,
                                                         const int *__restrict__ target,
                                                         const float *__restrict__ grad_logp,
                                                         int N,
                                                         int V,
                                                         int64_t probs_stride0,
                                                         int vocab_start_index) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(N) * V;
    if (idx >= total)
        return;
    const int row = static_cast<int>(idx / V);
    const int col = static_cast<int>(idx - static_cast<int64_t>(row) * V);
    const int64_t offset = static_cast<int64_t>(row) * probs_stride0 + col;
    float dz = -__bfloat162float(probs[offset]);
    if (target[row] == vocab_start_index + col)
        dz += 1.0f;
    probs[offset] = __float2bfloat16(dz * grad_logp[row]);
}

__global__ void linear_logp_local_probs_bf16_to_dlogits_kernel(
    nv_bfloat16 *__restrict__ probs,
    const int *__restrict__ target,
    const float *__restrict__ grad_logp,
    const float *__restrict__ local_lse,
    const float *__restrict__ global_lse,
    int N,
    int V,
    int64_t probs_stride0,
    int vocab_start_index) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(N) * V;
    if (idx >= total)
        return;
    const int row = static_cast<int>(idx / V);
    const int col = static_cast<int>(idx - static_cast<int64_t>(row) * V);
    const int64_t offset = static_cast<int64_t>(row) * probs_stride0 + col;
    const float local_to_global = __expf(local_lse[row] - global_lse[row]);
    float dz = -__bfloat162float(probs[offset]) * local_to_global;
    if (target[row] == vocab_start_index + col)
        dz += 1.0f;
    probs[offset] = __float2bfloat16(dz * grad_logp[row]);
}

__global__ void linear_logp_local_probs_bf16_to_dlogits_row_kernel(
    nv_bfloat16 *__restrict__ probs,
    const int *__restrict__ target,
    const float *__restrict__ grad_logp,
    const float *__restrict__ local_lse,
    const float *__restrict__ global_lse,
    int N,
    int V,
    int64_t probs_stride0,
    int vocab_start_index) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= N)
        return;

    __shared__ float local_to_global;
    __shared__ float row_grad;
    __shared__ int target_local;
    if (tid == 0) {
        local_to_global = __expf(local_lse[row] - global_lse[row]);
        row_grad = grad_logp[row];
        target_local = target[row] - vocab_start_index;
    }
    __syncthreads();

    const int64_t row_offset = static_cast<int64_t>(row) * probs_stride0;
    for (int col = tid; col < V; col += blockDim.x) {
        const int64_t offset = row_offset + col;
        float dz = -__bfloat162float(probs[offset]) * local_to_global;
        if (col == target_local)
            dz += 1.0f;
        probs[offset] = __float2bfloat16(dz * row_grad);
    }
}

__global__ void linear_logp_logits_bf16_to_dlogits_kernel(
    const nv_bfloat16 *__restrict__ logits,
    nv_bfloat16 *__restrict__ dlogits,
    const int *__restrict__ target,
    const float *__restrict__ grad_logp,
    const float *__restrict__ lse,
    int N,
    int V,
    int64_t logits_stride0,
    int64_t dlogits_stride0,
    int vocab_start_index) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(N) * V;
    if (idx >= total)
        return;
    const int row = static_cast<int>(idx / V);
    const int col = static_cast<int>(idx - static_cast<int64_t>(row) * V);
    const float logit =
        __bfloat162float(logits[static_cast<int64_t>(row) * logits_stride0 + col]);
    float dz = -__expf(logit - lse[row]);
    if (target[row] == vocab_start_index + col)
        dz += 1.0f;
    dlogits[static_cast<int64_t>(row) * dlogits_stride0 + col] =
        __float2bfloat16(dz * grad_logp[row]);
}

__global__ void linear_logp_logits_bf16_to_dlogits_row_kernel(
    const nv_bfloat16 *__restrict__ logits,
    nv_bfloat16 *__restrict__ dlogits,
    const int *__restrict__ target,
    const float *__restrict__ grad_logp,
    const float *__restrict__ lse,
    int N,
    int V,
    int64_t logits_stride0,
    int64_t dlogits_stride0,
    int vocab_start_index) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= N)
        return;

    __shared__ float row_lse;
    __shared__ float row_grad;
    __shared__ int target_local;
    if (tid == 0) {
        row_lse = lse[row];
        row_grad = grad_logp[row];
        target_local = target[row] - vocab_start_index;
    }
    __syncthreads();

    const int64_t logits_row_offset = static_cast<int64_t>(row) * logits_stride0;
    const int64_t dlogits_row_offset = static_cast<int64_t>(row) * dlogits_stride0;
    for (int col = tid; col < V; col += blockDim.x) {
        const float logit = __bfloat162float(logits[logits_row_offset + col]);
        float dz = -__expf(logit - row_lse);
        if (col == target_local)
            dz += 1.0f;
        dlogits[dlogits_row_offset + col] = __float2bfloat16(dz * row_grad);
    }
}

__global__ void fused_linear_logp_logits_tile_tf32_wmma_kernel(
    const float *__restrict__ hidden,
    const float *__restrict__ weight,
    float *__restrict__ logits,
    int N,
    int D,
    int V,
    int64_t logits_stride0) {
    const int warp_in_block = threadIdx.x / 32;
    const int tile_idx = blockIdx.x * LOGITS_WMMA_WARPS + warp_in_block;
    const int tiles_n = V / LOGITS_WMMA_N;
    const int total_tiles = (N / LOGITS_WMMA_M) * tiles_n;
    if (tile_idx >= total_tiles)
        return;

    const int row_tile = tile_idx / tiles_n;
    const int col_tile = tile_idx - row_tile * tiles_n;
    const int row0 = row_tile * LOGITS_WMMA_M;
    const int col0 = col_tile * LOGITS_WMMA_N;

    wmma::fragment<wmma::matrix_a, LOGITS_WMMA_M, LOGITS_WMMA_N, LOGITS_WMMA_K,
                   wmma::precision::tf32, wmma::row_major>
        a_frag;
    wmma::fragment<wmma::matrix_b, LOGITS_WMMA_M, LOGITS_WMMA_N, LOGITS_WMMA_K,
                   wmma::precision::tf32, wmma::col_major>
        b_frag;
    wmma::fragment<wmma::accumulator, LOGITS_WMMA_M, LOGITS_WMMA_N, LOGITS_WMMA_K, float>
        acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    for (int k = 0; k < D; k += LOGITS_WMMA_K) {
        wmma::load_matrix_sync(a_frag, hidden + static_cast<int64_t>(row0) * D + k, D);
        wmma::load_matrix_sync(b_frag, weight + static_cast<int64_t>(col0) * D + k, D);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    wmma::store_matrix_sync(logits + static_cast<int64_t>(row0) * logits_stride0 + col0,
                            acc_frag, logits_stride0, wmma::mem_row_major);
}

__global__ void fused_linear_logp_grad_weight_tile_wmma_kernel(
    const nv_bfloat16 *__restrict__ dlogits,
    const nv_bfloat16 *__restrict__ hidden,
    nv_bfloat16 *__restrict__ grad_weight,
    int N,
    int D,
    int V,
    int64_t dlogits_stride0,
    int64_t hidden_stride0,
    int64_t grad_weight_stride0) {
    const int warp_in_block = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int d_tiles = D / GRAD_W_WMMA_N;
    const int total_tiles = (V / GRAD_W_WMMA_M) * d_tiles;
    const int tile_idx = blockIdx.x * GRAD_W_WMMA_WARPS + warp_in_block;
    if (tile_idx >= total_tiles)
        return;

    const int v_tile = tile_idx / d_tiles;
    const int d_tile = tile_idx - v_tile * d_tiles;
    const int v0 = v_tile * GRAD_W_WMMA_M;
    const int d0 = d_tile * GRAD_W_WMMA_N;

    wmma::fragment<wmma::matrix_a, GRAD_W_WMMA_M, GRAD_W_WMMA_N, GRAD_W_WMMA_K,
                   nv_bfloat16, wmma::col_major>
        a_frag;
    wmma::fragment<wmma::matrix_b, GRAD_W_WMMA_M, GRAD_W_WMMA_N, GRAD_W_WMMA_K,
                   nv_bfloat16, wmma::row_major>
        b_frag;
    wmma::fragment<wmma::accumulator, GRAD_W_WMMA_M, GRAD_W_WMMA_N, GRAD_W_WMMA_K,
                   float>
        acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    for (int n0 = 0; n0 < N; n0 += GRAD_W_WMMA_K) {
        // A = dlogits.T tile [V, N]. dlogits is [N, V] row-major, so the
        // transposed A tile is naturally column-major with lda=dlogits_stride0.
        wmma::load_matrix_sync(
            a_frag,
            reinterpret_cast<const nv_bfloat16 *>(
                dlogits + static_cast<int64_t>(n0) * dlogits_stride0 + v0),
            dlogits_stride0);
        // B = hidden tile [N, D], row-major.
        wmma::load_matrix_sync(
            b_frag,
            reinterpret_cast<const nv_bfloat16 *>(
                hidden + static_cast<int64_t>(n0) * hidden_stride0 + d0),
            hidden_stride0);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    extern __shared__ char grad_w_smem_raw[];
    float *tile_out = reinterpret_cast<float *>(grad_w_smem_raw) +
                      warp_in_block * GRAD_W_WMMA_M * GRAD_W_WMMA_N;
    wmma::store_matrix_sync(tile_out, acc_frag, GRAD_W_WMMA_N, wmma::mem_row_major);
    __syncwarp();

    for (int idx = lane; idx < GRAD_W_WMMA_M * GRAD_W_WMMA_N; idx += 32) {
        const int vr = idx / GRAD_W_WMMA_N;
        const int dc = idx - vr * GRAD_W_WMMA_N;
        grad_weight[static_cast<int64_t>(v0 + vr) * grad_weight_stride0 + d0 + dc] =
            __float2bfloat16(tile_out[idx]);
    }
}

__global__ void fused_linear_logp_grad_weight_from_logits_tile_wmma_kernel(
    const float *__restrict__ logits_or_probs,
    const nv_bfloat16 *__restrict__ hidden,
    const int *__restrict__ target,
    const float *__restrict__ grad_logp,
    const float *__restrict__ lse,
    nv_bfloat16 *__restrict__ grad_weight,
    int N,
    int D,
    int V,
    int64_t logits_stride0,
    int64_t hidden_stride0,
    int64_t grad_weight_stride0,
    int vocab_start_index,
    bool logits_are_probs) {
    const int warp_in_block = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int d_tiles = D / GRAD_W_WMMA_N;
    const int d_groups = (d_tiles + GRAD_W_WMMA_WARPS - 1) / GRAD_W_WMMA_WARPS;
    const int block_tile = blockIdx.x;
    const int v_tile = block_tile / d_groups;
    const int d_group = block_tile - v_tile * d_groups;
    const int d_tile = d_group * GRAD_W_WMMA_WARPS + warp_in_block;
    const int v0 = v_tile * GRAD_W_WMMA_M;
    const int d0 = d_tile * GRAD_W_WMMA_N;
    const bool valid_d_tile = d_tile < d_tiles;

    extern __shared__ __align__(16) char smem_raw[];
    nv_bfloat16 *dlogits_tile = reinterpret_cast<nv_bfloat16 *>(smem_raw);
    float *tile_out = reinterpret_cast<float *>(dlogits_tile + GRAD_W_WMMA_M * GRAD_W_WMMA_K) +
                      warp_in_block * GRAD_W_WMMA_M * GRAD_W_WMMA_N;

    wmma::fragment<wmma::matrix_a, GRAD_W_WMMA_M, GRAD_W_WMMA_N, GRAD_W_WMMA_K,
                   nv_bfloat16, wmma::col_major>
        a_frag;
    wmma::fragment<wmma::matrix_b, GRAD_W_WMMA_M, GRAD_W_WMMA_N, GRAD_W_WMMA_K,
                   nv_bfloat16, wmma::row_major>
        b_frag;
    wmma::fragment<wmma::accumulator, GRAD_W_WMMA_M, GRAD_W_WMMA_N, GRAD_W_WMMA_K,
                   float>
        acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    for (int n0 = 0; n0 < N; n0 += GRAD_W_WMMA_K) {
        if (threadIdx.x < GRAD_W_WMMA_M * GRAD_W_WMMA_K) {
            const int vr = threadIdx.x % GRAD_W_WMMA_M;
            const int nk = threadIdx.x / GRAD_W_WMMA_M;
            const int n = n0 + nk;
            const int v = v0 + vr;
            const float src =
                logits_or_probs[static_cast<int64_t>(n) * logits_stride0 + v];
            float dz = logits_are_probs ? -src : -__expf(src - lse[n]);
            if (target[n] == vocab_start_index + v)
                dz += 1.0f;
            dlogits_tile[vr + nk * GRAD_W_WMMA_M] = __float2bfloat16(dz * grad_logp[n]);
        }
        __syncthreads();

        if (valid_d_tile) {
            wmma::load_matrix_sync(a_frag, dlogits_tile, GRAD_W_WMMA_M);
            wmma::load_matrix_sync(
                b_frag,
                reinterpret_cast<const nv_bfloat16 *>(
                    hidden + static_cast<int64_t>(n0) * hidden_stride0 + d0),
                hidden_stride0);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
        __syncthreads();
    }

    if (!valid_d_tile)
        return;

    wmma::store_matrix_sync(tile_out, acc_frag, GRAD_W_WMMA_N, wmma::mem_row_major);
    __syncwarp();

    for (int idx = lane; idx < GRAD_W_WMMA_M * GRAD_W_WMMA_N; idx += 32) {
        const int vr = idx / GRAD_W_WMMA_N;
        const int dc = idx - vr * GRAD_W_WMMA_N;
        grad_weight[static_cast<int64_t>(v0 + vr) * grad_weight_stride0 + d0 + dc] =
            __float2bfloat16(tile_out[idx]);
    }
}

bool fused_backward_use_input_precision_gemm(const torch::Tensor &hidden,
                                             const torch::Tensor &weight) {
    const char *precision_env = std::getenv("RL_KERNEL_LINEAR_LOGP_FUSED_BWD_PRECISION");
    if (precision_env != nullptr) {
        std::string precision(precision_env);
        std::transform(precision.begin(), precision.end(), precision.begin(), ::tolower);
        if (precision == "fp32")
            return false;
        if (precision == "input" || precision == "bf16" || precision == "auto")
            return true;
    }
    return hidden.scalar_type() == weight.scalar_type() &&
           (hidden.scalar_type() == at::kBFloat16 || hidden.scalar_type() == at::kHalf);
}

bool fused_backward_use_input_precision_logits() {
    const char *precision_env = std::getenv("RL_KERNEL_LINEAR_LOGP_FUSED_BWD_PRECISION");
    if (precision_env == nullptr)
        return false;
    std::string precision(precision_env);
    std::transform(precision.begin(), precision.end(), precision.begin(), ::tolower);
    return precision == "input_all" || precision == "bf16_all";
}

bool fused_backward_bf16_dlogits_enabled() {
    const char *enabled_env = std::getenv("RL_KERNEL_LINEAR_LOGP_FUSED_BWD_BF16_DLOGITS");
    if (enabled_env == nullptr)
        return true;
    std::string enabled(enabled_env);
    std::transform(enabled.begin(), enabled.end(), enabled.begin(), ::tolower);
    return enabled != "0" && enabled != "false" && enabled != "no" && enabled != "off";
}

bool fused_backward_rowwise_dlogits_enabled() {
    const char *enabled_env = std::getenv("RL_KERNEL_LINEAR_LOGP_FUSED_BWD_ROWWISE_DLOGITS");
    if (enabled_env == nullptr)
        return true;
    std::string enabled(enabled_env);
    std::transform(enabled.begin(), enabled.end(), enabled.begin(), ::tolower);
    return enabled != "0" && enabled != "false" && enabled != "no" && enabled != "off";
}

bool streaming_output_backward_enabled() {
    const char *precision_env = std::getenv("RL_KERNEL_LINEAR_LOGP_FUSED_BWD_PRECISION");
    if (precision_env != nullptr) {
        std::string precision(precision_env);
        std::transform(precision.begin(), precision.end(), precision.begin(), ::tolower);
        if (precision == "fp32")
            return false;
    }
    const char *enabled_env = std::getenv("RL_KERNEL_LINEAR_LOGP_STREAMING_BWD");
    if (enabled_env == nullptr)
        return false;
    std::string enabled(enabled_env);
    std::transform(enabled.begin(), enabled.end(), enabled.begin(), ::tolower);
    return enabled != "0" && enabled != "false" && enabled != "no" && enabled != "off";
}

bool streaming_output_backward_scalar_mode() {
    const char *mode_env = std::getenv("RL_KERNEL_LINEAR_LOGP_STREAMING_BWD_MODE");
    if (mode_env == nullptr)
        return false;
    std::string mode(mode_env);
    std::transform(mode.begin(), mode.end(), mode.begin(), ::tolower);
    return mode == "scalar";
}

std::string streaming_output_backward_logits_mode() {
    const char *mode_env = std::getenv("RL_KERNEL_LINEAR_LOGP_STREAMING_BWD_LOGITS");
    if (mode_env == nullptr)
        return "";
    std::string mode(mode_env);
    std::transform(mode.begin(), mode.end(), mode.begin(), ::tolower);
    return mode;
}

bool streaming_output_backward_cuda_logits_enabled() {
    const std::string mode = streaming_output_backward_logits_mode();
    return mode == "cuda_tf32" || mode == "tf32" || mode == "wmma_tf32";
}

bool streaming_output_backward_bf16_mma_logits_enabled() {
    const std::string mode = streaming_output_backward_logits_mode();
    return mode == "cuda_bf16_mma" || mode == "bf16_mma" || mode == "sm90_mma";
}

bool streaming_output_backward_bf16_mma_dz_enabled() {
    const std::string mode = streaming_output_backward_logits_mode();
    return mode == "cuda_bf16_mma_dz" || mode == "bf16_mma_dz" ||
           mode == "sm90_mma_dz";
}

std::string streaming_output_backward_grad_weight_mode() {
    const char *mode_env = std::getenv("RL_KERNEL_LINEAR_LOGP_STREAMING_BWD_GW");
    if (mode_env == nullptr)
        return "";
    std::string mode(mode_env);
    std::transform(mode.begin(), mode.end(), mode.begin(), ::tolower);
    return mode;
}

bool streaming_output_backward_grad_weight_wmma_enabled() {
    const std::string mode = streaming_output_backward_grad_weight_mode();
    return mode == "cuda_wmma" || mode == "wmma" || mode == "cuda_bf16_wmma";
}

bool fused_tile_backward_grad_weight_enabled() {
    const char *enabled_env = std::getenv("RL_KERNEL_LINEAR_LOGP_FUSED_TILE_BWD");
    if (enabled_env == nullptr)
        enabled_env = std::getenv("RL_KERNEL_LINEAR_LOGP_FUSED_TILE_DW");
    if (enabled_env != nullptr) {
        std::string enabled(enabled_env);
        std::transform(enabled.begin(), enabled.end(), enabled.begin(), ::tolower);
        return enabled != "0" && enabled != "false" && enabled != "no" &&
               enabled != "off";
    }
    const std::string mode = streaming_output_backward_grad_weight_mode();
    return mode == "fused_tile" || mode == "tile_fused" || mode == "cuda_fused_tile";
}

std::string fused_tile_backward_full_grad_mode() {
    const char *enabled_env = std::getenv("RL_KERNEL_LINEAR_LOGP_FUSED_TILE_BWD_FULL");
    if (enabled_env == nullptr)
        return "";
    std::string enabled(enabled_env);
    std::transform(enabled.begin(), enabled.end(), enabled.begin(), ::tolower);
    if (enabled == "0" || enabled == "false" || enabled == "no" || enabled == "off")
        return "";
    if (enabled == "1" || enabled == "true" || enabled == "yes" || enabled == "on" ||
        enabled == "auto")
        return "tile_cublas";
    return enabled;
}

bool save_probs_rowwise_dlogits_enabled() {
    const char *enabled_env = std::getenv("RL_KERNEL_LINEAR_LOGP_SAVE_PROBS_ROWWISE_DLOGITS");
    if (enabled_env == nullptr)
        return true;
    std::string enabled(enabled_env);
    std::transform(enabled.begin(), enabled.end(), enabled.begin(), ::tolower);
    return enabled != "0" && enabled != "false" && enabled != "no" && enabled != "off";
}

int streaming_output_backward_vocab_tile(int V) {
    const char *tile_env = std::getenv("RL_KERNEL_LINEAR_LOGP_STREAMING_BWD_VOCAB_TILE");
    int tile = 32768;
    if (tile_env != nullptr) {
        const int parsed = std::atoi(tile_env);
        if (parsed > 0)
            tile = parsed;
    }
    tile = std::max(1, tile);
    if (V > 1)
        tile = std::min(tile, V - 1);
    else
        tile = 1;
    return tile;
}

bool can_use_logits_tile_tf32_wmma(int N, int D, int V) {
    return N % LOGITS_WMMA_M == 0 && V % LOGITS_WMMA_N == 0 && D % LOGITS_WMMA_K == 0;
}

bool can_use_grad_weight_tile_wmma(int N, int D, int V) {
    return N % GRAD_W_WMMA_K == 0 && D % GRAD_W_WMMA_N == 0 &&
           V % GRAD_W_WMMA_M == 0;
}

void launch_logits_tile_tf32_wmma(torch::Tensor hidden,
                                  torch::Tensor weight,
                                  torch::Tensor logits) {
    const int N = hidden.size(0);
    const int D = hidden.size(1);
    const int V = weight.size(0);
    TORCH_CHECK(can_use_logits_tile_tf32_wmma(N, D, V),
                "TF32 WMMA logits tile requires N % ", LOGITS_WMMA_M, " == 0, V % ",
                LOGITS_WMMA_N, " == 0, and D % ", LOGITS_WMMA_K, " == 0");
    const int tile_count = (N / LOGITS_WMMA_M) * (V / LOGITS_WMMA_N);
    const int blocks = (tile_count + LOGITS_WMMA_WARPS - 1) / LOGITS_WMMA_WARPS;
    fused_linear_logp_logits_tile_tf32_wmma_kernel<<<
        blocks, LOGITS_WMMA_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        hidden.data_ptr<float>(), weight.data_ptr<float>(), logits.data_ptr<float>(), N, D, V,
        logits.stride(0));
}

void launch_logits_tile_bf16_mma(torch::Tensor hidden,
                                 torch::Tensor weight,
                                 torch::Tensor logits) {
    TORCH_CHECK(hidden.scalar_type() == at::kBFloat16 && weight.scalar_type() == at::kBFloat16,
                "bf16 MMA logits tile requires bf16 hidden and weight");
    TORCH_CHECK(hidden.is_contiguous() && weight.is_contiguous(),
                "bf16 MMA logits tile requires contiguous hidden and weight");
    const int N = hidden.size(0);
    const int D = hidden.size(1);
    const int V = weight.size(0);
    TORCH_CHECK(weight.size(1) == D, "hidden/weight hidden-dim mismatch");
    TORCH_CHECK(D % BK == 0, "D must be a multiple of ", BK, " for bf16 MMA logits tile");

    CUtensorMap h_tmap, w_tmap;
    init_tensor_map_noswizzle(
        &h_tmap, reinterpret_cast<const nv_bfloat16 *>(hidden.data_ptr<at::BFloat16>()), N, D, BM,
        BK);
    init_tensor_map_noswizzle(
        &w_tmap, reinterpret_cast<const nv_bfloat16 *>(weight.data_ptr<at::BFloat16>()), V, D, BN,
        BK);

    const int smem = STAGES * (BM * BK + BN * BK) * sizeof(nv_bfloat16) +
                     (BM * BN) * sizeof(float) + STAGES * 8;
    if (smem > 48 * 1024) {
        cudaFuncSetAttribute(fused_linear_logp_logits_tile_bf16_mma_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    }

    const int row_blocks = (N + BM - 1) / BM;
    const int total_vtiles = (V + BN - 1) / BN;
    dim3 grid(row_blocks, total_vtiles);
    fused_linear_logp_logits_tile_bf16_mma_kernel<<<
        grid, WG_THREADS, smem, at::cuda::getCurrentCUDAStream()>>>(
        h_tmap, w_tmap, logits.data_ptr<float>(), nullptr, nullptr, nullptr, nullptr, N, D, V,
        logits.stride(0), 0, 0, false);
}

void launch_dlogits_tile_bf16_mma(torch::Tensor hidden,
                                  torch::Tensor weight,
                                  torch::Tensor dlogits,
                                  torch::Tensor target,
                                  torch::Tensor grad_logp,
                                  torch::Tensor lse,
                                  int vocab_start_index) {
    TORCH_CHECK(hidden.scalar_type() == at::kBFloat16 && weight.scalar_type() == at::kBFloat16,
                "bf16 MMA dlogits tile requires bf16 hidden and weight");
    TORCH_CHECK(dlogits.scalar_type() == at::kBFloat16,
                "bf16 MMA dlogits tile requires bf16 dlogits output");
    TORCH_CHECK(hidden.is_contiguous() && weight.is_contiguous(),
                "bf16 MMA dlogits tile requires contiguous hidden and weight");
    TORCH_CHECK(target.is_contiguous() && grad_logp.is_contiguous() && lse.is_contiguous(),
                "bf16 MMA dlogits tile requires contiguous target, grad_logp, and lse");
    const int N = hidden.size(0);
    const int D = hidden.size(1);
    const int V = weight.size(0);
    TORCH_CHECK(weight.size(1) == D, "hidden/weight hidden-dim mismatch");
    TORCH_CHECK(dlogits.size(0) == N && dlogits.size(1) == V,
                "dlogits tile shape must be [N, V]");
    TORCH_CHECK(D % BK == 0, "D must be a multiple of ", BK, " for bf16 MMA dlogits tile");

    CUtensorMap h_tmap, w_tmap;
    init_tensor_map_noswizzle(
        &h_tmap, reinterpret_cast<const nv_bfloat16 *>(hidden.data_ptr<at::BFloat16>()), N, D, BM,
        BK);
    init_tensor_map_noswizzle(
        &w_tmap, reinterpret_cast<const nv_bfloat16 *>(weight.data_ptr<at::BFloat16>()), V, D, BN,
        BK);

    const int smem = STAGES * (BM * BK + BN * BK) * sizeof(nv_bfloat16) +
                     (BM * BN) * sizeof(float) + STAGES * 8;
    if (smem > 48 * 1024) {
        cudaFuncSetAttribute(fused_linear_logp_logits_tile_bf16_mma_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    }

    const int row_blocks = (N + BM - 1) / BM;
    const int total_vtiles = (V + BN - 1) / BN;
    dim3 grid(row_blocks, total_vtiles);
    fused_linear_logp_logits_tile_bf16_mma_kernel<<<
        grid, WG_THREADS, smem, at::cuda::getCurrentCUDAStream()>>>(
        h_tmap, w_tmap, nullptr,
        reinterpret_cast<nv_bfloat16 *>(dlogits.data_ptr<at::BFloat16>()),
        target.data_ptr<int>(), grad_logp.data_ptr<float>(), lse.data_ptr<float>(), N, D, V, 0,
        dlogits.stride(0), vocab_start_index, true);
}

void launch_grad_weight_tile_wmma(torch::Tensor dlogits,
                                  torch::Tensor hidden,
                                  torch::Tensor grad_weight) {
    TORCH_CHECK(dlogits.scalar_type() == at::kBFloat16 &&
                    hidden.scalar_type() == at::kBFloat16 &&
                    grad_weight.scalar_type() == at::kBFloat16,
                "WMMA grad_weight tile requires bf16 dlogits, hidden, and grad_weight");
    TORCH_CHECK(hidden.is_contiguous(), "WMMA grad_weight tile requires contiguous hidden");
    TORCH_CHECK(grad_weight.dim() == 2 && dlogits.dim() == 2 && hidden.dim() == 2,
                "WMMA grad_weight tile expects 2-D tensors");
    const int N = dlogits.size(0);
    const int V = dlogits.size(1);
    const int D = hidden.size(1);
    TORCH_CHECK(hidden.size(0) == N, "dlogits/hidden token dimension mismatch");
    TORCH_CHECK(grad_weight.size(0) == V && grad_weight.size(1) == D,
                "grad_weight tile shape must be [V, D]");
    TORCH_CHECK(can_use_grad_weight_tile_wmma(N, D, V),
                "WMMA grad_weight tile requires N % ", GRAD_W_WMMA_K, " == 0, D % ",
                GRAD_W_WMMA_N, " == 0, and V % ", GRAD_W_WMMA_M, " == 0");

    const int d_tiles = D / GRAD_W_WMMA_N;
    const int v_tiles = V / GRAD_W_WMMA_M;
    const int tile_count = d_tiles * v_tiles;
    const int blocks = (tile_count + GRAD_W_WMMA_WARPS - 1) / GRAD_W_WMMA_WARPS;
    const int smem = GRAD_W_WMMA_WARPS * GRAD_W_WMMA_M * GRAD_W_WMMA_N * sizeof(float);
    fused_linear_logp_grad_weight_tile_wmma_kernel<<<
        blocks, GRAD_W_WMMA_THREADS, smem, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const nv_bfloat16 *>(dlogits.data_ptr<at::BFloat16>()),
        reinterpret_cast<const nv_bfloat16 *>(hidden.data_ptr<at::BFloat16>()),
        reinterpret_cast<nv_bfloat16 *>(grad_weight.data_ptr<at::BFloat16>()), N, D, V,
        dlogits.stride(0), hidden.stride(0), grad_weight.stride(0));
}

void launch_grad_weight_from_logits_tile_wmma(torch::Tensor logits_or_probs,
                                              torch::Tensor hidden,
                                              torch::Tensor target,
                                              torch::Tensor grad_logp,
                                              torch::Tensor lse,
                                              torch::Tensor grad_weight,
                                              int vocab_start_index,
                                              bool logits_are_probs) {
    TORCH_CHECK(logits_or_probs.scalar_type() == at::kFloat,
                "fused tile grad_weight requires fp32 logits/probs");
    TORCH_CHECK(hidden.scalar_type() == at::kBFloat16 &&
                    grad_weight.scalar_type() == at::kBFloat16,
                "fused tile grad_weight requires bf16 hidden and grad_weight");
    TORCH_CHECK(hidden.is_contiguous(), "fused tile grad_weight requires contiguous hidden");
    TORCH_CHECK(logits_or_probs.dim() == 2 && hidden.dim() == 2 && grad_weight.dim() == 2,
                "fused tile grad_weight expects 2-D tensors");
    TORCH_CHECK(logits_or_probs.stride(1) == 1 && hidden.stride(1) == 1 &&
                    grad_weight.stride(1) == 1,
                "fused tile grad_weight requires unit stride on the last dimension");
    TORCH_CHECK(target.is_contiguous() && grad_logp.is_contiguous() && lse.is_contiguous(),
                "fused tile grad_weight requires contiguous target, grad_logp, and lse");
    TORCH_CHECK(target.scalar_type() == at::kInt,
                "fused tile grad_weight requires int32 target");
    TORCH_CHECK(grad_logp.scalar_type() == at::kFloat && lse.scalar_type() == at::kFloat,
                "fused tile grad_weight requires fp32 grad_logp and lse");

    const int N = logits_or_probs.size(0);
    const int V = logits_or_probs.size(1);
    const int D = hidden.size(1);
    TORCH_CHECK(hidden.size(0) == N, "logits/hidden token dimension mismatch");
    TORCH_CHECK(grad_weight.size(0) == V && grad_weight.size(1) == D,
                "grad_weight tile shape must be [V, D]");
    TORCH_CHECK(target.numel() == N && grad_logp.numel() == N && lse.numel() == N,
                "target, grad_logp, and lse must have one value per token");
    TORCH_CHECK(can_use_grad_weight_tile_wmma(N, D, V),
                "fused tile grad_weight requires N % ", GRAD_W_WMMA_K, " == 0, D % ",
                GRAD_W_WMMA_N, " == 0, and V % ", GRAD_W_WMMA_M, " == 0");

    const int d_tiles = D / GRAD_W_WMMA_N;
    const int d_groups = (d_tiles + GRAD_W_WMMA_WARPS - 1) / GRAD_W_WMMA_WARPS;
    const int v_tiles = V / GRAD_W_WMMA_M;
    const int blocks = v_tiles * d_groups;
    const int smem = GRAD_W_WMMA_M * GRAD_W_WMMA_K * sizeof(nv_bfloat16) +
                     GRAD_W_WMMA_WARPS * GRAD_W_WMMA_M * GRAD_W_WMMA_N * sizeof(float);
    fused_linear_logp_grad_weight_from_logits_tile_wmma_kernel<<<
        blocks, GRAD_W_WMMA_THREADS, smem, at::cuda::getCurrentCUDAStream()>>>(
        logits_or_probs.data_ptr<float>(),
        reinterpret_cast<const nv_bfloat16 *>(hidden.data_ptr<at::BFloat16>()),
        target.data_ptr<int>(), grad_logp.data_ptr<float>(), lse.data_ptr<float>(),
        reinterpret_cast<nv_bfloat16 *>(grad_weight.data_ptr<at::BFloat16>()), N, D, V,
        logits_or_probs.stride(0), hidden.stride(0), grad_weight.stride(0),
        vocab_start_index, logits_are_probs);
}

std::vector<torch::Tensor> linear_logp_probs_bf16_forward_impl(torch::Tensor logits,
                                                               torch::Tensor target,
                                                               int64_t vocab_start_index) {
    TORCH_CHECK(logits.is_cuda() && target.is_cuda(), "logits and target must be CUDA tensors");
    TORCH_CHECK(logits.scalar_type() == at::kBFloat16,
                "linear_logp_probs_bf16_forward requires bf16 logits");
    TORCH_CHECK(logits.dim() == 2, "logits must be 2-D [N, V]");
    TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");
    const int N = logits.size(0);
    const int V = logits.size(1);
    TORCH_CHECK(target.numel() == N, "target must have one id per row");

    c10::cuda::CUDAGuard device_guard(logits.device());
    auto target_i = target.reshape({N}).to(torch::kInt32).contiguous();
    auto out = torch::empty({N}, logits.options().dtype(torch::kFloat));
    auto probs = torch::empty_like(logits);
    linear_logp_probs_bf16_forward_kernel<<<
        N, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const nv_bfloat16 *>(logits.data_ptr<at::BFloat16>()),
        target_i.data_ptr<int>(), out.data_ptr<float>(), nullptr, nullptr,
        reinterpret_cast<nv_bfloat16 *>(probs.data_ptr<at::BFloat16>()), N, V,
        logits.stride(0), probs.stride(0), static_cast<int>(vocab_start_index));
    return {out, probs};
}

std::vector<torch::Tensor> linear_logp_bf16_forward_impl(torch::Tensor logits,
                                                         torch::Tensor target,
                                                         int64_t vocab_start_index) {
    TORCH_CHECK(logits.is_cuda() && target.is_cuda(), "logits and target must be CUDA tensors");
    TORCH_CHECK(logits.scalar_type() == at::kBFloat16,
                "linear_logp_bf16_forward requires bf16 logits");
    TORCH_CHECK(logits.dim() == 2, "logits must be 2-D [N, V]");
    TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");
    const int N = logits.size(0);
    const int V = logits.size(1);
    TORCH_CHECK(target.numel() == N, "target must have one id per row");

    c10::cuda::CUDAGuard device_guard(logits.device());
    auto target_i = target.reshape({N}).to(torch::kInt32).contiguous();
    auto opts_f = logits.options().dtype(torch::kFloat);
    auto out = torch::empty({N}, opts_f);
    auto lse = torch::empty({N}, opts_f);
    linear_logp_probs_bf16_forward_kernel<<<
        N, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const nv_bfloat16 *>(logits.data_ptr<at::BFloat16>()),
        target_i.data_ptr<int>(), out.data_ptr<float>(), nullptr, lse.data_ptr<float>(),
        nullptr, N, V, logits.stride(0), 0, static_cast<int>(vocab_start_index));
    return {out, lse};
}

std::vector<torch::Tensor> linear_logp_local_probs_bf16_forward_impl(
    torch::Tensor logits, torch::Tensor target, int64_t vocab_start_index) {
    TORCH_CHECK(logits.is_cuda() && target.is_cuda(), "logits and target must be CUDA tensors");
    TORCH_CHECK(logits.scalar_type() == at::kBFloat16,
                "linear_logp_local_probs_bf16_forward requires bf16 logits");
    TORCH_CHECK(logits.dim() == 2, "logits must be 2-D [N, V]");
    TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");
    const int N = logits.size(0);
    const int V = logits.size(1);
    TORCH_CHECK(target.numel() == N, "target must have one id per row");

    c10::cuda::CUDAGuard device_guard(logits.device());
    auto target_i = target.reshape({N}).to(torch::kInt32).contiguous();
    auto opts_f = logits.options().dtype(torch::kFloat);
    auto local_target_logit = torch::empty({N}, opts_f);
    auto local_lse = torch::empty({N}, opts_f);
    auto probs = torch::empty_like(logits);
    linear_logp_probs_bf16_forward_kernel<<<
        N, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const nv_bfloat16 *>(logits.data_ptr<at::BFloat16>()),
        target_i.data_ptr<int>(), nullptr, local_target_logit.data_ptr<float>(),
        local_lse.data_ptr<float>(),
        reinterpret_cast<nv_bfloat16 *>(probs.data_ptr<at::BFloat16>()), N, V,
        logits.stride(0), probs.stride(0), static_cast<int>(vocab_start_index));
    return {local_target_logit, local_lse, probs};
}

std::vector<torch::Tensor> linear_logp_local_bf16_forward_impl(
    torch::Tensor logits, torch::Tensor target, int64_t vocab_start_index) {
    TORCH_CHECK(logits.is_cuda() && target.is_cuda(), "logits and target must be CUDA tensors");
    TORCH_CHECK(logits.scalar_type() == at::kBFloat16,
                "linear_logp_local_bf16_forward requires bf16 logits");
    TORCH_CHECK(logits.dim() == 2, "logits must be 2-D [N, V]");
    TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");
    const int N = logits.size(0);
    const int V = logits.size(1);
    TORCH_CHECK(target.numel() == N, "target must have one id per row");

    c10::cuda::CUDAGuard device_guard(logits.device());
    auto target_i = target.reshape({N}).to(torch::kInt32).contiguous();
    auto opts_f = logits.options().dtype(torch::kFloat);
    auto local_target_logit = torch::empty({N}, opts_f);
    auto local_lse = torch::empty({N}, opts_f);
    linear_logp_probs_bf16_forward_kernel<<<
        N, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const nv_bfloat16 *>(logits.data_ptr<at::BFloat16>()),
        target_i.data_ptr<int>(), nullptr, local_target_logit.data_ptr<float>(),
        local_lse.data_ptr<float>(), nullptr, N, V, logits.stride(0), 0,
        static_cast<int>(vocab_start_index));
    return {local_target_logit, local_lse};
}

std::vector<torch::Tensor> linear_logp_top_p_local_bf16_forward_impl(
    torch::Tensor logits,
    torch::Tensor target,
    torch::Tensor replay_ids,
    torch::Tensor replay_logprobs,
    torch::Tensor temperature,
    int64_t vocab_start_index) {
    TORCH_CHECK(logits.is_cuda() && target.is_cuda() && replay_ids.is_cuda() &&
                    replay_logprobs.is_cuda() && temperature.is_cuda(),
                "top-p logits, target, replay tensors, and temperature must be CUDA tensors");
    TORCH_CHECK(logits.scalar_type() == at::kBFloat16,
                "top-p local logp requires bf16 logits");
    TORCH_CHECK(replay_ids.scalar_type() == at::kInt,
                "top-p replay ids must be int32");
    TORCH_CHECK(replay_logprobs.scalar_type() == at::kFloat &&
                    temperature.scalar_type() == at::kFloat,
                "top-p replay logprobs and temperature must be fp32");
    TORCH_CHECK(logits.dim() == 2 && replay_ids.dim() == 2 && replay_logprobs.dim() == 2,
                "top-p local logp expects 2-D logits and replay tensors");
    const int N = logits.size(0);
    const int V = logits.size(1);
    const int K = replay_ids.size(1);
    TORCH_CHECK(target.numel() == N && replay_ids.size(0) == N &&
                    replay_logprobs.sizes() == replay_ids.sizes(),
                "top-p replay rows must align with logits and target");
    TORCH_CHECK(K > 0 && K <= 64, "top-p replay width must be in [1, 64]");
    TORCH_CHECK(temperature.numel() == N, "temperature must have one value per row");
    TORCH_CHECK(logits.is_contiguous() && target.is_contiguous() &&
                    replay_ids.is_contiguous() && replay_logprobs.is_contiguous() &&
                    temperature.is_contiguous(),
                "top-p local logp inputs must be contiguous");

    c10::cuda::CUDAGuard device_guard(logits.device());
    auto target_i = target.reshape({N}).to(torch::kInt32).contiguous();
    auto opts_f = logits.options().dtype(torch::kFloat);
    auto local_target_logit = torch::empty({N}, opts_f);
    auto local_lse = torch::empty({N}, opts_f);
    linear_logp_top_p_local_bf16_forward_kernel<<<
        N, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const nv_bfloat16 *>(logits.data_ptr<at::BFloat16>()),
        target_i.data_ptr<int>(), replay_ids.data_ptr<int>(),
        replay_logprobs.data_ptr<float>(), temperature.data_ptr<float>(),
        local_target_logit.data_ptr<float>(), local_lse.data_ptr<float>(), N, V, K,
        logits.stride(0), replay_ids.stride(0), replay_logprobs.stride(0),
        static_cast<int>(vocab_start_index));
    return {local_target_logit, local_lse};
}

torch::Tensor linear_logp_probs_bf16_to_dlogits_impl(torch::Tensor probs,
                                                     torch::Tensor target,
                                                     torch::Tensor grad_logp,
                                                     int64_t vocab_start_index) {
    TORCH_CHECK(probs.is_cuda() && target.is_cuda() && grad_logp.is_cuda(),
                "probs, target, and grad_logp must be CUDA tensors");
    TORCH_CHECK(probs.scalar_type() == at::kBFloat16,
                "linear_logp_probs_bf16_to_dlogits_ requires bf16 probs");
    TORCH_CHECK(probs.dim() == 2, "probs must be 2-D [N, V]");
    const int N = probs.size(0);
    const int V = probs.size(1);
    TORCH_CHECK(target.numel() == N, "target must have one id per row");
    TORCH_CHECK(grad_logp.numel() == N, "grad_logp must have one value per row");

    c10::cuda::CUDAGuard device_guard(probs.device());
    auto target_i = target.reshape({N}).to(torch::kInt32).contiguous();
    auto grad_f = grad_logp.reshape({N}).to(torch::kFloat).contiguous();
    const int threads = 256;
    const int64_t total = static_cast<int64_t>(N) * V;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    linear_logp_probs_bf16_to_dlogits_kernel<<<
        blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<nv_bfloat16 *>(probs.data_ptr<at::BFloat16>()), target_i.data_ptr<int>(),
        grad_f.data_ptr<float>(), N, V, probs.stride(0), static_cast<int>(vocab_start_index));
    return probs;
}

torch::Tensor linear_logp_local_probs_bf16_to_dlogits_impl(torch::Tensor probs,
                                                           torch::Tensor target,
                                                           torch::Tensor grad_logp,
                                                           torch::Tensor local_lse,
                                                           torch::Tensor global_lse,
                                                           int64_t vocab_start_index) {
    TORCH_CHECK(probs.is_cuda() && target.is_cuda() && grad_logp.is_cuda() &&
                    local_lse.is_cuda() && global_lse.is_cuda(),
                "probs, target, grad_logp, local_lse, and global_lse must be CUDA tensors");
    TORCH_CHECK(probs.scalar_type() == at::kBFloat16,
                "linear_logp_local_probs_bf16_to_dlogits_ requires bf16 probs");
    TORCH_CHECK(local_lse.scalar_type() == at::kFloat && global_lse.scalar_type() == at::kFloat,
                "local_lse and global_lse must be fp32");
    TORCH_CHECK(probs.dim() == 2, "probs must be 2-D [N, V]");
    const int N = probs.size(0);
    const int V = probs.size(1);
    TORCH_CHECK(target.numel() == N, "target must have one id per row");
    TORCH_CHECK(grad_logp.numel() == N, "grad_logp must have one value per row");
    TORCH_CHECK(local_lse.numel() == N && global_lse.numel() == N,
                "local_lse/global_lse must have one value per row");

    c10::cuda::CUDAGuard device_guard(probs.device());
    auto target_i = target.reshape({N}).to(torch::kInt32).contiguous();
    auto grad_f = grad_logp.reshape({N}).to(torch::kFloat).contiguous();
    auto local_lse_f = local_lse.reshape({N}).to(torch::kFloat).contiguous();
    auto global_lse_f = global_lse.reshape({N}).to(torch::kFloat).contiguous();
    const int threads = 256;
    if (save_probs_rowwise_dlogits_enabled() && V >= threads) {
        linear_logp_local_probs_bf16_to_dlogits_row_kernel<<<
            N, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<nv_bfloat16 *>(probs.data_ptr<at::BFloat16>()),
            target_i.data_ptr<int>(), grad_f.data_ptr<float>(), local_lse_f.data_ptr<float>(),
            global_lse_f.data_ptr<float>(), N, V, probs.stride(0),
            static_cast<int>(vocab_start_index));
    } else {
        const int64_t total = static_cast<int64_t>(N) * V;
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        linear_logp_local_probs_bf16_to_dlogits_kernel<<<
            blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<nv_bfloat16 *>(probs.data_ptr<at::BFloat16>()),
            target_i.data_ptr<int>(), grad_f.data_ptr<float>(), local_lse_f.data_ptr<float>(),
            global_lse_f.data_ptr<float>(), N, V, probs.stride(0),
            static_cast<int>(vocab_start_index));
    }
    return probs;
}

torch::Tensor linear_logp_logits_bf16_to_dlogits_impl(torch::Tensor logits,
                                                      torch::Tensor dlogits,
                                                      torch::Tensor target,
                                                      torch::Tensor grad_logp,
                                                      torch::Tensor lse,
                                                      int64_t vocab_start_index) {
    TORCH_CHECK(logits.is_cuda() && dlogits.is_cuda() && target.is_cuda() &&
                    grad_logp.is_cuda() && lse.is_cuda(),
                "logits, dlogits, target, grad_logp, and lse must be CUDA tensors");
    TORCH_CHECK(logits.scalar_type() == at::kBFloat16 &&
                    dlogits.scalar_type() == at::kBFloat16,
                "linear_logp_logits_bf16_to_dlogits requires bf16 logits/dlogits");
    TORCH_CHECK(lse.scalar_type() == at::kFloat,
                "linear_logp_logits_bf16_to_dlogits requires fp32 lse");
    TORCH_CHECK(logits.dim() == 2 && dlogits.dim() == 2,
                "logits and dlogits must be 2-D [N, V]");
    const int N = logits.size(0);
    const int V = logits.size(1);
    TORCH_CHECK(dlogits.size(0) == N && dlogits.size(1) == V,
                "dlogits shape must match logits");
    TORCH_CHECK(logits.stride(1) == 1 && dlogits.stride(1) == 1,
                "linear_logp_logits_bf16_to_dlogits requires unit last-dim stride");
    TORCH_CHECK(target.numel() == N && grad_logp.numel() == N && lse.numel() == N,
                "target, grad_logp, and lse must have one value per row");

    c10::cuda::CUDAGuard device_guard(logits.device());
    auto target_i = target.reshape({N}).to(torch::kInt32).contiguous();
    auto grad_f = grad_logp.reshape({N}).to(torch::kFloat).contiguous();
    auto lse_f = lse.reshape({N}).to(torch::kFloat).contiguous();
    const int threads = 256;
    if (V >= threads) {
        linear_logp_logits_bf16_to_dlogits_row_kernel<<<
            N, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const nv_bfloat16 *>(logits.data_ptr<at::BFloat16>()),
            reinterpret_cast<nv_bfloat16 *>(dlogits.data_ptr<at::BFloat16>()),
            target_i.data_ptr<int>(), grad_f.data_ptr<float>(), lse_f.data_ptr<float>(), N, V,
            logits.stride(0), dlogits.stride(0), static_cast<int>(vocab_start_index));
    } else {
        const int64_t total = static_cast<int64_t>(N) * V;
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        linear_logp_logits_bf16_to_dlogits_kernel<<<
            blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const nv_bfloat16 *>(logits.data_ptr<at::BFloat16>()),
            reinterpret_cast<nv_bfloat16 *>(dlogits.data_ptr<at::BFloat16>()),
            target_i.data_ptr<int>(), grad_f.data_ptr<float>(), lse_f.data_ptr<float>(), N, V,
            logits.stride(0), dlogits.stride(0), static_cast<int>(vocab_start_index));
    }
    return dlogits;
}

__global__ void fused_linear_logp_output_only_bwd_stream_kernel(
    const nv_bfloat16 *__restrict__ hidden,
    const nv_bfloat16 *__restrict__ weight,
    const int *__restrict__ target,
    const float *__restrict__ grad_logp,
    const float *__restrict__ lse,
    const float *__restrict__ bias,
    nv_bfloat16 *__restrict__ grad_weight,
    float *__restrict__ grad_bias,
    int N,
    int D,
    int V,
    int vocab_start_index,
    bool has_bias,
    bool compute_grad_bias) {
    __shared__ float reduce_buf[STREAM_BWD_VECS][STREAM_BWD_THREADS];

    const int tid = threadIdx.x;
    const int v_base = blockIdx.x * STREAM_BWD_VECS;
    const int slot_count = (D + STREAM_BWD_THREADS - 1 - tid) / STREAM_BWD_THREADS;

    float acc[STREAM_BWD_VECS][STREAM_BWD_MAX_SLOTS];
#pragma unroll
    for (int j = 0; j < STREAM_BWD_VECS; ++j) {
#pragma unroll
        for (int s = 0; s < STREAM_BWD_MAX_SLOTS; ++s)
            acc[j][s] = 0.0f;
    }
    float bias_acc[STREAM_BWD_VECS] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int n = 0; n < N; ++n) {
        float hvals[STREAM_BWD_MAX_SLOTS];
#pragma unroll
        for (int s = 0; s < STREAM_BWD_MAX_SLOTS; ++s) {
            const int d = tid + s * STREAM_BWD_THREADS;
            hvals[s] = (s < slot_count && d < D)
                           ? __bfloat162float(hidden[static_cast<int64_t>(n) * D + d])
                           : 0.0f;
        }

#pragma unroll
        for (int j = 0; j < STREAM_BWD_VECS; ++j) {
            const int v = v_base + j;
            float partial = 0.0f;
            if (v < V) {
#pragma unroll
                for (int s = 0; s < STREAM_BWD_MAX_SLOTS; ++s) {
                    const int d = tid + s * STREAM_BWD_THREADS;
                    if (s < slot_count && d < D) {
                        partial +=
                            hvals[s] *
                            __bfloat162float(weight[static_cast<int64_t>(v) * D + d]);
                    }
                }
            }
            reduce_buf[j][tid] = partial;
        }
        __syncthreads();

        for (int offset = STREAM_BWD_THREADS / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
#pragma unroll
                for (int j = 0; j < STREAM_BWD_VECS; ++j)
                    reduce_buf[j][tid] += reduce_buf[j][tid + offset];
            }
            __syncthreads();
        }

#pragma unroll
        for (int j = 0; j < STREAM_BWD_VECS; ++j) {
            const int v = v_base + j;
            if (v >= V)
                continue;
            float logit = reduce_buf[j][0];
            if (has_bias)
                logit += bias[v];
            float dz = -__expf(logit - lse[n]);
            if (target[n] == vocab_start_index + v)
                dz += 1.0f;
            dz *= grad_logp[n];
            if (tid == 0 && compute_grad_bias)
                bias_acc[j] += dz;
#pragma unroll
            for (int s = 0; s < STREAM_BWD_MAX_SLOTS; ++s) {
                const int d = tid + s * STREAM_BWD_THREADS;
                if (s < slot_count && d < D)
                    acc[j][s] += dz * hvals[s];
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < STREAM_BWD_VECS; ++j) {
        const int v = v_base + j;
        if (v >= V)
            continue;
#pragma unroll
        for (int s = 0; s < STREAM_BWD_MAX_SLOTS; ++s) {
            const int d = tid + s * STREAM_BWD_THREADS;
            if (s < slot_count && d < D) {
                grad_weight[static_cast<int64_t>(v) * D + d] = __float2bfloat16(acc[j][s]);
            }
        }
        if (tid == 0 && compute_grad_bias)
            grad_bias[v] = bias_acc[j];
    }
}

// 2D bf16 tensor map with swizzle pinned to NONE. This kernel reads its tiles
// with plain row-major ldmatrix addressing, so the TMA must write them
// unswizzled -- the shared init_tensor_map auto-selects a swizzle from the row
// stride, which would not match. Kept local so the shared helper stays untouched.
inline void init_tensor_map_noswizzle(CUtensorMap *tmap, const nv_bfloat16 *gmem,
                                      uint64_t gmem_height, uint64_t gmem_width,
                                      uint32_t box_height, uint32_t box_width) {
    uint64_t size[2] = {gmem_width, gmem_height};
    uint64_t stride[1] = {gmem_width * sizeof(nv_bfloat16)};
    uint32_t box[2] = {box_width, box_height};
    uint32_t elem_stride[2] = {1, 1};
    CUresult res = cuTensorMapEncodeTiled(
        tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void *)gmem, size, stride, box, elem_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(res == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed for fused_linear_logp_sm90");
}

} // namespace

std::vector<torch::Tensor> linear_logp_probs_bf16_forward(torch::Tensor logits,
                                                          torch::Tensor target,
                                                          int64_t vocab_start_index) {
    return linear_logp_probs_bf16_forward_impl(logits, target, vocab_start_index);
}

std::vector<torch::Tensor> linear_logp_bf16_forward(torch::Tensor logits,
                                                    torch::Tensor target,
                                                    int64_t vocab_start_index) {
    return linear_logp_bf16_forward_impl(logits, target, vocab_start_index);
}

std::vector<torch::Tensor> linear_logp_local_probs_bf16_forward(torch::Tensor logits,
                                                                torch::Tensor target,
                                                                int64_t vocab_start_index) {
    return linear_logp_local_probs_bf16_forward_impl(logits, target, vocab_start_index);
}

std::vector<torch::Tensor> linear_logp_local_bf16_forward(torch::Tensor logits,
                                                          torch::Tensor target,
                                                          int64_t vocab_start_index) {
    return linear_logp_local_bf16_forward_impl(logits, target, vocab_start_index);
}

std::vector<torch::Tensor> linear_logp_top_p_local_bf16_forward(
    torch::Tensor logits,
    torch::Tensor target,
    torch::Tensor replay_ids,
    torch::Tensor replay_logprobs,
    torch::Tensor temperature,
    int64_t vocab_start_index) {
    return linear_logp_top_p_local_bf16_forward_impl(
        logits, target, replay_ids, replay_logprobs, temperature, vocab_start_index);
}

torch::Tensor linear_logp_probs_bf16_to_dlogits_(torch::Tensor probs,
                                                 torch::Tensor target,
                                                 torch::Tensor grad_logp,
                                                 int64_t vocab_start_index) {
    return linear_logp_probs_bf16_to_dlogits_impl(probs, target, grad_logp, vocab_start_index);
}

torch::Tensor linear_logp_local_probs_bf16_to_dlogits_(torch::Tensor probs,
                                                       torch::Tensor target,
                                                       torch::Tensor grad_logp,
                                                       torch::Tensor local_lse,
                                                       torch::Tensor global_lse,
                                                       int64_t vocab_start_index) {
    return linear_logp_local_probs_bf16_to_dlogits_impl(probs, target, grad_logp, local_lse,
                                                       global_lse, vocab_start_index);
}

torch::Tensor linear_logp_logits_bf16_to_dlogits(torch::Tensor logits,
                                                 torch::Tensor dlogits,
                                                 torch::Tensor target,
                                                 torch::Tensor grad_logp,
                                                 torch::Tensor lse,
                                                 int64_t vocab_start_index) {
    return linear_logp_logits_bf16_to_dlogits_impl(logits, dlogits, target, grad_logp, lse,
                                                  vocab_start_index);
}

std::vector<torch::Tensor> fused_linear_logp_sm90_forward_impl(
    torch::Tensor hidden, torch::Tensor weight, torch::Tensor target,
    torch::optional<torch::Tensor> bias, int64_t vocab_start_index, bool return_target_logit,
    torch::optional<torch::Tensor> temperature, bool return_logits, int64_t real_vocab_size) {
    TORCH_CHECK(hidden.is_cuda() && weight.is_cuda(), "hidden and weight must be CUDA tensors");
    TORCH_CHECK(weight.device() == hidden.device(),
                "lm_head_weight must be on the same device as hidden");
    TORCH_CHECK(hidden.scalar_type() == at::kBFloat16, "hidden must be bfloat16");
    TORCH_CHECK(weight.scalar_type() == at::kBFloat16, "weight must be bfloat16");
    TORCH_CHECK(hidden.is_contiguous() && weight.is_contiguous(), "inputs must be contiguous");
    const int N = hidden.size(0);
    const int D = hidden.size(1);
    const int V = weight.size(0);
    TORCH_CHECK(weight.size(1) == D, "hidden/weight hidden-dim mismatch");
    TORCH_CHECK(D % BK == 0, "D must be a multiple of ", BK, " for the SM90 kernel");
    TORCH_CHECK(target.numel() == N, "target must have one id per token: expected ", N,
                " (hidden rows), got ", target.numel());
    if (bias.has_value()) {
        TORCH_CHECK(bias->device() == hidden.device(),
                    "bias must be on the same device as hidden");
        TORCH_CHECK(bias->numel() == V, "bias must have V=", V, " elements, got ", bias->numel());
    }
    const float *temp_ptr = nullptr;
    torch::Tensor temp_c;
    if (temperature.has_value()) {
        TORCH_CHECK(temperature->device() == hidden.device(),
                    "temperature must be on the same device as hidden");
        TORCH_CHECK(temperature->scalar_type() == at::kFloat,
                    "temperature must be float32");
        TORCH_CHECK(temperature->numel() == N,
                    "temperature must have one value per token: expected ", N,
                    ", got ", temperature->numel());
        temp_c = temperature->contiguous();
        temp_ptr = temp_c.data_ptr<float>();
        const float tmin = temp_c.min().item<float>();
        TORCH_CHECK(tmin > 0.0f, "temperature must be strictly positive, got min ", tmin);
    }

    auto opts_f = hidden.options().dtype(torch::kFloat);
    auto out_value = torch::empty({N}, opts_f);
    auto lse = torch::empty({N}, opts_f);
    torch::Tensor logits;
    float *logits_ptr = nullptr;
    if (return_logits) {
        logits = torch::empty({N, V}, opts_f);
        logits_ptr = logits.data_ptr<float>();
    }
    // Real-vocab masking normalizes the padded Megatron domain (152064) and
    // the unpadded serving domain (151936) onto one contract: lanes at or
    // beyond the real vocab contribute exactly -inf to the max and 0 to the
    // sumexp on every rank. ``real_vocab_size < 0`` disables masking (V is
    // all real).
    const int real_vocab_end =
        real_vocab_size < 0
            ? V
            : static_cast<int>(
                  std::min<int64_t>(std::max<int64_t>(real_vocab_size - vocab_start_index, 0), V));

    // TMA descriptors: box [rows=BM/BN, cols=BK], unswizzled (see helper above).
    CUtensorMap h_tmap, w_tmap;
    init_tensor_map_noswizzle(
        &h_tmap, reinterpret_cast<const nv_bfloat16 *>(hidden.data_ptr<at::BFloat16>()), N, D, BM,
        BK);
    init_tensor_map_noswizzle(
        &w_tmap, reinterpret_cast<const nv_bfloat16 *>(weight.data_ptr<at::BFloat16>()), V, D, BN,
        BK);

    const float *bias_ptr = nullptr;
    torch::Tensor bias_f;
    if (bias.has_value()) {
        bias_f = bias->to(torch::kFloat).contiguous();
        bias_ptr = bias_f.data_ptr<float>();
    }

    const int smem = STAGES * (BM * BK + BN * BK) * sizeof(nv_bfloat16) +
                     (BM * BN) * sizeof(float) + 3 * BM * sizeof(float) + STAGES * 8;
    const int row_blocks = (N + BM - 1) / BM;
    const int total_vtiles = (V + BN - 1) / BN;
    auto target_i = target.to(torch::kInt32).contiguous();

    // [contract] The vocab-split count is a function of V only: the combine
    // tree must not change with batch shape, occupancy, or SM count. Filling
    // the GPU for large batches is recovered by launching more row blocks (a
    // bit-neutral axis), never by changing n_split.
    int n_split = std::max(1, std::min(N_SPLIT_CONTRACT, total_vtiles));

    auto part_max = torch::empty({n_split, N}, opts_f);
    auto part_sum = torch::empty({n_split, N}, opts_f);
    auto part_zt = torch::empty({n_split, N}, opts_f);

    if (smem > 48 * 1024) {
        cudaFuncSetAttribute(fused_linear_logp_sm90_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    }

    dim3 grid(row_blocks, n_split);
    fused_linear_logp_sm90_kernel<<<grid, WG_THREADS, smem>>>(
        h_tmap, w_tmap, target_i.data_ptr<int>(), bias_ptr, temp_ptr, logits_ptr,
        part_max.data_ptr<float>(), part_sum.data_ptr<float>(), part_zt.data_ptr<float>(), N, D,
        V, n_split, static_cast<int>(vocab_start_index), real_vocab_end);

    const int combine_threads = 256;
    const int combine_blocks = (N + combine_threads - 1) / combine_threads;
    fused_linear_logp_sm90_combine_kernel<<<combine_blocks, combine_threads>>>(
        part_max.data_ptr<float>(), part_sum.data_ptr<float>(), part_zt.data_ptr<float>(),
        out_value.data_ptr<float>(), lse.data_ptr<float>(), N, n_split, return_target_logit);

    if (return_logits)
        return {out_value, lse, logits};
    return {out_value, lse};
}

// Forward: hidden [N, D] bf16, weight [V, D] bf16, target [N] int32, optional
// bias [V] f32. Returns (logp [N] f32, lse [N] f32). Logits are never
// materialized; peak extra memory is the per-CTA shared-memory tiles.
std::vector<torch::Tensor> fused_linear_logp_sm90_forward(torch::Tensor hidden,
                                                          torch::Tensor weight,
                                                          torch::Tensor target,
                                                          torch::optional<torch::Tensor> bias,
                                                          torch::optional<torch::Tensor> temperature,
                                                          bool return_logits,
                                                          int64_t real_vocab_size) {
    return fused_linear_logp_sm90_forward_impl(hidden, weight, target, bias, 0, false,
                                                temperature, return_logits, real_vocab_size);
}

// Vocab-parallel local-shard forward. Target ids stay in global-vocab
// coordinates and the kernel captures only the target logit owned by this shard.
// Returns (local_target_logit [N] f32, local_lse [N] f32).
std::vector<torch::Tensor> fused_linear_logp_sm90_global_target_forward(
    torch::Tensor hidden, torch::Tensor weight, torch::Tensor target,
    torch::optional<torch::Tensor> bias, int64_t vocab_start_index,
    torch::optional<torch::Tensor> temperature, int64_t real_vocab_size) {
    return fused_linear_logp_sm90_forward_impl(
        hidden, weight, target, bias, vocab_start_index, true, temperature, false,
        real_vocab_size);
}

// Backward fast path: compute local dlogits in one CUDA kernel, then use GEMMs
// for the linear gradients. It intentionally materializes local logits/dlogits
// to remove Python chunk loops and many small matmul dispatches from the hot
// path. Non-TP recomputes lse from the backward logits so its gradients match
// the shared chunked backward numerics. For vocab-parallel TP, lse must already
// be the global lse and target is still in global-vocab coordinates.
std::vector<torch::Tensor> fused_linear_logp_sm90_backward(torch::Tensor grad_logp,
                                                           torch::Tensor hidden,
                                                           torch::Tensor weight,
                                                           torch::Tensor target,
                                                           torch::Tensor lse,
                                                           torch::optional<torch::Tensor> bias,
                                                           int64_t vocab_start_index,
                                                           bool compute_grad_hidden,
                                                           bool compute_grad_weight,
                                                           bool compute_grad_bias,
                                                           bool use_global_lse) {
    TORCH_CHECK(hidden.is_cuda() && weight.is_cuda(), "hidden and weight must be CUDA tensors");
    TORCH_CHECK(weight.device() == hidden.device(),
                "lm_head_weight must be on the same device as hidden");
    TORCH_CHECK(hidden.is_contiguous() && weight.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(hidden.dim() == 2 && weight.dim() == 2, "hidden/weight must be 2-D tensors");
    const int N = hidden.size(0);
    const int D = hidden.size(1);
    const int V = weight.size(0);
    TORCH_CHECK(weight.size(1) == D, "hidden/weight hidden-dim mismatch");
    TORCH_CHECK(target.numel() == N, "target must have one id per token");
    TORCH_CHECK(lse.numel() == N, "lse must have one value per token");
    TORCH_CHECK(grad_logp.numel() == N, "grad_logp must have one value per token");
    if (bias.has_value()) {
        TORCH_CHECK(bias->device() == hidden.device(),
                    "bias must be on the same device as hidden");
        TORCH_CHECK(bias->numel() == V, "bias must have V=", V, " elements, got ",
                    bias->numel());
    }

    c10::cuda::CUDAGuard device_guard(hidden.device());
    auto opts_f = hidden.options().dtype(torch::kFloat);
    auto empty = torch::empty({0}, opts_f);

    if (streaming_output_backward_enabled() && !compute_grad_hidden && compute_grad_weight &&
        hidden.scalar_type() == at::kBFloat16 && weight.scalar_type() == at::kBFloat16 &&
        D <= STREAM_BWD_MAX_D) {
        if (!streaming_output_backward_scalar_mode() && !compute_grad_bias) {
            auto grad_weight = torch::empty_like(weight);
            auto grad_f = grad_logp.reshape({N}).to(torch::kFloat).contiguous();
            auto lse_f = lse.reshape({N}).to(torch::kFloat).contiguous();
            auto target_i = target.reshape({N}).to(torch::kInt32).contiguous();
            auto hidden_gemm = hidden.contiguous();
            const int tile_v = streaming_output_backward_vocab_tile(V);
            const int threads = 256;
            const bool use_cuda_tf32_logits = streaming_output_backward_cuda_logits_enabled();
            const bool use_fused_tile_dw =
                fused_tile_backward_grad_weight_enabled() &&
                can_use_grad_weight_tile_wmma(N, D, V) &&
                (tile_v % GRAD_W_WMMA_M == 0);
            const bool requested_bf16_mma_dz =
                streaming_output_backward_bf16_mma_dz_enabled() && D % BK == 0 &&
                !bias.has_value();
            const bool use_bf16_mma_logits =
                (streaming_output_backward_bf16_mma_logits_enabled() ||
                 (use_fused_tile_dw && requested_bf16_mma_dz)) &&
                D % BK == 0;
            const bool use_bf16_mma_dz =
                requested_bf16_mma_dz && !use_fused_tile_dw;
            const bool use_bf16_mma_tile = use_bf16_mma_logits || use_bf16_mma_dz;
            const bool use_grad_weight_wmma =
                !use_fused_tile_dw && streaming_output_backward_grad_weight_wmma_enabled();
            torch::Tensor hidden_logits;
            if (!use_bf16_mma_tile)
                hidden_logits = hidden.to(torch::kFloat).contiguous();
            torch::Tensor weight_logits_workspace;
            if (!use_bf16_mma_tile)
                weight_logits_workspace = torch::empty({tile_v, D}, opts_f);
            torch::Tensor logits_workspace;
            if (!use_bf16_mma_dz)
                logits_workspace = torch::empty({N, tile_v}, opts_f);
            torch::Tensor dlogits_workspace;
            if (!use_fused_tile_dw)
                dlogits_workspace = torch::empty({N, tile_v}, hidden.options());

            for (int v0 = 0; v0 < V; v0 += tile_v) {
                const int vc = std::min(tile_v, V - v0);
                auto weight_tile = weight.narrow(0, v0, vc);
                torch::Tensor dlogits_gemm;
                if (!use_fused_tile_dw)
                    dlogits_gemm = dlogits_workspace.narrow(1, 0, vc);
                if (use_bf16_mma_dz) {
                    launch_dlogits_tile_bf16_mma(
                        hidden, weight_tile, dlogits_gemm, target_i, grad_f, lse_f,
                        static_cast<int>(vocab_start_index + v0));
                } else {
                    auto logits = logits_workspace.narrow(1, 0, vc);
                    if (use_bf16_mma_logits) {
                        launch_logits_tile_bf16_mma(hidden, weight_tile, logits);
                    } else {
                        auto weight_logits = weight_logits_workspace.narrow(0, 0, vc);
                        weight_logits.copy_(weight_tile);
                        if (use_cuda_tf32_logits && can_use_logits_tile_tf32_wmma(N, D, vc))
                            launch_logits_tile_tf32_wmma(hidden_logits, weight_logits, logits);
                        else
                            at::mm_out(logits, hidden_logits, weight_logits.transpose(0, 1));
                    }
                    if (bias.has_value()) {
                        auto bias_tile = bias->narrow(0, v0, vc).to(torch::kFloat).contiguous();
                        logits.add_(bias_tile);
                    }

                    auto grad_weight_view = grad_weight.narrow(0, v0, vc);
                    if (use_fused_tile_dw) {
                        launch_grad_weight_from_logits_tile_wmma(
                            logits, hidden_gemm, target_i, grad_f, lse_f, grad_weight_view,
                            static_cast<int>(vocab_start_index + v0), false);
                        continue;
                    }

                    const int64_t total = static_cast<int64_t>(N) * vc;
                    const int blocks = static_cast<int>((total + threads - 1) / threads);
                    fused_linear_logp_backward_dlogits_bf16_kernel<<<
                        blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                        logits.data_ptr<float>(),
                        reinterpret_cast<nv_bfloat16 *>(dlogits_gemm.data_ptr<at::BFloat16>()),
                        target_i.data_ptr<int>(), grad_f.data_ptr<float>(), lse_f.data_ptr<float>(),
                        N, vc, logits.stride(0), dlogits_gemm.stride(0),
                        static_cast<int>(vocab_start_index + v0), false);
                }

                auto grad_weight_view = grad_weight.narrow(0, v0, vc);
                if (use_grad_weight_wmma && can_use_grad_weight_tile_wmma(N, D, vc)) {
                    launch_grad_weight_tile_wmma(dlogits_gemm, hidden_gemm, grad_weight_view);
                } else {
                    auto grad_weight_tile =
                        at::matmul(dlogits_gemm.transpose(0, 1), hidden_gemm);
                    grad_weight_view.copy_(grad_weight_tile);
                }
            }
            return {empty, grad_weight, empty};
        }

        auto grad_weight = torch::empty_like(weight);
        torch::Tensor grad_bias = empty;
        if (compute_grad_bias)
            grad_bias = torch::empty({V}, opts_f);

        auto grad_f = grad_logp.reshape({N}).to(torch::kFloat).contiguous();
        auto lse_f = lse.reshape({N}).to(torch::kFloat).contiguous();
        auto target_i = target.reshape({N}).to(torch::kInt32).contiguous();
        torch::Tensor bias_f;
        const float *bias_ptr = nullptr;
        if (bias.has_value()) {
            bias_f = bias->to(torch::kFloat).contiguous();
            bias_ptr = bias_f.data_ptr<float>();
        }

        const int blocks = (V + STREAM_BWD_VECS - 1) / STREAM_BWD_VECS;
        fused_linear_logp_output_only_bwd_stream_kernel<<<
            blocks, STREAM_BWD_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const nv_bfloat16 *>(hidden.data_ptr<at::BFloat16>()),
            reinterpret_cast<const nv_bfloat16 *>(weight.data_ptr<at::BFloat16>()),
            target_i.data_ptr<int>(), grad_f.data_ptr<float>(), lse_f.data_ptr<float>(),
            bias_ptr, reinterpret_cast<nv_bfloat16 *>(grad_weight.data_ptr<at::BFloat16>()),
            compute_grad_bias ? grad_bias.data_ptr<float>() : nullptr, N, D, V,
            static_cast<int>(vocab_start_index), bias.has_value(), compute_grad_bias);
        return {empty, grad_weight, grad_bias};
    }

    const std::string full_fused_tile_mode = fused_tile_backward_full_grad_mode();
    if ((full_fused_tile_mode == "tile_cublas" || full_fused_tile_mode == "tile" ||
         full_fused_tile_mode == "streaming" || full_fused_tile_mode == "tiled") &&
        (compute_grad_hidden || compute_grad_weight) && !compute_grad_bias &&
        !bias.has_value() && hidden.scalar_type() == at::kBFloat16 &&
        weight.scalar_type() == at::kBFloat16 && D % BK == 0) {
        auto grad_f = grad_logp.reshape({N}).to(torch::kFloat).contiguous();
        auto lse_f = lse.reshape({N}).to(torch::kFloat).contiguous();
        auto target_i = target.reshape({N}).to(torch::kInt32).contiguous();
        auto hidden_gemm = hidden.contiguous();
        const int tile_v = streaming_output_backward_vocab_tile(V);
        auto dlogits_workspace = torch::empty({N, tile_v}, hidden.options());

        torch::Tensor grad_hidden = empty;
        torch::Tensor grad_weight = empty;
        bool grad_hidden_initialized = false;
        if (compute_grad_hidden)
            grad_hidden = torch::empty_like(hidden);
        if (compute_grad_weight)
            grad_weight = torch::empty_like(weight);

        for (int v0 = 0; v0 < V; v0 += tile_v) {
            const int vc = std::min(tile_v, V - v0);
            auto weight_tile = weight.narrow(0, v0, vc);
            auto dlogits_gemm = dlogits_workspace.narrow(1, 0, vc);
            launch_dlogits_tile_bf16_mma(
                hidden, weight_tile, dlogits_gemm, target_i, grad_f, lse_f,
                static_cast<int>(vocab_start_index + v0));

            if (compute_grad_hidden) {
                if (!grad_hidden_initialized) {
                    at::mm_out(grad_hidden, dlogits_gemm, weight_tile);
                    grad_hidden_initialized = true;
                } else {
                    at::addmm_out(grad_hidden, grad_hidden, dlogits_gemm, weight_tile);
                }
            }
            if (compute_grad_weight) {
                auto grad_weight_view = grad_weight.narrow(0, v0, vc);
                at::mm_out(grad_weight_view, dlogits_gemm.transpose(0, 1), hidden_gemm);
            }
        }
        return {grad_hidden, grad_weight, empty};
    }

    const bool use_input_precision_gemm =
        fused_backward_use_input_precision_gemm(hidden, weight);
    const bool use_input_precision_logits = fused_backward_use_input_precision_logits();
    auto hidden_logits =
        use_input_precision_logits || hidden.scalar_type() == at::kFloat
            ? hidden
            : hidden.to(torch::kFloat).contiguous();
    auto weight_logits =
        use_input_precision_logits || weight.scalar_type() == at::kFloat
            ? weight
            : weight.to(torch::kFloat).contiguous();
    auto hidden_grad_gemm =
        use_input_precision_gemm || hidden.scalar_type() == at::kFloat
            ? hidden
            : hidden.to(torch::kFloat).contiguous();
    auto weight_grad_gemm =
        use_input_precision_gemm || weight.scalar_type() == at::kFloat
            ? weight
            : weight.to(torch::kFloat).contiguous();
    if (!hidden_logits.is_contiguous())
        hidden_logits = hidden_logits.contiguous();
    if (!weight_logits.is_contiguous())
        weight_logits = weight_logits.contiguous();
    if (!hidden_grad_gemm.is_contiguous())
        hidden_grad_gemm = hidden_grad_gemm.contiguous();
    if (!weight_grad_gemm.is_contiguous())
        weight_grad_gemm = weight_grad_gemm.contiguous();

    // Recompute local logits once with cuBLAS, then overwrite a fp32 workspace
    // with dlogits = grad_logp * (onehot(target) - softmax(logits)). For bf16
    // training, keep the gradient GEMMs in bf16 Tensor Core precision like
    // native linear backward; only the logits/softmax/dlogits workspace stays
    // fp32 by default. Set RL_KERNEL_LINEAR_LOGP_FUSED_BWD_PRECISION=input_all
    // to also recompute logits in input precision.
    auto logits = at::matmul(hidden_logits, weight_logits.transpose(0, 1)).contiguous();
    if (bias.has_value()) {
        auto bias_gemm = bias->to(logits.scalar_type()).contiguous();
        logits.add_(bias_gemm);
    }

    auto grad_f = grad_logp.reshape({N}).to(torch::kFloat).contiguous();
    auto lse_f = lse.reshape({N}).to(torch::kFloat).contiguous();
    auto target_i = target.reshape({N}).to(torch::kInt32).contiguous();
    const int threads = 256;
    const int64_t total = static_cast<int64_t>(N) * V;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    const bool use_rowwise_dlogits =
        fused_backward_rowwise_dlogits_enabled() && V >= threads;
    const bool use_bf16_dlogits_gemm =
        fused_backward_bf16_dlogits_enabled() && use_input_precision_gemm &&
        hidden.scalar_type() == at::kBFloat16 && weight.scalar_type() == at::kBFloat16 &&
        logits.scalar_type() == at::kFloat && !compute_grad_bias;

    torch::Tensor dlogits;
    torch::Tensor dlogits_gemm;
    torch::Tensor dlogits_source_for_fused_tile;
    bool dlogits_source_for_fused_tile_are_probs = false;
    if (use_bf16_dlogits_gemm) {
        // Default bf16 training path: write the GEMM input directly in bf16.
        // This skips one full [N,V] fp32->bf16 cast/read/write round trip before
        // the Tensor Core linear backward GEMMs. Bias grad still uses the older
        // fp32 dlogits path so its reduction keeps fp32 accumulation.
        auto dlogits_source =
            use_global_lse ? logits : at::softmax(logits.to(torch::kFloat), 1).contiguous();
        dlogits_source_for_fused_tile = dlogits_source;
        dlogits_source_for_fused_tile_are_probs = !use_global_lse;
        dlogits_gemm = torch::empty({N, V}, hidden.options());
        if (use_rowwise_dlogits) {
            fused_linear_logp_backward_dlogits_bf16_row_kernel<<<
                N, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                dlogits_source.data_ptr<float>(),
                reinterpret_cast<nv_bfloat16 *>(dlogits_gemm.data_ptr<at::BFloat16>()),
                target_i.data_ptr<int>(), grad_f.data_ptr<float>(), lse_f.data_ptr<float>(), N,
                V, dlogits_source.stride(0), dlogits_gemm.stride(0),
                static_cast<int>(vocab_start_index), !use_global_lse);
        } else {
            fused_linear_logp_backward_dlogits_bf16_kernel<<<
                blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                dlogits_source.data_ptr<float>(),
                reinterpret_cast<nv_bfloat16 *>(dlogits_gemm.data_ptr<at::BFloat16>()),
                target_i.data_ptr<int>(), grad_f.data_ptr<float>(), lse_f.data_ptr<float>(), N, V,
                dlogits_source.stride(0), dlogits_gemm.stride(0),
                static_cast<int>(vocab_start_index), !use_global_lse);
        }
    } else {
        dlogits = use_global_lse ? logits.to(torch::kFloat).contiguous()
                                 : at::softmax(logits.to(torch::kFloat), 1).contiguous();
        if (use_rowwise_dlogits) {
            fused_linear_logp_backward_dlogits_row_kernel<<<
                N, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                dlogits.data_ptr<float>(), target_i.data_ptr<int>(), grad_f.data_ptr<float>(),
                lse_f.data_ptr<float>(), N, V, static_cast<int>(vocab_start_index),
                !use_global_lse);
        } else {
            fused_linear_logp_backward_dlogits_kernel<<<blocks, threads, 0,
                                                        at::cuda::getCurrentCUDAStream()>>>(
                dlogits.data_ptr<float>(), target_i.data_ptr<int>(), grad_f.data_ptr<float>(),
                lse_f.data_ptr<float>(), N, V, static_cast<int>(vocab_start_index),
                !use_global_lse);
        }
        dlogits_gemm = use_input_precision_gemm ? dlogits.to(hidden.scalar_type()).contiguous()
                                                : dlogits;
    }

    torch::Tensor grad_hidden = empty;
    torch::Tensor grad_weight = empty;
    torch::Tensor grad_bias = empty;
    if (compute_grad_hidden)
        grad_hidden = at::matmul(dlogits_gemm, weight_grad_gemm);
    if (compute_grad_weight) {
        const std::string full_fused_tile_mode = fused_tile_backward_full_grad_mode();
        const bool use_full_fused_tile_dw =
            !full_fused_tile_mode.empty() && use_bf16_dlogits_gemm &&
            hidden_grad_gemm.scalar_type() == at::kBFloat16 &&
            can_use_grad_weight_tile_wmma(N, D, V);
        if (use_full_fused_tile_dw) {
            grad_weight = torch::empty_like(weight);
            if (full_fused_tile_mode == "from_logits" ||
                full_fused_tile_mode == "logits" ||
                full_fused_tile_mode == "recompute") {
                TORCH_CHECK(dlogits_source_for_fused_tile.defined(),
                            "from_logits full fused tile requires fp32 logits/probs source");
                launch_grad_weight_from_logits_tile_wmma(
                    dlogits_source_for_fused_tile, hidden_grad_gemm, target_i, grad_f, lse_f,
                    grad_weight, static_cast<int>(vocab_start_index),
                    dlogits_source_for_fused_tile_are_probs);
            } else {
                launch_grad_weight_tile_wmma(dlogits_gemm, hidden_grad_gemm, grad_weight);
            }
        } else {
            grad_weight = at::matmul(dlogits_gemm.transpose(0, 1), hidden_grad_gemm);
        }
    }
    if (compute_grad_bias)
        grad_bias = dlogits.sum(0);

    return {grad_hidden, grad_weight, grad_bias};
}
