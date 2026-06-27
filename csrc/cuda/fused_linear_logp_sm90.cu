// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// Hopper (SM90) fused linear log-prob:
//     logp[n] = log_softmax(hidden[n] @ W^T + b)[target[n]]

#include "../utils/tma_utils.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <torch/extension.h>

namespace {

constexpr int BM = 256;       // tokens per CTA
constexpr int BN = 64;        // vocab per tile
constexpr int BK = 32;        // hidden-dim slice streamed per TMA load
constexpr int WARPS = 4;      // one warpgroup
constexpr int WG_THREADS = WARPS * 32; // 128
constexpr int STAGES = 2;     // double-buffering

constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;

constexpr int WARP_M = BM / WARPS;       // logit rows per warp
constexpr int M_TILES = WARP_M / MMA_M;  // MMA m-tiles each warp owns
constexpr int N_TILES = BN / MMA_N;      // 8 n-tiles per warp
constexpr int K_TILES = BK / MMA_K;      // MMA k-steps per TMA tile
constexpr int KK_GROUPS = BK / 32;       // 32-wide ldmatrix.x4 groups (2 k-steps each)

static_assert(WARP_M % MMA_M == 0, "rows per warp must be a multiple of MMA_M");
static_assert(BK % 32 == 0, "BK must be a multiple of 32 (ldmatrix.x4 spans 32 cols)");

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
                                              float *__restrict__ part_max,   // [n_split, N]
                                              float *__restrict__ part_sum,   // [n_split, N]
                                              float *__restrict__ part_zt,    // [n_split, N]
                                              int N, int D, int V, int n_split) {
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

        // Online softmax: threads stride over rows, each folding this tile's BN
        // columns into the running (max, sum) and capturing the target logit.
        for (int r = tid; r < num_rows; r += WG_THREADS) {
            const int tgt = target[row_base + r];
            float tmax = -CUDART_INF_F;
            for (int c = 0; c < BN; ++c) {
                const int col = col_base + c;
                if (col >= V)
                    break;
                float val = sLogits[r * BN + c];
                if (bias != nullptr)
                    val += bias[col];
                tmax = fmaxf(tmax, val);
                if (col == tgt)
                    sZt[r] = val;
            }
            float tsum = 0.0f;
            for (int c = 0; c < BN; ++c) {
                const int col = col_base + c;
                if (col >= V)
                    break;
                float val = sLogits[r * BN + c];
                if (bias != nullptr)
                    val += bias[col];
                tsum += __expf(val - tmax);
            }
            float old_max = sMax[r];
            float new_max = fmaxf(old_max, tmax);
            sSum[r] = sSum[r] * __expf(old_max - new_max) + tsum * __expf(tmax - new_max);
            sMax[r] = new_max;
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

// Merge per-split partials: M = max_s m_s, S = sum_s s_s*exp(m_s - M),
// zt = sum_s zt_s (exactly one split holds the target column), then
// logp = zt - (M + log S). One thread per token row.
__global__ void fused_linear_logp_sm90_combine_kernel(const float *__restrict__ part_max,
                                                      const float *__restrict__ part_sum,
                                                      const float *__restrict__ part_zt,
                                                      float *__restrict__ out_logp,
                                                      float *__restrict__ out_lse, int N,
                                                      int n_split) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= N)
        return;

    float M = -CUDART_INF_F;
    for (int s = 0; s < n_split; ++s)
        M = fmaxf(M, part_max[s * N + r]);

    float S = 0.0f;
    float zt = 0.0f;
    for (int s = 0; s < n_split; ++s) {
        const int idx = s * N + r;
        S += part_sum[idx] * __expf(part_max[idx] - M);
        zt += part_zt[idx];
    }
    const float lse = M + logf(S);
    out_logp[r] = zt - lse;
    out_lse[r] = lse;
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

// Forward: hidden [N, D] bf16, weight [V, D] bf16, target [N] int32, optional
// bias [V] f32. Returns (logp [N] f32, lse [N] f32). Logits are never
// materialized; peak extra memory is the per-CTA shared-memory tiles.
std::vector<torch::Tensor> fused_linear_logp_sm90_forward(torch::Tensor hidden,
                                                          torch::Tensor weight,
                                                          torch::Tensor target,
                                                          torch::optional<torch::Tensor> bias) {
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

    auto opts_f = hidden.options().dtype(torch::kFloat);
    auto logp = torch::empty({N}, opts_f);
    auto lse = torch::empty({N}, opts_f);

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

    // Split the vocab loop across CTAs so the grid fills the GPU: aim for a few
    // CTAs per SM, capped by the number of vocab tiles available to split.
    int sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    int target_ctas = sm_count * 4;
    int n_split = std::max(1, std::min(target_ctas / std::max(row_blocks, 1), total_vtiles));

    auto part_max = torch::empty({n_split, N}, opts_f);
    auto part_sum = torch::empty({n_split, N}, opts_f);
    auto part_zt = torch::empty({n_split, N}, opts_f);

    if (smem > 48 * 1024) {
        cudaFuncSetAttribute(fused_linear_logp_sm90_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    }

    dim3 grid(row_blocks, n_split);
    fused_linear_logp_sm90_kernel<<<grid, WG_THREADS, smem>>>(
        h_tmap, w_tmap, target_i.data_ptr<int>(), bias_ptr, part_max.data_ptr<float>(),
        part_sum.data_ptr<float>(), part_zt.data_ptr<float>(), N, D, V, n_split);

    const int combine_threads = 256;
    const int combine_blocks = (N + combine_threads - 1) / combine_threads;
    fused_linear_logp_sm90_combine_kernel<<<combine_blocks, combine_threads>>>(
        part_max.data_ptr<float>(), part_sum.data_ptr<float>(), part_zt.data_ptr<float>(),
        logp.data_ptr<float>(), lse.data_ptr<float>(), N, n_split);

    return {logp, lse};
}
