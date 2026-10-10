// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors

// Hopper (SM90) batch-invariant selected-token log-prob
// logp[n] = logits[n, target[n]] - logsumexp(logits[n, :])

#include "../utils/tma_utils.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <cub/cub.cuh>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <torch/extension.h>

namespace {

// A single TMA box dimension is capped at 256 elements
#define SMEM_TILE 4096
#define TMA_BOX 256
static constexpr int LOADS_PER_TILE = SMEM_TILE / TMA_BOX;

// CUDA 13's CCCL no longer exposes cub::Max/Sum as public functors. Keep the
// reduction tree explicit so the SM90 build remains portable and deterministic.
struct DeterministicMax {
    __device__ __forceinline__ float operator()(float lhs, float rhs) const {
        return lhs > rhs ? lhs : rhs;
    }
};

struct DeterministicSum {
    __device__ __forceinline__ float operator()(float lhs, float rhs) const {
        return lhs + rhs;
    }
};

template <typename T> __device__ __forceinline__ float to_float(T x);
template <> __device__ __forceinline__ float to_float<nv_bfloat16>(nv_bfloat16 x) {
    return __bfloat162float(x);
}
template <> __device__ __forceinline__ float to_float<float>(float x) { return x; }

template <typename T, int NUM_WARPS>
__global__ void batch_invariant_logp_sm90_kernel(const __grid_constant__ CUtensorMap logits_tmap,
                                                 const int *__restrict__ target,
                                                 const T *__restrict__ logits_gmem,
                                                 float *__restrict__ output_logp,
                                                 float *__restrict__ output_lse, int num_tokens,
                                                 int vocab_size, int ignore_index) {
    constexpr int NUM_THREADS = NUM_WARPS * 32;

    // one CTA per row; each warp streams a TMA tile of the row into shared memory
    const int tid = threadIdx.x;
    const int row_idx = blockIdx.x;

    extern __shared__ __align__(1024) char smem[];
    const int smem_addr = static_cast<int>(__cvta_generic_to_shared(smem));
    T *smem_logits = reinterpret_cast<T *>(smem);
    const int tma_mbar_addr = smem_addr + (SMEM_TILE * sizeof(T));

    if (tid == 0) {
        mbarrier_init(tma_mbar_addr, 1);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();

    const int v_aligned = (vocab_size / TMA_BOX) * TMA_BOX;
    const int num_tiles = (v_aligned + SMEM_TILE - 1) / SMEM_TILE;
    int phase = 0;

    using BlockReduce = cub::BlockReduce<float, NUM_THREADS>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float s_tile_max;

    float row_max = -CUDART_INF_F;
    float row_sum = 0.0f;

    for (int step = 0; step < num_tiles; ++step) {
        const int col_offset = step * SMEM_TILE;
        const int current_tile_size = min(SMEM_TILE, v_aligned - col_offset);

        if (tid == 0) {
            for (int j = 0; j < LOADS_PER_TILE; ++j) {
                const int col = col_offset + j * TMA_BOX;
                if (col < v_aligned) {
                    tma_2d_g2s(smem_addr + j * TMA_BOX * sizeof(T), &logits_tmap, col, row_idx,
                               tma_mbar_addr);
                }
            }
            mbarrier_arrive_expect_tx(tma_mbar_addr, current_tile_size * sizeof(T));
        }
        mbarrier_wait(tma_mbar_addr, phase);
        phase ^= 1;

        // tile max (fixed strided partition -> deterministic)
        float tile_max = -CUDART_INF_F;
        for (int i = tid; i < current_tile_size; i += NUM_THREADS) {
            tile_max = max(tile_max, to_float<T>(smem_logits[i]));
        }
        float block_tile_max = BlockReduce(temp_storage).Reduce(tile_max, DeterministicMax());
        if (tid == 0)
            s_tile_max = block_tile_max;
        __syncthreads();

        // tile sum of exp(x - tile_max).
        float tile_sum = 0.0f;
        for (int i = tid; i < current_tile_size; i += NUM_THREADS) {
            tile_sum += expf(to_float<T>(smem_logits[i]) - s_tile_max);
        }
        float block_tile_sum = BlockReduce(temp_storage).Reduce(tile_sum, DeterministicSum());

        // Online log-sum-exp merge of this tile into the running row state.
        if (tid == 0) {
            const float new_max = max(row_max, s_tile_max);
            row_sum = row_sum * expf(row_max - new_max) + block_tile_sum * expf(s_tile_max - new_max);
            row_max = new_max;
        }
        __syncthreads();
    }

    // Tail [v_aligned, V): fewer than TMA_BOX elements, read straight from global.
    const int tail = vocab_size - v_aligned;
    if (tail > 0) {
        const int64_t base = (int64_t)row_idx * vocab_size + v_aligned;
        float tail_max = -CUDART_INF_F;
        for (int i = tid; i < tail; i += NUM_THREADS) {
            tail_max = max(tail_max, to_float<T>(logits_gmem[base + i]));
        }
        float block_tail_max = BlockReduce(temp_storage).Reduce(tail_max, DeterministicMax());
        if (tid == 0)
            s_tile_max = block_tail_max;
        __syncthreads();

        float tail_sum = 0.0f;
        for (int i = tid; i < tail; i += NUM_THREADS) {
            tail_sum += expf(to_float<T>(logits_gmem[base + i]) - s_tile_max);
        }
        float block_tail_sum = BlockReduce(temp_storage).Reduce(tail_sum, DeterministicSum());
        if (tid == 0) {
            const float new_max = max(row_max, s_tile_max);
            row_sum = row_sum * expf(row_max - new_max) + block_tail_sum * expf(s_tile_max - new_max);
            row_max = new_max;
        }
        __syncthreads();
    }

    if (tid == 0) {
        const float lse = row_max + logf(row_sum);
        const int tgt = target[row_idx];
        if (tgt == ignore_index) {
            output_logp[row_idx] = 0.0f;
        } else {
            const float tgt_logit = to_float<T>(logits_gmem[(int64_t)row_idx * vocab_size + tgt]);
            output_logp[row_idx] = tgt_logit - lse;
        }
        output_lse[row_idx] = lse;
    }
}

template <typename T>
void launch_batch_invariant_logp_sm90(torch::Tensor logits, torch::Tensor target,
                                      torch::Tensor logp, torch::Tensor lse, int N, int V,
                                      int ignore_index) {
    constexpr int NUM_WARPS = 4;
    CUtensorMap logits_tmap;
    // Global [N, V]; TMA box = [1 row, TMA_BOX cols], unswizzled row-major.
    init_tensor_map<T>(&logits_tmap, reinterpret_cast<const T *>(logits.data_ptr<T>()), N, V, 1,
                       TMA_BOX);

    const int smem_size = (SMEM_TILE * sizeof(T)) + 16;
    batch_invariant_logp_sm90_kernel<T, NUM_WARPS><<<N, NUM_WARPS * 32, smem_size>>>(
        logits_tmap, target.data_ptr<int>(), reinterpret_cast<const T *>(logits.data_ptr<T>()),
        logp.data_ptr<float>(), lse.data_ptr<float>(), N, V, ignore_index);
}

// at::BFloat16 and nv_bfloat16 are layout-compatible; data_ptr<T> needs the ATen
// type, so bf16 gets its own launch that reinterprets the pointer.
void launch_batch_invariant_logp_sm90_bf16(torch::Tensor logits, torch::Tensor target,
                                           torch::Tensor logp, torch::Tensor lse, int N, int V,
                                           int ignore_index) {
    constexpr int NUM_WARPS = 4;
    CUtensorMap logits_tmap;
    init_tensor_map<nv_bfloat16>(
        &logits_tmap, reinterpret_cast<const nv_bfloat16 *>(logits.data_ptr<at::BFloat16>()), N, V,
        1, TMA_BOX);

    const int smem_size = (SMEM_TILE * sizeof(nv_bfloat16)) + 16;
    batch_invariant_logp_sm90_kernel<nv_bfloat16, NUM_WARPS><<<N, NUM_WARPS * 32, smem_size>>>(
        logits_tmap, target.data_ptr<int>(),
        reinterpret_cast<const nv_bfloat16 *>(logits.data_ptr<at::BFloat16>()),
        logp.data_ptr<float>(), lse.data_ptr<float>(), N, V, ignore_index);
}

} // namespace

std::vector<torch::Tensor> batch_invariant_logp_sm90_forward(torch::Tensor logits,
                                                             torch::Tensor target,
                                                             int64_t ignore_index) {
    TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor");
    const c10::cuda::CUDAGuard device_guard(logits.device());
    TORCH_CHECK(target.is_cuda() && target.device() == logits.device(),
                "target must be a CUDA tensor on the same device as logits");
    TORCH_CHECK(logits.dim() == 2, "logits must be 2-D [N, V]");
    TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");
    const int N = logits.size(0);
    const int V = logits.size(1);
    TORCH_CHECK(target.numel() == N, "target must have one id per row: expected ", N, ", got ",
                target.numel());

    // TMA requires the global row stride (V * elem_size) to be 16-byte aligned.
    const int elem_size = static_cast<int>(logits.element_size());
    TORCH_CHECK((static_cast<int64_t>(V) * elem_size) % 16 == 0,
                "batch_invariant_logp_sm90 requires the vocab row stride (V * elem_size) to be a "
                "multiple of 16 bytes; got V=",
                V, ", elem_size=", elem_size);

    auto opts_f = logits.options().dtype(torch::kFloat);
    auto logp = torch::empty({N}, opts_f);
    auto lse = torch::empty({N}, opts_f);
    auto target_i = target.to(torch::kInt32).contiguous();

    if (logits.scalar_type() == at::kBFloat16) {
        launch_batch_invariant_logp_sm90_bf16(logits, target_i, logp, lse, N, V,
                                              static_cast<int>(ignore_index));
    } else if (logits.scalar_type() == at::kFloat) {
        launch_batch_invariant_logp_sm90<float>(logits, target_i, logp, lse, N, V,
                                                static_cast<int>(ignore_index));
    } else {
        TORCH_CHECK(false, "batch_invariant_logp_sm90 supports only bfloat16 and float32 logits");
    }

    return {logp, lse};
}
