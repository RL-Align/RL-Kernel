// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors

#include "../utils/tma_utils.cuh"
#include <cstdint>
#include <math_constants.h>
#include <torch/extension.h>
#include <tuple>

// CUtensorMap box dimensions cannot exceed 256 elements.
#define TILE_V 256

__device__ __forceinline__ float warp_reduce_max(float value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    return value;
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

template<int NUM_WARPS>
__global__ void fused_logp_online_tma_kernel(
    const __grid_constant__ CUtensorMap logits_tmap,
    const int* __restrict__ labels,
    const nv_bfloat16* __restrict__ logits_gmem,
    float* __restrict__ output_logp,
    float* __restrict__ max_out,     // Optional [batch_size]
    float* __restrict__ logsum_out,  // Optional [batch_size]
    int batch_size,
    int vocab_size)
{
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int row_idx = blockIdx.x;

    extern __shared__ __align__(1024) char smem[];
    const int smem_addr = static_cast<int>(__cvta_generic_to_shared(smem));
    nv_bfloat16* smem_logits = reinterpret_cast<nv_bfloat16*>(smem);

    const int tma_mbar_addr = smem_addr + (TILE_V * sizeof(nv_bfloat16));
    const int mma_mbar_addr = tma_mbar_addr + 8;

    if (warp_id == 0 && lane_id == 0) {
        mbarrier_init(tma_mbar_addr, 1);
        mbarrier_init(mma_mbar_addr, (NUM_WARPS - 1) * 32);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();

    int num_tiles = (vocab_size + TILE_V - 1) / TILE_V;
    int phase = 0;

    if (warp_id == 0) {
        for (int step = 0; step < num_tiles; ++step) {
            int col_offset = step * TILE_V;

            if (step > 0) mbarrier_wait(mma_mbar_addr, phase ^ 1);

            if (lane_id == 0) {
                tma_2d_g2s(smem_addr, &logits_tmap, col_offset, row_idx, tma_mbar_addr);
                // The barrier expects a full tile even when the last tile is partial.
                mbarrier_arrive_expect_tx(tma_mbar_addr, TILE_V * sizeof(nv_bfloat16));
            }
            phase ^= 1;
        }
    }
    else {
        const int consumer_tid = (warp_id - 1) * 32 + lane_id;
        const int num_consumers = (NUM_WARPS - 1) * 32;
        // The producer warp does not enter this branch; reduce across consumers only.
        __shared__ float warp_partials[NUM_WARPS - 1];
        __shared__ float s_tile_max;
        __shared__ float s_tile_sum;

        float row_max = -CUDART_INF_F;
        float row_sum = 0.0f;

        for (int step = 0; step < num_tiles; ++step) {
            int current_tile_size = min(TILE_V, vocab_size - (step * TILE_V));

            mbarrier_wait(tma_mbar_addr, phase);

            float tile_max = -CUDART_INF_F;
            for (int i = consumer_tid; i < current_tile_size; i += num_consumers) {
                float val = __bfloat162float(smem_logits[i]);
                tile_max = max(tile_max, val);
            }
            tile_max = warp_reduce_max(tile_max);
            if (lane_id == 0) warp_partials[warp_id - 1] = tile_max;
            asm volatile("bar.sync 1, %0;" :: "n"(num_consumers));
            if (consumer_tid == 0) {
                float block_tile_max = -CUDART_INF_F;
#pragma unroll
                for (int i = 0; i < NUM_WARPS - 1; ++i) {
                    block_tile_max = max(block_tile_max, warp_partials[i]);
                }
                s_tile_max = block_tile_max;
            }
            asm volatile("bar.sync 1, %0;" :: "n"(num_consumers));

            float tile_sum = 0.0f;
            for (int i = consumer_tid; i < current_tile_size; i += num_consumers) {
                float val = __bfloat162float(smem_logits[i]);
                tile_sum += expf(val - s_tile_max);
            }
            tile_sum = warp_reduce_sum(tile_sum);
            if (lane_id == 0) warp_partials[warp_id - 1] = tile_sum;
            asm volatile("bar.sync 1, %0;" :: "n"(num_consumers));
            if (consumer_tid == 0) {
                float block_tile_sum = 0.0f;
#pragma unroll
                for (int i = 0; i < NUM_WARPS - 1; ++i) {
                    block_tile_sum += warp_partials[i];
                }
                s_tile_sum = block_tile_sum;
            }
            asm volatile("bar.sync 1, %0;" :: "n"(num_consumers));

            if (consumer_tid == 0) {
                float new_max = max(row_max, s_tile_max);
                row_sum = row_sum * expf(row_max - new_max) + s_tile_sum * expf(s_tile_max - new_max);
                row_max = new_max;
            }
            asm volatile("bar.sync 1, %0;" :: "n"(num_consumers));

            mbarrier_arrive(mma_mbar_addr);
            phase ^= 1;
        }

        if (consumer_tid == 0) {
            float log_sum = logf(row_sum);
            if (max_out != nullptr) {
                max_out[row_idx] = row_max;
                logsum_out[row_idx] = log_sum;
            }
            int label_idx = labels[row_idx];
            if (label_idx >= 0 && label_idx < vocab_size) {
                int64_t label_offset = static_cast<int64_t>(row_idx) * vocab_size + label_idx;
                float label_val = __bfloat162float(logits_gmem[label_offset]);
                output_logp[row_idx] = label_val - row_max - log_sum;
            } else {
                output_logp[row_idx] = 0.0f;
            }
        }
    }
}

namespace {

void launch_fused_logp_sm90(
    const torch::Tensor& logits,
    const torch::Tensor& labels,
    const torch::Tensor& output,
    float* max_ptr,
    float* logsum_ptr) {
    int B = logits.size(0);
    int V = logits.size(1);

    CUtensorMap logits_tmap;
    init_tensor_map(&logits_tmap,
                    reinterpret_cast<const nv_bfloat16 *>(logits.data_ptr<at::BFloat16>()), B, V, 1,
                    TILE_V);

    int smem_size = (TILE_V * sizeof(nv_bfloat16)) + 16;
    fused_logp_online_tma_kernel<4><<<B, 128, smem_size>>>(
        logits_tmap, labels.data_ptr<int>(),
        reinterpret_cast<const nv_bfloat16*>(logits.data_ptr<at::BFloat16>()),
        output.data_ptr<float>(), max_ptr, logsum_ptr, B, V
    );
}

} // namespace

torch::Tensor fused_logp_sm90_forward(torch::Tensor logits, torch::Tensor labels) {
    auto output = torch::empty({logits.size(0)}, logits.options().dtype(torch::kFloat));
    launch_fused_logp_sm90(logits, labels, output, nullptr, nullptr);
    return output;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fused_logp_sm90_forward_with_lse(
    torch::Tensor logits,
    torch::Tensor labels) {
    auto output = torch::empty({logits.size(0)}, logits.options().dtype(torch::kFloat));
    auto row_max = torch::empty({logits.size(0)}, logits.options().dtype(torch::kFloat));
    auto log_sum = torch::empty({logits.size(0)}, logits.options().dtype(torch::kFloat));
    launch_fused_logp_sm90(
        logits, labels, output, row_max.data_ptr<float>(), log_sum.data_ptr<float>());
    return {output, row_max, log_sum};
}
