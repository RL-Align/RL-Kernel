// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors

#include "../utils/tma_utils.cuh"
#include <math_constants.h>
#include <torch/extension.h>
#include <cub/cub.cuh>

#define TILE_V 4096

struct MaxOp {
    __device__ __forceinline__ float operator()(float a, float b) const {
        return fmaxf(a, b);
    }
};

template<int NUM_WARPS>
__global__ void fused_logp_online_tma_kernel(
    const __grid_constant__ CUtensorMap logits_tmap,
    const int* __restrict__ labels,
    const nv_bfloat16* __restrict__ logits_gmem,
    float* __restrict__ output_logp,
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
            int current_tile_size = min(TILE_V, vocab_size - col_offset);

            if (step > 0) mbarrier_wait(mma_mbar_addr, phase ^ 1);

            if (lane_id == 0) {
                tma_2d_g2s(smem_addr, &logits_tmap, col_offset, row_idx, tma_mbar_addr);
                mbarrier_arrive_expect_tx(tma_mbar_addr, current_tile_size * sizeof(nv_bfloat16));
            }
            phase ^= 1;
        }
    }
    else {
        const int consumer_tid = (warp_id - 1) * 32 + lane_id;
        const int num_consumers = (NUM_WARPS - 1) * 32;

        using BlockReduce = cub::BlockReduce<float, (NUM_WARPS - 1) * 32>;
        __shared__ typename BlockReduce::TempStorage temp_storage;

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
            float block_tile_max = BlockReduce(temp_storage).Reduce(tile_max, MaxOp{});
            __shared__ float s_tile_max;
            if (consumer_tid == 0) s_tile_max = block_tile_max;
            asm volatile("bar.sync 1, %0;" :: "n"(num_consumers));

            float tile_sum = 0.0f;
            for (int i = consumer_tid; i < current_tile_size; i += num_consumers) {
                float val = __bfloat162float(smem_logits[i]);
                tile_sum += expf(val - s_tile_max);
            }
            float block_tile_sum = BlockReduce(temp_storage).Sum(tile_sum);
            __shared__ float s_tile_sum;
            if (consumer_tid == 0) s_tile_sum = block_tile_sum;
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
            int label_idx = labels[row_idx];
            float label_val = __bfloat162float(logits_gmem[row_idx * vocab_size + label_idx]);
            output_logp[row_idx] = label_val - row_max - logf(row_sum);
        }
    }
}

torch::Tensor fused_logp_sm90_forward(torch::Tensor logits, torch::Tensor labels) {
    int B = logits.size(0);
    int V = logits.size(1);
    auto output = torch::empty({B}, logits.options().dtype(torch::kFloat));

    CUtensorMap logits_tmap;
    init_tensor_map(&logits_tmap,
                    reinterpret_cast<const nv_bfloat16 *>(logits.data_ptr<at::BFloat16>()), B, V, 1,
                    TILE_V);

    int smem_size = (TILE_V * sizeof(nv_bfloat16)) + 16;
    fused_logp_online_tma_kernel<4><<<B, 128, smem_size>>>(
        logits_tmap, labels.data_ptr<int>(),
        reinterpret_cast<const nv_bfloat16*>(logits.data_ptr<at::BFloat16>()),
        output.data_ptr<float>(), B, V
    );
    return output;
}
