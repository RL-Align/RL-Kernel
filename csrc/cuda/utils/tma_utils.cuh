// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors

#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cudaTypedefs.h>
#include <cstdint>
#include <iostream>

// Type Traits for TMA
template <typename T> struct TmaTypeTraits;

template<> struct TmaTypeTraits<nv_bfloat16> {
    static constexpr CUtensorMapDataType tmap_dtype = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
};
template<> struct TmaTypeTraits<float> {
    static constexpr CUtensorMapDataType tmap_dtype = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
};

// Host API
template <typename InType>
inline void init_tensor_map(
    CUtensorMap *tmap_ptr, const InType *gmem_ptr,
    uint64_t gmem_height, uint64_t gmem_width,
    uint32_t smem_height, uint32_t smem_width
) {
    constexpr uint32_t rank = 2;
    uint64_t size[rank]        = {gmem_width, gmem_height};
    uint64_t stride[rank - 1]  = {gmem_width * sizeof(InType)};
    uint32_t box_size[rank]    = {smem_width, smem_height};
    uint32_t elem_stride[rank] = {1, 1};

    const uint32_t smem_stride_B = smem_width * sizeof(InType);
    CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE;
    if (smem_stride_B == 32)       swizzle = CU_TENSOR_MAP_SWIZZLE_32B;
    else if (smem_stride_B == 64)  swizzle = CU_TENSOR_MAP_SWIZZLE_64B;
    else if (smem_stride_B == 128) swizzle = CU_TENSOR_MAP_SWIZZLE_128B;

    CUresult res = cuTensorMapEncodeTiled(
        tmap_ptr, TmaTypeTraits<InType>::tmap_dtype, rank,
        (void *)gmem_ptr, size, stride, box_size, elem_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    if (res != CUDA_SUCCESS) {
        std::cerr << "[RL-Kernel Error] cuTensorMapEncodeTiled failed!" << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Device API
__device__ inline void mbarrier_init(uint64_t addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "l"(addr), "r"(count));
}

__device__ inline void mbarrier_arrive(uint64_t addr) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" :: "l"(addr) : "memory");
}

__device__ inline void mbarrier_arrive_expect_tx(uint64_t addr, int size) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                 :: "l"(addr), "r"(size) : "memory");
}

__device__ inline void mbarrier_wait(uint64_t mbar_addr, int phase) {
    int ticks = 0x989680;
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n"
        "@!P1 bra.uni LAB_WAIT;\n"
        "}" :: "l"(mbar_addr), "r"(phase), "r"(ticks)
    );
}

__device__ inline void tma_2d_g2s(
    uint64_t dst_smem_addr,
    const void *tmap_ptr,
    int x,
    int y,
    uint64_t mbar_addr
) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3}], [%4];"
                 :: "l"(dst_smem_addr), "l"(tmap_ptr), "r"(x), "r"(y), "l"(mbar_addr) : "memory");
}
