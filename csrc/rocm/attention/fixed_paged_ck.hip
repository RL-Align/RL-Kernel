// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
// Instantiate the installed CK implementation with a fixed arithmetic schedule.
// CK headers remain external; no AITER/vLLM source or dispatcher is patched.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <array>
#include "fmha_fwd.hpp"

#ifndef RLK_CK_TILE_M
#define RLK_CK_TILE_M 128
#endif

namespace {
template <bool Causal, bool HasLSE, ck_tile::BlockAttentionKVCacheLoadModeEnum LoadMode>
void launch_fixed(fmha_batch_prefill_args a, hipStream_t stream) {
    using Config = FmhaFwdTypeConfig<FmhaFwdBf16>;
    constexpr int M = RLK_CK_TILE_M;
    constexpr int K = M == 64 ? 64 : 32;
    constexpr int W = M == 64 ? 16 : 32;
    using Shape = ck_tile::TileFmhaShape<
        ck_tile::sequence<M, 128, K, 128, K, 128>,
        ck_tile::sequence<4, 1, 1>, ck_tile::sequence<W, W, 16>,
        ck_tile::sequence<4, 1, 1>, ck_tile::sequence<W, W, 16>, true>;
    using Traits = ck_tile::TileFmhaBatchPrefillTraits<
        true, true, true, true, false, ck_tile::BlockAttentionBiasEnum::NO_BIAS,
        false, HasLSE, false, ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE,
        -1, false, false, 16,
        ck_tile::BlockAttentionKVCacheMemoryLayoutEnum::LINEAR_LAYOUT,
        ck_tile::BlockAttentionKVCacheLookupTableEnum::VLLM_BLOCK_TABLE_2D, LoadMode>;
    using Problem = ck_tile::BlockFmhaBatchPrefillPipelineProblem<
        Config::QDataType, Config::KDataType, Config::VDataType, Config::SaccDataType,
        Config::SMPLComputeDataType, Config::BiasDataType, Config::RandValOutputDataType,
        Config::LSEDataType, Config::PDataType, Config::OaccDataType, Config::ODataType,
        Shape, true, ck_tile::ComposedAttention<0, CK_TILE_FMHA_FWD_FAST_EXP2>,
        ck_tile::SimplifiedGenericAttentionMask<Causal>, false, 16, Traits>;
    using Pipeline = ck_tile::BlockFmhaBatchPrefillPipelineQRKSVSAsync<Problem>;
    using Epilogue = ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<
        Config::OaccDataType, Config::ODataType, true, true>>;
    using Kernel = ck_tile::FmhaBatchPrefillWithPagedKVCacheKernel<Pipeline, Epilogue>;
    auto [kargs, grid] = fmha_batch_prefill_create_kargs_and_grids<Kernel>(a);
    ck_tile::launch_kernel(ck_tile::stream_config{stream},
        ck_tile::make_kernel<Kernel::kBlockPerCu>(Kernel{}, grid, Kernel::BlockSize(), 0, kargs));
}

template <bool Causal, bool HasLSE>
void dispatch_load(fmha_batch_prefill_args a, hipStream_t stream) {
    using Mode = ck_tile::BlockAttentionKVCacheLoadModeEnum;
    // Keep 64-bit global addressing when either interleaved view exceeds 2GB.
    const auto k_bytes = int64_t(a.num_total_pages) * a.batch_stride_k * 2;
    const auto v_bytes = int64_t(a.num_total_pages) * a.batch_stride_v * 2;
    if (k_bytes > INT32_MAX || v_bytes > INT32_MAX)
        launch_fixed<Causal, HasLSE, Mode::GLOBAL_LOAD_LDS>(a, stream);
    else
        launch_fixed<Causal, HasLSE, Mode::BUFFER_LOAD>(a, stream);
}
} // namespace

// Pointer-only binding avoids importing the Torch C++ ABI into the CK build.
// Shape, dtype, device and metadata checks live at the RL-Kernel entry point.
void forward(const std::array<uint64_t, 9>& ptr,
             const std::array<int, 16>& dims,
             float scale, bool causal, bool has_lse) {
    fmha_batch_prefill_args a{};
    a.q_ptr = reinterpret_cast<void*>(ptr[0]);
    a.k_ptr = reinterpret_cast<void*>(ptr[1]);
    a.v_ptr = reinterpret_cast<void*>(ptr[2]);
    a.o_ptr = reinterpret_cast<void*>(ptr[3]);
    a.lse_ptr = has_lse ? reinterpret_cast<void*>(ptr[4]) : nullptr;
    a.seqstart_q_ptr = reinterpret_cast<void*>(ptr[5]);
    a.kv_page_indices = reinterpret_cast<void*>(ptr[6]);
    a.seqlen_k_ptr = reinterpret_cast<void*>(ptr[7]);
    auto stream = reinterpret_cast<hipStream_t>(ptr[8]);
    a.seqlen_q = dims[0];
    a.batch = dims[1];
    a.max_seqlen_q = dims[2];
    a.nhead_q = dims[3];
    a.nhead_k = dims[4];
    a.num_total_pages = dims[5];
    a.batch_stride_block_table = dims[6];
    a.stride_q = dims[7];
    a.stride_o = dims[7];
    a.stride_k = dims[8];
    a.stride_v = dims[9];
    a.nhead_stride_k = dims[10];
    a.nhead_stride_v = dims[11];
    a.batch_stride_k = dims[12];
    a.batch_stride_v = dims[13];
    a.nhead_stride_q = dims[14];
    a.nhead_stride_o = dims[14];
    a.nhead_stride_lse = dims[0];
    a.hdim_q = 128;
    a.hdim_v = 128;
    a.page_block_size = 16;
    a.kv_memory_layout = ck_tile::BlockAttentionKVCacheMemoryLayoutEnum::LINEAR_LAYOUT;
    a.kv_lookup_table = ck_tile::BlockAttentionKVCacheLookupTableEnum::VLLM_BLOCK_TABLE_2D;
    a.scale_s = scale;
    a.scale_p = 1;
    a.scale_o = 1;
    a.window_size_left = -1;
    a.window_size_right = causal ? 0 : -1;
    a.mask_type = static_cast<int>(causal ? ck_tile::GenericAttentionMaskEnum::MASK_FROM_BOTTOM_RIGHT
                                         : ck_tile::GenericAttentionMaskEnum::NO_MASK);
    if (causal) {
        if (has_lse) dispatch_load<true, true>(a, stream);
        else dispatch_load<true, false>(a, stream);
    } else {
        if (has_lse) dispatch_load<false, true>(a, stream);
        else dispatch_load<false, false>(a, stream);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward);
    m.attr("tile_m") = RLK_CK_TILE_M;
}
