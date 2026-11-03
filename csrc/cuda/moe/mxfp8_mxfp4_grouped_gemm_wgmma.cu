// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// P5-4 SM90 WGMMA grouped GEMM -- independent performance path (p5-wgmma-sm90-v1).
// Design: RL_KERNEL/P5_4_wgmma_plan.md. Build: KERNEL_ALIGN_MOE_SM90=1 ...
//
// Contract: a.codes uint8 E4M3 [M,K]; a.scales uint8 E8M0 [M,K/32];
// w.codes uint8 E2M1×2 [E,N,K/2] (low nibble=even k); w.scales uint8 E8M0 [E,N,K/32];
// expert_offsets int32 [E+1]; fwd->Y FP32 [M,N]; bwd->dX FP32 [M,K] (=dy·W, no dW).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/arch/barrier.h"
#include "cutlass/numeric_types.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/sm90_tile_scheduler_group.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/builders/sm90_common.inl"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/detail/blockwise_scale_layout.hpp"
#include "cute/tensor.hpp"
#include "cute/arch/mma_sm90.hpp"
#include "cute/arch/mma_sm90_gmma.hpp"

// Stock CUTLASS type probes; the compute path below uses CuTe GMMA directly.
namespace rl::moe::sm90 {

using ProblemShape =
    cutlass::gemm::GroupProblemShape<cute::Shape<int32_t, int32_t, int32_t>>;

// bwd: dX = dy*W; decoded B is RowMajor BF16.
using ElementA_bwd   = cutlass::bfloat16_t;
using ElementB_bwd   = cutlass::bfloat16_t;
using ElementC_bwd   = float;
using ElementAcc_bwd = float;
using LayoutA_bwd    = cutlass::layout::RowMajor;
using LayoutB_bwd    = cutlass::layout::RowMajor;
using LayoutC_bwd    = cutlass::layout::RowMajor;
using LayoutD_bwd    = LayoutC_bwd;

static constexpr int AlignA_bwd = 16;
static constexpr int AlignB_bwd = 16;
static constexpr int AlignC_bwd = 128 / cutlass::sizeof_bits<ElementC_bwd>::value;
static constexpr int AlignD_bwd = AlignC_bwd;

using TileShape_bwd    = cute::Shape<cute::_128, cute::_128, cute::_128>;
using ClusterShape_bwd = cute::Shape<cute::_1, cute::_1, cute::_1>;

using FusionOp_bwd = cutlass::epilogue::fusion::LinearCombination<
    ElementC_bwd, ElementAcc_bwd, ElementC_bwd, ElementAcc_bwd>;

using CollectiveEpilogue_bwd =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape_bwd, ClusterShape_bwd,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAcc_bwd, ElementAcc_bwd,
        ElementC_bwd, LayoutC_bwd*, AlignC_bwd,
        ElementC_bwd, LayoutD_bwd*, AlignD_bwd,
        cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative,
        FusionOp_bwd>::CollectiveOp;

using CollectiveMainloop_bwd =
    typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        ElementA_bwd, LayoutA_bwd*, AlignA_bwd,
        ElementB_bwd, LayoutB_bwd*, AlignB_bwd,
        ElementAcc_bwd, TileShape_bwd, ClusterShape_bwd,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue_bwd::SharedStorage))>,
        cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative>::CollectiveOp;

using GemmKernel_bwd =
    cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop_bwd,
                                         CollectiveEpilogue_bwd>;
using Gemm_bwd = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_bwd>;

static_assert(Gemm_bwd::GemmKernel::IsGroupedGemmKernel,
    "bwd must instantiate as a grouped (ptr-array) kernel");
static_assert(cute::is_base_of_v<
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative,
    typename Gemm_bwd::GemmKernel::CollectiveMainloop::DispatchPolicy::Schedule>,
    "bwd must use a ptr-array cooperative mainloop schedule");
static_assert(std::is_same_v<
    typename Gemm_bwd::GemmKernel::TileScheduler,
    cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Group<ProblemShape, 8>>,
    "bwd must use the static persistent group scheduler (stage=8)");

// Forward stock probe uses K=128 and a temporary E4M3 B.

using ElementA_fwd   = cutlass::float_e4m3_t;
using ElementB_fwd   = cutlass::float_e4m3_t;  // stand-in; 4d fork decodes E2M1 -> E4M3
using ElementC_fwd   = float;
using ElementAcc_fwd = float;
using LayoutA_fwd    = cutlass::layout::RowMajor;     // a.codes [M,K], K-contiguous
using LayoutB_fwd    = cutlass::layout::ColumnMajor;  // w.codes [E,N,K/2] -> B=[K,N]
using LayoutC_fwd    = cutlass::layout::RowMajor;
using LayoutD_fwd    = LayoutC_fwd;

static constexpr int AlignA_fwd = 32;  // E4M3 1byte, K%32 -> 32B
static constexpr int AlignB_fwd = 16;  // formal builder gate (sizeof(e4m3)*16 % 16 == 0)
static constexpr int AlignC_fwd = 128 / cutlass::sizeof_bits<ElementC_fwd>::value;  // = 4
static constexpr int AlignD_fwd = AlignC_fwd;

using TileShape_fwd    = cute::Shape<cute::_128, cute::_128, cute::_128>;
using ClusterShape_fwd = cute::Shape<cute::_1, cute::_1, cute::_1>;

using ScaleConfig_fwd = cutlass::detail::Sm90BlockwiseScaleConfig<1, 1, 128>;
using LayoutSFA_fwd = decltype(ScaleConfig_fwd::tile_atom_to_shape_SFA(
    cute::make_shape(cute::_128{}, cute::_128{}, cute::_128{}, cute::_1{})));
using LayoutSFB_fwd = decltype(ScaleConfig_fwd::tile_atom_to_shape_SFB(
    cute::make_shape(cute::_128{}, cute::_128{}, cute::_128{}, cute::_1{})));
using GmemLayoutPairA_fwd = cute::tuple<LayoutA_fwd*, LayoutSFA_fwd*>;
using GmemLayoutPairB_fwd = cute::tuple<LayoutB_fwd*, LayoutSFB_fwd*>;

using FusionOp_fwd = cutlass::epilogue::fusion::LinearCombination<
    ElementC_fwd, ElementAcc_fwd, ElementC_fwd, ElementAcc_fwd>;

using CollectiveEpilogue_fwd =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape_fwd, ClusterShape_fwd,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAcc_fwd, ElementAcc_fwd,
        ElementC_fwd, LayoutC_fwd*, AlignC_fwd,
        ElementC_fwd, LayoutD_fwd*, AlignD_fwd,
        cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative,
        FusionOp_fwd>::CollectiveOp;

using CollectiveMainloop_fwd =
    typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        ElementA_fwd, GmemLayoutPairA_fwd, AlignA_fwd,
        ElementB_fwd, GmemLayoutPairB_fwd, AlignB_fwd,
        ElementAcc_fwd, TileShape_fwd, ClusterShape_fwd,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue_fwd::SharedStorage))>,
        cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8Blockwise>::CollectiveOp;

using GemmKernel_fwd =
    cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop_fwd,
                                         CollectiveEpilogue_fwd>;
using Gemm_fwd = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_fwd>;

static_assert(Gemm_fwd::GemmKernel::IsGroupedGemmKernel,
    "fwd must instantiate as a grouped (ptr-array) kernel");
static_assert(cute::is_base_of_v<
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative,
    typename Gemm_fwd::GemmKernel::CollectiveMainloop::DispatchPolicy::Schedule>,
    "fwd must use a ptr-array cooperative mainloop schedule");
static_assert(std::is_same_v<
    typename Gemm_fwd::GemmKernel::TileScheduler,
    cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Group<ProblemShape, 8>>,
    "fwd must use the static persistent group scheduler (stage=8)");

}  // namespace rl::moe::sm90

// Device and shape checks.
namespace {

constexpr int kMoeMxBlockSize   = 32;  // MX block size (E8M0 granularity)
constexpr int kMoeBaseAlignBytes = 16;
constexpr int kMoePrepBlock     = 128;

__device__ __forceinline__ float moe_wgmma_e2m1(std::uint8_t nibble) {
  constexpr float magnitudes[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                                   2.0f, 3.0f, 4.0f, 6.0f};
  const float value = magnitudes[nibble & 7];
  return (nibble & 8) ? -value : value;
}

__device__ __forceinline__ float moe_wgmma_e8m0(std::uint8_t code) {
  return ldexpf(1.0f, static_cast<int>(code) - 127);
}

template <class SmemLayoutB>
__device__ __forceinline__ void moe_wgmma_decode_fwd_b_tile(
    const std::uint8_t* packed_stage,
    cutlass::float_e4m3_t* decoded,
    int valid_n, int valid_k, int stage, int thread_idx, int thread_count) {
  constexpr int tile_n = cute::size<0>(SmemLayoutB{});
  constexpr int tile_k = cute::size<1>(SmemLayoutB{});
  constexpr std::uint8_t e4m3_magnitude[8] = {
      0x00, 0x30, 0x38, 0x3c, 0x40, 0x44, 0x48, 0x4c};
  static_assert(tile_k % 2 == 0);
  auto sB = cute::make_tensor(cute::make_smem_ptr(decoded), SmemLayoutB{});
  for (int idx = thread_idx; idx < tile_n * tile_k; idx += thread_count) {
    const int n = idx / tile_k;
    const int k = idx % tile_k;
    std::uint8_t nibble = 0;
    if (n < valid_n && k < valid_k) {
      const std::uint8_t packed = packed_stage[n * (tile_k / 2) + k / 2];
      nibble = (packed >> ((k & 1) * 4)) & 15;
    }
    const std::uint8_t e4m3 = e4m3_magnitude[nibble & 7] | ((nibble & 8) << 4);
    sB(n, k, stage) = cutlass::float_e4m3_t::bitcast(e4m3);
  }
  cutlass::arch::fence_view_async_shared();
  __syncwarp();
}

template <class SmemLayoutB>
__device__ __forceinline__ void moe_wgmma_decode_bwd_b_tile(
    const std::uint8_t* packed_stage,
    const std::uint8_t* scale_stage,
    cutlass::bfloat16_t* decoded,
    int valid_k_model, int valid_n_model,
    int stage, int thread_idx, int thread_count) {
  constexpr int tile_k_model = cute::size<0>(SmemLayoutB{});
  constexpr int tile_n_model = cute::size<1>(SmemLayoutB{});
  static_assert(tile_k_model % 32 == 0);
  auto sB = cute::make_tensor(cute::make_smem_ptr(decoded), SmemLayoutB{});
  for (int idx = thread_idx; idx < tile_k_model * tile_n_model;
       idx += thread_count) {
    const int k_model = idx / tile_n_model;
    const int n_model = idx % tile_n_model;
    float value = 0.0f;
    if (k_model < valid_k_model && n_model < valid_n_model) {
      const std::uint8_t packed = packed_stage[
          n_model * (tile_k_model / 2) + k_model / 2];
      const std::uint8_t nibble = (packed >> ((k_model & 1) * 4)) & 15;
      const std::uint8_t sw = scale_stage[
          n_model * (tile_k_model / 32) + k_model / 32];
      value = __fmul_rn(moe_wgmma_e2m1(nibble), moe_wgmma_e8m0(sw));
    }
    sB(k_model, n_model, stage) = cutlass::bfloat16_t(value);
  }
  cutlass::arch::fence_view_async_shared();
  __syncwarp();
}

namespace wgmma_compute {

using FwdTile = cute::Shape<cute::_128, cute::_128, cute::_32>;
using BwdTile = cute::Shape<cute::_128, cute::_128, cute::_64>;
using WarpGroups = cute::Layout<cute::Shape<cute::_2, cute::_1, cute::_1>>;

using FwdMma = decltype(cute::make_tiled_mma(cute::GMMA::ss_op_selector<
    cutlass::float_e4m3_t, cutlass::float_e4m3_t, float, FwdTile,
    cute::GMMA::Major::K, cute::GMMA::Major::K>(), WarpGroups{}));
using BwdMma = decltype(cute::make_tiled_mma(cute::GMMA::ss_op_selector<
    cutlass::bfloat16_t, cutlass::bfloat16_t, float, BwdTile,
    cute::GMMA::Major::K, cute::GMMA::Major::MN>(), WarpGroups{}));

using FwdAtomA = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
    cute::GMMA::Major::K, cutlass::float_e4m3_t, cute::_128, cute::_32>());
using FwdAtomB = FwdAtomA;
using BwdAtomA = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
    cute::GMMA::Major::K, cutlass::bfloat16_t, cute::_128, cute::_64>());
using BwdAtomB = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
    cute::GMMA::Major::MN, cutlass::bfloat16_t, cute::_128, cute::_64>());

using FwdSmemA = decltype(cute::tile_to_shape(
    FwdAtomA{}, cute::make_shape(cute::_128{}, cute::_32{}, cute::_1{}),
    cute::Step<cute::_2, cute::_1, cute::_3>{}));
using FwdSmemB = decltype(cute::tile_to_shape(
    FwdAtomB{}, cute::make_shape(cute::_128{}, cute::_32{}, cute::_1{}),
    cute::Step<cute::_2, cute::_1, cute::_3>{}));
using BwdSmemA = decltype(cute::tile_to_shape(
    BwdAtomA{}, cute::make_shape(cute::_128{}, cute::_64{}, cute::_1{}),
    cute::Step<cute::_2, cute::_1, cute::_3>{}));
using BwdSmemB = decltype(cute::tile_to_shape(
    BwdAtomB{}, cute::make_shape(cute::_128{}, cute::_64{}, cute::_1{}),
    cute::Step<cute::_1, cute::_2, cute::_3>{}));

struct FwdShared {
  alignas(128) cutlass::float_e4m3_t a[cute::cosize_v<FwdSmemA>];
  alignas(128) std::uint8_t packed_b[128 * 16];
  alignas(128) cutlass::float_e4m3_t b[cute::cosize_v<FwdSmemB>];
  float sa[128];
  float sw[128];
};

struct BwdShared {
  alignas(128) cutlass::bfloat16_t a[cute::cosize_v<BwdSmemA>];
  alignas(128) std::uint8_t packed_b[64 * 64];
  alignas(128) std::uint8_t sw[64 * 4];
  alignas(128) cutlass::bfloat16_t b[cute::cosize_v<BwdSmemB>];
};

static_assert(cute::size(FwdMma{}) == 256);
static_assert(cute::size(BwdMma{}) == 256);

}  // namespace wgmma_compute

void moe_wgmma_check_cuda_contiguous(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void moe_wgmma_check_same_device(const torch::Tensor& first,
                                 const torch::Tensor& other, const char* name) {
  TORCH_CHECK(other.device() == first.device(), name,
              " must be on the same device as the first tensor");
}

void moe_wgmma_check_sm90(const torch::Tensor& t) {
  cudaDeviceProp properties{};
  TORCH_CHECK(cudaGetDeviceProperties(&properties, t.get_device()) == cudaSuccess,
              "P5-4 wgmma failed to query CUDA device properties");
  TORCH_CHECK(properties.major == 9,
              "P5-4 wgmma requires an H100/SM90 device, got compute capability ",
              properties.major, ".", properties.minor, ".");
}

void moe_wgmma_check_base_align(const torch::Tensor& t, const char* name) {
  const auto addr = reinterpret_cast<std::uintptr_t>(t.data_ptr());
  TORCH_CHECK(addr % kMoeBaseAlignBytes == 0,
              "P5-4 wgmma ", name, " base pointer must be ",
              kMoeBaseAlignBytes, "-byte aligned");
}

void moe_wgmma_check_int32_dim(std::int64_t value, const char* name) {
  TORCH_CHECK(value >= 0 && value <= std::numeric_limits<std::int32_t>::max(),
              name, " exceeds the int32 grouped-GEMM shape range");
}

// Offsets live on CUDA; copied to CPU only for wrapper validation.
std::vector<std::int32_t> moe_wgmma_copy_offsets(const torch::Tensor& offsets) {
  auto host = offsets.to(torch::kCPU).contiguous();
  const auto* data = host.data_ptr<std::int32_t>();
  return std::vector<std::int32_t>(data, data + host.numel());
}

std::vector<std::int32_t> moe_wgmma_tile_prefix(
    const std::vector<std::int32_t>& offsets) {
  std::vector<std::int32_t> prefix(offsets.size(), 0);
  std::int64_t count = 0;
  for (std::size_t e = 0; e + 1 < offsets.size(); ++e) {
    count += (static_cast<std::int64_t>(offsets[e + 1] - offsets[e]) + 127) / 128;
    TORCH_CHECK(count <= std::numeric_limits<std::int32_t>::max(),
                "P5-4 WGMMA tile prefix exceeds int32");
    prefix[e + 1] = static_cast<std::int32_t>(count);
  }
  return prefix;
}

}  // namespace

// One prep thread per expert.

__global__ void moe_get_group_gemm_starts_fwd(
    const cutlass::float_e4m3_t** a_ptrs,
    const std::uint8_t** b_ptrs,
    float** out_ptrs,
    const std::uint8_t** a_scales_ptrs,
    const std::uint8_t** b_scales_ptrs,
    std::int32_t* problem_sizes,
    const cutlass::float_e4m3_t* a_base,
    const std::uint8_t* b_base,
    float* out_base,
    const std::uint8_t* a_scales_base,
    const std::uint8_t* b_scales_base,
    const std::int32_t* expert_offsets,
    int E, int N, int K) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= E) return;
  const int row_begin = expert_offsets[e];
  const int m_e = expert_offsets[e + 1] - row_begin;
  const int blocks = K >> 5;  // K / 32

  a_ptrs[e]        = a_base        + static_cast<std::int64_t>(row_begin) * K;
  b_ptrs[e]        = b_base        + static_cast<std::int64_t>(e) * N * (K >> 1);
  out_ptrs[e]      = out_base      + static_cast<std::int64_t>(row_begin) * N;
  a_scales_ptrs[e] = a_scales_base + static_cast<std::int64_t>(row_begin) * blocks;
  b_scales_ptrs[e] = b_scales_base + static_cast<std::int64_t>(e) * N * blocks;

  problem_sizes[3 * e + 0] = m_e;
  problem_sizes[3 * e + 1] = N;
  problem_sizes[3 * e + 2] = K;
}

__global__ void moe_get_group_gemm_starts_bwd(
    const cutlass::bfloat16_t** dy_ptrs,
    const std::uint8_t** b_ptrs,
    const std::uint8_t** b_scales_ptrs,
    float** dx_ptrs,
    std::int32_t* problem_sizes,
    const cutlass::bfloat16_t* dy_base,
    const std::uint8_t* b_base,
    const std::uint8_t* b_scales_base,
    float* dx_base,
    const std::int32_t* expert_offsets,
    int E, int N, int K) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= E) return;
  const int row_begin = expert_offsets[e];
  const int m_e = expert_offsets[e + 1] - row_begin;
  const int blocks = K >> 5;  // K / 32

  dy_ptrs[e]       = dy_base       + static_cast<std::int64_t>(row_begin) * N;
  b_ptrs[e]        = b_base        + static_cast<std::int64_t>(e) * N * (K >> 1);
  b_scales_ptrs[e] = b_scales_base + static_cast<std::int64_t>(e) * N * blocks;
  dx_ptrs[e]       = dx_base       + static_cast<std::int64_t>(row_begin) * K;

  problem_sizes[3 * e + 0] = m_e;
  problem_sizes[3 * e + 1] = K;
  problem_sizes[3 * e + 2] = N;
}

__global__ void moe_grouped_gemm_fwd_wgmma_compute(
    const cutlass::float_e4m3_t* const* a_ptrs,
    const std::uint8_t* const* b_ptrs,
    float* const* out_ptrs,
    const std::uint8_t* const* a_scales_ptrs,
    const std::uint8_t* const* b_scales_ptrs,
    const std::int32_t* problem_sizes,
    const std::int32_t* tile_prefix,
    int E, int N, int K, int n_tiles) {
  using namespace cute;
  using namespace wgmma_compute;
  const int tid = threadIdx.x;
  const int global_m_tile = blockIdx.x / n_tiles;
  const int n_start = (blockIdx.x % n_tiles) * 128;
  int lo = 0, hi = E;
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    if (global_m_tile < tile_prefix[mid + 1]) hi = mid;
    else lo = mid + 1;
  }
  if (lo >= E) return;
  const int e = lo;
  const int m_start = (global_m_tile - tile_prefix[e]) * 128;
  const int m_e = problem_sizes[3 * e];
  if (m_start >= m_e) return;
  const int valid_m = min(128, m_e - m_start);
  const int valid_n = min(128, N - n_start);
  const auto* a = a_ptrs[e];
  const auto* b = b_ptrs[e];
  const auto* sa = a_scales_ptrs[e];
  const auto* sw = b_scales_ptrs[e];
  auto* out = out_ptrs[e];

  __shared__ FwdShared storage;
  auto sA = make_tensor(make_smem_ptr(storage.a), FwdSmemA{});
  auto sB = make_tensor(make_smem_ptr(storage.b), FwdSmemB{});
  FwdMma mma;
  auto wg_layout = make_layout(Int<2>{}, Int<128>{});
  auto thread_mma = mma.get_slice(wg_layout(tid / 128));
  auto tCrA = thread_mma.make_fragment_A(thread_mma.partition_A(sA));
  auto tCrB = thread_mma.make_fragment_B(thread_mma.partition_B(sB));
  auto accum = partition_fragment_C(mma, make_shape(_128{}, _128{}));
  auto partial = partition_fragment_C(mma, make_shape(_128{}, _128{}));
  auto identity = make_identity_tensor(make_shape(_128{}, _128{}));
  auto coords = mma.get_thread_slice(tid).partition_C(identity);
  for (int i = 0; i < size(accum); ++i) {
    accum(i) = 0.0f;
    partial(i) = 0.0f;
  }

  for (int kb = 0; kb < K / 32; ++kb) {
    const int k_start = kb * 32;
    for (int idx = tid; idx < 128 * 32; idx += 256) {
      const int m = idx / 32, k = idx % 32;
      sA(m, k, 0) = m < valid_m
          ? a[static_cast<std::int64_t>(m_start + m) * K + k_start + k]
                                : cutlass::float_e4m3_t::bitcast(0);
    }
    for (int idx = tid; idx < 128 * 16; idx += 256) {
      const int n = idx / 16, h = idx % 16;
      storage.packed_b[idx] = n < valid_n
          ? b[static_cast<std::int64_t>(n_start + n) * (K / 2)
              + k_start / 2 + h] : 0;
    }
    if (tid < 128) {
      storage.sa[tid] = tid < valid_m
          ? moe_wgmma_e8m0(sa[static_cast<std::int64_t>(m_start + tid)
              * (K / 32) + kb]) : 1.0f;
      storage.sw[tid] = tid < valid_n
          ? moe_wgmma_e8m0(sw[static_cast<std::int64_t>(n_start + tid)
              * (K / 32) + kb]) : 1.0f;
    }
    cutlass::arch::fence_view_async_shared();
    __syncthreads();
    moe_wgmma_decode_fwd_b_tile<FwdSmemB>(
        storage.packed_b, storage.b, valid_n, 32, 0, tid, 256);
    __syncthreads();

    mma.accumulate_ = GMMA::ScaleOut::Zero;
    warpgroup_fence_operand(partial);
    warpgroup_arrive();
    for (int k_atom = 0; k_atom < size<2>(tCrA); ++k_atom) {
      gemm(mma, tCrA(_, _, k_atom, 0), tCrB(_, _, k_atom, 0), partial);
      mma.accumulate_ = GMMA::ScaleOut::One;
    }
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    for (int i = 0; i < size(accum); ++i) {
      const auto coord = coords(i);
      const int m = get<0>(coord), n = get<1>(coord);
      const float scaled = __fmul_rn(
          __fmul_rn(partial(i), storage.sa[m]), storage.sw[n]);
      accum(i) = __fadd_rn(accum(i), scaled);
    }
    __syncthreads();
  }
  for (int i = 0; i < size(accum); ++i) {
    const auto coord = coords(i);
    const int m = get<0>(coord), n = get<1>(coord);
    if (m < valid_m && n < valid_n) {
      out[static_cast<std::int64_t>(m_start + m) * N + n_start + n] = accum(i);
    }
  }
}

__global__ void moe_grouped_gemm_bwd_wgmma_compute(
    const cutlass::bfloat16_t* const* dy_ptrs,
    const std::uint8_t* const* b_ptrs,
    const std::uint8_t* const* b_scales_ptrs,
    float* const* dx_ptrs,
    const std::int32_t* problem_sizes,
    const std::int32_t* tile_prefix,
    int E, int N, int K, int k_tiles) {
  using namespace cute;
  using namespace wgmma_compute;
  const int tid = threadIdx.x;
  const int global_m_tile = blockIdx.x / k_tiles;
  const int k_start = (blockIdx.x % k_tiles) * 128;
  int lo = 0, hi = E;
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    if (global_m_tile < tile_prefix[mid + 1]) hi = mid;
    else lo = mid + 1;
  }
  if (lo >= E) return;
  const int e = lo;
  const int m_start = (global_m_tile - tile_prefix[e]) * 128;
  const int m_e = problem_sizes[3 * e];
  if (m_start >= m_e) return;
  const int valid_m = min(128, m_e - m_start);
  const int valid_k = min(128, K - k_start);
  const auto* dy = dy_ptrs[e];
  const auto* b = b_ptrs[e];
  const auto* sw = b_scales_ptrs[e];
  auto* dx = dx_ptrs[e];

  __shared__ BwdShared storage;
  auto sA = make_tensor(make_smem_ptr(storage.a), BwdSmemA{});
  auto sB = make_tensor(make_smem_ptr(storage.b), BwdSmemB{});
  BwdMma mma;
  auto wg_layout = make_layout(Int<2>{}, Int<128>{});
  auto thread_mma = mma.get_slice(wg_layout(tid / 128));
  auto tCrA = thread_mma.make_fragment_A(thread_mma.partition_A(sA));
  auto tCrB = thread_mma.make_fragment_B(thread_mma.partition_B(sB));
  auto accum = partition_fragment_C(mma, make_shape(_128{}, _128{}));
  auto identity = make_identity_tensor(make_shape(_128{}, _128{}));
  auto coords = mma.get_thread_slice(tid).partition_C(identity);
  for (int i = 0; i < size(accum); ++i) accum(i) = 0.0f;
  mma.accumulate_ = GMMA::ScaleOut::Zero;

  for (int n_start = 0; n_start < N; n_start += 64) {
    const int valid_n = min(64, N - n_start);
    for (int idx = tid; idx < 128 * 64; idx += 256) {
      const int m = idx / 64, n = idx % 64;
      sA(m, n, 0) = m < valid_m && n < valid_n
          ? dy[static_cast<std::int64_t>(m_start + m) * N + n_start + n]
          : cutlass::bfloat16_t(0.0f);
    }
    for (int idx = tid; idx < 64 * 64; idx += 256) {
      const int n = idx / 64, h = idx % 64;
      storage.packed_b[idx] = n < valid_n && 2 * h < valid_k
          ? b[static_cast<std::int64_t>(n_start + n) * (K / 2)
              + k_start / 2 + h] : 0;
    }
    for (int idx = tid; idx < 64 * 4; idx += 256) {
      const int n = idx / 4, block = idx % 4;
      storage.sw[idx] = n < valid_n && block * 32 < valid_k
          ? sw[static_cast<std::int64_t>(n_start + n) * (K / 32)
              + k_start / 32 + block] : 127;
    }
    cutlass::arch::fence_view_async_shared();
    __syncthreads();
    moe_wgmma_decode_bwd_b_tile<BwdSmemB>(
        storage.packed_b, storage.sw, storage.b, valid_k, valid_n, 0, tid, 256);
    __syncthreads();

    warpgroup_fence_operand(accum);
    warpgroup_arrive();
    for (int k_atom = 0; k_atom < size<2>(tCrA); ++k_atom) {
      gemm(mma, tCrA(_, _, k_atom, 0), tCrB(_, _, k_atom, 0), accum);
      mma.accumulate_ = GMMA::ScaleOut::One;
    }
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    __syncthreads();
  }
  for (int i = 0; i < size(accum); ++i) {
    const auto coord = coords(i);
    const int m = get<0>(coord), k = get<1>(coord);
    if (m < valid_m && k < valid_k) {
      dx[static_cast<std::int64_t>(m_start + m) * K + k_start + k] = accum(i);
    }
  }
}

// Host compute launch.

void moe_launch_grouped_gemm_fwd_wgmma(
    const cutlass::float_e4m3_t** a_ptrs,
    const std::uint8_t** b_ptrs,
    float** out_ptrs,
    const std::uint8_t** a_scales_ptrs,
    const std::uint8_t** b_scales_ptrs,
    const std::int32_t* problem_sizes,
    const std::int32_t* tile_prefix,
    int total_m_tiles, int E, int N, int K,
    cudaStream_t stream) {
  const std::int64_t n_tiles = (static_cast<std::int64_t>(N) + 127) / 128;
  const std::int64_t grid = n_tiles * total_m_tiles;
  TORCH_CHECK(grid > 0 && grid <= std::numeric_limits<std::int32_t>::max(),
              "P5-4 WGMMA forward grid exceeds CUDA x dimension");
  moe_grouped_gemm_fwd_wgmma_compute<<<static_cast<int>(grid), 256, 0, stream>>>(
      a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs,
      problem_sizes, tile_prefix, E, N, K, static_cast<int>(n_tiles));
  const auto status = cudaGetLastError();
  TORCH_CHECK(status == cudaSuccess, "P5-4 WGMMA forward launch failed: ",
              cudaGetErrorString(status));
}

void moe_launch_grouped_gemm_bwd_wgmma(
    const cutlass::bfloat16_t** dy_ptrs,
    const std::uint8_t** b_ptrs,
    const std::uint8_t** b_scales_ptrs,
    float** dx_ptrs,
    const std::int32_t* problem_sizes,
    const std::int32_t* tile_prefix,
    int total_m_tiles, int E, int N, int K,
    cudaStream_t stream) {
  const std::int64_t k_tiles = (static_cast<std::int64_t>(K) + 127) / 128;
  const std::int64_t grid = k_tiles * total_m_tiles;
  TORCH_CHECK(grid > 0 && grid <= std::numeric_limits<std::int32_t>::max(),
              "P5-4 WGMMA backward grid exceeds CUDA x dimension");
  moe_grouped_gemm_bwd_wgmma_compute<<<static_cast<int>(grid), 256, 0, stream>>>(
      dy_ptrs, b_ptrs, b_scales_ptrs, dx_ptrs, problem_sizes,
      tile_prefix, E, N, K, static_cast<int>(k_tiles));
  const auto status = cudaGetLastError();
  TORCH_CHECK(status == cudaSuccess, "P5-4 WGMMA backward launch failed: ",
              cudaGetErrorString(status));
}

// --- Host entry points (ops.cpp bindings). ---

torch::Tensor moe_mxfp8_mxfp4_grouped_gemm_forward_wgmma(
    torch::Tensor activation_codes,
    torch::Tensor activation_scales,
    torch::Tensor packed_weight_codes,
    torch::Tensor weight_scales,
    torch::Tensor expert_offsets) {
  moe_wgmma_check_cuda_contiguous(activation_codes, "activation_codes");
  moe_wgmma_check_cuda_contiguous(activation_scales, "activation_scales");
  moe_wgmma_check_cuda_contiguous(packed_weight_codes, "packed_weight_codes");
  moe_wgmma_check_cuda_contiguous(weight_scales, "weight_scales");
  moe_wgmma_check_cuda_contiguous(expert_offsets, "expert_offsets");
  moe_wgmma_check_same_device(activation_codes, activation_scales, "activation_scales");
  moe_wgmma_check_same_device(activation_codes, packed_weight_codes, "packed_weight_codes");
  moe_wgmma_check_same_device(activation_codes, weight_scales, "weight_scales");
  moe_wgmma_check_same_device(activation_codes, expert_offsets, "expert_offsets");

  TORCH_CHECK(activation_codes.scalar_type() == torch::kUInt8,
              "activation_codes must be uint8 E4M3");
  TORCH_CHECK(activation_scales.scalar_type() == torch::kUInt8,
              "activation_scales must be uint8 E8M0");
  TORCH_CHECK(packed_weight_codes.scalar_type() == torch::kUInt8,
              "packed_weight_codes must be uint8 E2M1x2");
  TORCH_CHECK(weight_scales.scalar_type() == torch::kUInt8,
              "weight_scales must be uint8 E8M0");
  TORCH_CHECK(expert_offsets.scalar_type() == torch::kInt32,
              "expert_offsets must be int32");
  TORCH_CHECK(activation_codes.dim() == 2, "activation_codes must be [M,K]");
  TORCH_CHECK(activation_scales.dim() == 2, "activation_scales must be [M,K/32]");
  TORCH_CHECK(packed_weight_codes.dim() == 3, "packed_weight_codes must be [E,N,K/2]");
  TORCH_CHECK(weight_scales.dim() == 3, "weight_scales must be [E,N,K/32]");
  TORCH_CHECK(expert_offsets.dim() == 1, "expert_offsets must be [E+1]");

  const auto M = activation_codes.size(0);
  const auto K = activation_codes.size(1);
  const auto E = packed_weight_codes.size(0);
  const auto N = packed_weight_codes.size(1);
  moe_wgmma_check_int32_dim(M, "M");
  moe_wgmma_check_int32_dim(N, "N");
  moe_wgmma_check_int32_dim(K, "K");
  moe_wgmma_check_int32_dim(E, "E");
  TORCH_CHECK(E > 0 && E <= std::numeric_limits<std::int32_t>::max() / 3,
              "E must be positive and 3*E must fit int32");
  TORCH_CHECK(N > 0, "N must be positive");
  TORCH_CHECK(K > 0 && K % kMoeMxBlockSize == 0,
              "K must be positive and divisible by 32");
  TORCH_CHECK(activation_scales.size(0) == M &&
                  activation_scales.size(1) == K / kMoeMxBlockSize,
              "activation_scales must be [M,K/32]");
  TORCH_CHECK(packed_weight_codes.size(2) == K / 2,
              "packed_weight_codes must be [E,N,K/2]");
  TORCH_CHECK(weight_scales.size(0) == E && weight_scales.size(1) == N &&
                  weight_scales.size(2) == K / kMoeMxBlockSize,
              "weight_scales must be [E,N,K/32]");
  TORCH_CHECK(expert_offsets.size(0) == E + 1,
              "expert_offsets must have E+1 entries");

  moe_wgmma_check_sm90(activation_codes);
  moe_wgmma_check_base_align(activation_codes, "activation_codes");
  moe_wgmma_check_base_align(packed_weight_codes, "packed_weight_codes");

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(activation_codes));
  const auto offsets = moe_wgmma_copy_offsets(expert_offsets);
  TORCH_CHECK(offsets.front() == 0, "expert_offsets must start at 0");
  TORCH_CHECK(offsets.back() == M, "expert_offsets must end at M");
  for (int64_t e = 0; e < E; ++e) {
    TORCH_CHECK(offsets[e] <= offsets[e + 1], "expert_offsets must be non-decreasing");
  }

  auto output = torch::empty({M, N}, activation_codes.options().dtype(torch::kFloat32));
  moe_wgmma_check_base_align(output, "output");
  if (M == 0) return output;

  const auto tile_prefix = moe_wgmma_tile_prefix(offsets);
  auto tile_prefix_cpu = torch::empty(
      {E + 1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
  std::memcpy(tile_prefix_cpu.data_ptr<std::int32_t>(), tile_prefix.data(),
              tile_prefix.size() * sizeof(std::int32_t));
  auto tile_prefix_gpu = tile_prefix_cpu.to(activation_codes.device());

  // Per-expert pointer arrays [E] (int64) + problem sizes [3E] (int32), on device.
  auto opts = activation_codes.options();
  auto a_ptrs_t        = torch::empty({E}, opts.dtype(torch::kInt64));
  auto b_ptrs_t        = torch::empty({E}, opts.dtype(torch::kInt64));
  auto out_ptrs_t      = torch::empty({E}, opts.dtype(torch::kInt64));
  auto a_scales_ptrs_t = torch::empty({E}, opts.dtype(torch::kInt64));
  auto b_scales_ptrs_t = torch::empty({E}, opts.dtype(torch::kInt64));
  auto problem_sizes_t = torch::empty({3 * E}, opts.dtype(torch::kInt32));

  auto a_ptrs        = reinterpret_cast<const cutlass::float_e4m3_t**>(a_ptrs_t.data_ptr<std::int64_t>());
  auto b_ptrs        = reinterpret_cast<const std::uint8_t**>(b_ptrs_t.data_ptr<std::int64_t>());
  auto out_ptrs      = reinterpret_cast<float**>(out_ptrs_t.data_ptr<std::int64_t>());
  auto a_scales_ptrs = reinterpret_cast<const std::uint8_t**>(a_scales_ptrs_t.data_ptr<std::int64_t>());
  auto b_scales_ptrs = reinterpret_cast<const std::uint8_t**>(b_scales_ptrs_t.data_ptr<std::int64_t>());
  auto problem_sizes = problem_sizes_t.data_ptr<std::int32_t>();

  auto a_base        = reinterpret_cast<const cutlass::float_e4m3_t*>(activation_codes.data_ptr<std::uint8_t>());
  auto b_base        = packed_weight_codes.data_ptr<std::uint8_t>();
  auto out_base      = output.data_ptr<float>();
  auto a_scales_base = activation_scales.data_ptr<std::uint8_t>();
  auto b_scales_base = weight_scales.data_ptr<std::uint8_t>();
  auto offsets_ptr   = expert_offsets.data_ptr<std::int32_t>();

  auto stream = at::cuda::getCurrentCUDAStream();
  const int Ei = static_cast<int>(E), Ni = static_cast<int>(N), Ki = static_cast<int>(K);
  const int grid = (Ei + kMoePrepBlock - 1) / kMoePrepBlock;
  moe_get_group_gemm_starts_fwd<<<grid, kMoePrepBlock, 0, stream>>>(
      a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs, problem_sizes,
      a_base, b_base, out_base, a_scales_base, b_scales_base, offsets_ptr,
      Ei, Ni, Ki);
  const auto prep_status = cudaGetLastError();
  TORCH_CHECK(prep_status == cudaSuccess, "P5-4 WGMMA fwd prep launch failed: ",
              cudaGetErrorString(prep_status));
  moe_launch_grouped_gemm_fwd_wgmma(a_ptrs, b_ptrs, out_ptrs,
      a_scales_ptrs, b_scales_ptrs, problem_sizes,
      tile_prefix_gpu.data_ptr<std::int32_t>(), tile_prefix.back(),
      Ei, Ni, Ki, stream);
  return output;
}

torch::Tensor moe_mxfp8_mxfp4_grouped_gemm_backward_wgmma(
    torch::Tensor dy,
    torch::Tensor packed_weight_codes,
    torch::Tensor weight_scales,
    torch::Tensor expert_offsets) {
  moe_wgmma_check_cuda_contiguous(dy, "dy");
  moe_wgmma_check_cuda_contiguous(packed_weight_codes, "packed_weight_codes");
  moe_wgmma_check_cuda_contiguous(weight_scales, "weight_scales");
  moe_wgmma_check_cuda_contiguous(expert_offsets, "expert_offsets");
  moe_wgmma_check_same_device(dy, packed_weight_codes, "packed_weight_codes");
  moe_wgmma_check_same_device(dy, weight_scales, "weight_scales");
  moe_wgmma_check_same_device(dy, expert_offsets, "expert_offsets");

  TORCH_CHECK(dy.scalar_type() == torch::kBFloat16, "dy must be BF16 [M,N]");
  TORCH_CHECK(packed_weight_codes.scalar_type() == torch::kUInt8,
              "packed_weight_codes must be uint8 E2M1x2");
  TORCH_CHECK(weight_scales.scalar_type() == torch::kUInt8,
              "weight_scales must be uint8 E8M0");
  TORCH_CHECK(expert_offsets.scalar_type() == torch::kInt32,
              "expert_offsets must be int32");
  TORCH_CHECK(dy.dim() == 2, "dy must be [M,N]");
  TORCH_CHECK(packed_weight_codes.dim() == 3, "packed_weight_codes must be [E,N,K/2]");
  TORCH_CHECK(weight_scales.dim() == 3, "weight_scales must be [E,N,K/32]");
  TORCH_CHECK(expert_offsets.dim() == 1, "expert_offsets must be [E+1]");

  const auto M = dy.size(0);
  const auto N = dy.size(1);
  const auto E = packed_weight_codes.size(0);
  const auto K = packed_weight_codes.size(2) * 2;  // packed [E,N,K/2] -> K
  moe_wgmma_check_int32_dim(M, "M");
  moe_wgmma_check_int32_dim(N, "N");
  moe_wgmma_check_int32_dim(K, "K");
  moe_wgmma_check_int32_dim(E, "E");
  TORCH_CHECK(E > 0 && E <= std::numeric_limits<std::int32_t>::max() / 3,
              "E must be positive and 3*E must fit int32");
  TORCH_CHECK(N > 0, "backward N must be positive");
  TORCH_CHECK(K > 0 && K % kMoeMxBlockSize == 0,
              "K must be positive and divisible by 32");
  TORCH_CHECK(packed_weight_codes.size(1) == N, "dy N must match packed weight N");
  TORCH_CHECK(weight_scales.size(0) == E && weight_scales.size(1) == N &&
                  weight_scales.size(2) == K / kMoeMxBlockSize,
              "weight_scales must be [E,N,K/32]");
  TORCH_CHECK(expert_offsets.size(0) == E + 1,
              "expert_offsets must have E+1 entries");

  moe_wgmma_check_sm90(dy);
  moe_wgmma_check_base_align(dy, "dy");
  moe_wgmma_check_base_align(packed_weight_codes, "packed_weight_codes");

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(dy));
  const auto offsets = moe_wgmma_copy_offsets(expert_offsets);
  TORCH_CHECK(offsets.front() == 0, "expert_offsets must start at 0");
  TORCH_CHECK(offsets.back() == M, "expert_offsets must end at M");
  for (int64_t e = 0; e < E; ++e) {
    TORCH_CHECK(offsets[e] <= offsets[e + 1], "expert_offsets must be non-decreasing");
  }

  auto dx = torch::empty({M, K}, dy.options().dtype(torch::kFloat32));
  moe_wgmma_check_base_align(dx, "dx");
  if (M == 0) return dx;

  const auto tile_prefix = moe_wgmma_tile_prefix(offsets);
  auto tile_prefix_cpu = torch::empty(
      {E + 1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
  std::memcpy(tile_prefix_cpu.data_ptr<std::int32_t>(), tile_prefix.data(),
              tile_prefix.size() * sizeof(std::int32_t));
  auto tile_prefix_gpu = tile_prefix_cpu.to(dy.device());

  // Per-expert pointer arrays [E] (int64) + problem sizes [3E] (int32), on device.
  auto opts = dy.options();
  auto dy_ptrs_t       = torch::empty({E}, opts.dtype(torch::kInt64));
  auto b_ptrs_t        = torch::empty({E}, opts.dtype(torch::kInt64));
  auto b_scales_ptrs_t = torch::empty({E}, opts.dtype(torch::kInt64));
  auto dx_ptrs_t       = torch::empty({E}, opts.dtype(torch::kInt64));
  auto problem_sizes_t = torch::empty({3 * E}, opts.dtype(torch::kInt32));

  auto dy_ptrs       = reinterpret_cast<const cutlass::bfloat16_t**>(dy_ptrs_t.data_ptr<std::int64_t>());
  auto b_ptrs        = reinterpret_cast<const std::uint8_t**>(b_ptrs_t.data_ptr<std::int64_t>());
  auto b_scales_ptrs = reinterpret_cast<const std::uint8_t**>(b_scales_ptrs_t.data_ptr<std::int64_t>());
  auto dx_ptrs       = reinterpret_cast<float**>(dx_ptrs_t.data_ptr<std::int64_t>());
  auto problem_sizes = problem_sizes_t.data_ptr<std::int32_t>();

  auto dy_base       = reinterpret_cast<const cutlass::bfloat16_t*>(dy.data_ptr<at::BFloat16>());
  auto b_base        = packed_weight_codes.data_ptr<std::uint8_t>();
  auto b_scales_base = weight_scales.data_ptr<std::uint8_t>();
  auto dx_base       = dx.data_ptr<float>();
  auto offsets_ptr   = expert_offsets.data_ptr<std::int32_t>();

  auto stream = at::cuda::getCurrentCUDAStream();
  const int Ei = static_cast<int>(E), Ni = static_cast<int>(N), Ki = static_cast<int>(K);
  const int grid = (Ei + kMoePrepBlock - 1) / kMoePrepBlock;
  moe_get_group_gemm_starts_bwd<<<grid, kMoePrepBlock, 0, stream>>>(
      dy_ptrs, b_ptrs, b_scales_ptrs, dx_ptrs, problem_sizes,
      dy_base, b_base, b_scales_base, dx_base, offsets_ptr,
      Ei, Ni, Ki);
  const auto prep_status = cudaGetLastError();
  TORCH_CHECK(prep_status == cudaSuccess, "P5-4 WGMMA bwd prep launch failed: ",
              cudaGetErrorString(prep_status));
  moe_launch_grouped_gemm_bwd_wgmma(dy_ptrs, b_ptrs, b_scales_ptrs,
      dx_ptrs, problem_sizes, tile_prefix_gpu.data_ptr<std::int32_t>(),
      tile_prefix.back(), Ei, Ni, Ki, stream);
  return dx;
}
