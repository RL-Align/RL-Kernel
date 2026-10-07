// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// Hopper (SM90) `wgmma` path for the MLP down projection, contract
// ``mlp-down-gemm-mma-v1``.
//
// Compiled into the extension through the repo-wide SM90 source set (setup.py
// `sm90_srcs`, selected by ``KERNEL_ALIGN_FORCE_SM90=1``), the same gate as the
// other ``*_sm90.cu`` rows: the default build does not compile this file, so the
// mma.sync kernel in mlp_down_gemm.cu is the only MLP down path there.
//
// Arithmetic contract (unchanged, and the reason this file is a drop-in
// replacement for the `mma.sync` kernel): the K reduction walks ascending
// k-chunks of 16, every output element is chained through ONE fp32 accumulator,
// there is no split-K and no atomics, bias is added once in fp32 after the whole
// reduction, and exactly one bf16 cast happens at the store. Tiling, staging and
// pipelining are performance only -- they never reorder work inside or across a
// k-chunk, so every output bit is unchanged. This is the same premise the row's
// Triton backend already relies on (wgmma.mma_async.m64n256k16 measured
// byte-identical to the hand-written mma.sync schedule).
//
// Staging: 2-D TMA bulk-tensor loads (`cp.async.bulk.tensor`) with mbarrier
// completion, TMA swizzle pinned to SWIZZLE_128B. The box inner extent is
// exactly the 128 B swizzle span, so every smem row is one contiguous global
// request -- the earlier 16-B-row (swizzle NONE) variant was correct but spent
// 8x the TMA requests and measured 120 TFLOP/s on this GPU, which is why it was
// replaced. TMA zero-fills every out-of-bounds element, which is exactly the
// masked-row / short-tail semantics the mma.sync path implements by hand.
//
// Shared-memory layout: the canonical 128B-swizzled GMMA layout. TMA's
// SWIZZLE_128B and the descriptor's layout type B128 are the *same* permutation
// of the same bytes (16-byte chunk c of a 128 B row is stored at c ^ (row % 8)),
// so one describes exactly what the other produced.
//
//   K-major operand [MN][K] (K contiguous): the tile is MN rows of 128 B, each
//     row holding 64 contiguous K elements. A k16 slab starts 32 B into the row.
//     Descriptor: layout B128, LBO = 1 (unused when swizzled), SBO = 64
//     (8 rows x 128 B = 1024 B), start += 2 per k16 slab, += 512 per 64 rows.
//
//   MN-major operand [K][MN] (MN contiguous): the tile is MN/64 regions of 64
//     reduction rows x 128 B, swizzled the same way. The 8 reduction rows of a
//     descriptor atom are 128 B apart, so the k8-chunk stride is 1024 B
//     (SBO = 64) and the 64-column MN-block stride is 8192 B (LBO = 512); one
//     k16 slab is 16 reduction rows = 2 KB, and a warpgroup is one MN block.
//
// Both operand majors therefore use the same 2-D TMA box (64 reduction elements
// x MN rows); only the tensor map's minor axis differs.
//
// Pipeline: STAGES smem slots, one mbarrier per slot, a single producer thread
// issuing the TMA for a slot, and per-warpgroup wgmma commit groups with
// `wgmma.wait_group<KG>` gating the refill: the slot being refilled is the one
// whose commit group already retired, so the async MMA can never read a tile
// that is being overwritten.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstdint>
#include <optional>

using nv_bf16 = __nv_bfloat16;

namespace sm90 {

using nv_bf16 = __nv_bfloat16;

// ---------------------------------------------------------------------------
// Tile geometry (per CTA: TM x TN outputs, BK-deep stages), instantiated per
// contraction rather than shared. The forward reads a K-major pair whose wide
// panel is B, so it runs Triton's 128x256 footprint (two warpgroups of
// m64n256k16); dx and dW read at least one MN-major operand and keep the
// 256x128 / four-warpgroup shape whose larger TM amortizes those gathers.
// ---------------------------------------------------------------------------
constexpr int FWD_STAGES = 4;      // forward ring depth (swept)
constexpr int FWD_BK = 64;         // forward stage width in K (swept)
constexpr int FWD_KG = 1;          // forward wgmma groups in flight (swept)
constexpr int FWD_GM = 8;          // forward raster group (swept)
constexpr int BWD_STAGES = 4;      // dx/dW ring depth

template <int TMv, int TNv, int NWGv, int STAGESv, int GMv, int BKv = 64, int KGv = 1>
struct Geo {
    static constexpr int TM = TMv;             // output rows per CTA
    static constexpr int TN = TNv;             // output cols per CTA
    static constexpr int NWG = NWGv;           // warpgroups stacked along M
    static constexpr int STAGES = STAGESv;
    static constexpr int RASTER_GM = GMv;
    static constexpr int BK = BKv;             // K elements staged per stage
    static constexpr int KC = BKv / 8;
    static constexpr int KG = KGv;             // wgmma groups kept in flight
    // The stage width fixes the TMA swizzle and therefore the descriptor: 64 K
    // elements is a 128 B row (SWIZZLE_128B, layout type 1), 32 is a 64 B row
    // (SWIZZLE_64B, layout type 2). Everything else in the descriptor scales off
    // the row pitch: the 8-row core-matrix stride and a warpgroup's 64-row block.
    static constexpr int ROW_BYTES = BKv * 2;
    static constexpr int LAYOUT_TYPE = (ROW_BYTES == 128) ? 1 : 2;
    static constexpr int SWIZZLE = (ROW_BYTES == 128) ? 3 : 2;  // CU_TENSOR_MAP_SWIZZLE_*
    static constexpr int SBO_U128 = 8 * ROW_BYTES / 16;
    static constexpr int AWG_U128 = 64 * ROW_BYTES / 16;
    static constexpr int THREADS = NWGv * 128;
    static constexpr int A_BYTES = TMv * BKv * 2;
    static constexpr int B_BYTES = TNv * BKv * 2;
    static constexpr int STAGE_BYTES = A_BYTES + B_BYTES;
    static constexpr int SMEM_BYTES = STAGESv * STAGE_BYTES + STAGESv * 8 + 64;
    static_assert(NWGv == TMv / 64, "one warpgroup per 64-row block");
    static_assert(SMEM_BYTES <= 227 * 1024, "stage ring exceeds the SM90 dynamic smem limit");
};

// Forward: the narrow-stage configuration, so STAGES can be deep without the
// prefetch collapsing (BK=32 -> 24 KB per stage -> 8 stages in 192 KB).
using GeoFwd = Geo<128, 256, 2, FWD_STAGES, FWD_GM, FWD_BK, FWD_KG>;
// dx/dW: unchanged 64-wide stages, 4 deep.
using GeoBwd = Geo<256, 128, 4, (BWD_STAGES), 4>;

// wgmma.mma_async is sm_90a only. The extension is built with both a plain and
// an arch-native gencode; this file's device code is inert in the plain pass
// (the runtime selects the sm_90a cubin on a cc 9.x device).
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define RL_MLP_WGMMA 1
#else
#define RL_MLP_WGMMA 0
#endif

#if RL_MLP_WGMMA

// ---------------------------------------------------------------------------
// GMMA shared-memory descriptor, swizzled to the 128B TMA layout
// ---------------------------------------------------------------------------
// Layout type B128 (1): rows are 128 B (64 bf16) apart and the two k8 chunks of
// a k16 slab sit 16 B apart inside the row, both XOR-swizzled by (row % 8) --
// the identical permutation TMA's CU_TENSOR_MAP_SWIZZLE_128B applies, so the two
// describe the same bytes.
__device__ __forceinline__ uint64_t gmma_desc(uint32_t addr_u128, uint32_t lbo_u128,
                                             uint32_t sbo_u128, int layout) {
    return static_cast<uint64_t>(addr_u128 & 0x3FFFu) |
           (static_cast<uint64_t>(lbo_u128 & 0x3FFFu) << 16) |
           (static_cast<uint64_t>(sbo_u128 & 0x3FFFu) << 32) |
           (static_cast<uint64_t>(layout & 0x3u) << 62);
}

// ---------------------------------------------------------------------------
// mbarrier + TMA
// ---------------------------------------------------------------------------
__device__ __forceinline__ void mbar_init(uint32_t addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(addr), "r"(count));
}

__device__ __forceinline__ void mbar_arrive_expect_tx(uint32_t addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(addr),
        "r"(bytes)
        : "memory");
}

__device__ __forceinline__ void mbar_wait(uint32_t addr, int phase) {
    // Non-blocking check first (the common case: the stage is already in), then a
    // short-suspend poll loop so a spinning warp does not burn the issue slots.
    asm volatile(
        "{\n\t"
        ".reg .pred P;\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, [%0], %1;\n\t"
        "@P bra DONE_%=;\n\t"
        "POLL_%=:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, [%0], %1, 1000;\n\t"
        "@!P bra POLL_%=;\n\t"
        "DONE_%=:\n\t"
        "}\n" ::"r"(addr),
        "r"(phase)
        : "memory");
}

// One 2-D TMA box: dim0 = 64 reduction elements (128 B, the swizzle span),
// dim1 = the MN rows covered by the box, swizzled to SWIZZLE_128B.
__device__ __forceinline__ void tma_2d(uint32_t dst, const void* tmap, int x, int y,
                                       uint32_t mbar) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];" ::"r"(dst),
        "l"(tmap), "r"(x), "r"(y), "r"(mbar)
        : "memory");
}

// ---------------------------------------------------------------------------
// wgmma
// ---------------------------------------------------------------------------
__device__ __forceinline__ void wgmma_fence() {
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void wgmma_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
}

template <int PENDING>
__device__ __forceinline__ void wgmma_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(PENDING) : "memory");
}

// D[64x256] += A[64x16] * B[256x16]; n256 halves the wgmma issue count per unit
// of work versus two n128 issues and matches the tile Triton picks.
template <int TA, int TB>
__device__ __forceinline__ void wgmma_m64n256k16(uint64_t da, uint64_t db, float (&d)[128]) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, 1, 0;\n\t"
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16\n\t"
                "    {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15,\n\t"
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31,\n\t"
        "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47,\n\t"
        "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63,\n\t"
        "%64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79,\n\t"
        "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95,\n\t"
        "%96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111,\n\t"
        "%112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, %128, %129, p, 1, 1, %130, %131;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]), "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
        : "l"(da), "l"(db), "n"(TA), "n"(TB));
}

// D[64xTN] += A[64x16] * B[TN x16], fp32 accumulate, bf16 inputs.
template <int TA, int TB>
__device__ __forceinline__ void wgmma_m64n128k16(uint64_t da, uint64_t db, float (&d)[64]) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, 1, 0;\n\t"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16\n\t"
        "    {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15,\n\t"
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31,\n\t"
        "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47,\n\t"
        "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63},"
        " %64, %65, p, 1, 1, %66, %67;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
        : "l"(da), "l"(db), "n"(TA), "n"(TB));
}

__device__ __forceinline__ void store_pair(nv_bf16* dst, float lo, float hi) {
    const uint32_t v = static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16(lo))) |
                       (static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16(hi))) << 16);
    *reinterpret_cast<uint32_t*>(dst) = v;
}

#endif  // RL_MLP_WGMMA


// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
template <bool A_MN, bool B_MN, bool HAS_BIAS, class G>
__global__ void __launch_bounds__(G::THREADS, 1)
mlp_down_gemm_sm90_kernel(const __grid_constant__ CUtensorMap tmap_a,
                          const __grid_constant__ CUtensorMap tmap_b,
                          const float* __restrict__ bias, nv_bf16* __restrict__ out, int M, int N,
                          int K) {
#if RL_MLP_WGMMA
    constexpr int TM = G::TM;
    constexpr int TN = G::TN;
    constexpr int NWG = G::NWG;
    constexpr int STAGES = G::STAGES;
    constexpr int RASTER_GM = G::RASTER_GM;
    constexpr int THREADS = G::THREADS;
    constexpr int A_BYTES = G::A_BYTES;
    constexpr int B_BYTES = G::B_BYTES;
    constexpr int STAGE_BYTES = G::STAGE_BYTES;
    constexpr int BK = G::BK;
    constexpr int KC = G::KC;
    constexpr int KG = G::KG;
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    uint64_t* const mbar_base = reinterpret_cast<uint64_t*>(smem_raw + STAGES * STAGE_BYTES);

    const int tid = threadIdx.x;
    const uint32_t smem_u = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw));
    const uint32_t mbar_u = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_base));

    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; ++s) {
            mbar_init(mbar_u + static_cast<uint32_t>(s * 8), 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    // Grouped rasterization: consecutive CTAs stay on one N tile and walk
    // RASTER_GM M tiles, so a wave's working set is a few A panels plus one B
    // panel instead of the whole B operand. Purely a scheduling choice.
    const int my_tiles = (M + TM - 1) / TM;
    const int lin = blockIdx.y * gridDim.x + blockIdx.x;
    const int per_group = RASTER_GM * gridDim.x;
    const int grp = lin / per_group;
    const int w = lin % per_group;
    const int m_tile = grp * RASTER_GM + (w % RASTER_GM);
    const int n_tile = w / RASTER_GM;
    if (m_tile >= my_tiles) {
        return;  // CTA from the padded grouped launch
    }
    const int m_base = m_tile * TM;
    const int n_base = n_tile * TN;
    const int k_blocks = (K + BK - 1) / BK;

    float acc[TN / 2];
#pragma unroll
    for (int i = 0; i < TN / 2; ++i) {
        acc[i] = 0.0f;
    }

    // Producer: one 128-B swizzled TMA box per 64-element reduction slab / MN
    // block; the whole stage is covered by one box per K-major operand and
    // MN_TILE/64 boxes per MN-major operand.
    const auto issue = [&](int slot, int kb) {
        if (tid != 0) {
            return;
        }
        const uint32_t ab = smem_u + static_cast<uint32_t>(slot * STAGE_BYTES);
        const uint32_t bb = ab + A_BYTES;
        const uint32_t mb = mbar_u + static_cast<uint32_t>(slot * 8);
        mbar_arrive_expect_tx(mb, STAGE_BYTES);
        const int k0 = kb * BK;
        if (A_MN) {
#pragma unroll
            for (int z = 0; z < TM / 64; ++z) {
                tma_2d(ab + static_cast<uint32_t>(z * 8192), &tmap_a, m_base + 64 * z, k0, mb);
            }
        } else {
            tma_2d(ab, &tmap_a, k0, m_base, mb);
        }
        if (B_MN) {
#pragma unroll
            for (int z = 0; z < TN / 64; ++z) {
                tma_2d(bb + static_cast<uint32_t>(z * 8192), &tmap_b, n_base + 64 * z, k0, mb);
            }
        } else {
            tma_2d(bb, &tmap_b, k0, n_base, mb);
        }
    };

    // Ring prologue: everything the first KG commit groups do not need to release.
    const int prologue = (STAGES - KG < k_blocks) ? (STAGES - KG) : k_blocks;
    for (int s = 0; s < prologue; ++s) {
        issue(s, s);
    }

    int phase[STAGES];
#pragma unroll
    for (int s = 0; s < STAGES; ++s) {
        phase[s] = 0;
    }

    const int wg = tid >> 7;

    for (int kb = 0; kb < k_blocks; ++kb) {
        const int slot = kb % STAGES;
        mbar_wait(mbar_u + static_cast<uint32_t>(slot * 8), phase[slot]);
        phase[slot] ^= 1;

        wgmma_fence();
        const uint32_t a_tile = (smem_u + static_cast<uint32_t>(slot * STAGE_BYTES)) >> 4;
        const uint32_t b_tile = a_tile + (A_BYTES >> 4);
        // A warpgroup is 64 rows: 64 x 128 B for a K-major tile, one 64-row MN
        // block (8 KB) for an MN-major tile; both are 512 16-B units.
        const uint32_t awg = G::AWG_U128 * static_cast<uint32_t>(wg);
        // A k16 slab is 16 reduction rows: 32 B inside a K-major row, 2 KB for
        // an MN-major tile.
        constexpr uint32_t A_KS = A_MN ? 128u : 2u;
        constexpr uint32_t B_KS = B_MN ? 128u : 2u;
        constexpr uint32_t A_LBO = A_MN ? 512u : 1u;
        constexpr uint32_t B_LBO = B_MN ? 512u : 1u;
        // Four ascending k16 slabs: the reduction order is exactly the mma.sync one.
#pragma unroll
        for (int sl = 0; sl < KC / 2; ++sl) {
            const uint32_t da = a_tile + awg + A_KS * static_cast<uint32_t>(sl);
            const uint32_t db = b_tile + B_KS * static_cast<uint32_t>(sl);
            if constexpr (TN == 128) {
                wgmma_m64n128k16<A_MN ? 1 : 0, B_MN ? 1 : 0>(gmma_desc(da, A_LBO, G::SBO_U128, G::LAYOUT_TYPE),
                                                             gmma_desc(db, B_LBO, G::SBO_U128, G::LAYOUT_TYPE), acc);
            } else {
                wgmma_m64n256k16<A_MN ? 1 : 0, B_MN ? 1 : 0>(gmma_desc(da, A_LBO, G::SBO_U128, G::LAYOUT_TYPE),
                                                             gmma_desc(db, B_LBO, G::SBO_U128, G::LAYOUT_TYPE), acc);
            }
        }
        wgmma_commit();
        wgmma_wait<KG>();
        __syncthreads();
        // The slot whose commit group has retired is exactly the one (kb + STAGES - KG) needs.
        const int nxt = kb + (STAGES - KG);
        if (nxt < k_blocks) {
            issue(nxt % STAGES, nxt);
        }
    }
    wgmma_wait<0>();

    // Epilogue: bias once in fp32 (after the complete reduction), one bf16 cast.
    const int lane = tid & 31;
    const int warp = (tid >> 5) & 3;
    const int g = lane >> 2;
    const int q = lane & 3;
    const int row0 = m_base + 64 * wg + 16 * warp + g;
    const int row1 = row0 + 8;
    // Interior specialisation: when the whole CTA tile lies inside the matrix no
    // per-element guard can fire, so the store path drops every comparison and
    // every predicated fallback. The test is uniform across the CTA, so it costs
    // one branch once; edge tiles keep exactly the guarded path.
    if (m_base + TM <= M && n_base + TN <= N) {
#pragma unroll
        for (int j = 0; j < TN / 8; ++j) {
            const int col = n_base + 8 * j + 2 * q;
            float v00 = acc[4 * j + 0];
            float v01 = acc[4 * j + 1];
            float v10 = acc[4 * j + 2];
            float v11 = acc[4 * j + 3];
            if (HAS_BIAS) {
                const float b0 = bias[col];
                const float b1 = bias[col + 1];
                v00 += b0;
                v01 += b1;
                v10 += b0;
                v11 += b1;
            }
            store_pair(out + static_cast<long>(row0) * N + col, v00, v01);
            store_pair(out + static_cast<long>(row1) * N + col, v10, v11);
        }
    } else {
#pragma unroll
        for (int j = 0; j < TN / 8; ++j) {
            const int col = n_base + 8 * j + 2 * q;
            const bool c_ok = col < N;
            const bool c2_ok = (col + 1) < N;
            float b0 = 0.0f, b1 = 0.0f;
            if (HAS_BIAS) {
                if (c_ok) {
                    b0 = bias[col];
                }
                if (c2_ok) {
                    b1 = bias[col + 1];
                }
            }
            float v00 = acc[4 * j + 0];
            float v01 = acc[4 * j + 1];
            float v10 = acc[4 * j + 2];
            float v11 = acc[4 * j + 3];
            if (HAS_BIAS) {
                v00 += b0;
                v01 += b1;
                v10 += b0;
                v11 += b1;
            }
            if (row0 < M) {
                if (c2_ok) {
                    store_pair(out + static_cast<long>(row0) * N + col, v00, v01);
                } else if (c_ok) {
                    out[static_cast<long>(row0) * N + col] = __float2bfloat16(v00);
                }
            }
            if (row1 < M) {
                if (c2_ok) {
                    store_pair(out + static_cast<long>(row1) * N + col, v10, v11);
                } else if (c_ok) {
                    out[static_cast<long>(row1) * N + col] = __float2bfloat16(v10);
                }
            }
        }
    }

#else
    (void)tmap_a;
    (void)tmap_b;
    (void)bias;
    (void)out;
    (void)M;
    (void)N;
    (void)K;
#endif  // RL_MLP_WGMMA
}

// ---------------------------------------------------------------------------
// Host: tensor maps + launch
// ---------------------------------------------------------------------------

// `cuTensorMapEncodeTiled` is a driver-API call and needs the primary context to
// be current on the calling thread. PyTorch's autograd engine runs user
// backward functions on its own thread, where the CUDA runtime may not have
// bound the context yet (cudaSetDevice alone does not), so bind it explicitly.
void ensure_cuda_context(const c10::Device& device) {
    const c10::cuda::CUDAGuard guard(device);
    CUcontext ctx = nullptr;
    if (cuCtxGetCurrent(&ctx) != CUDA_SUCCESS || ctx == nullptr) {
        cudaFree(nullptr);  // canonical runtime call that creates/binds the context
    }
}

// Both operand majors use the same 2-D TMA box: 64 reduction elements (128 B,
// the SWIZZLE_128B span) by `box_mn` rows. The tensor maps differ only in which
// axis is the tensor's minor (contiguous) axis.
void init_tmap(CUtensorMap* tmap, const void* base, uint64_t minor_elems, uint64_t rows,
               uint32_t box_rows, uint32_t box_inner, int swizzle) {
    uint64_t gdim[2] = {minor_elems, rows};
    uint64_t gstr[1] = {minor_elems * 2};
    uint32_t box[2] = {box_inner, box_rows};
    uint32_t estr[2] = {1, 1};
    const CUresult res = cuTensorMapEncodeTiled(
        tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(base), gdim, gstr, box, estr,
        CU_TENSOR_MAP_INTERLEAVE_NONE, static_cast<CUtensorMapSwizzle>(swizzle),
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(res == CUDA_SUCCESS, "mlp_down_gemm_sm90: cuTensorMapEncodeTiled failed (",
                static_cast<int>(res), ")");
}

// K-major operand [MN][red]: dim0 = the reduction (contiguous), dim1 = MN rows.
void init_kmap(CUtensorMap* tmap, const void* base, uint64_t red, uint64_t mn, uint32_t box_mn,
               uint32_t box_inner, int swizzle) {
    init_tmap(tmap, base, red, mn, box_mn, box_inner, swizzle);
}

// MN-major operand [red][MN]: dim0 = MN (contiguous), dim1 = reduction rows.
void init_mnmap(CUtensorMap* tmap, const void* base, uint64_t red, uint64_t mn, int swizzle) {
    init_tmap(tmap, base, mn, red, 64, 64, swizzle);
}

// D[M][N] = sum_k A * B, with each operand either K-major or MN-major.
template <bool A_MN, bool B_MN, bool HAS_BIAS, class G = GeoBwd>
void launch_wgmma(const nv_bf16* a, const nv_bf16* b, const float* bias, nv_bf16* out, int M, int N,
                  int K, cudaStream_t stream) {
    CUtensorMap tmap_a;
    CUtensorMap tmap_b;
    if (A_MN) {
        init_mnmap(&tmap_a, a, K, M, G::SWIZZLE);
    } else {
        init_kmap(&tmap_a, a, K, M, G::TM, G::BK, G::SWIZZLE);
    }
    if (B_MN) {
        init_mnmap(&tmap_b, b, K, N, G::SWIZZLE);
    } else {
        init_kmap(&tmap_b, b, K, N, G::TN, G::BK, G::SWIZZLE);
    }

    const int my_tiles = (M + G::TM - 1) / G::TM;
    const dim3 grid((N + G::TN - 1) / G::TN,
                    ((my_tiles + G::RASTER_GM - 1) / G::RASTER_GM) * G::RASTER_GM);
    auto* kernel = mlp_down_gemm_sm90_kernel<A_MN, B_MN, HAS_BIAS, G>;
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM_BYTES);
        configured = true;
    }
    kernel<<<grid, G::THREADS, G::SMEM_BYTES, stream>>>(tmap_a, tmap_b, bias, out, M, N, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

bool sm90_device_ok() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) {
        return false;
    }
    int major = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) {
        return false;
    }
    return major >= 9;
}

void check_sm90_operands(const void* a, const void* b, int64_t K, int64_t N) {
    TORCH_CHECK(K > 0 && N > 0, "mlp_down_gemm_sm90: K and N must be positive");
    TORCH_CHECK(K % 8 == 0 && N % 8 == 0,
                "mlp_down_gemm_sm90 needs K and N multiples of 8 (TMA row strides); got K=", K,
                ", N=", N);
    TORCH_CHECK(reinterpret_cast<uintptr_t>(a) % 16 == 0 &&
                    reinterpret_cast<uintptr_t>(b) % 16 == 0,
                "mlp_down_gemm_sm90: operands must be 16-byte aligned");
}

// The tensor maps describe the operands as dense 2-D arrays with the contiguous
// extent as the row stride, so a non-contiguous operand would be read with the
// wrong stride. Fail loudly instead of computing a wrong result.
void check_sm90_contiguous(const torch::Tensor& a, const torch::Tensor& b) {
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(),
                "mlp_down_gemm_sm90: operands must be contiguous");
}

// ---------------------------------------------------------------------------
// db: ascending-row fp32 left fold (contract c-prime: exactly one correctly
// rounded fp32 add per row, in row order, one chain per output column).
//
}  // namespace sm90

torch::Tensor mlp_down_gemm_cuda_forward_sm90(torch::Tensor x, torch::Tensor weight,
                                             std::optional<torch::Tensor> bias) {
    const c10::cuda::CUDAGuard guard(x.device());
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "x and weight must be CUDA tensors");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && weight.scalar_type() == at::kBFloat16,
                "mlp-down-gemm-mma-v1 is a bf16 contract");
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 2, "x and weight must be 2-D");
    TORCH_CHECK(x.size(1) == weight.size(1), "x K must match weight K");
    const int64_t S = x.size(0);
    const int64_t N = weight.size(0);
    const int64_t K = x.size(1);
    auto out = torch::empty({S, N}, x.options());
    if (out.numel() == 0) {
        return out;
    }
    TORCH_CHECK(sm90::sm90_device_ok(), "mlp_down_gemm_sm90 requires a compute capability 9.x device");
    sm90::ensure_cuda_context(x.device());
    const auto* a = reinterpret_cast<const nv_bf16*>(x.data_ptr<at::BFloat16>());
    const auto* b = reinterpret_cast<const nv_bf16*>(weight.data_ptr<at::BFloat16>());
    sm90::check_sm90_contiguous(x, weight);
    sm90::check_sm90_operands(a, b, K, N);
    const float* bias_ptr = nullptr;
    torch::Tensor bias_f;
    if (bias.has_value()) {
        TORCH_CHECK(bias->numel() == N, "bias must have N elements");
        TORCH_CHECK(bias->device() == x.device(), "bias must live on x's device");
        bias_f = bias->to(at::kFloat).contiguous();
        bias_ptr = bias_f.data_ptr<float>();
    }
    auto* o = reinterpret_cast<nv_bf16*>(out.data_ptr<at::BFloat16>());
    const int M = static_cast<int>(S);
    const int Nn = static_cast<int>(N);
    const int Kk = static_cast<int>(K);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (bias_ptr != nullptr) {
        sm90::launch_wgmma<false, false, true, sm90::GeoFwd>(a, b, bias_ptr, o, M, Nn, Kk, stream);
    } else {
        sm90::launch_wgmma<false, false, false, sm90::GeoFwd>(a, b, bias_ptr, o, M, Nn, Kk, stream);
    }
    return out;
}

torch::Tensor mlp_down_gemm_cuda_dx_sm90(torch::Tensor g, torch::Tensor weight) {
    const c10::cuda::CUDAGuard guard(g.device());
    TORCH_CHECK(g.is_cuda() && weight.is_cuda(), "g and weight must be CUDA tensors");
    TORCH_CHECK(g.scalar_type() == at::kBFloat16 && weight.scalar_type() == at::kBFloat16,
                "mlp-down-gemm-mma-v1 is a bf16 contract");
    TORCH_CHECK(g.dim() == 2 && weight.dim() == 2, "g and weight must be 2-D");
    TORCH_CHECK(g.size(1) == weight.size(0), "g N must match weight rows");
    const int64_t S = g.size(0);
    const int64_t Ng = g.size(1);
    const int64_t Kw = weight.size(1);
    auto out = torch::empty({S, Kw}, g.options());
    if (out.numel() == 0) {
        return out;
    }
    TORCH_CHECK(sm90::sm90_device_ok(), "mlp_down_gemm_sm90 requires a compute capability 9.x device");
    sm90::ensure_cuda_context(g.device());
    // dx is A = g [S, Ng] K-major over the reduction Ng, B = w [Ng, Kw] MN-major.
    const auto* a = reinterpret_cast<const nv_bf16*>(g.data_ptr<at::BFloat16>());
    const auto* b = reinterpret_cast<const nv_bf16*>(weight.data_ptr<at::BFloat16>());
    sm90::check_sm90_contiguous(g, weight);
    sm90::check_sm90_operands(a, b, Ng, Kw);
    sm90::launch_wgmma<false, true, false>(a, b, nullptr,
                                     reinterpret_cast<nv_bf16*>(out.data_ptr<at::BFloat16>()),
                                     static_cast<int>(S), static_cast<int>(Kw),
                                     static_cast<int>(Ng), at::cuda::getCurrentCUDAStream());
    return out;
}

torch::Tensor mlp_down_gemm_cuda_dw_sm90(torch::Tensor g, torch::Tensor x) {
    const c10::cuda::CUDAGuard guard(g.device());
    TORCH_CHECK(g.is_cuda() && x.is_cuda(), "g and x must be CUDA tensors");
    TORCH_CHECK(g.scalar_type() == at::kBFloat16 && x.scalar_type() == at::kBFloat16,
                "mlp-down-gemm-mma-v1 is a bf16 contract");
    TORCH_CHECK(g.dim() == 2 && x.dim() == 2, "g and x must be 2-D");
    TORCH_CHECK(g.size(0) == x.size(0), "g S must match x S");
    const int64_t S = g.size(0);
    const int64_t Ng = g.size(1);
    const int64_t Kx = x.size(1);
    auto out = torch::empty({Ng, Kx}, g.options());
    if (out.numel() == 0) {
        return out;
    }
    TORCH_CHECK(sm90::sm90_device_ok(), "mlp_down_gemm_sm90 requires a compute capability 9.x device");
    sm90::ensure_cuda_context(g.device());
    // dW is A = g [S, Ng] MN-major, B = x [S, Kx] MN-major, reduction S.
    const auto* a = reinterpret_cast<const nv_bf16*>(g.data_ptr<at::BFloat16>());
    const auto* b = reinterpret_cast<const nv_bf16*>(x.data_ptr<at::BFloat16>());
    sm90::check_sm90_contiguous(g, x);
    sm90::check_sm90_operands(a, b, Kx, Ng);
    sm90::launch_wgmma<true, true, false>(a, b, nullptr,
                                    reinterpret_cast<nv_bf16*>(out.data_ptr<at::BFloat16>()),
                                    static_cast<int>(Ng), static_cast<int>(Kx),
                                    static_cast<int>(S), at::cuda::getCurrentCUDAStream());
    return out;
}

// Bias gradient: the same ascending-row fp32 left fold as the mma.sync entry
// point, computed by a smem-staged schedule (bit-identical; see the kernel
// comment). Only the fold's *schedule* differs, not a single add.
