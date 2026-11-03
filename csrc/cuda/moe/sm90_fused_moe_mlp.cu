// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// SM90 fused routed-expert MLP (DSv4 MoE): MXFP8 activation x MXFP4 frozen
// weight grouped GEMM with FP8 WGMMA, SwiGLU + MX re-quantization fused into
// the fc1 epilogue, and an fc3 (down projection) GEMM on the same core.
//
// Numeric profile ``p5-sm90-fused-mlp-v1`` (NOT oracle-aligned):
//   * e4m3 x e4m3 WGMMA with FP32 accumulators. Each 32-wide MX block is
//     accumulated by one k32 WGMMA into a fresh accumulator and then promoted
//     on CUDA cores with the exact power-of-two product of the activation and
//     weight E8M0 scales. So MX block semantics are exact; only the order of
//     the 32 products inside a block is the tensor core's.
//   * MXFP4 (E2M1) weights are converted to E4M3 bytes in shared memory with
//     the block scale folded in relative to a per-column reference exponent
//     ref_c = max_j sw[c][j] - 6: value' = e2m1 * 2^(sw - ref_c). This is exact
//     as long as the residual sw - ref_c stays in [-8, 6] (checked fail-closed
//     by sm90_moe_prepare_weight_ref). The promote then only applies the
//     activation scale, and 2^(ref_c - 127) is applied once in the epilogue.
//     Since every term of a column is scaled by the same power of two, the
//     FP32 accumulation is bit-identical to promoting both scales per block.
//   * Batch invariance by construction: fixed tile constants (never chosen by
//     shape), every output element reduced over the full K inside one CTA in
//     ascending block order, no split-K, no atomics. A row's bytes depend only
//     on its own row, its expert's weights and the constants, so
//     fwd(x)[t] == fwd(x[t:t+1]).
//
// Layouts (P5 contract tensors):
//   A codes  : uint8 E4M3 [M, K], K contiguous.       A scales: uint8 E8M0 [M, K/32]
//   W codes  : uint8 E2M1 [E, N, K/2], low nibble = even k.  W scales: [E, N, K/32]
//   offsets  : int32 [E+1], non-decreasing, [0] == 0, [E] == M.
//
// Tiles: BM = 64 tokens, BN = 128 weight rows, BK = 128 (four 32-wide MX
// blocks per stage), STAGES-deep TMA/mbarrier pipeline. One producer
// warpgroup (TMA issue for the A tile and the raw packed-FP4 W tile one stage
// ahead, then FP4 -> FP8 conversion from shared memory) and two consumer warpgroups
// (m64n64k32 WGMMA on 64 columns each, double-buffered block accumulators so
// the promote of block j overlaps the WGMMA of block j+1), 384 threads, one
// CTA per SM.
//
// Modes:
//   FC1_Z    : W1 rows paired per h-column tile as [gate 0-31 | up 0-31 | gate 32-63 |
//              up 32-63] so each consumer warpgroup holds matching gate/up columns;
//              writes FP32 z [M, 2F].
//   FC1_HQ   : same GEMM; epilogue = clamp-SwiGLU * p_s -> BF16 -> MX quant; writes
//              h codes [M, F] + h scales [M, F/32]  (phase 2, two-launch pipeline).
//   FC3_Y    : plain 128-row tiles of W2; A = h codes; writes BF16 y [M, H].
//   FUSED    : phase 3, single kernel; fc1 tiles fill an smem-resident h_q, then
//              fc3 tiles read it. (Implemented in a separate kernel below.)

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

#if defined(RL_KERNEL_ENABLE_SM90)
#include <cuda.h>
#include <cudaTypedefs.h>
#include "../gemm/det_gemm_tma.cuh"
#endif

namespace {

constexpr int kBM = 64;       // tokens per CTA tile
constexpr int kBN = 128;      // weight rows per tile (fc1: 64 gate + 64 up)
constexpr int kBK = 128;      // K elements per pipeline stage (4 MX blocks)
constexpr int kMxBlock = 32;
constexpr int kNumThreads = 384;  // warpgroup 0 = producer, warpgroups 1..2 = consumers (64 cols each)

constexpr int kStageABytes = kBM * kBK;          // 8 KB  (E4M3 bytes, 128B swizzle)
constexpr int kStageBBytes = kBN * kBK;          // 16 KB (E4M3 bytes, 128B swizzle)
constexpr int kStageWBytes = kBN * (kBK / 2);    // 8 KB  raw packed E2M1 tile (TMA staging)
constexpr int kStageSABytes = (kBK / 32) * kBM * 4;  // 1 KB: FP32 2^(sa-127), [block][row]
constexpr int kStageSBBytes = 0;                     // weight scales are folded into B

constexpr float kGateClampMax = 10.0f;
constexpr float kUpClampMin = -10.0f;
constexpr float kUpClampMax = 10.0f;

enum class Mode : int { FC1_Z = 0, FC1_HQ = 1, FC3_Y = 2 };

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}

// E2M1 magnitudes {0,.5,1,1.5,2,3,4,6} * 2^res as E4M3 bytes (lo: idx 0..3, hi: 4..7),
// res = -8 .. +6 (all exact in E4M3). Index = res + 8.
__constant__ uint32_t kFoldLut[15][2] = {
    {0x03020100u, 0x0C080604u},  // 2^-8
    {0x06040200u, 0x14100C08u},  // 2^-7
    {0x0C080400u, 0x1C181410u},  // 2^-6
    {0x14100800u, 0x24201C18u},  // 2^-5
    {0x1C181000u, 0x2C282420u},  // 2^-4
    {0x24201800u, 0x34302C28u},  // 2^-3
    {0x2C282000u, 0x3C383430u},  // 2^-2
    {0x34302800u, 0x44403C38u},  // 2^-1
    {0x3C383000u, 0x4C484440u},  // 2^+0
    {0x44403800u, 0x54504C48u},  // 2^+1
    {0x4C484000u, 0x5C585450u},  // 2^+2
    {0x54504800u, 0x64605C58u},  // 2^+3
    {0x5C585000u, 0x6C686460u},  // 2^+4
    {0x64605800u, 0x74706C68u},  // 2^+5
    {0x6C686000u, 0x7C787470u},  // 2^+6
};

// E2M1 -> E4M3 for the four packed nibbles in bits 0..15 of ``p`` (exact).
// Magnitude: the 3 low bits of each nibble select from the 8-entry byte table
// {0,.5,1,1.5,2,3,4,6} -> {00,30,38,3C,40,44,48,4C} with one PRMT. Sign: a
// second PRMT in sign-replicate mode picks bytes whose msb is the E2M1 sign
// bit (bit 3 of nibble i sits at the msb of byte i/2 of p<<4 or p), then bit 7
// is masked. Output byte i = converted nibble i.
__device__ __forceinline__ uint32_t e2m1x4_packed_to_e4m3x4(uint32_t p, uint32_t lut_lo,
                                                            uint32_t lut_hi) {
  const uint32_t mag = prmt(lut_lo, lut_hi, p & 0x7777u);
  const uint32_t sgn = prmt(p << 4, p, 0xD9C8u) & 0x80808080u;
  return mag | sgn;
}

// 2^(code - 127) for an E8M0 code, branch-free: codes 1..254 are the FP32
// exponent field directly; code 0 is the subnormal 2^-127.
__device__ __forceinline__ float e8m0_to_f32(uint32_t code) {
  return __uint_as_float(code ? (code << 23) : 0x00400000u);
}

#if defined(RL_KERNEL_ENABLE_SM90)

// ------------------------------------------------------------------ PTX ----

__device__ __forceinline__ void mbar_arrive(uint32_t addr) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(addr) : "memory");
}

// Register expected TMA bytes on a barrier without arriving (the arrival comes
// later, after the same thread has finished its own shared-memory writes).
__device__ __forceinline__ void mbar_expect_tx(uint32_t addr, uint32_t bytes) {
  asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;" ::"r"(addr), "r"(bytes) : "memory");
}

// Generic-proxy shared-memory writes (st.shared) must be fenced before the
// async proxy (wgmma / TMA) reads them.
__device__ __forceinline__ void fence_proxy_async() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void wgmma_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
}
template <int N>
__device__ __forceinline__ void wgmma_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory");
}

// K-major, 128B-swizzled shared-memory matrix descriptor. The tile base must
// be 1024-byte aligned. SBO = 1024 B (8 rows x 128 B); LBO is unused for
// swizzled K-major layouts and set to 1 (16 B) per the CUTLASS convention.
__device__ __forceinline__ uint64_t make_smem_desc_sw128(uint32_t smem_addr) {
  uint64_t d = 0;
  d |= static_cast<uint64_t>((smem_addr & 0x3FFFFu) >> 4);
  d |= static_cast<uint64_t>(1) << 16;
  d |= static_cast<uint64_t>(1024 >> 4) << 32;
  d |= static_cast<uint64_t>(1) << 62;  // SWIZZLE_128B
  return d;
}


// D[64x64] (+)= A[64x32] * B[32x64], e4m3 x e4m3 -> f32. Both operands are
// K-major shared-memory descriptors (128B swizzle). scale_d == 0 zeroes D.
__device__ __forceinline__ void wgmma_m64n64k32_e4m3(float* d, uint64_t desc_a,
                                                     uint64_t desc_b, int scale_d) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, p, 1, 1;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "l"(desc_a), "l"(desc_b), "r"(scale_d));
}


// Byte (row, col) of a [rows x 128 B] tile in the 128B-swizzle layout that
// both TMA (CU_TENSOR_MAP_SWIZZLE_128B) and the wgmma descriptor above use:
// the 16-byte chunk index is XORed with (row % 8).
__device__ __forceinline__ uint32_t swz128(uint32_t row, uint32_t col) {
  return row * 128u + ((((col >> 4) ^ (row & 7u)) << 4) | (col & 15u));
}

// host helper

inline void init_tmap_u8_sw128(CUtensorMap* tmap, const void* gmem, uint64_t rows,
                               uint64_t cols_bytes, uint32_t box_rows, uint32_t box_cols) {
  uint64_t size[2] = {cols_bytes, rows};
  uint64_t stride[1] = {cols_bytes};
  uint32_t box[2] = {box_cols, box_rows};
  uint32_t estride[2] = {1, 1};
  const CUresult res = cuTensorMapEncodeTiled(
      tmap, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(gmem), size, stride, box,
      estride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(res == CUDA_SUCCESS, "sm90_fused_moe_mlp: cuTensorMapEncodeTiled failed (", res, ")");
}

inline void init_tmap_u8_plain(CUtensorMap* tmap, const void* gmem, uint64_t rows,
                               uint64_t cols_bytes, uint32_t box_rows, uint32_t box_cols) {
  uint64_t size[2] = {cols_bytes, rows};
  uint64_t stride[1] = {cols_bytes};
  uint32_t box[2] = {box_cols, box_rows};
  uint32_t estride[2] = {1, 1};
  const CUresult res = cuTensorMapEncodeTiled(
      tmap, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(gmem), size, stride, box,
      estride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(res == CUDA_SUCCESS, "sm90_fused_moe_mlp: cuTensorMapEncodeTiled (weights) failed (", res, ")");
}

// ------------------------------------------------------------- kernel -----

struct GemmArgs {
  const uint8_t* a_scales;   // [M, K/32]
  const uint8_t* w_codes;    // [E, N, K/2]
  const uint8_t* w_scales;   // [E, N, K/32]
  const uint8_t* w_ref;      // [E, N] per-column reference exponent code (see prepare)
  const uint8_t* w_res;      // [E, N, K/32] residual (sw - ref) + 8, in 0..14 (see prepare)
  const int32_t* offsets;    // [E+1]
  const float* p_s;          // [M] (FC1_HQ only)
  float* z;                  // [M, 2F] (FC1_Z)
  uint8_t* h_codes;          // [M, F]   (FC1_HQ)
  uint8_t* h_scales;         // [M, F/32](FC1_HQ)
  __nv_bfloat16* y;          // [M, N]   (FC3_Y)
  int M, N, K, E;            // N = weight rows per expert (2F for fc1, H for fc3)
};

template <int STAGES>
struct SmemLayout {
  static constexpr int kA = 0;
  static constexpr int kB = kA + STAGES * kStageABytes;
  static constexpr int kW = kB + STAGES * kStageBBytes;
  static constexpr int kSA = kW + STAGES * kStageWBytes;
  static constexpr int kBar = kSA + STAGES * kStageSABytes;
  static constexpr int kTotal = kBar + 3 * STAGES * 8;   // full[], empty[], raw[]
};

// Map blockIdx.x -> (expert, token block). Returns false if past the end.
__device__ __forceinline__ bool locate_token_block(const int32_t* offsets, int E, int bid,
                                                   int& expert, int& row0, int& row_end) {
  int acc = 0;
  for (int e = 0; e < E; ++e) {
    const int lo = offsets[e], hi = offsets[e + 1];
    const int nb = (hi - lo + kBM - 1) / kBM;
    if (bid < acc + nb) {
      expert = e;
      row0 = lo + (bid - acc) * kBM;
      row_end = hi;
      return true;
    }
    acc += nb;
  }
  return false;
}

// Global weight row (output column) held by B-tile row r of this n-tile.
template <Mode MODE>
__device__ __forceinline__ int b_row_to_weight_row(int r, int n_tile, int N) {
  if constexpr (MODE == Mode::FC3_Y) {
    return n_tile * kBN + r;
  } else {
    // B rows: [gate 0-31 | up 0-31 | gate 32-63 | up 32-63] of this h-column tile.
    const int F = N >> 1;
    const int half = r >> 6, within = r & 63;
    return (within < 32) ? (n_tile * 64 + half * 32 + within)
                         : (F + n_tile * 64 + half * 32 + (within - 32));
  }
}

// Producer: convert one stage of packed E2M1 weight rows (already staged in
// shared memory by TMA, 64 bytes per row) into E4M3 bytes in the swizzled B
// buffer, folding the residual block scale. 128 threads x 4 chunks of 16 B;
// thread t owns k-chunk q = t & 3 of rows (t >> 2) + 32 i.
__device__ __forceinline__ void convert_b_stage(const uint8_t* sW, uint8_t* sB,
                                                const uint32_t res_plus8[4], int tid) {
  const int q = tid & 3;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int r = (tid >> 2) + 32 * i;
    const uint4 packed = *reinterpret_cast<const uint4*>(sW + r * (kBK / 2) + q * 16);
    const uint32_t lut_lo = kFoldLut[res_plus8[i]][0], lut_hi = kFoldLut[res_plus8[i]][1];
    const uint32_t pk[4] = {packed.x, packed.y, packed.z, packed.w};
    uint32_t out[8];
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      out[2 * w] = e2m1x4_packed_to_e4m3x4(pk[w], lut_lo, lut_hi);          // k = 8w+0 .. 8w+3
      out[2 * w + 1] = e2m1x4_packed_to_e4m3x4(pk[w] >> 16, lut_lo, lut_hi);  // k = 8w+4 .. 8w+7
    }
    // 32 output bytes = two 16-byte chunks at columns q*32 and q*32+16.
    const uint32_t c0 = q * 32, c1 = q * 32 + 16;
    *reinterpret_cast<uint4*>(sB + swz128(r, c0)) = make_uint4(out[0], out[1], out[2], out[3]);
    *reinterpret_cast<uint4*>(sB + swz128(r, c1)) = make_uint4(out[4], out[5], out[6], out[7]);
  }
}

// Residual codes (sw - ref + 8) for this thread's 4 chunks of stage kb.
template <Mode MODE>
__device__ __forceinline__ void load_residuals(const GemmArgs& a, int expert, int n_tile, int kb,
                                               int tid, uint32_t out[4]) {
  const int blocks = a.K >> 5;
  const int q = tid & 3;
  const uint8_t* res = a.w_res + static_cast<int64_t>(expert) * a.N * blocks;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int grow = b_row_to_weight_row<MODE>((tid >> 2) + 32 * i, n_tile, a.N);
    out[i] = __ldg(res + static_cast<int64_t>(grow) * blocks + kb * 4 + q);
  }
}

// Consumer helpers -----------------------------------------------------------

// Fragment coordinates for the m64n64 f32 accumulator: register 4*i + {0,1,2,3}
// holds (row, col), (row, col+1), (row+8, col), (row+8, col+1) with
// row = warp*16 + lane/4 and col = 8*i + (lane%4)*2, i = 0..7.
__device__ __forceinline__ void frag_rows(int tid_in_wg, int& r0, int& r1) {
  const int warp = tid_in_wg >> 5, lane = tid_in_wg & 31;
  r0 = warp * 16 + (lane >> 2);
  r1 = r0 + 8;
}

// h = BF16(SiLU(min(g,10)) * clamp(u,-10,10) * p) as FP32.
__device__ __forceinline__ float swiglu_bf16(float g, float u, float p) {
  g = fminf(g, kGateClampMax);
  u = fminf(fmaxf(u, kUpClampMin), kUpClampMax);
  const float sig = 1.0f / (1.0f + expf(-g));
  const float h = (g * sig) * u * p;
  return __bfloat162float(__float2bfloat16(h));
}

// E8M0 code for a block amax: floor(log2(amax)) - 8, clamped to
// [-127, 127], bias 127; amax == 0 -> 127.
__device__ __forceinline__ int e8m0_code_from_amax(float amax) {
  if (amax == 0.0f) return 127;
  int ex;
  (void)frexpf(fmaxf(amax, 1.17549435e-38f), &ex);   // amax = f * 2^ex, f in [0.5, 1)
  int e = (ex - 1) - 8;
  e = max(-127, min(127, e));
  return e + 127;
}

__device__ __forceinline__ uint8_t f32_to_e4m3_sat(float v) {
  return static_cast<uint8_t>(__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3));
}

template <Mode MODE, int STAGES>
__global__ void __launch_bounds__(kNumThreads, 1)
sm90_moe_gemm_kernel(const __grid_constant__ CUtensorMap a_tmap,
                     const __grid_constant__ CUtensorMap w_tmap, GemmArgs args) {
  using L = SmemLayout<STAGES>;
  extern __shared__ __align__(1024) uint8_t smem[];
  uint8_t* sA = smem + L::kA;
  uint8_t* sB = smem + L::kB;
  uint8_t* sW = smem + L::kW;
  uint8_t* sSA = smem + L::kSA;
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + L::kBar);   // full[], empty[], raw[]

  const int tid = threadIdx.x;
  const int wg = tid >> 7;         // 0 producer, 1..2 consumers
  const int tig = tid & 127;
  const int n_tile = blockIdx.y;

  int expert, row0, row_end;
  if (!locate_token_block(args.offsets, args.E, blockIdx.x, expert, row0, row_end)) return;

  const uint32_t bar0 = static_cast<uint32_t>(__cvta_generic_to_shared(bars));
  auto full = [&](int s) { return bar0 + 8u * s; };
  auto empty = [&](int s) { return bar0 + 8u * (STAGES + s); };
  auto raw = [&](int s) { return bar0 + 8u * (2 * STAGES + s); };
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < STAGES; ++s) {
      det_gemm::mbar_init(full(s), 128);    // 128 producer arrivals + A tile TMA bytes
      det_gemm::mbar_init(empty(s), 256);   // 256 consumer arrivals
      det_gemm::mbar_init(raw(s), 1);       // TMA-issuing thread + raw W tile bytes
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  const int K = args.K;
  const int num_kb = K / kBK;
  const int blocks = K >> 5;

  if (wg == 0) {
    // ------------------------------------------------------------ producer
    // Thread 0 issues TMA for stage kb+1 (A tile -> full, raw W tile -> raw)
    // before the warpgroup converts stage kb, so the loads of the next stage
    // overlap this stage's conversion. Small per-thread global loads (activation
    // scale codes, weight residuals) are prefetched one stage ahead as well.
    const uint32_t sA_base = static_cast<uint32_t>(__cvta_generic_to_shared(sA));
    const uint32_t sW_base = static_cast<uint32_t>(__cvta_generic_to_shared(sW));
    auto issue_tma = [&](int kb) {
      const int s = kb % STAGES;
      mbar_expect_tx(full(s), kStageABytes);
      det_gemm::tma_2d_g2s(sA_base + s * kStageABytes, &a_tmap, kb * kBK, row0, full(s));
      det_gemm::mbar_arrive_expect_tx(raw(s), kStageWBytes);
      const uint32_t dst = sW_base + s * kStageWBytes;
      const int koff = kb * (kBK / 2);   // packed bytes
      if constexpr (MODE == Mode::FC3_Y) {
        det_gemm::tma_2d_g2s(dst, &w_tmap, koff, expert * args.N + n_tile * kBN, raw(s));
      } else {
        // Four 32-row boxes: gate 0-31, up 0-31, gate 32-63, up 32-63.
        const int F = args.N >> 1;
        const int64_t e0 = static_cast<int64_t>(expert) * args.N;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          const int half = b >> 1;
          const int grow = (b & 1) ? (F + n_tile * 64 + half * 32) : (n_tile * 64 + half * 32);
          det_gemm::tma_2d_g2s(dst + b * 32 * (kBK / 2), &w_tmap, koff,
                               static_cast<int>(e0 + grow), raw(s));
        }
      }
    };
    auto load_sa = [&](int kb) -> uint32_t {
      uint32_t v = 127u * 0x01010101u;   // rows past M: factor 1.0 (results are masked)
      if (tig < kBM) {
        const int m = row0 + tig;
        if (m < args.M) {
          v = __ldg(reinterpret_cast<const uint32_t*>(
              args.a_scales + static_cast<int64_t>(m) * blocks + kb * 4));
        }
      }
      return v;
    };

    uint32_t sa_cur = load_sa(0);
    uint32_t res_cur[4];
    load_residuals<MODE>(args, expert, n_tile, 0, tig, res_cur);
    if (tig == 0) {
      det_gemm::mbar_wait(empty(0), 1);
      issue_tma(0);
    }
    for (int kb = 0; kb < num_kb; ++kb) {
      const int s = kb % STAGES;
      // Producer-wide barrier: every producer thread has finished converting
      // stage kb-1 (hence all older stages), so the raw buffer of stage kb+1
      // (last used by stage kb+1-STAGES) may be refilled, and thread 0's
      // empty() wait for stage kb (done in iteration kb-1) orders everyone's
      // writes into sB[s] after the consumer released it.
      asm volatile("bar.sync 1, 128;" ::: "memory");
      uint32_t sa_nxt = sa_cur, res_nxt[4] = {res_cur[0], res_cur[1], res_cur[2], res_cur[3]};
      if (kb + 1 < num_kb) {
        sa_nxt = load_sa(kb + 1);
        load_residuals<MODE>(args, expert, n_tile, kb + 1, tig, res_nxt);
        if (tig == 0) {
          const int s1 = (kb + 1) % STAGES;
          det_gemm::mbar_wait(empty(s1), (((kb + 1) / STAGES) & 1) ^ 1);
          issue_tma(kb + 1);
        }
      }
      // Activation scale factors for this stage: [block][row] FP32.
      if (tig < kBM) {
        float* sf = reinterpret_cast<float*>(sSA + s * kStageSABytes);
#pragma unroll
        for (int j = 0; j < 4; ++j) sf[j * kBM + tig] = e8m0_to_f32((sa_cur >> (8 * j)) & 0xFF);
      }
      det_gemm::mbar_wait(raw(s), (kb / STAGES) & 1);
      convert_b_stage(sW + s * kStageWBytes, sB + s * kStageBBytes, res_cur, tig);
      fence_proxy_async();
      mbar_arrive(full(s));
      sa_cur = sa_nxt;
#pragma unroll
      for (int i = 0; i < 4; ++i) res_cur[i] = res_nxt[i];
    }
    return;
  }

  // -------------------------------------------------------------- consumer
  const int cw = wg - 1;           // which 64-column half of the B tile
  const int lane = tig & 31;
  int r0, r1;
  frag_rows(tig, r0, r1);
  const int c0 = (lane & 3) * 2;   // column offset inside each 8-wide slice

  float acc[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) acc[i] = 0.0f;

  const uint32_t sA_base = static_cast<uint32_t>(__cvta_generic_to_shared(sA));
  const uint32_t sB_base = static_cast<uint32_t>(__cvta_generic_to_shared(sB)) + cw * (64 * 128);

  // acc += blk * 2^(sa-127) for MX block j of stage s (weight scales are folded
  // into B; the per-column reference factor is applied once in the epilogue).
  auto promote = [&](const float* blk, int s, int j) {
    const float* fa = reinterpret_cast<const float*>(sSA + s * kStageSABytes) + j * kBM;
    const float fa0 = fa[r0], fa1 = fa[r1];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      acc[4 * i + 0] = fmaf(blk[4 * i + 0], fa0, acc[4 * i + 0]);
      acc[4 * i + 1] = fmaf(blk[4 * i + 1], fa0, acc[4 * i + 1]);
      acc[4 * i + 2] = fmaf(blk[4 * i + 2], fa1, acc[4 * i + 2]);
      acc[4 * i + 3] = fmaf(blk[4 * i + 3], fa1, acc[4 * i + 3]);
    }
  };

  // Software pipeline over MX blocks b = 0 .. NB-1 (block b lives in stage
  // (b/4) % STAGES at k-offset (b%4)*32). Three rotating block accumulators keep
  // two WGMMAs in flight while a third block is promoted, across stage
  // boundaries, so tensor-core latency is hidden behind the CUDA-core promote.
  const int NB = num_kb * (kBK / kMxBlock);
  float blk0[32], blk1[32], blk2[32];

  // Descriptor of stage s, k-block j == base descriptor + (s * bytes + j * 32) >> 4.
  const uint64_t da0 = make_smem_desc_sw128(sA_base), db0 = make_smem_desc_sw128(sB_base);
  auto issue = [&](int b, float* buf) {
    const int stage_idx = b >> 2;
    const int s = stage_idx % STAGES, j = b & 3;
    if (j == 0) det_gemm::mbar_wait(full(s), (stage_idx / STAGES) & 1);
    wgmma_fence();
    wgmma_m64n64k32_e4m3(buf, da0 + (s * (kStageABytes >> 4)) + 2 * j,
                         db0 + (s * (kStageBBytes >> 4)) + 2 * j, 0);
    wgmma_commit();
  };
  auto step = [&](float* buf, int b) {
    // Groups pending after the prologue / previous steps: W(b) .. W(min(NB, b+3)-1).
    const int ahead = min(NB - b - 1, 2);
    if (ahead >= 2) wgmma_wait<2>();
    else if (ahead == 1) wgmma_wait<1>();
    else wgmma_wait<0>();
    promote(buf, (b >> 2) % STAGES, b & 3);
    if ((b & 3) == 3) mbar_arrive(empty((b >> 2) % STAGES));
    if (b + 3 < NB) issue(b + 3, buf);
  };

  issue(0, blk0);
  if (NB > 1) issue(1, blk1);
  if (NB > 2) issue(2, blk2);
  int b = 0;
  for (; b + 3 <= NB; b += 3) {
    step(blk0, b);
    step(blk1, b + 1);
    step(blk2, b + 2);
  }
  if (b < NB) { step(blk0, b); ++b; }
  if (b < NB) { step(blk1, b); ++b; }

  // -------------------------------------------------------------- epilogue
  const int m0 = row0 + r0, m1 = row0 + r1;
  const bool v0 = m0 < row_end, v1 = m1 < row_end;

  // Undo the per-column reference exponent folded into the weights.
  {
    const uint8_t* wref = args.w_ref + static_cast<int64_t>(expert) * args.N;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      int grow0, grow1;   // weight rows (output columns) of accumulator slice i
      if constexpr (MODE == Mode::FC3_Y) {
        grow0 = n_tile * kBN + cw * 64 + 8 * i + c0;
        grow1 = grow0 + 1;
      } else {
        const int F = args.N >> 1;
        const int base = (i < 4) ? (n_tile * 64 + cw * 32 + 8 * i + c0)
                                 : (F + n_tile * 64 + cw * 32 + 8 * (i - 4) + c0);
        grow0 = base;
        grow1 = base + 1;
      }
      const float f0 = e8m0_to_f32(__ldg(wref + grow0));
      const float f1 = e8m0_to_f32(__ldg(wref + grow1));
      acc[4 * i + 0] *= f0; acc[4 * i + 1] *= f1;
      acc[4 * i + 2] *= f0; acc[4 * i + 3] *= f1;
    }
  }

  if constexpr (MODE == Mode::FC1_Z) {
    const int F = args.N >> 1;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int col = (i < 4) ? (n_tile * 64 + cw * 32 + 8 * i + c0)
                              : (F + n_tile * 64 + cw * 32 + 8 * (i - 4) + c0);
      if (v0) *reinterpret_cast<float2*>(args.z + static_cast<int64_t>(m0) * args.N + col) =
          make_float2(acc[4 * i + 0], acc[4 * i + 1]);
      if (v1) *reinterpret_cast<float2*>(args.z + static_cast<int64_t>(m1) * args.N + col) =
          make_float2(acc[4 * i + 2], acc[4 * i + 3]);
    }
  } else if constexpr (MODE == Mode::FC3_Y) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int col = n_tile * kBN + cw * 64 + 8 * i + c0;
      if (v0) *reinterpret_cast<__nv_bfloat162*>(args.y + static_cast<int64_t>(m0) * args.N + col) =
          __floats2bfloat162_rn(acc[4 * i + 0], acc[4 * i + 1]);
      if (v1) *reinterpret_cast<__nv_bfloat162*>(args.y + static_cast<int64_t>(m1) * args.N + col) =
          __floats2bfloat162_rn(acc[4 * i + 2], acc[4 * i + 3]);
    }
  } else {  // FC1_HQ: this warpgroup owns h columns [n_tile*64 + cw*32, +32) == one MX block.
    const int F = args.N >> 1;
    const float p0 = v0 ? __ldg(args.p_s + m0) : 0.0f;
    const float p1 = v1 ? __ldg(args.p_s + m1) : 0.0f;
    // Slice i (gate) pairs with slice i+4 (up) at the same column.
    float h0[8], h1[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      h0[2 * i + 0] = swiglu_bf16(acc[4 * i + 0], acc[4 * (i + 4) + 0], p0);
      h0[2 * i + 1] = swiglu_bf16(acc[4 * i + 1], acc[4 * (i + 4) + 1], p0);
      h1[2 * i + 0] = swiglu_bf16(acc[4 * i + 2], acc[4 * (i + 4) + 2], p1);
      h1[2 * i + 1] = swiglu_bf16(acc[4 * i + 3], acc[4 * (i + 4) + 3], p1);
    }
    float a0 = 0.0f, a1 = 0.0f;
#pragma unroll
    for (int t = 0; t < 8; ++t) {
      a0 = fmaxf(a0, fabsf(h0[t]));
      a1 = fmaxf(a1, fabsf(h1[t]));
    }
    // The 32-column block is spread over the 4 lanes of a quad (same row).
    a0 = fmaxf(a0, __shfl_xor_sync(0xFFFFFFFFu, a0, 1));
    a0 = fmaxf(a0, __shfl_xor_sync(0xFFFFFFFFu, a0, 2));
    a1 = fmaxf(a1, __shfl_xor_sync(0xFFFFFFFFu, a1, 1));
    a1 = fmaxf(a1, __shfl_xor_sync(0xFFFFFFFFu, a1, 2));
    const int code0 = e8m0_code_from_amax(a0);
    const int code1 = e8m0_code_from_amax(a1);
    const float inv0 = e8m0_to_f32(254 - code0);   // 1 / 2^(code-127) == 2^(127-code)
    const float inv1 = e8m0_to_f32(254 - code1);
    const int hcol0 = n_tile * 64 + cw * 32;
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      const int col = hcol0 + 8 * t + c0;
      if (v0) {
        const uint8_t q0 = f32_to_e4m3_sat(h0[2 * t] * inv0);
        const uint8_t q1 = f32_to_e4m3_sat(h0[2 * t + 1] * inv0);
        *reinterpret_cast<uint16_t*>(args.h_codes + static_cast<int64_t>(m0) * F + col) =
            static_cast<uint16_t>(q0 | (q1 << 8));
      }
      if (v1) {
        const uint8_t q0 = f32_to_e4m3_sat(h1[2 * t] * inv1);
        const uint8_t q1 = f32_to_e4m3_sat(h1[2 * t + 1] * inv1);
        *reinterpret_cast<uint16_t*>(args.h_codes + static_cast<int64_t>(m1) * F + col) =
            static_cast<uint16_t>(q0 | (q1 << 8));
      }
    }
    if ((lane & 3) == 0) {
      const int sidx = hcol0 / kMxBlock;
      if (v0) args.h_scales[static_cast<int64_t>(m0) * (F / kMxBlock) + sidx] = static_cast<uint8_t>(code0);
      if (v1) args.h_scales[static_cast<int64_t>(m1) * (F / kMxBlock) + sidx] = static_cast<uint8_t>(code1);
    }
  }
}


// ======================================================================
// Phase 3: single fused kernel. One CTA per token block. Phase A runs the
// fc1 GEMM over every h-column tile and writes the quantized activation into
// a shared-memory-resident h_q (E4M3 in the 128B-swizzle layout the WGMMA A
// descriptor expects, plus E8M0 codes per (block, row)). Phase B runs the fc3
// GEMM over every H tile with A read straight from that buffer. The TMA /
// mbarrier pipeline is shared by both phases through one global stage counter.
// Shared memory at F = 2048: 128 KB h_q + 4 KB codes + STAGES x 33 KB.
// ======================================================================

struct FusedArgs {
  const uint8_t* a_scales;   // [M, H/32]
  const uint8_t* w1_codes;   // [E, 2F, H/2]
  const uint8_t* w1_res;     // [E, 2F, H/32]
  const uint8_t* w1_ref;     // [E, 2F]
  const uint8_t* w2_codes;   // [E, H, F/2]
  const uint8_t* w2_res;     // [E, H, F/32]
  const uint8_t* w2_ref;     // [E, H]
  const int32_t* offsets;    // [E+1]
  const float* p_s;          // [M]
  __nv_bfloat16* y;          // [M, H]
  int M, H, F, E;
};

template <int STAGES, int F_MAX>
struct FusedSmemLayout {
  static constexpr int kHq = 0;                                    // [F/128 atoms][64 x 128B]
  static constexpr int kHs = kHq + kBM * F_MAX;                    // E8M0 codes [F/32][64]
  static constexpr int kA = kHs + (F_MAX / kMxBlock) * kBM;
  static constexpr int kB = kA + STAGES * kStageABytes;
  static constexpr int kW = kB + STAGES * kStageBBytes;
  static constexpr int kSA = kW + STAGES * kStageWBytes;
  static constexpr int kBar = kSA + STAGES * kStageSABytes;
  static constexpr int kTotal = kBar + 3 * STAGES * 8;
};

template <int STAGES, int F_MAX>
__global__ void __launch_bounds__(kNumThreads, 1)
sm90_moe_fused_mlp_kernel(const __grid_constant__ CUtensorMap a_tmap,
                          const __grid_constant__ CUtensorMap w1_tmap,
                          const __grid_constant__ CUtensorMap w2_tmap, FusedArgs args) {
  using L = FusedSmemLayout<STAGES, F_MAX>;
  extern __shared__ __align__(1024) uint8_t smem[];
  uint8_t* sHq = smem + L::kHq;
  uint8_t* sHs = smem + L::kHs;
  uint8_t* sA = smem + L::kA;
  uint8_t* sB = smem + L::kB;
  uint8_t* sW = smem + L::kW;
  uint8_t* sSA = smem + L::kSA;
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + L::kBar);

  const int tid = threadIdx.x;
  const int wg = tid >> 7;
  const int tig = tid & 127;

  int expert, row0, row_end;
  if (!locate_token_block(args.offsets, args.E, blockIdx.x, expert, row0, row_end)) return;

  const uint32_t bar0 = static_cast<uint32_t>(__cvta_generic_to_shared(bars));
  auto full = [&](int s) { return bar0 + 8u * s; };
  auto empty = [&](int s) { return bar0 + 8u * (STAGES + s); };
  auto raw = [&](int s) { return bar0 + 8u * (2 * STAGES + s); };
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < STAGES; ++s) {
      det_gemm::mbar_init(full(s), 128);
      det_gemm::mbar_init(empty(s), 256);
      det_gemm::mbar_init(raw(s), 1);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  const int H = args.H, F = args.F;
  const int nkb1 = H / kBK, nkb2 = F / kBK;       // K stages per tile in each phase
  const int nt1 = F / 64, nt2 = H / kBN;          // tiles per phase
  const int G1 = nt1 * nkb1;                      // global stage count of phase A
  const int G = G1 + nt2 * nkb2;
  const int blocks1 = H >> 5, blocks2 = F >> 5;

  // Global stage g -> (phase, tile, kb).
  auto decode = [&](int g, int& phase, int& tile, int& kb) {
    if (g < G1) { phase = 0; tile = g / nkb1; kb = g - tile * nkb1; }
    else { phase = 1; const int r = g - G1; tile = r / nkb2; kb = r - tile * nkb2; }
  };

  if (wg == 0) {
    // ------------------------------------------------------------ producer
    const uint32_t sA_base = static_cast<uint32_t>(__cvta_generic_to_shared(sA));
    const uint32_t sW_base = static_cast<uint32_t>(__cvta_generic_to_shared(sW));
    auto issue_tma = [&](int g) {
      int phase, tile, kb;
      decode(g, phase, tile, kb);
      const int s = g % STAGES;
      const uint32_t dst = sW_base + s * kStageWBytes;
      const int koff = kb * (kBK / 2);
      if (phase == 0) {
        mbar_expect_tx(full(s), kStageABytes);
        det_gemm::tma_2d_g2s(sA_base + s * kStageABytes, &a_tmap, kb * kBK, row0, full(s));
        det_gemm::mbar_arrive_expect_tx(raw(s), kStageWBytes);
        const int64_t e0 = static_cast<int64_t>(expert) * (2 * F);
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          const int half = b >> 1;
          const int grow = (b & 1) ? (F + tile * 64 + half * 32) : (tile * 64 + half * 32);
          det_gemm::tma_2d_g2s(dst + b * 32 * (kBK / 2), &w1_tmap, koff,
                               static_cast<int>(e0 + grow), raw(s));
        }
      } else {
        det_gemm::mbar_arrive_expect_tx(raw(s), kStageWBytes);
        det_gemm::tma_2d_g2s(dst, &w2_tmap, koff, expert * H + tile * kBN, raw(s));
      }
    };
    auto load_sa = [&](int g) -> uint32_t {   // phase A only
      uint32_t v = 127u * 0x01010101u;
      int phase, tile, kb;
      decode(g, phase, tile, kb);
      if (phase == 0 && tig < kBM) {
        const int m = row0 + tig;
        if (m < args.M)
          v = __ldg(reinterpret_cast<const uint32_t*>(args.a_scales + static_cast<int64_t>(m) * blocks1 + kb * 4));
      }
      return v;
    };
    auto load_res = [&](int g, uint32_t out[4]) {
      int phase, tile, kb;
      decode(g, phase, tile, kb);
      const int q = tig & 3;
      if (phase == 0) {
        const uint8_t* res = args.w1_res + static_cast<int64_t>(expert) * (2 * F) * blocks1;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int grow = b_row_to_weight_row<Mode::FC1_HQ>((tig >> 2) + 32 * i, tile, 2 * F);
          out[i] = __ldg(res + static_cast<int64_t>(grow) * blocks1 + kb * 4 + q);
        }
      } else {
        const uint8_t* res = args.w2_res + static_cast<int64_t>(expert) * H * blocks2;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int grow = tile * kBN + (tig >> 2) + 32 * i;
          out[i] = __ldg(res + static_cast<int64_t>(grow) * blocks2 + kb * 4 + q);
        }
      }
    };

    uint32_t sa_cur = load_sa(0);
    uint32_t res_cur[4];
    load_res(0, res_cur);
    if (tig == 0) {
      det_gemm::mbar_wait(empty(0), 1);
      issue_tma(0);
    }
    for (int g = 0; g < G; ++g) {
      const int s = g % STAGES;
      asm volatile("bar.sync 1, 128;" ::: "memory");
      if (g < G1 && tig < kBM) {
        float* sf = reinterpret_cast<float*>(sSA + s * kStageSABytes);
#pragma unroll
        for (int j = 0; j < 4; ++j) sf[j * kBM + tig] = e8m0_to_f32((sa_cur >> (8 * j)) & 0xFF);
      }
      det_gemm::mbar_wait(raw(s), (g / STAGES) & 1);
      convert_b_stage(sW + s * kStageWBytes, sB + s * kStageBBytes, res_cur, tig);
      fence_proxy_async();
      mbar_arrive(full(s));
      // With only two stages the consumer's block pipeline needs stage g before
      // it releases stage g-1, so the next stage's buffer wait must come AFTER
      // this stage is complete (otherwise thread 0 and the consumer deadlock).
      if (g + 1 < G) {
        sa_cur = load_sa(g + 1);
        load_res(g + 1, res_cur);
        if (tig == 0) {
          det_gemm::mbar_wait(empty((g + 1) % STAGES), (((g + 1) / STAGES) & 1) ^ 1);
          issue_tma(g + 1);
        }
      }
    }
    return;
  }

  // -------------------------------------------------------------- consumer
  const int cw = wg - 1;
  const int lane = tig & 31;
  int r0, r1;
  frag_rows(tig, r0, r1);
  const int c0 = (lane & 3) * 2;
  const int m0 = row0 + r0, m1 = row0 + r1;
  const bool v0 = m0 < row_end, v1 = m1 < row_end;
  const float p0 = v0 ? __ldg(args.p_s + m0) : 0.0f;
  const float p1 = v1 ? __ldg(args.p_s + m1) : 0.0f;

  const uint32_t sA_base = static_cast<uint32_t>(__cvta_generic_to_shared(sA));
  const uint32_t sB_base = static_cast<uint32_t>(__cvta_generic_to_shared(sB)) + cw * (64 * 128);
  const uint32_t sHq_base = static_cast<uint32_t>(__cvta_generic_to_shared(sHq));
  const uint64_t da0 = make_smem_desc_sw128(sA_base), db0 = make_smem_desc_sw128(sB_base);
  const uint64_t dh0 = make_smem_desc_sw128(sHq_base);

  float acc[32];
  float blk0[32], blk1[32], blk2[32];

  // Block b of tile (phase, tile): stage g = gbase + (b >> 2), k-block j = b & 3.
  auto issue = [&](int phase, int gbase, int b, float* buf) {
    const int g = gbase + (b >> 2), j = b & 3;
    const int s = g % STAGES;
    if (j == 0) det_gemm::mbar_wait(full(s), (g / STAGES) & 1);
    const uint64_t da = (phase == 0) ? (da0 + s * (kStageABytes >> 4) + 2 * j)
                                     : (dh0 + (b >> 2) * (kStageABytes >> 4) + 2 * j);
    wgmma_fence();
    wgmma_m64n64k32_e4m3(buf, da, db0 + s * (kStageBBytes >> 4) + 2 * j, 0);
    wgmma_commit();
  };
  auto promote = [&](int phase, int gbase, const float* blk, int b) {
    const int g = gbase + (b >> 2), j = b & 3;
    float fa0, fa1;
    if (phase == 0) {
      const float* fa = reinterpret_cast<const float*>(sSA + (g % STAGES) * kStageSABytes) + j * kBM;
      fa0 = fa[r0]; fa1 = fa[r1];
    } else {
      const uint8_t* hs = sHs + (b * kBM);   // block b of F: [block][row] codes
      fa0 = e8m0_to_f32(hs[r0]); fa1 = e8m0_to_f32(hs[r1]);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      acc[4 * i + 0] = fmaf(blk[4 * i + 0], fa0, acc[4 * i + 0]);
      acc[4 * i + 1] = fmaf(blk[4 * i + 1], fa0, acc[4 * i + 1]);
      acc[4 * i + 2] = fmaf(blk[4 * i + 2], fa1, acc[4 * i + 2]);
      acc[4 * i + 3] = fmaf(blk[4 * i + 3], fa1, acc[4 * i + 3]);
    }
  };
  auto run_tile = [&](int phase, int gbase, int NB) {
#pragma unroll
    for (int i = 0; i < 32; ++i) acc[i] = 0.0f;
    auto step = [&](float* buf, int b) {
      const int ahead = min(NB - b - 1, 2);
      if (ahead >= 2) wgmma_wait<2>();
      else if (ahead == 1) wgmma_wait<1>();
      else wgmma_wait<0>();
      promote(phase, gbase, buf, b);
      if ((b & 3) == 3) mbar_arrive(empty((gbase + (b >> 2)) % STAGES));
      if (b + 3 < NB) issue(phase, gbase, b + 3, buf);
    };
    issue(phase, gbase, 0, blk0);
    issue(phase, gbase, 1, blk1);
    issue(phase, gbase, 2, blk2);
    int b = 0;
    for (; b + 3 <= NB; b += 3) { step(blk0, b); step(blk1, b + 1); step(blk2, b + 2); }
    if (b < NB) { step(blk0, b); ++b; }
    if (b < NB) { step(blk1, b); ++b; }
  };

  // ---------------- phase A: fc1 tiles -> h_q in shared memory
  const uint8_t* w1ref = args.w1_ref + static_cast<int64_t>(expert) * (2 * F);
  for (int tile = 0; tile < nt1; ++tile) {
    run_tile(0, tile * nkb1, nkb1 * 4);
    // per-column reference factors, then SwiGLU + quant into h_q
    float h0[8], h1[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int gcol = tile * 64 + cw * 32 + 8 * i + c0;
      const float fg0 = e8m0_to_f32(__ldg(w1ref + gcol)), fg1 = e8m0_to_f32(__ldg(w1ref + gcol + 1));
      const float fu0 = e8m0_to_f32(__ldg(w1ref + F + gcol)), fu1 = e8m0_to_f32(__ldg(w1ref + F + gcol + 1));
      h0[2 * i + 0] = swiglu_bf16(acc[4 * i + 0] * fg0, acc[4 * (i + 4) + 0] * fu0, p0);
      h0[2 * i + 1] = swiglu_bf16(acc[4 * i + 1] * fg1, acc[4 * (i + 4) + 1] * fu1, p0);
      h1[2 * i + 0] = swiglu_bf16(acc[4 * i + 2] * fg0, acc[4 * (i + 4) + 2] * fu0, p1);
      h1[2 * i + 1] = swiglu_bf16(acc[4 * i + 3] * fg1, acc[4 * (i + 4) + 3] * fu1, p1);
    }
    float a0 = 0.0f, a1 = 0.0f;
#pragma unroll
    for (int q = 0; q < 8; ++q) { a0 = fmaxf(a0, fabsf(h0[q])); a1 = fmaxf(a1, fabsf(h1[q])); }
    a0 = fmaxf(a0, __shfl_xor_sync(0xFFFFFFFFu, a0, 1));
    a0 = fmaxf(a0, __shfl_xor_sync(0xFFFFFFFFu, a0, 2));
    a1 = fmaxf(a1, __shfl_xor_sync(0xFFFFFFFFu, a1, 1));
    a1 = fmaxf(a1, __shfl_xor_sync(0xFFFFFFFFu, a1, 2));
    const int code0 = e8m0_code_from_amax(a0), code1 = e8m0_code_from_amax(a1);
    const float inv0 = e8m0_to_f32(254 - code0), inv1 = e8m0_to_f32(254 - code1);
    const int hcol0 = tile * 64 + cw * 32;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int col = hcol0 + 8 * q + c0;
      const uint32_t atom = col >> 7, cc = col & 127;
      const uint16_t v0b = static_cast<uint16_t>(f32_to_e4m3_sat(h0[2 * q] * inv0) | (f32_to_e4m3_sat(h0[2 * q + 1] * inv0) << 8));
      const uint16_t v1b = static_cast<uint16_t>(f32_to_e4m3_sat(h1[2 * q] * inv1) | (f32_to_e4m3_sat(h1[2 * q + 1] * inv1) << 8));
      *reinterpret_cast<uint16_t*>(sHq + atom * kStageABytes + swz128(r0, cc)) = v0b;
      *reinterpret_cast<uint16_t*>(sHq + atom * kStageABytes + swz128(r1, cc)) = v1b;
    }
    if ((lane & 3) == 0) {
      sHs[(hcol0 / kMxBlock) * kBM + r0] = static_cast<uint8_t>(code0);
      sHs[(hcol0 / kMxBlock) * kBM + r1] = static_cast<uint8_t>(code1);
    }
  }
  // h_q complete: make generic-proxy writes visible to WGMMA and sync consumers.
  fence_proxy_async();
  asm volatile("bar.sync 2, 256;" ::: "memory");

  // ---------------- phase B: fc3 tiles from h_q -> y
  const uint8_t* w2ref = args.w2_ref + static_cast<int64_t>(expert) * H;
  for (int tile = 0; tile < nt2; ++tile) {
    run_tile(1, G1 + tile * nkb2, nkb2 * 4);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int col = tile * kBN + cw * 64 + 8 * i + c0;
      const float f0 = e8m0_to_f32(__ldg(w2ref + col)), f1 = e8m0_to_f32(__ldg(w2ref + col + 1));
      if (v0) *reinterpret_cast<__nv_bfloat162*>(args.y + static_cast<int64_t>(m0) * H + col) =
          __floats2bfloat162_rn(acc[4 * i + 0] * f0, acc[4 * i + 1] * f1);
      if (v1) *reinterpret_cast<__nv_bfloat162*>(args.y + static_cast<int64_t>(m1) * H + col) =
          __floats2bfloat162_rn(acc[4 * i + 2] * f0, acc[4 * i + 3] * f1);
    }
  }
}

// --------------------------------------------------------------- host ------

constexpr int kStages = 4;
constexpr int kFusedStages = 2;      // 128 KB h_q + 4 KB codes + 2 x 33 KB stages < 227 KB
constexpr int kFusedFMax = 2048;

void check_u8(const torch::Tensor& t, int dims, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.scalar_type() == torch::kUInt8, name, " must be uint8");
  TORCH_CHECK(t.dim() == dims, name, " must be ", dims, "-D");
}

void check_grouped_inputs(const torch::Tensor& a_codes, const torch::Tensor& a_scales,
                          const torch::Tensor& w_codes, const torch::Tensor& w_scales,
                          const torch::Tensor& w_ref, const torch::Tensor& w_res,
                          const torch::Tensor& offsets, int64_t& M, int64_t& K, int64_t& E,
                          int64_t& N) {
  check_u8(w_ref, 2, "w_ref");
  check_u8(w_res, 3, "w_res");
  check_u8(a_codes, 2, "a_codes");
  check_u8(a_scales, 2, "a_scales");
  check_u8(w_codes, 3, "w_codes");
  check_u8(w_scales, 3, "w_scales");
  TORCH_CHECK(offsets.is_cuda() && offsets.is_contiguous() && offsets.scalar_type() == torch::kInt32 &&
                  offsets.dim() == 1,
              "expert_offsets must be a contiguous int32 CUDA vector");
  M = a_codes.size(0);
  K = a_codes.size(1);
  E = w_codes.size(0);
  N = w_codes.size(1);
  TORCH_CHECK(K > 0 && K % kBK == 0, "K must be a positive multiple of ", kBK, ", got ", K);
  TORCH_CHECK(a_scales.size(0) == M && a_scales.size(1) == K / kMxBlock, "a_scales must be [M, K/32]");
  TORCH_CHECK(w_codes.size(2) == K / 2, "w_codes must be [E, N, K/2]");
  TORCH_CHECK(w_scales.size(0) == E && w_scales.size(1) == N && w_scales.size(2) == K / kMxBlock,
              "w_scales must be [E, N, K/32]");
  TORCH_CHECK(offsets.size(0) == E + 1, "expert_offsets must have E+1 entries");
  TORCH_CHECK(w_ref.size(0) == E && w_ref.size(1) == N, "w_ref must be [E, N] (see sm90_moe_prepare_weight_ref)");
  TORCH_CHECK(w_ref.device() == a_codes.device(), "w_ref must be on the same device");
  TORCH_CHECK(w_res.sizes() == w_scales.sizes() && w_res.device() == a_codes.device(),
              "w_res must be [E, N, K/32] on the same device (see sm90_moe_prepare_weight_ref)");
  TORCH_CHECK(K % 2 == 0 && (K / 2) % 16 == 0, "K/2 must be a multiple of 16 bytes for TMA");
  TORCH_CHECK(a_scales.device() == a_codes.device() && w_codes.device() == a_codes.device() &&
                  w_scales.device() == a_codes.device() && offsets.device() == a_codes.device(),
              "all inputs must be on the same device");
  TORCH_CHECK(M < (1LL << 31) && N < (1LL << 31), "M and N must fit in int32");
}

template <Mode MODE>
void launch_gemm(const torch::Tensor& a_codes, const torch::Tensor& w_codes, GemmArgs args,
                 int64_t M, int64_t E, int64_t K, int grid_y, cudaStream_t stream) {
  CUtensorMap a_tmap, w_tmap;
  init_tmap_u8_sw128(&a_tmap, a_codes.data_ptr(), static_cast<uint64_t>(M),
                     static_cast<uint64_t>(K), kBM, kBK);
  // Packed weights viewed as [E*N rows, K/2 bytes]; boxes of 64 bytes x 32 or 128 rows.
  init_tmap_u8_plain(&w_tmap, w_codes.data_ptr(), static_cast<uint64_t>(E) * args.N,
                     static_cast<uint64_t>(K / 2), MODE == Mode::FC3_Y ? kBN : 32, kBK / 2);
  using L = SmemLayout<kStages>;
  auto* kernel = sm90_moe_gemm_kernel<MODE, kStages>;
  static bool attr_set = false;   // per MODE instantiation
  if (!attr_set) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, L::kTotal));
    attr_set = true;
  }
  // Upper bound on the number of token blocks over all experts.
  const int64_t max_blocks = (M + kBM - 1) / kBM + E;
  dim3 grid(static_cast<unsigned>(max_blocks), static_cast<unsigned>(grid_y));
  kernel<<<grid, kNumThreads, L::kTotal, stream>>>(a_tmap, w_tmap, args);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int sm_major_of(const torch::Tensor& t) {
  return at::cuda::getDeviceProperties(t.get_device())->major;
}

#endif  // RL_KERNEL_ENABLE_SM90

}  // namespace

// Per-column reference exponent for the folded weight scales: ref = max_j sw - 6,
// so residuals sw - ref lie in [-(range), 0] and are exact in E4M3 iff range <= 8
// (values down to 0.5 * 2^-8 = 2^-9, the smallest E4M3 subnormal). Weights are
// frozen, so this runs once per weight. Fails closed if any column exceeds the
// range instead of silently losing precision.
std::vector<torch::Tensor> sm90_moe_prepare_weight_ref(torch::Tensor w_scales) {
  TORCH_CHECK(w_scales.is_cuda() && w_scales.scalar_type() == torch::kUInt8 && w_scales.dim() == 3,
              "w_scales must be uint8 [E, N, K/32] on CUDA");
  TORCH_CHECK(!(w_scales == 255).any().item<bool>(), "E8M0 NaN code 255 is not allowed");
  auto s = w_scales.to(torch::kInt32);
  auto mx = std::get<0>(s.max(-1));
  auto mn = std::get<0>(s.min(-1));
  const auto range = (mx - mn).max().item<int64_t>();
  TORCH_CHECK(range <= 8,
              "sm90_moe_prepare_weight_ref: weight block scales within one column span ", range,
              " binades (> 8); the folded-scale kernel would lose precision (fail-closed)");
  auto ref = (mx - 6).clamp(0, 254);
  auto res = (s - ref.unsqueeze(-1) + 8);   // in [0, 14]
  return {ref.to(torch::kUInt8).contiguous(), res.to(torch::kUInt8).contiguous()};
}

// fc1: z = x_q @ W1^T per expert, FP32 [M, 2F] (gate columns then up columns).
torch::Tensor sm90_moe_fc1_forward(torch::Tensor a_codes, torch::Tensor a_scales,
                                   torch::Tensor w1_codes, torch::Tensor w1_scales,
                                   torch::Tensor w_ref, torch::Tensor w_res, torch::Tensor expert_offsets) {
#if defined(RL_KERNEL_ENABLE_SM90)
  int64_t M, K, E, N;
  check_grouped_inputs(a_codes, a_scales, w1_codes, w1_scales, w_ref, w_res, expert_offsets, M, K, E, N);
  TORCH_CHECK(N % 128 == 0, "2F must be a multiple of 128, got ", N);
  TORCH_CHECK(sm_major_of(a_codes) == 9, "sm90_moe_fc1_forward requires an SM90 (Hopper) device");
  const c10::cuda::OptionalCUDAGuard guard(device_of(a_codes));
  auto z = torch::empty({M, N}, a_codes.options().dtype(torch::kFloat32));
  if (M == 0) return z;
  GemmArgs args{};
  args.a_scales = a_scales.data_ptr<uint8_t>();
  args.w_codes = w1_codes.data_ptr<uint8_t>();
  args.w_scales = w1_scales.data_ptr<uint8_t>();
  args.w_ref = w_ref.data_ptr<uint8_t>();
  args.w_res = w_res.data_ptr<uint8_t>();
  args.offsets = expert_offsets.data_ptr<int32_t>();
  args.z = z.data_ptr<float>();
  args.M = static_cast<int>(M); args.N = static_cast<int>(N); args.K = static_cast<int>(K); args.E = static_cast<int>(E);
  launch_gemm<Mode::FC1_Z>(a_codes, w1_codes, args, M, E, K, static_cast<int>(N / 128),
                           at::cuda::getCurrentCUDAStream());
  return z;
#else
  TORCH_CHECK(false, "sm90_moe_fc1_forward: not compiled (build with KERNEL_ALIGN_MOE_SM90=1)");
#endif
}

// fc1 + clamp-SwiGLU * p_s + MX quant: returns (h_codes [M, F], h_scales [M, F/32]).
std::vector<torch::Tensor> sm90_moe_fc1_swiglu_quant_forward(
    torch::Tensor a_codes, torch::Tensor a_scales, torch::Tensor w1_codes,
    torch::Tensor w1_scales, torch::Tensor w_ref, torch::Tensor w_res, torch::Tensor expert_offsets, torch::Tensor p_s) {
#if defined(RL_KERNEL_ENABLE_SM90)
  int64_t M, K, E, N;
  check_grouped_inputs(a_codes, a_scales, w1_codes, w1_scales, w_ref, w_res, expert_offsets, M, K, E, N);
  TORCH_CHECK(N % 128 == 0, "2F must be a multiple of 128, got ", N);
  TORCH_CHECK(p_s.is_cuda() && p_s.is_contiguous() && p_s.scalar_type() == torch::kFloat32 &&
                  p_s.dim() == 1 && p_s.size(0) == M,
              "p_s must be a contiguous FP32 CUDA vector [M]");
  TORCH_CHECK(sm_major_of(a_codes) == 9, "requires an SM90 (Hopper) device");
  const c10::cuda::OptionalCUDAGuard guard(device_of(a_codes));
  const int64_t F = N / 2;
  auto h_codes = torch::empty({M, F}, a_codes.options());
  auto h_scales = torch::empty({M, F / kMxBlock}, a_codes.options());
  if (M == 0) return {h_codes, h_scales};
  GemmArgs args{};
  args.a_scales = a_scales.data_ptr<uint8_t>();
  args.w_codes = w1_codes.data_ptr<uint8_t>();
  args.w_scales = w1_scales.data_ptr<uint8_t>();
  args.w_ref = w_ref.data_ptr<uint8_t>();
  args.w_res = w_res.data_ptr<uint8_t>();
  args.offsets = expert_offsets.data_ptr<int32_t>();
  args.p_s = p_s.data_ptr<float>();
  args.h_codes = h_codes.data_ptr<uint8_t>();
  args.h_scales = h_scales.data_ptr<uint8_t>();
  args.M = static_cast<int>(M); args.N = static_cast<int>(N); args.K = static_cast<int>(K); args.E = static_cast<int>(E);
  launch_gemm<Mode::FC1_HQ>(a_codes, w1_codes, args, M, E, K, static_cast<int>(F / 64),
                            at::cuda::getCurrentCUDAStream());
  return {h_codes, h_scales};
#else
  TORCH_CHECK(false, "sm90_moe_fc1_swiglu_quant_forward: not compiled (build with KERNEL_ALIGN_MOE_SM90=1)");
#endif
}

// fc3: y = h_q @ W2^T per expert, BF16 [M, H].
torch::Tensor sm90_moe_fc3_forward(torch::Tensor h_codes, torch::Tensor h_scales,
                                   torch::Tensor w2_codes, torch::Tensor w2_scales,
                                   torch::Tensor w_ref, torch::Tensor w_res, torch::Tensor expert_offsets) {
#if defined(RL_KERNEL_ENABLE_SM90)
  int64_t M, K, E, N;
  check_grouped_inputs(h_codes, h_scales, w2_codes, w2_scales, w_ref, w_res, expert_offsets, M, K, E, N);
  TORCH_CHECK(N % kBN == 0, "H must be a multiple of ", kBN, ", got ", N);
  TORCH_CHECK(sm_major_of(h_codes) == 9, "requires an SM90 (Hopper) device");
  const c10::cuda::OptionalCUDAGuard guard(device_of(h_codes));
  auto y = torch::empty({M, N}, h_codes.options().dtype(torch::kBFloat16));
  if (M == 0) return y;
  GemmArgs args{};
  args.a_scales = h_scales.data_ptr<uint8_t>();
  args.w_codes = w2_codes.data_ptr<uint8_t>();
  args.w_scales = w2_scales.data_ptr<uint8_t>();
  args.w_ref = w_ref.data_ptr<uint8_t>();
  args.w_res = w_res.data_ptr<uint8_t>();
  args.offsets = expert_offsets.data_ptr<int32_t>();
  args.y = reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>());
  args.M = static_cast<int>(M); args.N = static_cast<int>(N); args.K = static_cast<int>(K); args.E = static_cast<int>(E);
  launch_gemm<Mode::FC3_Y>(h_codes, w2_codes, args, M, E, K, static_cast<int>(N / kBN),
                           at::cuda::getCurrentCUDAStream());
  return y;
#else
  TORCH_CHECK(false, "sm90_moe_fc3_forward: not compiled (build with KERNEL_ALIGN_MOE_SM90=1)");
#endif
}

// Fused forward: y = fc3(mx_quant(swiglu(fc1(x_q)) * p_s)) in one launch (inference only).
torch::Tensor sm90_moe_fused_mlp_forward(
    torch::Tensor a_codes, torch::Tensor a_scales, torch::Tensor w1_codes, torch::Tensor w1_scales,
    torch::Tensor w1_ref, torch::Tensor w1_res, torch::Tensor w2_codes, torch::Tensor w2_scales,
    torch::Tensor w2_ref, torch::Tensor w2_res, torch::Tensor expert_offsets, torch::Tensor p_s) {
#if defined(RL_KERNEL_ENABLE_SM90)
  int64_t M, H, E, N1;
  check_grouped_inputs(a_codes, a_scales, w1_codes, w1_scales, w1_ref, w1_res, expert_offsets, M, H, E, N1);
  const int64_t F = N1 / 2;
  check_u8(w2_codes, 3, "w2_codes"); check_u8(w2_scales, 3, "w2_scales");
  check_u8(w2_ref, 2, "w2_ref"); check_u8(w2_res, 3, "w2_res");
  TORCH_CHECK(w2_codes.size(0) == E && w2_codes.size(1) == H && w2_codes.size(2) == F / 2, "w2_codes must be [E, H, F/2]");
  TORCH_CHECK(w2_scales.size(0) == E && w2_scales.size(1) == H && w2_scales.size(2) == F / kMxBlock, "w2_scales must be [E, H, F/32]");
  TORCH_CHECK(w2_ref.size(0) == E && w2_ref.size(1) == H && w2_res.sizes() == w2_scales.sizes(), "w2_ref/w2_res shapes");
  TORCH_CHECK(N1 % 128 == 0 && H % kBN == 0 && F % kBK == 0, "2F % 128, H % 128 and F % 128 must be 0");
  TORCH_CHECK(F <= kFusedFMax, "fused kernel supports F <= ", kFusedFMax, " (shared-memory resident h_q), got ", F);
  TORCH_CHECK(p_s.is_cuda() && p_s.is_contiguous() && p_s.scalar_type() == torch::kFloat32 && p_s.dim() == 1 && p_s.size(0) == M,
              "p_s must be a contiguous FP32 CUDA vector [M]");
  TORCH_CHECK(sm_major_of(a_codes) == 9, "requires an SM90 (Hopper) device");
  const c10::cuda::OptionalCUDAGuard guard(device_of(a_codes));
  auto y = torch::empty({M, H}, a_codes.options().dtype(torch::kBFloat16));
  if (M == 0) return y;
  FusedArgs args{};
  args.a_scales = a_scales.data_ptr<uint8_t>();
  args.w1_codes = w1_codes.data_ptr<uint8_t>(); args.w1_res = w1_res.data_ptr<uint8_t>(); args.w1_ref = w1_ref.data_ptr<uint8_t>();
  args.w2_codes = w2_codes.data_ptr<uint8_t>(); args.w2_res = w2_res.data_ptr<uint8_t>(); args.w2_ref = w2_ref.data_ptr<uint8_t>();
  args.offsets = expert_offsets.data_ptr<int32_t>();
  args.p_s = p_s.data_ptr<float>();
  args.y = reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>());
  args.M = static_cast<int>(M); args.H = static_cast<int>(H); args.F = static_cast<int>(F); args.E = static_cast<int>(E);

  CUtensorMap a_tmap, w1_tmap, w2_tmap;
  init_tmap_u8_sw128(&a_tmap, a_codes.data_ptr(), static_cast<uint64_t>(M), static_cast<uint64_t>(H), kBM, kBK);
  init_tmap_u8_plain(&w1_tmap, w1_codes.data_ptr(), static_cast<uint64_t>(E) * N1, static_cast<uint64_t>(H / 2), 32, kBK / 2);
  init_tmap_u8_plain(&w2_tmap, w2_codes.data_ptr(), static_cast<uint64_t>(E) * H, static_cast<uint64_t>(F / 2), kBN, kBK / 2);
  using L = FusedSmemLayout<kFusedStages, kFusedFMax>;
  auto* kernel = sm90_moe_fused_mlp_kernel<kFusedStages, kFusedFMax>;
  static bool attr_set = false;
  if (!attr_set) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, L::kTotal));
    attr_set = true;
  }
  const int64_t max_blocks = (M + kBM - 1) / kBM + E;
  kernel<<<dim3(static_cast<unsigned>(max_blocks)), kNumThreads, L::kTotal, at::cuda::getCurrentCUDAStream()>>>(
      a_tmap, w1_tmap, w2_tmap, args);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
#else
  TORCH_CHECK(false, "sm90_moe_fused_mlp_forward: not compiled (build with KERNEL_ALIGN_MOE_SM90=1)");
#endif
}
