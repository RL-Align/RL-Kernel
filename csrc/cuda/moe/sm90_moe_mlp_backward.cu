// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// SM90 routed-expert MLP backward (DSv4 MoE, frozen MXFP4 base weights, no dW).
// Follows the P5 oracle chain (rl_engine/moe/oracle.py routed_expert_backward):
//
//   z     = fc1(x_q)                       recomputed by sm90_moe_fc1_forward (FP32)
//   dh    = BF16(dy_bf16 @ W2[e])          [M, F]
//   dgate, dup, dp_s = clamp_swiglu_weighted_bwd(dh, z, p_s)
//   dz    = BF16([dgate | dup])            [M, 2F]
//   dx    = dz @ W1[e]                     [M, H], FP32
//
// One grouped GEMM shape serves both products: A BF16 [M, Nw] times the MXFP4
// weight W [E, Nw, Kw] reduced over its rows, out [M, Kw]. The weight is
// expanded to BF16 in shared memory, exactly (e2m1 * 2^(s-127) needs at most 2
// significant bits; scale codes are restricted to [2, 252] so every value is a
// normal BF16, checked fail-closed on the host), transposed into the K-major
// operand layout. BF16 WGMMA m64n64k16 with FP32 accumulators.
//
// Numeric profile ``p5-sm90-bwd-bf16-v1`` (not oracle byte-aligned; the oracle
// reduces serially): fixed tiles (BM 128 tokens, 128 output columns, 64 reduced
// rows per stage), the whole reduction inside one CTA in ascending order, no
// split-K, no atomics. A row's bytes depend only on that row, its expert and
// the constants, so the backward is batch-invariant and run-to-run identical.
// dp_s: each CTA sums its 128 columns per row in a fixed order into a partial; a
// second kernel adds the partials in ascending order.
//
// Layout of the producer / consumer pipeline is the forward kernel's
// (sm90_fused_moe_mlp.cu): warpgroup 0 issues TMA (A tile, raw packed W tile)
// one stage ahead and converts W; warpgroups 1..2 run WGMMA m64n128k16 on 64 token
// rows each, sharing the converted weight tile (CTA tile 128 tokens x 128 columns).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cstdint>

#if defined(RL_KERNEL_ENABLE_MOE_SM90)
#include <cuda.h>
#include <cudaTypedefs.h>
#include "../gemm/det_gemm_tma.cuh"
#endif

namespace {

constexpr int kBM = 128;   // tokens per CTA tile (2 consumer warpgroups x 64 rows)
constexpr int kBO = 128;   // output columns per CTA tile
constexpr int kBR = 64;    // reduced weight rows per stage (one 128-byte BF16 row of A)
constexpr int kStages = 5;
constexpr int kNumThreads = 384;

constexpr int kStageABytes = kBM * kBR * 2;      // 16 KB, 128B swizzle
constexpr int kStageBBytes = kBO * kBR * 2;      // 16 KB, 128B swizzle (rows = output cols)
constexpr int kStageWBytes = kBR * (kBO / 2);    // 4 KB raw packed E2M1
constexpr int kStageSBytes = kBR * 4;            // 256 B scale codes [row][4 blocks]

constexpr float kGateClampMax = 10.0f;
constexpr float kUpClampMin = -10.0f;
constexpr float kUpClampMax = 10.0f;

enum class Mode : int { DX = 0, DH_SWIGLU = 1 };

#if defined(RL_KERNEL_ENABLE_MOE_SM90)

__device__ __forceinline__ void mbar_arrive(uint32_t addr) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(addr) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t addr, uint32_t bytes) {
  asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;" ::"r"(addr), "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void cp_async4(uint32_t dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}
__device__ __forceinline__ void fence_proxy_async() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
}
template <int N>
__device__ __forceinline__ void wgmma_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory");
}

// K-major, 128B-swizzled descriptor (tile base 1024-byte aligned, SBO 1024 B).
__device__ __forceinline__ uint64_t make_smem_desc_sw128(uint32_t smem_addr) {
  uint64_t d = 0;
  d |= static_cast<uint64_t>((smem_addr & 0x3FFFFu) >> 4);
  d |= static_cast<uint64_t>(1) << 16;
  d |= static_cast<uint64_t>(1024 >> 4) << 32;
  d |= static_cast<uint64_t>(1) << 62;
  return d;
}

// D[64x128] += A[64x16] * B[16x128], bf16 x bf16 -> f32, both K-major.
__device__ __forceinline__ void wgmma_m64n128k16_bf16(float* d, uint64_t desc_a, uint64_t desc_b) {
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, 1, 1, 1, 0, 0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "l"(desc_a), "l"(desc_b));
}

__device__ __forceinline__ uint32_t swz128(uint32_t row, uint32_t col) {
  return row * 128u + ((((col >> 4) ^ (row & 7u)) << 4) | (col & 15u));
}

inline void init_tmap_u8(CUtensorMap* tmap, const void* gmem, uint64_t rows, uint64_t cols_bytes,
                         uint32_t box_rows, uint32_t box_cols, bool swizzle) {
  uint64_t size[2] = {cols_bytes, rows};
  uint64_t stride[1] = {cols_bytes};
  uint32_t box[2] = {box_cols, box_rows};
  uint32_t estride[2] = {1, 1};
  const CUresult res = cuTensorMapEncodeTiled(
      tmap, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(gmem), size, stride, box, estride,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      swizzle ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(res == CUDA_SUCCESS, "sm90_moe_mlp_backward: cuTensorMapEncodeTiled failed (", res, ")");
}

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}

// Four E2M1 nibbles (bits 0..15 of q, nibble i = column i) -> two BF16x2 words
// (columns 0,1 and 2,3) of e2m1 * 2^(s - 127), s in [2, 252] (exact, normal BF16).
// Magnitudes {0,.5,1,1.5,2,3,4,6} as BF16 bytes: high {00,3F,3F,3F,40,40,40,40},
// low {00,00,80,C0,00,40,80,C0}, each selected for 4 nibbles by one PRMT; the sign
// (nibble bit 3) lands on the high byte's msb (the forward kernel's PRMT trick);
// the exponent shift is a 16-bit add on non-zero magnitudes.
__device__ __forceinline__ void e2m1x4_to_bf16x4(uint32_t q, uint32_t shift2, uint32_t& w01,
                                                 uint32_t& w23) {
  const uint32_t sel = q & 0x7777u;
  const uint32_t sgn = prmt(q << 4, q, 0xD9C8u) & 0x80808080u;
  const uint32_t hb = prmt(0x3F3F3F00u, 0x40404040u, sel) | sgn;
  const uint32_t lb = prmt(0xC0800000u, 0xC0804000u, sel);
  uint32_t a = prmt(lb, hb, 0x5140u), b = prmt(lb, hb, 0x7362u);
  a = __vadd2(a, shift2 & __vcmpne2(a & 0x7FFF7FFFu, 0u));
  b = __vadd2(b, shift2 & __vcmpne2(b & 0x7FFF7FFFu, 0u));
  w01 = a;
  w23 = b;
}

struct BwdArgs {
  const uint8_t* w_scales;  // [E, Nw, Kw/32]
  const int32_t* offsets;   // [E+1]
  float* out_f32;           // DX: [M, Kw]
  __nv_bfloat16* dz;        // DH_SWIGLU: [M, 2F] (Kw == F)
  const float* z;           // DH_SWIGLU: [M, 2F]
  const float* p_s;         // DH_SWIGLU: [M]
  float* dps_part;          // DH_SWIGLU: [M, Kw / 64]
  int M, Nw, Kw, E;
  int num_tb;               // token-block upper bound (grid rows)
};

struct Smem {
  static constexpr int kA = 0;
  static constexpr int kB = kA + kStages * kStageABytes;
  static constexpr int kW = kB + kStages * kStageBBytes;
  static constexpr int kS = kW + kStages * kStageWBytes;
  static constexpr int kBar = kS + kStages * kStageSBytes;
  static constexpr int kTotal = kBar + 3 * kStages * 8;
};

__device__ __forceinline__ bool locate_token_block(const int32_t* offsets, int E, int bid, int& expert,
                                                   int& row0, int& row_end) {
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

// Producer: expand one stage of packed E2M1 rows (kBR = 64 reduced rows x 64
// bytes = 128 output columns) into the K-major BF16 B tile (row = output column,
// 64 reduced rows = 128 bytes). Thread t owns an 8 x 8 block: output columns
// 8 cg .. 8 cg + 7 (cg = t >> 3, one 32-bit word per reduced row, all in MX block
// cg / 4) and reduced rows 8 ng .. 8 ng + 7 (ng = t & 7), transposed in registers.
// The 8 threads of a store phase share cg and write 8 different 16-byte chunks of
// one swizzled row (conflict-free).
__device__ __forceinline__ void convert_stage(const uint8_t* sW, const uint8_t* sS, uint8_t* sB,
                                              int tid) {
  const int cg = tid >> 3, ng = tid & 7;
  const int blk = cg >> 2;
  uint32_t w[8][4];  // [reduced row][column pair]
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int n = ng * 8 + j;
    const uint32_t p = *reinterpret_cast<const uint32_t*>(sW + n * (kBO / 2) + cg * 4);
    const uint32_t sh = static_cast<uint32_t>((static_cast<int>(sS[n * 4 + blk]) - 127) << 7) & 0xFFFFu;
    const uint32_t shift2 = sh | (sh << 16);
    e2m1x4_to_bf16x4(p, shift2, w[j][0], w[j][1]);
    e2m1x4_to_bf16x4(p >> 16, shift2, w[j][2], w[j][3]);
  }
#pragma unroll
  for (int kk = 0; kk < 8; ++kk) {
    const uint32_t sel = (kk & 1) ? 0x7632u : 0x5410u;
    uint32_t o[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = prmt(w[2 * j][kk >> 1], w[2 * j + 1][kk >> 1], sel);
    *reinterpret_cast<uint4*>(sB + swz128(cg * 8 + kk, ng * 16)) = make_uint4(o[0], o[1], o[2], o[3]);
  }
}

// kPreB: W was expanded ahead of time (sm90_moe_dequant_t: the same convert_stage,
// so the same BF16 values) and w_tmap maps that [E * Kw, Nw] BF16 tensor; the B
// tile arrives by TMA and the producer warpgroup does no conversion. The WGMMA
// sequence is unchanged, so both modes give the same bytes.
template <Mode MODE, bool kPreB>
__global__ void __launch_bounds__(kNumThreads, 1)
sm90_moe_bwd_gemm_kernel(const __grid_constant__ CUtensorMap a_tmap,
                         const __grid_constant__ CUtensorMap w_tmap, BwdArgs args) {
  extern __shared__ __align__(1024) uint8_t smem[];
  uint8_t* sA = smem + Smem::kA;
  uint8_t* sB = smem + Smem::kB;
  uint8_t* sW = smem + Smem::kW;
  uint8_t* sS = smem + Smem::kS;
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + Smem::kBar);

  const int tid = threadIdx.x;
  const int wg = tid >> 7;
  const int tig = tid & 127;
  // Grouped rasterization: kGroup token blocks share a group and the output tile
  // varies fastest inside it, so a wave reuses its A tiles from L2. Launch order
  // only; every CTA's bytes are independent of it.
  constexpr int kGroup = 8;
  const int n_ot = args.Kw / kBO;
  const int bid = blockIdx.x;
  const int group = bid / (kGroup * n_ot);
  const int g_rows = min(kGroup, args.num_tb - group * kGroup);
  const int within = bid - group * kGroup * n_ot;
  const int tb = group * kGroup + within % g_rows;
  const int o_tile = within / g_rows;

  int expert, row0, row_end;
  if (!locate_token_block(args.offsets, args.E, tb, expert, row0, row_end)) return;

  const uint32_t bar0 = static_cast<uint32_t>(__cvta_generic_to_shared(bars));
  auto full = [&](int s) { return bar0 + 8u * s; };
  auto empty = [&](int s) { return bar0 + 8u * (kStages + s); };
  auto raw = [&](int s) { return bar0 + 8u * (2 * kStages + s); };
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < kStages; ++s) {
      det_gemm::mbar_init(full(s), kPreB ? 1 : 128);
      det_gemm::mbar_init(empty(s), 256);
      det_gemm::mbar_init(raw(s), 1);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  const int num_kb = args.Nw / kBR;
  const int blocks = args.Kw >> 5;

  if (wg == 0 && kPreB) {
    // ------------------------------------------------- producer (B by TMA)
    if (tig == 0) {
      const uint32_t sA_base = static_cast<uint32_t>(__cvta_generic_to_shared(sA));
      const uint32_t sB_base = static_cast<uint32_t>(__cvta_generic_to_shared(sB));
      const int brow = expert * args.Kw + o_tile * kBO;
      for (int kb = 0; kb < num_kb; ++kb) {
        const int s = kb % kStages;
        det_gemm::mbar_wait(empty(s), ((kb / kStages) & 1) ^ 1);
        det_gemm::mbar_arrive_expect_tx(full(s), kStageABytes + kStageBBytes);
        det_gemm::tma_2d_g2s(sA_base + s * kStageABytes, &a_tmap, kb * kBR * 2, row0, full(s));
        det_gemm::tma_2d_g2s(sB_base + s * kStageBBytes, &w_tmap, kb * kBR * 2, brow, full(s));
      }
    }
    return;
  }
  if (wg == 0) {
    // ------------------------------------------------------------ producer
    const uint32_t sA_base = static_cast<uint32_t>(__cvta_generic_to_shared(sA));
    const uint32_t sW_base = static_cast<uint32_t>(__cvta_generic_to_shared(sW));
    const int64_t wrow0 = static_cast<int64_t>(expert) * args.Nw;
    auto issue_tma = [&](int kb) {
      const int s = kb % kStages;
      mbar_expect_tx(full(s), kStageABytes);
      det_gemm::tma_2d_g2s(sA_base + s * kStageABytes, &a_tmap, kb * kBR * 2, row0, full(s));
      det_gemm::mbar_arrive_expect_tx(raw(s), kStageWBytes);
      det_gemm::tma_2d_g2s(sW_base + s * kStageWBytes, &w_tmap, o_tile * (kBO / 2),
                           static_cast<int>(wrow0 + kb * kBR), raw(s));
    };
    // Scale codes of reduced row tig (< 64) for this tile's 4 MX blocks, copied
    // asynchronously into the stage's scale buffer.
    const uint32_t sS_base = static_cast<uint32_t>(__cvta_generic_to_shared(sS));
    auto load_scales = [&](int kb) {
      if (tig < kBR) {
        const uint8_t* p = args.w_scales + (wrow0 + kb * kBR + tig) * blocks + o_tile * 4;
        cp_async4(sS_base + (kb % kStages) * kStageSBytes + tig * 4, p);
      }
      cp_async_commit();  // every thread commits, so group counts stay uniform
    };

    // Prologue: stages 0 .. kStages-2 in flight. Stage kb + kStages - 1 is issued in
    // iteration kb, once every producer thread has finished converting kb - 1 (the
    // last user of its raw and scale buffers) and the consumers released it.
    constexpr int kAhead = kStages - 1;
    for (int p = 0; p < kAhead; ++p) {
      if (p < num_kb) {
        if (tig == 0) {
          det_gemm::mbar_wait(empty(p), 1);
          issue_tma(p);
        }
        load_scales(p);
      } else {
        cp_async_commit();
      }
    }
    // Iteration kb converts stage kb, publishes it, then refills the buffers of stage
    // kb - 1 with stage kb + kAhead. The refill must come after full(kb): consumers
    // release stage kb - 1 only once their kb WGMMAs are issued, so waiting on that
    // release before publishing kb would deadlock.
    for (int kb = 0; kb < num_kb; ++kb) {
      const int s = kb % kStages;
      cp_async_wait<kAhead - 1>();  // this thread's scale copy for stage kb has landed
      // Scales of kb visible to all; every thread is done converting kb - 1.
      asm volatile("bar.sync 1, 128;" ::: "memory");
      det_gemm::mbar_wait(raw(s), (kb / kStages) & 1);
      convert_stage(sW + s * kStageWBytes, sS + s * kStageSBytes, sB + s * kStageBBytes, tig);
      fence_proxy_async();
      mbar_arrive(full(s));
      const int nxt = kb + kAhead;
      if (nxt < num_kb) {
        if (tig == 0) {
          det_gemm::mbar_wait(empty(nxt % kStages), ((nxt / kStages) & 1) ^ 1);
          issue_tma(nxt);
        }
        load_scales(nxt);
      } else {
        cp_async_commit();
      }
    }
    return;
  }

  // -------------------------------------------------------------- consumer
  const int cw = wg - 1;  // rows cw * 64 .. cw * 64 + 63 of the token tile
  const int lane = tig & 31;
  const int r0 = cw * 64 + (tig >> 5) * 16 + (lane >> 2), r1 = r0 + 8;
  const int c0 = (lane & 3) * 2;

  float acc[64];
#pragma unroll
  for (int i = 0; i < 64; ++i) acc[i] = 0.0f;

  const uint32_t sA_base = static_cast<uint32_t>(__cvta_generic_to_shared(sA)) + cw * (64 * 128);
  const uint32_t sB_base = static_cast<uint32_t>(__cvta_generic_to_shared(sB));
  const uint64_t da0 = make_smem_desc_sw128(sA_base), db0 = make_smem_desc_sw128(sB_base);

  for (int kb = 0; kb < num_kb; ++kb) {
    const int s = kb % kStages;
    det_gemm::mbar_wait(full(s), (kb / kStages) & 1);
    wgmma_fence();
#pragma unroll
    for (int j = 0; j < kBR / 16; ++j) {
      wgmma_m64n128k16_bf16(acc, da0 + (s * (kStageABytes >> 4)) + 2 * j,
                            db0 + (s * (kStageBBytes >> 4)) + 2 * j);
    }
    wgmma_commit();
    if (kb > 0) {
      wgmma_wait<1>();
      mbar_arrive(empty((kb - 1) % kStages));
    }
  }
  wgmma_wait<0>();
  mbar_arrive(empty((num_kb - 1) % kStages));

  // -------------------------------------------------------------- epilogue
  const int m0 = row0 + r0, m1 = row0 + r1;
  const bool v0 = m0 < row_end, v1 = m1 < row_end;
  const int col_base = o_tile * kBO;

  if constexpr (MODE == Mode::DX) {
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int col = col_base + 8 * i + c0;
      if (v0) *reinterpret_cast<float2*>(args.out_f32 + static_cast<int64_t>(m0) * args.Kw + col) =
          make_float2(acc[4 * i + 0], acc[4 * i + 1]);
      if (v1) *reinterpret_cast<float2*>(args.out_f32 + static_cast<int64_t>(m1) * args.Kw + col) =
          make_float2(acc[4 * i + 2], acc[4 * i + 3]);
    }
  } else {
    const int F = args.Kw;
    const int64_t zs = 2 * static_cast<int64_t>(F);
    float part[2] = {0.0f, 0.0f};
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int m = h ? m1 : m0;
      const bool v = h ? v1 : v0;
      if (!v) continue;
      const float p = __ldg(args.p_s + m);
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const int col = col_base + 8 * i + c0;
        const float2 zg = *reinterpret_cast<const float2*>(args.z + m * zs + col);
        const float2 zu = *reinterpret_cast<const float2*>(args.z + m * zs + F + col);
        float dg[2], du[2];
#pragma unroll
        for (int c = 0; c < 2; ++c) {
          const float gate = c ? zg.y : zg.x;
          const float up = c ? zu.y : zu.x;
          const float dh = __bfloat162float(__float2bfloat16_rn(acc[4 * i + 2 * h + c]));
          const float g = fminf(gate, kGateClampMax);
          const float u = fminf(fmaxf(up, kUpClampMin), kUpClampMax);
          const float sig = 1.0f / (1.0f + expf(-g));
          const float silu = g * sig;
          const float w = dh * p;
          const float dsilu = sig * (1.0f + g * (1.0f - sig));
          dg[c] = (gate < kGateClampMax) ? (w * u) * dsilu : 0.0f;
          du[c] = (up > kUpClampMin && up < kUpClampMax) ? w * silu : 0.0f;
          part[h] = __fadd_rn(part[h], __fmul_rn(__fmul_rn(dh, silu), u));
        }
        *reinterpret_cast<__nv_bfloat162*>(args.dz + m * zs + col) = __floats2bfloat162_rn(dg[0], dg[1]);
        *reinterpret_cast<__nv_bfloat162*>(args.dz + m * zs + F + col) =
            __floats2bfloat162_rn(du[0], du[1]);
      }
    }
    // Quad (same row) reduce in a fixed butterfly order.
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      part[h] = __fadd_rn(part[h], __shfl_xor_sync(0xFFFFFFFFu, part[h], 1));
      part[h] = __fadd_rn(part[h], __shfl_xor_sync(0xFFFFFFFFu, part[h], 2));
    }
    if ((lane & 3) == 0) {
      const int parts = F / kBO;
      const int idx = o_tile;
      if (v0) args.dps_part[static_cast<int64_t>(m0) * parts + idx] = part[0];
      if (v1) args.dps_part[static_cast<int64_t>(m1) * parts + idx] = part[1];
    }
  }
}

__global__ void dps_reduce_kernel(const float* __restrict__ part, int parts, int64_t rows,
                                  float* __restrict__ dp_s) {
  const int64_t m = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (m >= rows) return;
  float acc = part[m * parts];
  for (int j = 1; j < parts; ++j) acc = __fadd_rn(acc, part[m * parts + j]);
  dp_s[m] = acc;
}

// W [E, Nw, Kw] MXFP4 -> W^T BF16 [E, Kw, Nw] (row = output column) through the
// GEMM's own convert_stage, so the values are the ones the inline path builds in
// shared memory. One CTA per (64 reduced rows, 128 columns, expert); experts without
// rows are skipped and left unwritten (the GEMM never reads them).
__global__ void __launch_bounds__(128)
sm90_moe_dequant_t_kernel(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales,
                          const int32_t* __restrict__ offsets, int Nw, int Kw,
                          uint8_t* __restrict__ out) {
  __shared__ __align__(1024) uint8_t sB[kStageBBytes];
  __shared__ __align__(16) uint8_t sW[kStageWBytes];
  __shared__ __align__(16) uint8_t sS[kStageSBytes];
  const int nb = blockIdx.x, ot = blockIdx.y, e = blockIdx.z, tid = threadIdx.x;
  if (offsets[e + 1] == offsets[e]) return;
  const int64_t row0 = static_cast<int64_t>(e) * Nw + nb * kBR;
  for (int i = tid; i < kBR * 4; i += 128) {
    const int r = i >> 2, c = i & 3;
    *reinterpret_cast<uint4*>(sW + r * (kBO / 2) + c * 16) = *reinterpret_cast<const uint4*>(
        codes + (row0 + r) * (Kw / 2) + ot * (kBO / 2) + c * 16);
  }
  if (tid < kBR) {
    *reinterpret_cast<uint32_t*>(sS + tid * 4) =
        *reinterpret_cast<const uint32_t*>(scales + (row0 + tid) * (Kw >> 5) + ot * 4);
  }
  __syncthreads();
  convert_stage(sW, sS, sB, tid);
  __syncthreads();
  for (int i = tid; i < kBO * 8; i += 128) {
    const int r = i >> 3, c = i & 7;
    const int64_t dst = ((static_cast<int64_t>(e) * Kw + ot * kBO + r) * Nw + nb * kBR) * 2 + c * 16;
    *reinterpret_cast<uint4*>(out + dst) = *reinterpret_cast<const uint4*>(sB + swz128(r, c * 16));
  }
}

void check_weight(const torch::Tensor& codes, const torch::Tensor& scales, int64_t& E, int64_t& Nw,
                  int64_t& Kw) {
  TORCH_CHECK(codes.is_cuda() && codes.is_contiguous() && codes.scalar_type() == torch::kUInt8 &&
                  codes.dim() == 3,
              "w_codes must be contiguous uint8 [E, Nw, Kw/2]");
  E = codes.size(0);
  Nw = codes.size(1);
  Kw = codes.size(2) * 2;
  TORCH_CHECK(scales.is_cuda() && scales.is_contiguous() && scales.scalar_type() == torch::kUInt8 &&
                  scales.dim() == 3 && scales.size(0) == E && scales.size(1) == Nw &&
                  scales.size(2) == Kw / 32,
              "w_scales must be contiguous uint8 [E, Nw, Kw/32]");
  TORCH_CHECK(Nw % kBR == 0 && Kw % kBO == 0, "Nw % 64 and Kw % 128 must be 0, got ", Nw, " / ", Kw);
}

template <Mode MODE, bool kPreB>
void launch_impl(const torch::Tensor& a, const void* w_ptr, BwdArgs args, int64_t M, int64_t E,
                 cudaStream_t stream) {
  CUtensorMap a_tmap, w_tmap;
  init_tmap_u8(&a_tmap, a.data_ptr(), static_cast<uint64_t>(M), static_cast<uint64_t>(args.Nw) * 2,
               kBM, kBR * 2, true);
  if (kPreB) {
    init_tmap_u8(&w_tmap, w_ptr, static_cast<uint64_t>(E) * args.Kw,
                 static_cast<uint64_t>(args.Nw) * 2, kBO, kBR * 2, true);
  } else {
    init_tmap_u8(&w_tmap, w_ptr, static_cast<uint64_t>(E) * args.Nw,
                 static_cast<uint64_t>(args.Kw / 2), kBR, kBO / 2, false);
  }
  auto* kernel = sm90_moe_bwd_gemm_kernel<MODE, kPreB>;
  static bool attr_set = false;
  if (!attr_set) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Smem::kTotal));
    attr_set = true;
  }
  args.num_tb = static_cast<int>((M + kBM - 1) / kBM + E);
  const int64_t blocks = static_cast<int64_t>(args.num_tb) * (args.Kw / kBO);
  kernel<<<static_cast<unsigned>(blocks), kNumThreads, Smem::kTotal, stream>>>(a_tmap, w_tmap, args);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <Mode MODE>
void launch(const torch::Tensor& a, const torch::Tensor& w_codes,
            const c10::optional<torch::Tensor>& w_t, BwdArgs args, int64_t M, int64_t E,
            cudaStream_t stream) {
  if (w_t.has_value()) {
    const auto& t = *w_t;
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == torch::kBFloat16 &&
                    t.dim() == 3 && t.size(0) == E && t.size(1) == args.Kw && t.size(2) == args.Nw,
                "w_t must be the contiguous BF16 [E, Kw, Nw] from sm90_moe_dequant_t");
    launch_impl<MODE, true>(a, t.data_ptr(), args, M, E, stream);
  } else {
    launch_impl<MODE, false>(a, w_codes.data_ptr(), args, M, E, stream);
  }
}

void check_a(const torch::Tensor& a, int64_t Nw, const char* name) {
  TORCH_CHECK(a.is_cuda() && a.is_contiguous() && a.scalar_type() == torch::kBFloat16 &&
                  a.dim() == 2 && a.size(1) == Nw,
              name, " must be contiguous BF16 [M, ", Nw, "]");
}

void check_offsets(const torch::Tensor& offsets, int64_t E) {
  TORCH_CHECK(offsets.is_cuda() && offsets.is_contiguous() && offsets.scalar_type() == torch::kInt32 &&
                  offsets.dim() == 1 && offsets.size(0) == E + 1,
              "expert_offsets must be a contiguous int32 CUDA vector [E+1]");
}

#endif  // RL_KERNEL_ENABLE_MOE_SM90

}  // namespace

// dx = a @ W[e] per expert row group: a BF16 [M, Nw], W MXFP4 [E, Nw, Kw] -> FP32 [M, Kw].
torch::Tensor sm90_moe_bwd_dx(torch::Tensor a, torch::Tensor w_codes, torch::Tensor w_scales,
                              torch::Tensor expert_offsets, c10::optional<torch::Tensor> w_t) {
#if defined(RL_KERNEL_ENABLE_MOE_SM90)
  int64_t E, Nw, Kw;
  check_weight(w_codes, w_scales, E, Nw, Kw);
  check_a(a, Nw, "a");
  check_offsets(expert_offsets, E);
  const c10::cuda::OptionalCUDAGuard guard(device_of(a));
  const int64_t M = a.size(0);
  auto out = torch::empty({M, Kw}, a.options().dtype(torch::kFloat32));
  if (M == 0) return out;
  BwdArgs args{};
  args.w_scales = w_scales.data_ptr<uint8_t>();
  args.offsets = expert_offsets.data_ptr<int32_t>();
  args.out_f32 = out.data_ptr<float>();
  args.M = static_cast<int>(M); args.Nw = static_cast<int>(Nw); args.Kw = static_cast<int>(Kw);
  args.E = static_cast<int>(E);
  launch<Mode::DX>(a, w_codes, w_t, args, M, E, at::cuda::getCurrentCUDAStream());
  return out;
#else
  TORCH_CHECK(false, "sm90_moe_bwd_dx: not compiled (build with KERNEL_ALIGN_MOE_SM90=1)");
#endif
}

// dh = BF16(dy @ W2[e]) fused with the clamp-SwiGLU backward: returns
// (dz BF16 [M, 2F] = [dgate | dup], dp_s FP32 [M]).
std::vector<torch::Tensor> sm90_moe_bwd_dh_swiglu(torch::Tensor dy, torch::Tensor w2_codes,
                                                  torch::Tensor w2_scales,
                                                  torch::Tensor expert_offsets, torch::Tensor z,
                                                  torch::Tensor p_s,
                                                  c10::optional<torch::Tensor> w2_t) {
#if defined(RL_KERNEL_ENABLE_MOE_SM90)
  int64_t E, H, F;
  check_weight(w2_codes, w2_scales, E, H, F);
  check_a(dy, H, "dy");
  check_offsets(expert_offsets, E);
  const int64_t M = dy.size(0);
  TORCH_CHECK(z.is_cuda() && z.is_contiguous() && z.scalar_type() == torch::kFloat32 && z.dim() == 2 &&
                  z.size(0) == M && z.size(1) == 2 * F,
              "z must be contiguous FP32 [M, 2F]");
  TORCH_CHECK(p_s.is_cuda() && p_s.is_contiguous() && p_s.scalar_type() == torch::kFloat32 &&
                  p_s.dim() == 1 && p_s.size(0) == M,
              "p_s must be contiguous FP32 [M]");
  const c10::cuda::OptionalCUDAGuard guard(device_of(dy));
  auto dz = torch::empty({M, 2 * F}, dy.options());
  auto dp_s = torch::empty({M}, dy.options().dtype(torch::kFloat32));
  if (M == 0) return {dz, dp_s};
  const int parts = static_cast<int>(F / kBO);
  auto part = torch::empty({M, parts}, dy.options().dtype(torch::kFloat32));
  BwdArgs args{};
  args.w_scales = w2_scales.data_ptr<uint8_t>();
  args.offsets = expert_offsets.data_ptr<int32_t>();
  args.dz = reinterpret_cast<__nv_bfloat16*>(dz.data_ptr<at::BFloat16>());
  args.z = z.data_ptr<float>();
  args.p_s = p_s.data_ptr<float>();
  args.dps_part = part.data_ptr<float>();
  args.M = static_cast<int>(M); args.Nw = static_cast<int>(H); args.Kw = static_cast<int>(F);
  args.E = static_cast<int>(E);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch<Mode::DH_SWIGLU>(dy, w2_codes, w2_t, args, M, E, stream);
  dps_reduce_kernel<<<static_cast<unsigned>((M + 255) / 256), 256, 0, stream>>>(
      part.data_ptr<float>(), parts, M, dp_s.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dz, dp_s};
#else
  TORCH_CHECK(false, "sm90_moe_bwd_dh_swiglu: not compiled (build with KERNEL_ALIGN_MOE_SM90=1)");
#endif
}

// W MXFP4 [E, Nw, Kw] -> W^T BF16 [E, Kw, Nw] for the experts that have rows (the
// others stay uninitialised); the ``w_t`` operand of the two GEMMs above.
torch::Tensor sm90_moe_dequant_t(torch::Tensor w_codes, torch::Tensor w_scales,
                                 torch::Tensor expert_offsets) {
#if defined(RL_KERNEL_ENABLE_MOE_SM90)
  int64_t E, Nw, Kw;
  check_weight(w_codes, w_scales, E, Nw, Kw);
  check_offsets(expert_offsets, E);
  const c10::cuda::OptionalCUDAGuard guard(device_of(w_codes));
  auto out = torch::empty({E, Kw, Nw}, w_codes.options().dtype(torch::kBFloat16));
  if (E == 0) return out;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(static_cast<unsigned>(Nw / kBR), static_cast<unsigned>(Kw / kBO), static_cast<unsigned>(E));
  sm90_moe_dequant_t_kernel<<<grid, 128, 0, stream>>>(
      w_codes.data_ptr<uint8_t>(), w_scales.data_ptr<uint8_t>(),
      expert_offsets.data_ptr<int32_t>(), static_cast<int>(Nw), static_cast<int>(Kw),
      reinterpret_cast<uint8_t*>(out.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
#else
  TORCH_CHECK(false, "sm90_moe_dequant_t: not compiled (build with KERNEL_ALIGN_MOE_SM90=1)");
#endif
}
