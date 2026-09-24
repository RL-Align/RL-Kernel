// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// P1-2 fp32_gemm_rms: deterministic FP32 controller projection + controller
// RMS scale, forward and backward, byte-equal to the FP32 oracle
// (rl_engine/mhc/oracle.py, numeric profile oracle-fp32-mhc-v1).
//
// Determinism by construction: every reduced output element is produced by a
// single FP32 accumulator walking its reduction axis in ascending order with
// __fmul_rn / __fadd_rn (mul and add round separately, no FMA). Parallelism
// comes only from independent output elements, so there is no Split-K /
// Stream-K / atomic partial accumulation and the bytes cannot depend on batch
// size, token count, SM count or launch geometry.
//
// Performance notes (bytes never change): the optimized paths only widen
// loads (float4 over consecutive ascending k), stage W through shared memory
// (exact copies), and prefetch the next iteration's operands into registers
// so global-memory latency stays off the serial accumulation chain. The
// per-element accumulation order is identical to the scalar fallback paths.
//
// Compiled into the standalone extension rl_engine._C_mhc with -fmad=false
// and no fast-math (see setup.py).

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_pipeline.h>

#include <cmath>

namespace {

constexpr int kMaxWarpN = 31;  // forward kernels: lanes 0..N-1 + one sumsq lane

// Oracle epilogue: norm = sqrt(s); q = norm * (1/sqrt(K)); r = 1 / (q + eps).
// Controller RMS, deliberately NOT rsqrt(mean + eps). Torch executes the
// scalar division by sqrt(K) as a multiply by the FP32-rounded double
// reciprocal (measured on-device), so the kernel does exactly that.
__device__ __forceinline__ void write_rms_epilogue(
    float ss, float recip_sqrt_k, float eps, long t,
    float* s_out, float* norm_out, float* q_out, float* r_out) {
  const float norm = __fsqrt_rn(ss);
  const float q = __fmul_rn(norm, recip_sqrt_k);
  const float r = __fdiv_rn(1.0f, __fadd_rn(q, eps));
  s_out[t] = ss;
  norm_out[t] = norm;
  q_out[t] = q;
  r_out[t] = r;
}

// ---------------------------------------------------------------------------
// Forward, scalar fallback (any K): one 32-thread block per token. Lane
// n < N accumulates P[t, n] serially over k; lane N accumulates sum(x^2).
// ---------------------------------------------------------------------------
__global__ void gemm_rms_fwd_warp_kernel(
    const float* __restrict__ x,   // [T, K]
    const float* __restrict__ w,   // [N, K]
    float* __restrict__ p,         // [T, N]
    float* __restrict__ s_out,     // [T]
    float* __restrict__ norm_out,  // [T]
    float* __restrict__ q_out,     // [T]
    float* __restrict__ r_out,     // [T]
    long tokens,
    long n_dim,
    long k_dim,
    float recip_sqrt_k,
    float eps) {
  const long t = blockIdx.x;
  if (t >= tokens) return;
  const int lane = threadIdx.x;
  const float* x_row = x + t * k_dim;

  if (lane < n_dim) {
    const float* w_row = w + static_cast<long>(lane) * k_dim;
    float acc = 0.0f;
    for (long k = 0; k < k_dim; ++k) {
      acc = __fadd_rn(acc, __fmul_rn(__ldg(x_row + k), __ldg(w_row + k)));
    }
    p[t * n_dim + lane] = acc;
  } else if (lane == n_dim) {
    float ss = 0.0f;
    for (long k = 0; k < k_dim; ++k) {
      const float v = __ldg(x_row + k);
      ss = __fadd_rn(ss, __fmul_rn(v, v));
    }
    write_rms_epilogue(ss, recip_sqrt_k, eps, t, s_out, norm_out, q_out, r_out);
  }
}

// ---------------------------------------------------------------------------
// Forward fast paths (K % 4 == 0). All three kernels share one structure:
//
// - Every serial chain (24 projection chains + 1 sum-of-squares chain per
//   token) is carried by one lane, and the chain walks k strictly ascending
//   with one __fadd_rn at a time (bytes identical to the scalar fallback).
// - The sum-of-squares chain uses the same inner loop as the projection
//   chains: its "weight" pointer aliases the staged X row.
// - Phase-separated staging (measured: on this part staging and compute do
//   not overlap anyway — the wall clock is additive in MIO wavefronts), so
//   each large chunk is staged once by EVERY thread of the block at the
//   multi-warp cp.async rate, then consumed after a single barrier. No
//   double buffering, no wait bookkeeping.
// - Tile rows are contiguous with an odd float4 row stride (pad), so staging
//   stores are contiguous and chain reads spread across banks.
// ---------------------------------------------------------------------------
constexpr int kSoloC4 = 512;               // 2048 k elements per staged chunk
constexpr int kSoloRS = kSoloC4 + 1;       // padded row stride (float4s)
constexpr int kGnC4 = 128;                 // 512 k elements (triple-buffered G2)
constexpr int kGnRS = kGnC4 + 1;
constexpr int kG8C4 = 64;                  // 256 k elements (triple-buffered K3)
constexpr int kG8RS = kG8C4 + 1;
constexpr long kSoloMaxTokens = 32;
constexpr long kG2MaxTokens = 288;

__device__ __forceinline__ void fwd_store_chain(
    float acc, long t, int c, long n_dim, float recip_sqrt_k, float eps,
    float* __restrict__ p, float* s_out, float* norm_out, float* q_out,
    float* r_out) {
  if (c < n_dim) {
    p[t * n_dim + c] = acc;
  } else {
    write_rms_epilogue(acc, recip_sqrt_k, eps, t, s_out, norm_out, q_out, r_out);
  }
}

// The shared unified chain loop over one staged chunk: fully unrolled
// 128-float4 blocks with pointer advance (every LDS keeps an immediate
// offset) and a 4-deep rotating register prefetch. Adds stay in ascending k
// order; xr/wr are per-lane smem row pointers (wr aliases xr for the
// sum-of-squares chain).
__device__ __forceinline__ float fwd_chain_chunk(
    float acc, const float4* __restrict__ xr, const float4* __restrict__ wr,
    int cur4) {
  int blk = 0;
  for (; blk + 128 <= cur4; blk += 128) {
    const float4* xp = xr + blk;
    const float4* wp = wr + blk;
    float4 xb[4], wb[4];
#pragma unroll
    for (int d = 0; d < 4; ++d) {
      xb[d] = xp[d];
      wb[d] = wp[d];
    }
#pragma unroll
    for (int j4 = 0; j4 < 128; ++j4) {
      const int sl = j4 & 3;
      const float4 xv = xb[sl];
      const float4 wv = wb[sl];
      if (j4 + 4 < 128) {
        xb[sl] = xp[j4 + 4];
        wb[sl] = wp[j4 + 4];
      }
      acc = __fadd_rn(acc, __fmul_rn(xv.x, wv.x));
      acc = __fadd_rn(acc, __fmul_rn(xv.y, wv.y));
      acc = __fadd_rn(acc, __fmul_rn(xv.z, wv.z));
      acc = __fadd_rn(acc, __fmul_rn(xv.w, wv.w));
    }
  }
  for (; blk < cur4; ++blk) {
    const float4 xv = xr[blk];
    const float4 wv = wr[blk];
    acc = __fadd_rn(acc, __fmul_rn(xv.x, wv.x));
    acc = __fadd_rn(acc, __fmul_rn(xv.y, wv.y));
    acc = __fadd_rn(acc, __fmul_rn(xv.z, wv.z));
    acc = __fadd_rn(acc, __fmul_rn(xv.w, wv.w));
  }
  return acc;
}

// Stage `rows` W rows starting at `row0` plus `xrows` X rows into the tile,
// cooperatively over all block threads (16B contiguous copies).
template <int kC4, int kRS>
__device__ __forceinline__ void fwd_stage(
    const float* __restrict__ w, const float* __restrict__ x, float4* w_tile,
    float4* x_tile, int tid, int nthreads, int row0, int rows, long t0,
    int xrows, long tokens, long k_dim, long k0, int cur4) {
  constexpr int kShift = kC4 == 64 ? 6 : (kC4 == 128 ? 7 : (kC4 == 256 ? 8 : 9));
  if (cur4 == kC4) {
    for (int idx = tid; idx < (rows << kShift); idx += nthreads) {
      const int r = idx >> kShift;
      const int j4 = idx & (kC4 - 1);
      __pipeline_memcpy_async(
          w_tile + r * kRS + j4,
          w + static_cast<long>(row0 + r) * k_dim + k0 + (j4 << 2),
          sizeof(float4));
    }
    for (int idx = tid; idx < (xrows << kShift); idx += nthreads) {
      const int r = idx >> kShift;
      const int j4 = idx & (kC4 - 1);
      if (t0 + r < tokens) {
        __pipeline_memcpy_async(x_tile + r * kRS + j4,
                                x + (t0 + r) * k_dim + k0 + (j4 << 2),
                                sizeof(float4));
      }
    }
  } else {
    for (int idx = tid; idx < rows * cur4; idx += nthreads) {
      const int r = idx / cur4;
      const int j4 = idx - r * cur4;
      __pipeline_memcpy_async(
          w_tile + r * kRS + j4,
          w + static_cast<long>(row0 + r) * k_dim + k0 + (j4 << 2),
          sizeof(float4));
    }
    for (int idx = tid; idx < xrows * cur4; idx += nthreads) {
      const int r = idx / cur4;
      const int j4 = idx - r * cur4;
      if (t0 + r < tokens) {
        __pipeline_memcpy_async(x_tile + r * kRS + j4,
                                x + (t0 + r) * k_dim + k0 + (j4 << 2),
                                sizeof(float4));
      }
    }
  }
}

// --- K1: solo-split, T <= 32. Four blocks per token; block q owns chains
// q*ceil((N+1)/4).. and stages only its own W rows. 128 threads: warp 0
// carries the chains, all four warps stage (multi-warp cp.async rate).
__global__ void __launch_bounds__(128, 1) gemm_rms_fwd_solo_kernel(
    const float* __restrict__ x, const float* __restrict__ w,
    float* __restrict__ p, float* __restrict__ s_out,
    float* __restrict__ norm_out, float* __restrict__ q_out,
    float* __restrict__ r_out, long tokens, long n_dim, long k_dim,
    float recip_sqrt_k, float eps) {
  extern __shared__ char sm[];
  const int cpt = static_cast<int>(n_dim) + 1;
  const int per_block = (cpt + 3) / 4;
  const long t = blockIdx.x >> 2;
  const int quarter = blockIdx.x & 3;
  const int c0 = quarter * per_block;
  const int n_chains = min(per_block, cpt - c0);
  if (n_chains <= 0) return;
  const int w_rows = max(0, min(per_block, static_cast<int>(n_dim) - c0));
  const int tid = threadIdx.x;
  const int chunk = kSoloC4 << 2;
  const int n_chunks = static_cast<int>((k_dim + chunk - 1) / chunk);
  float4* w_tile = reinterpret_cast<float4*>(sm);
  float4* x_tile = w_tile + static_cast<size_t>(w_rows) * kSoloRS;

  float acc = 0.0f;
  const int c_lane = c0 + tid;
  for (int c = 0; c < n_chunks; ++c) {
    const long k0 = static_cast<long>(c) * chunk;
    const int cur4 =
        static_cast<int>(min(static_cast<long>(chunk), k_dim - k0)) >> 2;
    fwd_stage<kSoloC4, kSoloRS>(w, x, w_tile, x_tile, tid, 128, c0, w_rows, t,
                                1, tokens, k_dim, k0, cur4);
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    if (tid < n_chains) {
      const float4* xr = x_tile;
      const float4* wr = c_lane < n_dim ? w_tile + tid * kSoloRS : xr;
      acc = fwd_chain_chunk(acc, xr, wr, cur4);
    }
    __syncthreads();
  }
  if (tid < n_chains) {
    fwd_store_chain(acc, t, c_lane, n_dim, recip_sqrt_k, eps, p, s_out,
                    norm_out, q_out, r_out);
  }
}

// The CPL2 chain loop: one lane carries two chains (tokens g and g+4) with
// the same weight column, so a single 16B W read feeds both accumulators.
// Each chain's adds stay strictly ascending in k; interleaving the two
// independent chains does not touch either rounding sequence.
__device__ __forceinline__ void fwd_chain_chunk_cpl2(
    float& acc_a, float& acc_b, const float4* __restrict__ xa,
    const float4* __restrict__ xb2, const float4* __restrict__ wr, int cur4) {
  int blk = 0;
  for (; blk + 128 <= cur4; blk += 128) {
    const float4* xap = xa + blk;
    const float4* xbp = xb2 + blk;
    const float4* wp = wr + blk;
    float4 ra[4], rb[4], rw[4];
#pragma unroll
    for (int d = 0; d < 4; ++d) {
      ra[d] = xap[d];
      rb[d] = xbp[d];
      rw[d] = wp[d];
    }
#pragma unroll
    for (int j4 = 0; j4 < 128; ++j4) {
      const int sl = j4 & 3;
      const float4 va = ra[sl];
      const float4 vb = rb[sl];
      const float4 wv = rw[sl];
      if (j4 + 4 < 128) {
        ra[sl] = xap[j4 + 4];
        rb[sl] = xbp[j4 + 4];
        rw[sl] = wp[j4 + 4];
      }
      acc_a = __fadd_rn(acc_a, __fmul_rn(va.x, wv.x));
      acc_b = __fadd_rn(acc_b, __fmul_rn(vb.x, wv.x));
      acc_a = __fadd_rn(acc_a, __fmul_rn(va.y, wv.y));
      acc_b = __fadd_rn(acc_b, __fmul_rn(vb.y, wv.y));
      acc_a = __fadd_rn(acc_a, __fmul_rn(va.z, wv.z));
      acc_b = __fadd_rn(acc_b, __fmul_rn(vb.z, wv.z));
      acc_a = __fadd_rn(acc_a, __fmul_rn(va.w, wv.w));
      acc_b = __fadd_rn(acc_b, __fmul_rn(vb.w, wv.w));
    }
  }
  for (; blk < cur4; ++blk) {
    const float4 va = xa[blk];
    const float4 vb = xb2[blk];
    const float4 wv = wr[blk];
    acc_a = __fadd_rn(acc_a, __fmul_rn(va.x, wv.x));
    acc_b = __fadd_rn(acc_b, __fmul_rn(vb.x, wv.x));
    acc_a = __fadd_rn(acc_a, __fmul_rn(va.y, wv.y));
    acc_b = __fadd_rn(acc_b, __fmul_rn(vb.y, wv.y));
    acc_a = __fadd_rn(acc_a, __fmul_rn(va.z, wv.z));
    acc_b = __fadd_rn(acc_b, __fmul_rn(vb.z, wv.z));
    acc_a = __fadd_rn(acc_a, __fmul_rn(va.w, wv.w));
    acc_b = __fadd_rn(acc_b, __fmul_rn(vb.w, wv.w));
  }
}

// --- K3: eight tokens per block with two chains per lane, T > 288. CPL2
// halves per-chain W read wavefronts (verified: 2.2x fewer, short-
// scoreboard 49% -> 1%); the last two warps are dedicated staging warps
// driving a TRIPLE-buffered tile with a single barrier per chunk (the
// mod-3 write/read distance makes the trailing barrier unnecessary — the
// same change that took G2 past torch-native). Sum-of-squares chains sit
// on the slots after the 4N projection slots (their own warp at N = 24).
template <int kThreads>
__global__ void __launch_bounds__(kThreads, 1) gemm_rms_fwd_g8cpl2_kernel(
    const float* __restrict__ x, const float* __restrict__ w,
    float* __restrict__ p, float* __restrict__ s_out,
    float* __restrict__ norm_out, float* __restrict__ q_out,
    float* __restrict__ r_out, long tokens, long n_dim, long k_dim,
    float recip_sqrt_k, float eps) {
  constexpr int kG = 8;
  constexpr int kHalfG = 4;
  extern __shared__ char sm[];
  const int n_i = static_cast<int>(n_dim);
  const long t0 = static_cast<long>(blockIdx.x) * kG;
  const int tid = threadIdx.x;
  const int chunk = kG8C4 << 2;
  const int n_chunks = static_cast<int>((k_dim + chunk - 1) / chunk);
  const size_t buf_f4 = static_cast<size_t>(n_i + kG) * kG8RS;
  auto w_tile = [&](int buf) {
    return reinterpret_cast<float4*>(sm) + static_cast<size_t>(buf) * buf_f4;
  };
  auto x_tile = [&](int buf) {
    return w_tile(buf) + static_cast<size_t>(n_i) * kG8RS;
  };
  const int stage_tid = tid - (kThreads - 64);
  const bool stager = stage_tid >= 0;

  auto issue = [&](int c) {
    if (!stager) return;
    const long k0 = static_cast<long>(c) * chunk;
    const int cur4 =
        static_cast<int>(min(static_cast<long>(chunk), k_dim - k0)) >> 2;
    const int buf = c % 3;
    fwd_stage<kG8C4, kG8RS>(w, x, w_tile(buf), x_tile(buf), stage_tid, 64, 0,
                            n_i, t0, kG, tokens, k_dim, k0, cur4);
    __pipeline_commit();
  };

  const int p_slots = kHalfG * n_i;
  const bool p_lane = tid < p_slots;
  const int g = p_lane ? tid / n_i : 0;
  const int c = p_lane ? tid - g * n_i : 0;
  const int ss_t = tid - p_slots;
  const bool ss_lane = !p_lane && !stager && ss_t < kG;
  const bool a_ok = p_lane && t0 + g < tokens;
  const bool b_ok = p_lane && t0 + kHalfG + g < tokens;

  float acc_a = 0.0f;
  float acc_b = 0.0f;
  issue(0);
  for (int cc = 0; cc < n_chunks; ++cc) {
    if (cc + 1 < n_chunks) {
      issue(cc + 1);
      if (stager) __pipeline_wait_prior(1);
    } else {
      if (stager) __pipeline_wait_prior(0);
    }
    __syncthreads();
    const long k0 = static_cast<long>(cc) * chunk;
    const int cur4 =
        static_cast<int>(min(static_cast<long>(chunk), k_dim - k0)) >> 2;
    const int buf = cc % 3;
    if (p_lane) {
      const float4* xa = x_tile(buf) + g * kG8RS;
      const float4* xb2 = x_tile(buf) + (kHalfG + g) * kG8RS;
      const float4* wr = w_tile(buf) + c * kG8RS;
      fwd_chain_chunk_cpl2(acc_a, acc_b, xa, xb2, wr, cur4);
    } else if (ss_lane && t0 + ss_t < tokens) {
      const float4* xr = x_tile(buf) + ss_t * kG8RS;
      acc_a = fwd_chain_chunk(acc_a, xr, xr, cur4);
    }
  }
  if (a_ok) {
    p[(t0 + g) * n_dim + c] = acc_a;
  }
  if (b_ok) {
    p[(t0 + kHalfG + g) * n_dim + c] = acc_b;
  }
  if (ss_lane && t0 + ss_t < tokens) {
    write_rms_epilogue(acc_a, recip_sqrt_k, eps, t0 + ss_t, s_out, norm_out,
                       q_out, r_out);
  }
}

// --- K2 family: kG tokens per block sharing one W tile. Compute lanes
// s = tid -> (t_local = s / (N+1), c = s % (N+1)); the LAST TWO WARPS are
// dedicated staging warps driving a double-buffered tile. The compute
// phase here is issue/chain bound (not LSU bound), so staging genuinely
// overlaps it — unlike same-warp staging, which the MIO additivity made
// pointless.
template <int kG, int kThreads>
__global__ void __launch_bounds__(kThreads, 1) gemm_rms_fwd_gN_kernel(
    const float* __restrict__ x, const float* __restrict__ w,
    float* __restrict__ p, float* __restrict__ s_out,
    float* __restrict__ norm_out, float* __restrict__ q_out,
    float* __restrict__ r_out, long tokens, long n_dim, long k_dim,
    float recip_sqrt_k, float eps) {
  extern __shared__ char sm[];
  const int n_i = static_cast<int>(n_dim);
  const int cpt = n_i + 1;
  const long t0 = static_cast<long>(blockIdx.x) * kG;
  const int tid = threadIdx.x;
  const int chunk = kGnC4 << 2;
  const int n_chunks = static_cast<int>((k_dim + chunk - 1) / chunk);
  // Triple-buffered with a SINGLE barrier per chunk: the staging warps
  // write buffer (cc+1) % 3 while the chain warps read (cc) % 3, and the
  // buffer being written was last read two chunks ago (mod-3 distance 2),
  // so no trailing barrier is needed and the sync count halves.
  const size_t buf_f4 = static_cast<size_t>(n_i + kG) * kGnRS;
  auto w_tile = [&](int buf) {
    return reinterpret_cast<float4*>(sm) + static_cast<size_t>(buf) * buf_f4;
  };
  auto x_tile = [&](int buf) {
    return w_tile(buf) + static_cast<size_t>(n_i) * kGnRS;
  };
  const int stage_tid = tid - (kThreads - 64);
  const bool stager = stage_tid >= 0;

  auto issue = [&](int c) {
    if (!stager) return;
    const long k0 = static_cast<long>(c) * chunk;
    const int cur4 =
        static_cast<int>(min(static_cast<long>(chunk), k_dim - k0)) >> 2;
    const int buf = c % 3;
    fwd_stage<kGnC4, kGnRS>(w, x, w_tile(buf), x_tile(buf), stage_tid, 64, 0,
                            n_i, t0, kG, tokens, k_dim, k0, cur4);
    __pipeline_commit();
  };

  const int s = tid;
  const int t_local = s / cpt;
  const int c = s - t_local * cpt;
  const bool active = t_local < kG && t0 + t_local < tokens;

  float acc = 0.0f;
  issue(0);
  for (int cc = 0; cc < n_chunks; ++cc) {
    if (cc + 1 < n_chunks) {
      issue(cc + 1);
      if (stager) __pipeline_wait_prior(1);
    } else {
      if (stager) __pipeline_wait_prior(0);
    }
    __syncthreads();
    const long k0 = static_cast<long>(cc) * chunk;
    const int cur4 =
        static_cast<int>(min(static_cast<long>(chunk), k_dim - k0)) >> 2;
    const int buf = cc % 3;
    if (active) {
      const float4* xr = x_tile(buf) + t_local * kGnRS;
      const float4* wr = c < n_i ? w_tile(buf) + c * kGnRS : xr;
      acc = fwd_chain_chunk(acc, xr, wr, cur4);
    }
  }
  if (active) {
    fwd_store_chain(acc, t0 + t_local, c, n_dim, recip_sqrt_k, eps, p, s_out,
                    norm_out, q_out, r_out);
  }
}

// ---------------------------------------------------------------------------
// Generic fixed-K GEMM: one thread per output element, K ascending.
// a: [M, K], b: [N, K] -> out: [M, N] (out[m, n] = fold_k a[m, k] * b[n, k]).
// ---------------------------------------------------------------------------
__global__ void fixed_k_gemm_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    long m_dim,
    long n_dim,
    long k_dim) {
  const long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= m_dim * n_dim) return;
  const long m = idx / n_dim;
  const long n = idx % n_dim;
  const float* a_row = a + m * k_dim;
  const float* b_row = b + n * k_dim;
  float acc = 0.0f;
  for (long k = 0; k < k_dim; ++k) {
    acc = __fadd_rn(acc, __fmul_rn(__ldg(a_row + k), __ldg(b_row + k)));
  }
  out[idx] = acc;
}

// ---------------------------------------------------------------------------
// Backward dX, scalar fallback: one thread per (t, k). GEMM leg folds n
// ascending (dX_gemm[t, k] = fold_n dP[t, n] * W[n, k]), then, when has_rms,
// the RMS leg in the oracle's exact association:
// dX_rms = dr * ((-(r*r) * x) / (K * q)); dX = dX_gemm + dX_rms.
// ---------------------------------------------------------------------------
__global__ void gemm_rms_bwd_dx_kernel(
    const float* __restrict__ dp,  // [T, N]
    const float* __restrict__ dr,  // [T] (null when !has_rms)
    const float* __restrict__ x,   // [T, K]
    const float* __restrict__ w,   // [N, K]
    const float* __restrict__ q,   // [T] (null when !has_rms)
    const float* __restrict__ r,   // [T] (null when !has_rms)
    float* __restrict__ dx,        // [T, K]
    long tokens,
    long n_dim,
    long k_dim,
    bool has_rms) {
  const long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= tokens * k_dim) return;
  const long t = idx / k_dim;
  const long k = idx % k_dim;

  float acc = 0.0f;
  const float* dp_row = dp + t * n_dim;
  for (long n = 0; n < n_dim; ++n) {
    acc = __fadd_rn(acc, __fmul_rn(__ldg(dp_row + n), __ldg(w + n * k_dim + k)));
  }
  if (has_rms) {
    const float r_t = __ldg(r + t);
    const float neg_r2 = -__fmul_rn(r_t, r_t);
    const float denom = __fmul_rn(static_cast<float>(k_dim), __ldg(q + t));
    const float scaled = __fdiv_rn(__fmul_rn(neg_r2, __ldg(x + idx)), denom);
    acc = __fadd_rn(acc, __fmul_rn(__ldg(dr + t), scaled));
  }
  dx[idx] = acc;
}

// ---------------------------------------------------------------------------
// Backward dX, float4 fast path (K % 4 == 0): one thread per (t, 4k). Four
// independent accumulator chains per thread with fully coalesced W/X/dX
// accesses. Per-element order identical to the fallback.
// ---------------------------------------------------------------------------
__global__ void gemm_rms_bwd_dx_vec4_kernel(
    const float* __restrict__ dp,
    const float* __restrict__ dr,
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ q,
    const float* __restrict__ r,
    float* __restrict__ dx,
    long tokens,
    long n_dim,
    long k4_dim,  // K / 4
    bool has_rms) {
  const long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= tokens * k4_dim) return;
  const long t = idx / k4_dim;
  const long k4 = idx % k4_dim;
  const long k_dim = k4_dim << 2;
  const float* dp_row = dp + t * n_dim;

  float4 a = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll 8
  for (long n = 0; n < n_dim; ++n) {
    const float d = __ldg(dp_row + n);
    const float4 wv = __ldg(reinterpret_cast<const float4*>(w + n * k_dim) + k4);
    a.x = __fadd_rn(a.x, __fmul_rn(d, wv.x));
    a.y = __fadd_rn(a.y, __fmul_rn(d, wv.y));
    a.z = __fadd_rn(a.z, __fmul_rn(d, wv.z));
    a.w = __fadd_rn(a.w, __fmul_rn(d, wv.w));
  }
  if (has_rms) {
    const float r_t = __ldg(r + t);
    const float neg_r2 = -__fmul_rn(r_t, r_t);
    const float denom = __fmul_rn(static_cast<float>(k_dim), __ldg(q + t));
    const float dr_t = __ldg(dr + t);
    const float4 xv = __ldg(reinterpret_cast<const float4*>(x + t * k_dim) + k4);
    a.x = __fadd_rn(a.x, __fmul_rn(dr_t, __fdiv_rn(__fmul_rn(neg_r2, xv.x), denom)));
    a.y = __fadd_rn(a.y, __fmul_rn(dr_t, __fdiv_rn(__fmul_rn(neg_r2, xv.y), denom)));
    a.z = __fadd_rn(a.z, __fmul_rn(dr_t, __fdiv_rn(__fmul_rn(neg_r2, xv.z), denom)));
    a.w = __fadd_rn(a.w, __fmul_rn(dr_t, __fdiv_rn(__fmul_rn(neg_r2, xv.w), denom)));
  }
  reinterpret_cast<float4*>(dx + t * k_dim)[k4] = a;
}

// ---------------------------------------------------------------------------
// Backward dW, float4 fast path (K % 4 == 0): one thread per (n, 4k), t
// ascending, coalesced. Per-element order identical to the fallback.
// ---------------------------------------------------------------------------
__global__ void gemm_rms_bwd_dw_vec4_kernel(
    const float* __restrict__ dp,
    const float* __restrict__ x,
    float* __restrict__ dw,
    long tokens,
    long n_dim,
    long k4_dim) {
  const long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n_dim * k4_dim) return;
  const long n = idx / k4_dim;
  const long k4 = idx % k4_dim;
  const long k_dim = k4_dim << 2;

  float4 a = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll 4
  for (long t = 0; t < tokens; ++t) {
    const float d = __ldg(dp + t * n_dim + n);
    const float4 xv = __ldg(reinterpret_cast<const float4*>(x + t * k_dim) + k4);
    a.x = __fadd_rn(a.x, __fmul_rn(d, xv.x));
    a.y = __fadd_rn(a.y, __fmul_rn(d, xv.y));
    a.z = __fadd_rn(a.z, __fmul_rn(d, xv.z));
    a.w = __fadd_rn(a.w, __fmul_rn(d, xv.w));
  }
  reinterpret_cast<float4*>(dw + n * k_dim)[k4] = a;
}

// ---------------------------------------------------------------------------
// Backward dW, scalar fallback: one thread per (n, k), t ascending
// (dW[n, k] = fold_t dP[t, n] * X[t, k]).
// ---------------------------------------------------------------------------
__global__ void gemm_rms_bwd_dw_kernel(
    const float* __restrict__ dp,  // [T, N]
    const float* __restrict__ x,   // [T, K]
    float* __restrict__ dw,        // [N, K]
    long tokens,
    long n_dim,
    long k_dim) {
  const long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n_dim * k_dim) return;
  const long n = idx / k_dim;
  const long k = idx % k_dim;
  float acc = 0.0f;
  for (long t = 0; t < tokens; ++t) {
    acc = __fadd_rn(acc, __fmul_rn(__ldg(dp + t * n_dim + n), __ldg(x + t * k_dim + k)));
  }
  dw[idx] = acc;
}

void check_input(const torch::Tensor& t, const char* name, int dim) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, " must be FP32, got ", t.scalar_type());
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.dim() == dim, name, " must be ", dim, "-D, got ", t.dim(), "-D");
}

long grid_1d(long total, int threads) {
  return (total + threads - 1) / threads;
}

}  // namespace

std::vector<torch::Tensor> fp32_gemm_rms_forward(
    torch::Tensor x, torch::Tensor w, double eps) {
  check_input(x, "x_flat", 2);
  check_input(w, "weight", 2);
  const long tokens = x.size(0);
  const long k_dim = x.size(1);
  const long n_dim = w.size(0);
  TORCH_CHECK(w.size(1) == k_dim, "weight K ", w.size(1), " != x K ", k_dim);
  TORCH_CHECK(k_dim >= 1, "K must be >= 1");
  TORCH_CHECK(n_dim >= 1 && n_dim <= kMaxWarpN,
              "fp32_gemm_rms CUDA kernel supports 1 <= N <= ", kMaxWarpN,
              ", got ", n_dim, " (fail-closed)");
  TORCH_CHECK(std::isfinite(eps) && eps > 0.0, "eps must be positive and finite");

  const at::cuda::CUDAGuard guard(x.device());
  auto p = torch::empty({tokens, n_dim}, x.options());
  auto s = torch::empty({tokens}, x.options());
  auto norm = torch::empty({tokens}, x.options());
  auto q = torch::empty({tokens}, x.options());
  auto r = torch::empty({tokens}, x.options());
  if (tokens > 0) {
    const float recip_sqrt_k =
        static_cast<float>(1.0 / std::sqrt(static_cast<double>(k_dim)));
    auto stream = at::cuda::getCurrentCUDAStream();
    if ((k_dim & 3) == 0) {
      const int n_i = static_cast<int>(n_dim);
      if (tokens <= kSoloMaxTokens) {
        const int per_block = (n_i + 1 + 3) / 4;
        const size_t smem =
            static_cast<size_t>(per_block + 1) * kSoloRS * sizeof(float4);
        static bool solo_opt = [] {
          cudaFuncSetAttribute(gemm_rms_fwd_solo_kernel,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               10 * kSoloRS * sizeof(float4));
          return true;
        }();
        (void)solo_opt;
        gemm_rms_fwd_solo_kernel<<<static_cast<unsigned>(tokens * 4), 128,
                                   smem, stream>>>(
            x.data_ptr<float>(), w.data_ptr<float>(), p.data_ptr<float>(),
            s.data_ptr<float>(), norm.data_ptr<float>(), q.data_ptr<float>(),
            r.data_ptr<float>(), tokens, n_dim, k_dim, recip_sqrt_k,
            static_cast<float>(eps));
      } else if (tokens <= kG2MaxTokens) {
        const size_t smem =
            3 * static_cast<size_t>(n_i + 2) * kGnRS * sizeof(float4);
        static bool g2_opt = [] {
          cudaFuncSetAttribute(gemm_rms_fwd_gN_kernel<2, 128>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               3 * 33 * kGnRS * sizeof(float4));
          return true;
        }();
        (void)g2_opt;
        gemm_rms_fwd_gN_kernel<2, 128>
            <<<static_cast<unsigned>(grid_1d(tokens, 2)), 128, smem, stream>>>(
                x.data_ptr<float>(), w.data_ptr<float>(), p.data_ptr<float>(),
                s.data_ptr<float>(), norm.data_ptr<float>(), q.data_ptr<float>(),
                r.data_ptr<float>(), tokens, n_dim, k_dim, recip_sqrt_k,
                static_cast<float>(eps));
      } else {
        const size_t smem =
            3 * static_cast<size_t>(n_i + 8) * kG8RS * sizeof(float4);
        if (smem <= 224 * 1024) {
          static bool g8_opt = [] {
            cudaFuncSetAttribute(gemm_rms_fwd_g8cpl2_kernel<192>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 224 * 1024);
            return true;
          }();
          (void)g8_opt;
          // 4*N projection + kG sum-of-squares slots (N <= 31 -> 132, warps
          // 0..3) plus two dedicated staging warps.
          gemm_rms_fwd_g8cpl2_kernel<192>
              <<<static_cast<unsigned>(grid_1d(tokens, 8)), 192, smem, stream>>>(
                  x.data_ptr<float>(), w.data_ptr<float>(), p.data_ptr<float>(),
                  s.data_ptr<float>(), norm.data_ptr<float>(),
                  q.data_ptr<float>(), r.data_ptr<float>(), tokens, n_dim,
                  k_dim, recip_sqrt_k, static_cast<float>(eps));
        } else {
          // Very wide N would overflow the triple-buffered tile; the G2
          // kernel handles any token count correctly.
          const size_t smem2 =
              3 * static_cast<size_t>(n_i + 2) * kGnRS * sizeof(float4);
          gemm_rms_fwd_gN_kernel<2, 128>
              <<<static_cast<unsigned>(grid_1d(tokens, 2)), 128, smem2,
                 stream>>>(
                  x.data_ptr<float>(), w.data_ptr<float>(), p.data_ptr<float>(),
                  s.data_ptr<float>(), norm.data_ptr<float>(),
                  q.data_ptr<float>(), r.data_ptr<float>(), tokens, n_dim,
                  k_dim, recip_sqrt_k, static_cast<float>(eps));
        }
      }
    } else {
      gemm_rms_fwd_warp_kernel<<<static_cast<unsigned>(tokens), 32, 0, stream>>>(
          x.data_ptr<float>(), w.data_ptr<float>(), p.data_ptr<float>(),
          s.data_ptr<float>(), norm.data_ptr<float>(), q.data_ptr<float>(),
          r.data_ptr<float>(), tokens, n_dim, k_dim, recip_sqrt_k,
          static_cast<float>(eps));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {p, r, s, norm, q};
}

namespace {

void launch_bwd(
    const torch::Tensor& dp, const float* dr, const torch::Tensor& x,
    const torch::Tensor& w, const float* q, const float* r, torch::Tensor& dx,
    torch::Tensor& dw, long tokens, long n_dim, long k_dim, bool has_rms) {
  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int kThreads = 256;
  if ((k_dim & 3) == 0) {
    // dX and dW are fully independent deterministic kernels (disjoint
    // outputs, read-only shared inputs). For larger batches they run on two
    // streams purely for concurrency; each kernel's bytes are unchanged
    // either way, and small batches skip the event overhead.
    const long k4 = k_dim >> 2;
    const bool dual_stream = tokens >= 64;
    at::cuda::CUDAStream side =
        dual_stream ? at::cuda::getStreamFromPool() : stream;
    if (dual_stream) {
      at::cuda::CUDAEvent inputs_ready;
      inputs_ready.record(stream);
      inputs_ready.block(side);
    }
    gemm_rms_bwd_dx_vec4_kernel<<<grid_1d(tokens * k4, kThreads), kThreads, 0, stream>>>(
        dp.data_ptr<float>(), dr, x.data_ptr<float>(), w.data_ptr<float>(), q, r,
        dx.data_ptr<float>(), tokens, n_dim, k4, has_rms);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    gemm_rms_bwd_dw_vec4_kernel<<<grid_1d(n_dim * k4, kThreads), kThreads, 0, side>>>(
        dp.data_ptr<float>(), x.data_ptr<float>(), dw.data_ptr<float>(),
        tokens, n_dim, k4);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (dual_stream) {
      for (const auto& t : {dp, x, dw}) {
        c10::cuda::CUDACachingAllocator::recordStream(t.storage().data_ptr(), side);
      }
      at::cuda::CUDAEvent dw_done;
      dw_done.record(side);
      dw_done.block(stream);
    }
  } else {
    gemm_rms_bwd_dx_kernel<<<grid_1d(tokens * k_dim, kThreads), kThreads, 0, stream>>>(
        dp.data_ptr<float>(), dr, x.data_ptr<float>(), w.data_ptr<float>(), q, r,
        dx.data_ptr<float>(), tokens, n_dim, k_dim, has_rms);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    gemm_rms_bwd_dw_kernel<<<grid_1d(n_dim * k_dim, kThreads), kThreads, 0, stream>>>(
        dp.data_ptr<float>(), x.data_ptr<float>(), dw.data_ptr<float>(),
        tokens, n_dim, k_dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

}  // namespace

std::vector<torch::Tensor> fp32_gemm_rms_backward(
    torch::Tensor dp, torch::Tensor dr, torch::Tensor x, torch::Tensor w,
    torch::Tensor q, torch::Tensor r) {
  check_input(dp, "dp", 2);
  check_input(dr, "dr", 1);
  check_input(x, "x_flat", 2);
  check_input(w, "weight", 2);
  check_input(q, "q", 1);
  check_input(r, "r", 1);
  const long tokens = x.size(0);
  const long k_dim = x.size(1);
  const long n_dim = w.size(0);
  TORCH_CHECK(w.size(1) == k_dim, "weight K ", w.size(1), " != x K ", k_dim);
  TORCH_CHECK(dp.size(0) == tokens && dp.size(1) == n_dim, "dp shape mismatch");
  TORCH_CHECK(dr.size(0) == tokens && q.size(0) == tokens && r.size(0) == tokens,
              "dr/q/r must be [T]");

  const at::cuda::CUDAGuard guard(x.device());
  auto dx = torch::empty({tokens, k_dim}, x.options());
  auto dw = tokens > 0 ? torch::empty({n_dim, k_dim}, x.options())
                       : torch::zeros({n_dim, k_dim}, x.options());
  if (tokens > 0) {
    launch_bwd(dp, dr.data_ptr<float>(), x, w, q.data_ptr<float>(),
               r.data_ptr<float>(), dx, dw, tokens, n_dim, k_dim,
               /*has_rms=*/true);
  }
  return {dx, dw};
}

torch::Tensor fixed_k_gemm_forward(torch::Tensor x, torch::Tensor w) {
  check_input(x, "x", 2);
  check_input(w, "w", 2);
  const long m_dim = x.size(0);
  const long k_dim = x.size(1);
  const long n_dim = w.size(0);
  TORCH_CHECK(w.size(1) == k_dim, "w K ", w.size(1), " != x K ", k_dim);

  const at::cuda::CUDAGuard guard(x.device());
  auto out = torch::empty({m_dim, n_dim}, x.options());
  if (m_dim > 0 && n_dim > 0) {
    auto stream = at::cuda::getCurrentCUDAStream();
    constexpr int kThreads = 256;
    fixed_k_gemm_kernel<<<grid_1d(m_dim * n_dim, kThreads), kThreads, 0, stream>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), out.data_ptr<float>(),
        m_dim, n_dim, k_dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return out;
}

std::vector<torch::Tensor> fixed_k_gemm_backward(
    torch::Tensor dy, torch::Tensor x, torch::Tensor w) {
  check_input(dy, "dy", 2);
  check_input(x, "x", 2);
  check_input(w, "w", 2);
  const long tokens = x.size(0);
  const long k_dim = x.size(1);
  const long n_dim = w.size(0);
  TORCH_CHECK(w.size(1) == k_dim, "w K ", w.size(1), " != x K ", k_dim);
  TORCH_CHECK(dy.size(0) == tokens && dy.size(1) == n_dim, "dy shape mismatch");

  const at::cuda::CUDAGuard guard(x.device());
  auto dx = torch::empty({tokens, k_dim}, x.options());
  auto dw = tokens > 0 ? torch::empty({n_dim, k_dim}, x.options())
                       : torch::zeros({n_dim, k_dim}, x.options());
  if (tokens > 0) {
    launch_bwd(dy, nullptr, x, w, nullptr, nullptr, dx, dw, tokens, n_dim,
               k_dim, /*has_rms=*/false);
  }
  return {dx, dw};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp32_gemm_rms_forward", &fp32_gemm_rms_forward,
        "Deterministic FP32 controller GEMM + RMS scale forward (P1-2)");
  m.def("fp32_gemm_rms_backward", &fp32_gemm_rms_backward,
        "Deterministic FP32 controller GEMM + RMS scale backward (P1-2)");
  m.def("fixed_k_gemm_forward", &fixed_k_gemm_forward,
        "Deterministic fixed-K GEMM forward (P1-2 / P1-D6 core)");
  m.def("fixed_k_gemm_backward", &fixed_k_gemm_backward,
        "Deterministic fixed-K GEMM backward (P1-2 / P1-D6 core)");
}
