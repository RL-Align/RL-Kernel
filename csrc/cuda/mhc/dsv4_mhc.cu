// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// DSv4 manifold hyper-connections (mHC), numeric profile dsv4-mhc-v1. Byte-equal to the
// golden rl_engine.kernels.ops.pytorch.mhc_dsv4 (its docstring is the arithmetic
// contract). Forward: mhc_pre, mhc_post, hc_head. Backward: dX only (frozen parameters).
//
// Every FP32 step outside the long reductions is spelled __f*_rn, so nothing is contracted
// into an FMA; the backward's long reductions use fmaf on purpose (the golden emulates it).
// expf is libdevice, like PyTorch's CUDA exp.
//
// Kernels
//   (GEMM):       DeepGEMM tf32_hc_prenorm_gemm, 16 fixed splits, called from Python.
//   pre_epilogue: one block per token. Reduces the 16 split partials, runs the controller
//                 (sigmoid, softmax, Sinkhorn) on one warp, then mixes the four streams.
//   post:         elementwise.
//   post_bwd:     one block per token: dx, d_residual and the H-long dot products for
//                 d_post / d_comb (128 slices).
//   pre_bwd:      one block per token: d_pre dot products, controller backward, the
//                 aggregate leg of d_residual; d_raw / d_ss to a workspace.
//   gemm_bwd:     d_residual += d_raw @ fn + 2 x d_ss, tiled over tokens so fn is reused;
//                 the residual / dy rows are staged by cp.async.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <utility>

#if defined(__USE_FAST_MATH__)
#error "dsv4_mhc.cu must not be built with --use_fast_math (dsv4-mhc-v1 is bitwise)"
#endif

namespace {

constexpr int kHc = 4;
constexpr int kSplits = 16;      // DeepGEMM tf32_hc_prenorm_gemm split-K (fixed)
constexpr int kSlices = 128;      // H-long dot products (backward)
constexpr int kMaxSteps = 2 * 20; // Sinkhorn normalizations (1 + 2 * (iters - 1)) <= 40

__device__ __forceinline__ float tree4(float a0, float a1, float a2, float a3) {
  return __fadd_rn(__fadd_rn(a0, a1), __fadd_rn(a2, a3));
}

__device__ __forceinline__ float sigmoid_rn(float x) {
  return __fdiv_rn(1.f, __fadd_rn(1.f, expf(-x)));
}

__device__ __forceinline__ float warp_halve(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, off));
  return v;
}

// ------------------------------------------------------------------ forward kernels

// Reduce the kSplits GEMM partials of one value (element stride ``stride``) by halving:
// p[0:8] + p[8:16], then 4, 2, 1.
__device__ __forceinline__ float split_sum(const float* p, int64_t stride) {
  float a[kSplits];
#pragma unroll
  for (int i = 0; i < kSplits; ++i) a[i] = __ldg(p + i * stride);
#pragma unroll
  for (int half = kSplits / 2; half > 0; half /= 2)
#pragma unroll
    for (int i = 0; i < half; ++i) a[i] = __fadd_rn(a[i], a[i + half]);
  return a[0];
}

__device__ __forceinline__ float shx(float v, int m) { return __shfl_xor_sync(0xffffffffu, v, m); }

// Per-lane controller state the backward needs.
struct LaneCtl {
  float raw;   // this lane's raw mix (n = lane_n(lane))
  float r;     // rsqrt of the mean square
  float sig;   // sigmoid of the pre / post logit (lanes 0-7)
  float soft;  // softmax of the comb logits (lanes 16-31)
};

// Lane -> controller index n: lanes 0-3 pre, 4-7 post, 16-31 comb (n = lane - 8).
template <int N>
__device__ __forceinline__ int lane_n(int lane) {
  if (N == kHc) return lane < kHc ? lane : 0;
  return lane < 2 * kHc ? lane : (lane >= 16 ? lane - 8 : 0);
}

// Controller, part 1 (one warp): rsqrt of the mean square, logits, pre (lanes 0-3) and post
// (lanes 4-7). ``v`` holds raw[lane] on lanes < N and ss on lane N. Returns this lane's logit.
template <int N>
__device__ float controller_head(float v, const float* scale, const float* base, int k,
                                 float rms_eps, float eps, float post_mult, int lane, float& pre,
                                 float& post, LaneCtl* save) {
  const float ss = __shfl_sync(0xffffffffu, v, N);
  const float mean = __fdiv_rn(ss, static_cast<float>(k));
  const float r = __fdiv_rn(1.f, __fsqrt_rn(__fadd_rn(mean, rms_eps)));
  const int n = lane_n<N>(lane);
  const float raw_n = __shfl_sync(0xffffffffu, v, n);
  const int seg = (N == kHc) ? 0 : (n < kHc ? 0 : (n < 2 * kHc ? 1 : 2));
  const float logit = __fadd_rn(__fmul_rn(__fmul_rn(raw_n, r), scale[seg]), base[n]);
  const float sig = sigmoid_rn(logit);
  pre = __fadd_rn(sig, eps);
  post = __fmul_rn(post_mult, sig);
  if (save) {
    save->raw = raw_n;
    save->r = r;
    save->sig = sig;
  }
  return logit;
}

// Controller, part 2 (one warp, N = 24): softmax of the comb logits, then the Sinkhorn
// normalizations; lanes 16-31 return comb[(lane-16)/4][lane%4]. Row / column sums of the 4x4
// matrix use xor-shuffles (1, 2) / (4, 8): each lane gets (a0 + a1) + (a2 + a3) up to the
// order of a commutative add, which is exact. With ``steps_m`` / ``steps_d`` (shared,
// kMaxSteps x 16) every normalization's input and this lane's denominator are saved.
__device__ float controller_sinkhorn(float logit, float eps, int iters, int lane, LaneCtl* save,
                                     float* steps_m, float* steps_d) {
  const float mx0 = fmaxf(logit, shx(logit, 1));
  const float mx = fmaxf(mx0, shx(mx0, 2));
  const float e = expf(__fsub_rn(logit, mx));
  const float e1 = __fadd_rn(e, shx(e, 1));
  const float den = __fadd_rn(e1, shx(e1, 2));
  const float soft = __fdiv_rn(e, den);
  if (save) save->soft = soft;
  float m = __fadd_rn(soft, eps);
  const int l = lane & 15;
  int step = 0;
  auto keep = [&](float d) {
    if (steps_m != nullptr && lane >= 16) {
      steps_m[step * 16 + l] = m;
      steps_d[step * 16 + l] = d;
    }
    ++step;
  };
  auto col = [&]() {
    const float c1 = __fadd_rn(m, shx(m, 4));
    const float d = __fadd_rn(__fadd_rn(c1, shx(c1, 8)), eps);
    keep(d);
    m = __fdiv_rn(m, d);
  };
  auto row = [&]() {
    const float r1 = __fadd_rn(m, shx(m, 1));
    const float d = __fadd_rn(__fadd_rn(r1, shx(r1, 2)), eps);
    keep(d);
    m = __fdiv_rn(m, d);
  };
  col();
  for (int it = 0; it < iters - 1; ++it) {
    row();
    col();
  }
  return m;
}

// Backward of the controller on one warp (golden _controller_backward), same lane layout.
// g holds this lane's incoming gradient: d_pre (lanes 0-3), d_post (4-7), d_comb (16-31).
// Returns d_raw for this lane's n; *dss is valid on every lane.
template <int N>
__device__ float controller_bwd_warp(const LaneCtl& c, float g, const float* scale, int k,
                                     int iters, float post_mult, int lane,
                                     const float* steps_m, const float* steps_d, float* dss) {
  const int n = lane_n<N>(lane);
  float dlog;
  const float dsig = __fmul_rn(c.sig, __fsub_rn(1.f, c.sig));
  if (lane < kHc) dlog = __fmul_rn(g, dsig);
  else dlog = __fmul_rn(__fmul_rn(g, post_mult), dsig);
  if (N != kHc) {
    const int l = lane & 15;
    const int nsteps = 1 + 2 * (iters - 1);
    for (int s = nsteps - 1; s >= 0; --s) {  // steps: col, row, col, ..., col
      const float mi = steps_m[s * 16 + l];
      const float dd = steps_d[s * 16 + l];
      const float p = __fmul_rn(g, mi);
      float sum;
      if ((s % 2) == 0) {
        const float c1 = __fadd_rn(p, shx(p, 4));
        sum = __fadd_rn(c1, shx(c1, 8));
      } else {
        const float r1 = __fadd_rn(p, shx(p, 1));
        sum = __fadd_rn(r1, shx(r1, 2));
      }
      g = __fsub_rn(__fdiv_rn(g, dd), __fdiv_rn(sum, __fmul_rn(dd, dd)));
    }
    const float p = __fmul_rn(g, c.soft);
    const float r1 = __fadd_rn(p, shx(p, 1));
    const float gs = __fadd_rn(r1, shx(r1, 2));
    if (lane >= 16) dlog = __fmul_rn(c.soft, __fsub_rn(g, gs));
  }
  const int seg = (N == kHc) ? 0 : (n < kHc ? 0 : (n < 2 * kHc ? 1 : 2));
  const float dm = __fmul_rn(dlog, scale[seg]);
  const float prod = __fmul_rn(dm, c.raw);
  // dr = prod_0 + prod_1 + ... + prod_{N-1}, left to right.
  float dr = __shfl_sync(0xffffffffu, prod, 0);
#pragma unroll
  for (int q = 1; q < N; ++q) {
    const int src = (N == kHc || q < 2 * kHc) ? q : q + 8;
    dr = __fadd_rn(dr, __shfl_sync(0xffffffffu, prod, src));
  }
  const float r3 = __fmul_rn(__fmul_rn(c.r, c.r), c.r);
  *dss = __fdiv_rn(__fmul_rn(__fmul_rn(dr, -0.5f), r3), static_cast<float>(k));
  return __fmul_rn(dm, c.r);
}

// Runs of 8: thread i of a 128-thread block owns columns [8 i + 1024 j, + 8) (j = chunk).
constexpr int kRun = 8;
constexpr int kSpan = 128 * kRun;
constexpr int kMaxChunks = 4;  // H <= 4096

__device__ __forceinline__ void load8(const __nv_bfloat16* p, float (&v)[kRun]) {
  const uint4 q = *reinterpret_cast<const uint4*>(p);
  const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&q);
#pragma unroll
  for (int e = 0; e < kRun; ++e) v[e] = __bfloat162float(b[e]);
}
__device__ __forceinline__ void load8(const float* p, float (&v)[kRun]) {
  const float4 a = *reinterpret_cast<const float4*>(p);
  const float4 b = *reinterpret_cast<const float4*>(p + 4);
  v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}
__device__ __forceinline__ void store8(__nv_bfloat16* p, const float (&v)[kRun]) {
  uint4 q;
  __nv_bfloat16* b = reinterpret_cast<__nv_bfloat16*>(&q);
#pragma unroll
  for (int e = 0; e < kRun; ++e) b[e] = __float2bfloat16_rn(v[e]);
  *reinterpret_cast<uint4*>(p) = q;
}
__device__ __forceinline__ void store8(float* p, const float (&v)[kRun]) {
  *reinterpret_cast<float4*>(p) = make_float4(v[0], v[1], v[2], v[3]);
  *reinterpret_cast<float4*>(p + 4) = make_float4(v[4], v[5], v[6], v[7]);
}

// Block layout of the per-token kernels: warps 0-3 (128 threads) work along H; warp 4 runs
// the controller. Named barriers: kBarPre (warp 4 -> warps 0-3: pre is ready), kBarMix
// (warps 0-3 only), kBarDpre (warps 0-3 -> warp 4: d_pre is ready).
constexpr int kMixThreads = 128;
constexpr int kCtlThreads = 160;
constexpr int kBarPre = 1, kBarMix = 2, kBarDpre = 3;

__device__ __forceinline__ void bar_sync(int id, int n) {
  asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ void bar_arrive(int id, int n) {
  asm volatile("bar.arrive %0, %1;\n" ::"r"(id), "r"(n) : "memory");
}
template <int BAR>
__device__ __forceinline__ void mix_sync() {
  if (BAR == 0) __syncthreads();
  else bar_sync(BAR, kMixThreads);
}

// Reduction of NV values per thread over the 128 slices (threads 0-127, 4 warps): halving
// within each warp, then (g0 + g1) + (g2 + g3). Result in out[] on every thread. BAR = 0 uses
// __syncthreads (128-thread blocks), else that named barrier over threads 0-127.
template <int NV, int BAR = 0>
__device__ void slice_reduce(float (&v)[NV], float (&out)[NV], float (*sh)[4]) {
  const int tid = threadIdx.x, w = tid / 32, lane = tid % 32;
#pragma unroll
  for (int q = 0; q < NV; ++q) {
    const float r = warp_halve(v[q]);
    if (lane == 0) sh[q][w] = r;
  }
  mix_sync<BAR>();
#pragma unroll
  for (int q = 0; q < NV; ++q) out[q] = tree4(sh[q][0], sh[q][1], sh[q][2], sh[q][3]);
  mix_sync<BAR>();
}

__device__ __forceinline__ uint4 ld128(const __nv_bfloat16* p) {
  return __ldg(reinterpret_cast<const uint4*>(p));
}
__device__ __forceinline__ float bf_at(const uint4& q, int e) {
  return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&q)[e]);
}

// y = (p0 x0 + p1 x1) + (p2 x2 + p3 x3) from the four streams' 8 packed BF16 values,
// rounded to BF16 and returned as FP32.
__device__ __forceinline__ void mix8q(const uint4 (&q)[kHc], const float* p, float (&y)[kRun]) {
#pragma unroll
  for (int e = 0; e < kRun; ++e)
    y[e] = __bfloat162float(__float2bfloat16_rn(
        tree4(__fmul_rn(p[0], bf_at(q[0], e)), __fmul_rn(p[1], bf_at(q[1], e)),
              __fmul_rn(p[2], bf_at(q[2], e)), __fmul_rn(p[3], bf_at(q[3], e)))));
}

__device__ __forceinline__ void mix8(const __nv_bfloat16* r, int h, int c, const float* p,
                                     float (&y)[kRun]) {
  const uint4 q[kHc] = {ld128(r + c), ld128(r + h + c), ld128(r + 2 * h + c), ld128(r + 3 * h + c)};
  mix8q(q, p, y);
}

// Programmatic dependent launch: wait for the preceding kernel (the DeepGEMM partials).
__device__ __forceinline__ void grid_dep_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

// RMSNorm scale of the BF16 row held in yv: 1 / sqrt(sum(y^2) / h + eps), threads 0-127.
template <int BAR>
__device__ __forceinline__ float norm_scale(const float (&yv)[kMaxChunks][kRun], int h, float eps,
                                            float (*sh)[4]) {
  float acc[1] = {0.f};
#pragma unroll
  for (int it = 0; it < kMaxChunks; ++it) {
    if ((it * 128 + static_cast<int>(threadIdx.x)) * kRun >= h) continue;
#pragma unroll
    for (int e = 0; e < kRun; ++e) acc[0] = fmaf(yv[it][e], yv[it][e], acc[0]);
  }
  float ss[1];
  slice_reduce<1, BAR>(acc, ss, sh);
  return __fdiv_rn(1.f, __fsqrt_rn(__fadd_rn(__fdiv_rn(ss[0], static_cast<float>(h)), eps)));
}

// One block per token (kCtlThreads). Warp 4: GEMM split sums and the controller; it hands pre
// to warps 0-3 before the Sinkhorn iterations, so the four-stream mix and the fused RMSNorm
// (with a norm weight) run alongside them. HEAD: N = 4, no post / comb.
template <int N>
__global__ void __launch_bounds__(kCtlThreads)
pre_epilogue_kernel(const float* __restrict__ pmix, const float* __restrict__ pss, int t_total,
                    int np, const __nv_bfloat16* __restrict__ res,
                    const float* __restrict__ scale, const float* __restrict__ base, int h,
                    float rms_eps, float eps, int iters, float post_mult,
                    const __nv_bfloat16* __restrict__ norm_w, float norm_eps,
                    __nv_bfloat16* __restrict__ out, float* __restrict__ raw_out,
                    float* __restrict__ ss_out, float* __restrict__ post_out,
                    float* __restrict__ comb_out) {
  __shared__ float s_pre[kHc];
  __shared__ float sh[1][4];
  const int t = blockIdx.x, tid = threadIdx.x;
  grid_dep_wait();
  if (tid >= kMixThreads) {
    const int lane = tid - kMixThreads;
    float v = 0.f;
    if (lane < N) v = split_sum(pmix + static_cast<int64_t>(t) * np + lane,
                                static_cast<int64_t>(t_total) * np);
    else if (lane == N) v = split_sum(pss + t, t_total);
    float pre, post;
    const float logit = controller_head<N>(v, scale, base, kHc * h, rms_eps, eps, post_mult, lane,
                                           pre, post, nullptr);
    if (lane < kHc) s_pre[lane] = pre;
    bar_arrive(kBarPre, kCtlThreads);
    if (lane < N) raw_out[static_cast<int64_t>(t) * N + lane] = v;
    if (lane == N) ss_out[t] = v;
    if constexpr (N != kHc) {
      if (lane >= kHc && lane < 2 * kHc) post_out[static_cast<int64_t>(t) * kHc + lane - kHc] = post;
      const float comb = controller_sinkhorn(logit, eps, iters, lane, nullptr, nullptr, nullptr);
      if (lane >= 16) comb_out[static_cast<int64_t>(t) * kHc * kHc + lane - 16] = comb;
    }
    return;
  }
  bar_sync(kBarPre, kCtlThreads);
  const float p[kHc] = {s_pre[0], s_pre[1], s_pre[2], s_pre[3]};
  const __nv_bfloat16* r = res + static_cast<int64_t>(t) * kHc * h;
  __nv_bfloat16* ot = out + static_cast<int64_t>(t) * h;
  float yv[kMaxChunks][kRun];
#pragma unroll
  for (int it = 0; it < kMaxChunks; ++it) {
    const int c = (it * 128 + tid) * kRun;
    if (c >= h) continue;
    mix8(r, h, c, p, yv[it]);
    if (norm_w == nullptr) store8(ot + c, yv[it]);
  }
  if (norm_w == nullptr) return;
  const float rn = norm_scale<kBarMix>(yv, h, norm_eps, sh);
#pragma unroll
  for (int it = 0; it < kMaxChunks; ++it) {
    const int c = (it * 128 + tid) * kRun;
    if (c >= h) continue;
    float w[kRun], o[kRun];
    load8(norm_w + c, w);
#pragma unroll
    for (int e = 0; e < kRun; ++e) o[e] = __fmul_rn(__fmul_rn(yv[it][e], rn), w[e]);
    store8(ot + c, o);
  }
}

// Block (128 threads) covers 1024 columns of one token, 8 per thread.
__global__ void __launch_bounds__(128)
post_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ res,
            const float* __restrict__ post, const float* __restrict__ comb, int h,
            __nv_bfloat16* __restrict__ out) {
  const int t = blockIdx.x;
  const int c = (blockIdx.y * blockDim.x + threadIdx.x) * 8;
  if (c >= h) return;
  float pt[kHc], cm[kHc * kHc];
#pragma unroll
  for (int i = 0; i < kHc; ++i) pt[i] = __ldg(post + static_cast<int64_t>(t) * kHc + i);
#pragma unroll
  for (int q = 0; q < kHc * kHc; ++q) cm[q] = __ldg(comb + static_cast<int64_t>(t) * kHc * kHc + q);
  const __nv_bfloat16* r = res + static_cast<int64_t>(t) * kHc * h;
  const uint4 xq = *reinterpret_cast<const uint4*>(x + static_cast<int64_t>(t) * h + c);
  const __nv_bfloat16* xb = reinterpret_cast<const __nv_bfloat16*>(&xq);
  uint4 rq[kHc];
#pragma unroll
  for (int i = 0; i < kHc; ++i) rq[i] = *reinterpret_cast<const uint4*>(r + i * h + c);
#pragma unroll
  for (int j = 0; j < kHc; ++j) {
    uint4 o;
    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&o);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      float rv[kHc];
#pragma unroll
      for (int i = 0; i < kHc; ++i) rv[i] = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&rq[i])[e]);
      const float mix = tree4(__fmul_rn(cm[j], rv[0]), __fmul_rn(cm[kHc + j], rv[1]),
                              __fmul_rn(cm[2 * kHc + j], rv[2]), __fmul_rn(cm[3 * kHc + j], rv[3]));
      ob[e] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(pt[j], __bfloat162float(xb[e])), mix));
    }
    *reinterpret_cast<uint4*>(out + (static_cast<int64_t>(t) * kHc + j) * h + c) = o;
  }
}

// ------------------------------------------------------------------ backward kernels
// One block (128 threads) per token; columns in runs of 8. TG: gradient type (BF16 / FP32),
// TO: dx / d_residual type (BF16 = the FP32 result rounded once).
template <typename TG, typename TO>
__global__ void __launch_bounds__(kSlices)
post_bwd_kernel(const TG* __restrict__ dnew, const __nv_bfloat16* __restrict__ x,
                const __nv_bfloat16* __restrict__ res, const float* __restrict__ post,
                const float* __restrict__ comb, int h, TO* __restrict__ dx,
                TO* __restrict__ dres, float* __restrict__ dpost, float* __restrict__ dcomb) {
  __shared__ float sh[kHc + kHc * kHc][4];
  const int t = blockIdx.x, s = threadIdx.x;
  float pt[kHc], cm[kHc * kHc];
#pragma unroll
  for (int i = 0; i < kHc; ++i) pt[i] = __ldg(post + static_cast<int64_t>(t) * kHc + i);
#pragma unroll
  for (int q = 0; q < kHc * kHc; ++q) cm[q] = __ldg(comb + static_cast<int64_t>(t) * kHc * kHc + q);
  const TG* d = dnew + static_cast<int64_t>(t) * kHc * h;
  const __nv_bfloat16* r = res + static_cast<int64_t>(t) * kHc * h;
  const __nv_bfloat16* xt = x + static_cast<int64_t>(t) * h;
  float acc[kHc + kHc * kHc];
#pragma unroll
  for (int q = 0; q < kHc + kHc * kHc; ++q) acc[q] = 0.f;
#pragma unroll
  for (int it = 0; it < kMaxChunks; ++it) {
    const int c = (it * 128 + s) * kRun;
    if (c >= h) continue;
    float dv[kHc][kRun], rv[kHc][kRun], xv[kRun], o[kRun];
#pragma unroll
    for (int j = 0; j < kHc; ++j) load8(d + j * h + c, dv[j]);
#pragma unroll
    for (int i = 0; i < kHc; ++i) load8(r + i * h + c, rv[i]);
    load8(xt + c, xv);
#pragma unroll
    for (int e = 0; e < kRun; ++e)
      o[e] = tree4(__fmul_rn(pt[0], dv[0][e]), __fmul_rn(pt[1], dv[1][e]),
                   __fmul_rn(pt[2], dv[2][e]), __fmul_rn(pt[3], dv[3][e]));
    store8(dx + static_cast<int64_t>(t) * h + c, o);
#pragma unroll
    for (int i = 0; i < kHc; ++i) {
#pragma unroll
      for (int e = 0; e < kRun; ++e)
        o[e] = tree4(__fmul_rn(cm[kHc * i], dv[0][e]), __fmul_rn(cm[kHc * i + 1], dv[1][e]),
                     __fmul_rn(cm[kHc * i + 2], dv[2][e]), __fmul_rn(cm[kHc * i + 3], dv[3][e]));
      store8(dres + (static_cast<int64_t>(t) * kHc + i) * h + c, o);
    }
#pragma unroll
    for (int e = 0; e < kRun; ++e) {
#pragma unroll
      for (int j = 0; j < kHc; ++j) acc[j] = fmaf(dv[j][e], xv[e], acc[j]);
#pragma unroll
      for (int i = 0; i < kHc; ++i)
#pragma unroll
        for (int j = 0; j < kHc; ++j)
          acc[kHc + kHc * i + j] = fmaf(rv[i][e], dv[j][e], acc[kHc + kHc * i + j]);
    }
  }
  float out[kHc + kHc * kHc];
  slice_reduce<kHc + kHc * kHc>(acc, out, sh);
#pragma unroll
  for (int q = 0; q < kHc + kHc * kHc; ++q) {  // static indices keep out[] in registers
    if (s != q) continue;
    if (q < kHc) dpost[static_cast<int64_t>(t) * kHc + q] = out[q];
    else dcomb[static_cast<int64_t>(t) * kHc * kHc + q - kHc] = out[q];
  }
}

// One block per token (kCtlThreads). dz is the gradient of the op's output (of the normed
// output when norm_w is given). Warp 4 recomputes the controller (pre first, then the
// Sinkhorn steps, saved) while warps 0-3 form the gradient of the stream mix y (through the
// fused RMSNorm) and d_pre; warp 4 then runs the controller backward. Writes pre, d_raw,
// d_ss and, with a norm, dy_out (FP32); gemm_bwd_kernel then forms d_residual.
template <int N, typename TG>
__global__ void __launch_bounds__(kCtlThreads)
pre_bwd_kernel(const TG* __restrict__ dz, const float* __restrict__ dpost,
               const float* __restrict__ dcomb, const __nv_bfloat16* __restrict__ res,
               const float* __restrict__ scale, const float* __restrict__ base,
               const float* __restrict__ raw, const float* __restrict__ ss, int h, float rms_eps,
               float eps, int iters, float post_mult, const __nv_bfloat16* __restrict__ norm_w,
               float norm_eps, float* __restrict__ dy_out, float* __restrict__ pre_out,
               float* __restrict__ draw_out, float* __restrict__ dss_out) {
  __shared__ float sh[kHc][4];
  __shared__ float s_pre[kHc];
  __shared__ float s_dpre[kHc];
  __shared__ float steps_m[kMaxSteps * 16];
  __shared__ float steps_d[kMaxSteps * 16];
  const int t = blockIdx.x, s = threadIdx.x;
  const int k = kHc * h;
  if (s >= kMixThreads) {
    const int lane = s - kMixThreads;
    float v = 0.f;
    if (lane < N) v = raw[static_cast<int64_t>(t) * N + lane];
    else if (lane == N) v = ss[t];
    LaneCtl lc;
    float pre, post;
    const float logit = controller_head<N>(v, scale, base, k, rms_eps, eps, post_mult, lane, pre,
                                           post, &lc);
    if (lane < kHc) {
      s_pre[lane] = pre;
      pre_out[static_cast<int64_t>(t) * kHc + lane] = pre;
    }
    bar_arrive(kBarPre, kCtlThreads);
    float gin = 0.f;
    if constexpr (N != kHc) {
      controller_sinkhorn(logit, eps, iters, lane, &lc, steps_m, steps_d);
      __syncwarp();
      if (lane >= kHc && lane < 2 * kHc) gin = dpost[static_cast<int64_t>(t) * kHc + lane - kHc];
      else if (lane >= 16) gin = dcomb[static_cast<int64_t>(t) * kHc * kHc + lane - 16];
    }
    bar_sync(kBarDpre, kCtlThreads);
    if (lane < kHc) gin = s_dpre[lane];
    float dss;
    const float dr = controller_bwd_warp<N>(lc, gin, scale, k, iters, post_mult, lane, steps_m,
                                            steps_d, &dss);
    const bool live = (N == kHc) ? lane < kHc : (lane < 2 * kHc || lane >= 16);
    if (live) draw_out[static_cast<int64_t>(t) * N + lane_n<N>(lane)] = dr;
    if (lane == 0) dss_out[t] = dss;
    return;
  }
  bar_sync(kBarPre, kCtlThreads);
  const float p[kHc] = {s_pre[0], s_pre[1], s_pre[2], s_pre[3]};
  const __nv_bfloat16* r = res + static_cast<int64_t>(t) * kHc * h;
  const TG* dzt = dz + static_cast<int64_t>(t) * h;
  float dy[kMaxChunks][kRun];
#pragma unroll
  for (int it = 0; it < kMaxChunks; ++it) {
    const int c = (it * 128 + s) * kRun;
    if (c >= h) continue;
    load8(dzt + c, dy[it]);
  }
  if (norm_w != nullptr) {  // through the fused RMSNorm: dy = u r - (sum(u y) r^3 / H) y
    float yv[kMaxChunks][kRun];
#pragma unroll
    for (int it = 0; it < kMaxChunks; ++it) {
      const int c = (it * 128 + s) * kRun;
      if (c >= h) continue;
      mix8(r, h, c, p, yv[it]);
    }
    const float rn = norm_scale<kBarMix>(yv, h, norm_eps, sh);
    float dot[1] = {0.f};
#pragma unroll
    for (int it = 0; it < kMaxChunks; ++it) {
      const int c = (it * 128 + s) * kRun;
      if (c >= h) continue;
      float w[kRun];
      load8(norm_w + c, w);
#pragma unroll
      for (int e = 0; e < kRun; ++e) {
        dy[it][e] = __fmul_rn(dy[it][e], w[e]);  // u
        dot[0] = fmaf(dy[it][e], yv[it][e], dot[0]);
      }
    }
    float tot[1];
    slice_reduce<1, kBarMix>(dot, tot, sh);
    const float cc = __fdiv_rn(__fmul_rn(__fmul_rn(tot[0], rn), __fmul_rn(rn, rn)), static_cast<float>(h));
#pragma unroll
    for (int it = 0; it < kMaxChunks; ++it) {
      if ((it * 128 + s) * kRun >= h) continue;
#pragma unroll
      for (int e = 0; e < kRun; ++e)
        dy[it][e] = __fsub_rn(__fmul_rn(dy[it][e], rn), __fmul_rn(cc, yv[it][e]));
      store8(dy_out + static_cast<int64_t>(t) * h + (it * 128 + s) * kRun, dy[it]);
    }
  }
  float acc[kHc] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int it = 0; it < kMaxChunks; ++it) {
    const int c = (it * 128 + s) * kRun;
    if (c >= h) continue;
    float rv[kHc][kRun];
#pragma unroll
    for (int i = 0; i < kHc; ++i) load8(r + i * h + c, rv[i]);
#pragma unroll
    for (int e = 0; e < kRun; ++e)
#pragma unroll
      for (int i = 0; i < kHc; ++i) acc[i] = fmaf(rv[i][e], dy[it][e], acc[i]);
  }
  float dpre[kHc];
  slice_reduce<kHc, kBarMix>(acc, dpre, sh);
  if (s == 0) {
#pragma unroll
    for (int i = 0; i < kHc; ++i) s_dpre[i] = dpre[i];
  }
  bar_arrive(kBarDpre, kCtlThreads);
}

// dres[t][i][c] = pre_i dy[t][c] + ((fma chain over n of draw[t][n] fn[n][k]) + (2 x) dss),
// k = i H + c (gemm_bwd_kernel below).
template <int V>
struct Vec;
template <>
struct Vec<4> {
  static __device__ __forceinline__ void load(const float* p, float (&v)[4]) {
    const float4 a = *reinterpret_cast<const float4*>(p);
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
  }
  static __device__ __forceinline__ void load(const __nv_bfloat16* p, float (&v)[4]) {
    const uint2 q = *reinterpret_cast<const uint2*>(p);
    const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&q);
#pragma unroll
    for (int e = 0; e < 4; ++e) v[e] = __bfloat162float(b[e]);
  }
  static __device__ __forceinline__ void store(float* p, const float (&v)[4]) {
    *reinterpret_cast<float4*>(p) = make_float4(v[0], v[1], v[2], v[3]);
  }
  static __device__ __forceinline__ void store(__nv_bfloat16* p, const float (&v)[4]) {
    uint2 q;
    __nv_bfloat16* b = reinterpret_cast<__nv_bfloat16*>(&q);
#pragma unroll
    for (int e = 0; e < 4; ++e) b[e] = __float2bfloat16_rn(v[e]);
    *reinterpret_cast<uint2*>(p) = q;
  }
};

template <int BYTES>
__device__ __forceinline__ void cp_async_n(void* dst, const void* src) {
  const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  if constexpr (BYTES == 16)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(src));
  else
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::"r"(d), "l"(src), "n"(BYTES));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// Block: 256 threads over k (4 consecutive k each), TOK tokens (16 below 1024 tokens so
// small batches still fill the GPU, 64 above so fn is reloaded less). The thread's 24 x 4
// fn values are loaded once and held in registers. The residual and dy of the next
// kGemmDepth - 1 tokens are staged in shared memory by cp.async: each thread fetches and
// reads only its own elements, so no block barrier is needed, and enough bytes stay in
// flight without more registers (register prefetch would halve the occupancy).
constexpr int kGemmDepth = 6;
template <int N, int TOK, typename TD, typename TO>
__global__ void __launch_bounds__(256)
gemm_bwd_kernel(const __nv_bfloat16* __restrict__ res, const float* __restrict__ fn,
                      const TD* __restrict__ dy, const float* __restrict__ pre,
                      const float* __restrict__ draw, const float* __restrict__ dss, int t_total,
                      int h, TO* __restrict__ dres) {
  constexpr int V = 4, DEPTH = kGemmDepth;
  static_assert(N % 4 == 0, "draw rows are read as float4");
  __shared__ __align__(16) float s_draw[TOK][N];
  __shared__ float s_dss[TOK];
  __shared__ float s_pre[TOK][kHc];
  __shared__ __align__(16) __nv_bfloat16 s_res[DEPTH][256 * V];
  __shared__ __align__(16) TD s_dy[DEPTH][256 * V];
  const int t0 = blockIdx.y * TOK;
  const int k = kHc * h;
  for (int q = threadIdx.x; q < TOK * N; q += blockDim.x) {
    const int i = q / N, nn = q % N;
    s_draw[i][nn] = (t0 + i < t_total) ? draw[static_cast<int64_t>(t0 + i) * N + nn] : 0.f;
  }
  for (int q = threadIdx.x; q < TOK * kHc; q += blockDim.x) {
    const int i = q / kHc, j = q % kHc;
    s_pre[i][j] = (t0 + i < t_total) ? pre[static_cast<int64_t>(t0 + i) * kHc + j] : 0.f;
  }
  for (int i = threadIdx.x; i < TOK; i += blockDim.x) s_dss[i] = (t0 + i < t_total) ? dss[t0 + i] : 0.f;
  __syncthreads();
  const int kk = (blockIdx.x * blockDim.x + threadIdx.x) * V;
  if (kk >= k) return;
  const int stream = kk / h, c = kk - stream * h;
  const int tn = min(TOK, t_total - t0);
  auto fetch = [&](int i) {
    if (i < tn) {
      const int slot = i % DEPTH;
      cp_async_n<8>(&s_res[slot][threadIdx.x * V], res + static_cast<int64_t>(t0 + i) * k + kk);
      cp_async_n<sizeof(TD) * V>(&s_dy[slot][threadIdx.x * V],
                                 dy + static_cast<int64_t>(t0 + i) * h + c);
    }
    cp_commit();  // possibly empty: keeps the group count uniform
  };
#pragma unroll
  for (int i = 0; i < DEPTH - 1; ++i) fetch(i);
  float f[N][V];
#pragma unroll
  for (int nn = 0; nn < N; ++nn) {
    float v[V];
    Vec<V>::load(fn + static_cast<int64_t>(nn) * k + kk, v);
#pragma unroll
    for (int e = 0; e < V; ++e) f[nn][e] = v[e];
  }
  for (int i = 0; i < tn; ++i) {
    fetch(i + DEPTH - 1);
    cp_wait<DEPTH - 1>();
    const int slot = i % DEPTH;
    float xv[V], dyv[V];
    Vec<V>::load(&s_res[slot][threadIdx.x * V], xv);
    Vec<V>::load(&s_dy[slot][threadIdx.x * V], dyv);
    float g[V] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int n4 = 0; n4 < N; n4 += 4) {
      const float4 d = *reinterpret_cast<const float4*>(&s_draw[i][n4]);
      const float dv[4] = {d.x, d.y, d.z, d.w};
#pragma unroll
      for (int q = 0; q < 4; ++q)
#pragma unroll
        for (int e = 0; e < V; ++e) g[e] = fmaf(dv[q], f[n4 + q][e], g[e]);
    }
    const float ds = s_dss[i], pi = s_pre[i][stream];
    float out[V];
#pragma unroll
    for (int e = 0; e < V; ++e) {
      const float ge = __fadd_rn(g[e], __fmul_rn(__fmul_rn(xv[e], 2.f), ds));
      out[e] = __fadd_rn(__fmul_rn(pi, dyv[e]), ge);
    }
    Vec<V>::store(dres + static_cast<int64_t>(t0 + i) * k + kk, out);
  }
  cp_wait<0>();
}


// ------------------------------------------------------------------ host helpers
void check_res(const torch::Tensor& res, int64_t& t, int64_t& h) {
  TORCH_CHECK(res.is_cuda() && res.is_contiguous() && res.scalar_type() == torch::kBFloat16 &&
                  res.dim() == 3 && res.size(1) == kHc,
              "residual must be contiguous BF16 [T, 4, H]");
  t = res.size(0);
  h = res.size(2);
  TORCH_CHECK(h % kSlices == 0 && h <= kMaxChunks * kSpan,
              "H must be a multiple of 128 and at most 4096, got ", h);
}

void check_f32(const torch::Tensor& x, std::initializer_list<int64_t> shape, const char* name) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kFloat32, name,
              " must be contiguous FP32 on CUDA");
  TORCH_CHECK(x.sizes() == torch::IntArrayRef(shape), name, " has shape ", x.sizes(),
              ", expected ", torch::IntArrayRef(shape));
}

// Gradient inputs: contiguous BF16 or FP32 on CUDA.
void check_grad(const torch::Tensor& x, std::initializer_list<int64_t> shape, const char* name) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() &&
                  (x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kBFloat16),
              name, " must be contiguous BF16 or FP32 on CUDA");
  TORCH_CHECK(x.sizes() == torch::IntArrayRef(shape), name, " has shape ", x.sizes(),
              ", expected ", torch::IntArrayRef(shape));
}

template <typename F>
void dispatch_grad(const torch::Tensor& g, F&& f) {
  if (g.scalar_type() == torch::kBFloat16)
    f(reinterpret_cast<const __nv_bfloat16*>(g.data_ptr<at::BFloat16>()));
  else
    f(static_cast<const float*>(g.data_ptr<float>()));
}

template <typename F>
void dispatch_out(bool bf16, torch::Tensor& o, F&& f) {
  if (bf16)
    f(reinterpret_cast<__nv_bfloat16*>(o.data_ptr<at::BFloat16>()));
  else
    f(o.data_ptr<float>());
}

// Launch with programmatic stream serialization so the kernel starts (and prefetches) while
// the preceding DeepGEMM kernel drains; the kernel waits with griddepcontrol.wait.
template <typename... K, typename... A>
void launch_pdl(void (*kernel)(K...), unsigned grid, unsigned block, cudaStream_t stream,
                A&&... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(block);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, std::forward<A>(args)...));
}

// Optional BF16 [H] RMSNorm weight -> device pointer or nullptr.
const __nv_bfloat16* norm_ptr(const std::optional<torch::Tensor>& w, int64_t h) {
  if (!w.has_value()) return nullptr;
  TORCH_CHECK(w->is_cuda() && w->is_contiguous() && w->scalar_type() == torch::kBFloat16 &&
                  w->dim() == 1 && w->size(0) == h,
              "norm_weight must be contiguous BF16 [H]");
  return reinterpret_cast<const __nv_bfloat16*>(w->data_ptr<at::BFloat16>());
}

}  // namespace

// residual BF16 [T, 4, H] and the DeepGEMM tf32_hc_prenorm_gemm partials pmix [16, T, 24],
// pss [16, T] -> (out BF16 [T, H], post [T, 4], comb [T, 4, 4], raw [T, 24], ss [T]);
// out is the stream mix y, or RMSNorm(y) * norm_weight when a weight is given.
std::vector<torch::Tensor> dsv4_mhc_pre(torch::Tensor res, torch::Tensor pmix, torch::Tensor pss,
                                        torch::Tensor scale, torch::Tensor base, double rms_eps,
                                        double eps, int64_t iters, double post_mult,
                                        std::optional<torch::Tensor> norm_weight,
                                        double norm_eps) {
  int64_t t, h;
  check_res(res, t, h);
  check_f32(pmix, {kSplits, t, 24}, "pmix");
  check_f32(pss, {kSplits, t}, "pss");
  check_f32(scale, {3}, "scale");
  check_f32(base, {24}, "base");
  TORCH_CHECK(iters >= 1 && 1 + 2 * (iters - 1) <= kMaxSteps, "sinkhorn iters out of range");
  const __nv_bfloat16* nw = norm_ptr(norm_weight, h);
  const c10::cuda::OptionalCUDAGuard guard(device_of(res));
  auto f32 = res.options().dtype(torch::kFloat32);
  auto y = torch::empty({t, h}, res.options());
  auto post = torch::empty({t, kHc}, f32);
  auto comb = torch::empty({t, kHc, kHc}, f32);
  auto raw = torch::empty({t, 24}, f32);
  auto ss = torch::empty({t}, f32);
  if (t == 0) return {y, post, comb, raw, ss};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_pdl(pre_epilogue_kernel<24>, static_cast<unsigned>(t), kCtlThreads, stream,
      pmix.data_ptr<float>(), pss.data_ptr<float>(), static_cast<int>(t), 24, reinterpret_cast<const __nv_bfloat16*>(res.data_ptr<at::BFloat16>()),
      scale.data_ptr<float>(), base.data_ptr<float>(), static_cast<int>(h),
      static_cast<float>(rms_eps), static_cast<float>(eps), static_cast<int>(iters),
      static_cast<float>(post_mult), nw, static_cast<float>(norm_eps),
      reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()), raw.data_ptr<float>(), ss.data_ptr<float>(), post.data_ptr<float>(), comb.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y, post, comb, raw, ss};
}

// residual BF16 [T, 4, H], partials of the 8-row padded head fn pmix [16, T, 8], pss [16, T],
// scale [1], base [4] -> (out BF16 [T, H], raw [T, 4], ss [T]) (out normed if a weight is given).
std::vector<torch::Tensor> dsv4_mhc_head(torch::Tensor res, torch::Tensor pmix, torch::Tensor pss,
                                         torch::Tensor scale, torch::Tensor base, double rms_eps,
                                         double eps, std::optional<torch::Tensor> norm_weight,
                                         double norm_eps) {
  int64_t t, h;
  check_res(res, t, h);
  check_f32(pmix, {kSplits, t, 8}, "pmix");
  check_f32(pss, {kSplits, t}, "pss");
  check_f32(scale, {1}, "scale");
  check_f32(base, {kHc}, "base");
  const __nv_bfloat16* nw = norm_ptr(norm_weight, h);
  const c10::cuda::OptionalCUDAGuard guard(device_of(res));
  auto f32 = res.options().dtype(torch::kFloat32);
  auto y = torch::empty({t, h}, res.options());
  auto raw = torch::empty({t, kHc}, f32);
  auto ss = torch::empty({t}, f32);
  if (t == 0) return {y, raw, ss};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_pdl(pre_epilogue_kernel<kHc>, static_cast<unsigned>(t), kCtlThreads, stream,
      pmix.data_ptr<float>(), pss.data_ptr<float>(), static_cast<int>(t), 8, reinterpret_cast<const __nv_bfloat16*>(res.data_ptr<at::BFloat16>()),
      scale.data_ptr<float>(), base.data_ptr<float>(), static_cast<int>(h),
      static_cast<float>(rms_eps), static_cast<float>(eps), 1, 1.f, nw,
      static_cast<float>(norm_eps), reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()), raw.data_ptr<float>(),
      ss.data_ptr<float>(), nullptr, nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y, raw, ss};
}

// x BF16 [T, H], residual BF16 [T, 4, H], post [T, 4], comb [T, 4, 4] -> BF16 [T, 4, H].
torch::Tensor dsv4_mhc_post(torch::Tensor x, torch::Tensor res, torch::Tensor post,
                            torch::Tensor comb) {
  int64_t t, h;
  check_res(res, t, h);
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kBFloat16 &&
                  x.dim() == 2 && x.size(0) == t && x.size(1) == h,
              "x must be contiguous BF16 [T, H]");
  check_f32(post, {t, kHc}, "post");
  check_f32(comb, {t, kHc, kHc}, "comb");
  const c10::cuda::OptionalCUDAGuard guard(device_of(res));
  auto out = torch::empty({t, kHc, h}, res.options());
  if (t == 0) return out;
  dim3 grid(static_cast<unsigned>(t), static_cast<unsigned>((h + 1023) / 1024));
  post_kernel<<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(res.data_ptr<at::BFloat16>()), post.data_ptr<float>(),
      comb.data_ptr<float>(), static_cast<int>(h),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// d_new BF16 / FP32 [T, 4, H] -> (dx [T, H], d_residual [T, 4, H], d_post [T, 4],
// d_comb [T, 4, 4]); dx and d_residual are BF16 when bf16_out, else FP32.
std::vector<torch::Tensor> dsv4_mhc_post_backward(torch::Tensor dnew, torch::Tensor x,
                                                  torch::Tensor res, torch::Tensor post,
                                                  torch::Tensor comb, bool bf16_out) {
  int64_t t, h;
  check_res(res, t, h);
  check_grad(dnew, {t, kHc, h}, "d_new");
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kBFloat16 &&
                  x.dim() == 2 && x.size(0) == t && x.size(1) == h,
              "x must be contiguous BF16 [T, H]");
  check_f32(post, {t, kHc}, "post");
  check_f32(comb, {t, kHc, kHc}, "comb");
  const c10::cuda::OptionalCUDAGuard guard(device_of(res));
  auto f32 = res.options().dtype(torch::kFloat32);
  auto oo = res.options().dtype(bf16_out ? torch::kBFloat16 : torch::kFloat32);
  auto dx = torch::empty({t, h}, oo);
  auto dres = torch::empty({t, kHc, h}, oo);
  auto dpost = torch::empty({t, kHc}, f32);
  auto dcomb = torch::empty({t, kHc, kHc}, f32);
  if (t == 0) return {dx, dres, dpost, dcomb};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dispatch_grad(dnew, [&](auto* g) {
    using TG = std::remove_const_t<std::remove_pointer_t<decltype(g)>>;
    dispatch_out(bf16_out, dx, [&](auto* o) {
      using TO = std::remove_pointer_t<decltype(o)>;
      post_bwd_kernel<TG, TO><<<static_cast<unsigned>(t), kSlices, 0, stream>>>(
          g, reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
          reinterpret_cast<const __nv_bfloat16*>(res.data_ptr<at::BFloat16>()),
          post.data_ptr<float>(), comb.data_ptr<float>(), static_cast<int>(h), o,
          reinterpret_cast<TO*>(dres.data_ptr()), dpost.data_ptr<float>(),
          dcomb.data_ptr<float>());
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dx, dres, dpost, dcomb};
}

namespace {

template <int N>
torch::Tensor pre_backward_impl(const torch::Tensor& dy, const float* dpost, const float* dcomb,
                                const torch::Tensor& res, const torch::Tensor& fn,
                                const torch::Tensor& scale, const torch::Tensor& base,
                                const torch::Tensor& raw, const torch::Tensor& ss, double rms_eps,
                                double eps, int64_t iters, double post_mult,
                                const __nv_bfloat16* nw, double norm_eps, bool bf16_out,
                                int64_t t, int64_t h) {
  const int64_t k = kHc * h;
  auto f32 = res.options().dtype(torch::kFloat32);
  auto dres = torch::empty({t, kHc, h}, res.options().dtype(bf16_out ? torch::kBFloat16 : torch::kFloat32));
  if (t == 0) return dres;
  auto draw = torch::empty({t, N}, f32);
  auto dss = torch::empty({t}, f32);
  auto pre = torch::empty({t, kHc}, f32);
  torch::Tensor dmix;  // gradient of the stream mix y when it differs from dy (fused norm)
  if (nw != nullptr) dmix = torch::empty({t, h}, f32);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dispatch_grad(dy, [&](auto* g) {
    using TG = std::remove_const_t<std::remove_pointer_t<decltype(g)>>;
    pre_bwd_kernel<N, TG><<<static_cast<unsigned>(t), kCtlThreads, 0, stream>>>(
        g, dpost, dcomb, reinterpret_cast<const __nv_bfloat16*>(res.data_ptr<at::BFloat16>()),
        scale.data_ptr<float>(), base.data_ptr<float>(), raw.data_ptr<float>(),
        ss.data_ptr<float>(), static_cast<int>(h), static_cast<float>(rms_eps),
        static_cast<float>(eps), static_cast<int>(iters), static_cast<float>(post_mult), nw,
        static_cast<float>(norm_eps), nw != nullptr ? dmix.data_ptr<float>() : nullptr,
        pre.data_ptr<float>(), draw.data_ptr<float>(), dss.data_ptr<float>());
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const torch::Tensor& dsrc = nw != nullptr ? dmix : dy;
  auto launch = [&](auto tok) {
    constexpr int TOK = decltype(tok)::value;
    dim3 grid(static_cast<unsigned>((k / 4 + 255) / 256),
              static_cast<unsigned>((t + TOK - 1) / TOK));
    dispatch_grad(dsrc, [&](auto* d) {
      using TD = std::remove_const_t<std::remove_pointer_t<decltype(d)>>;
      dispatch_out(bf16_out, dres, [&](auto* o) {
        using TO = std::remove_pointer_t<decltype(o)>;
        gemm_bwd_kernel<N, TOK, TD, TO><<<grid, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(res.data_ptr<at::BFloat16>()),
            fn.data_ptr<float>(), d, pre.data_ptr<float>(), draw.data_ptr<float>(),
            dss.data_ptr<float>(), static_cast<int>(t), static_cast<int>(h), o);
      });
    });
  };
  if (t < 1024)
    launch(std::integral_constant<int, 16>{});
  else
    launch(std::integral_constant<int, 64>{});
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return dres;
}

}  // namespace

// d_residual [T, 4, H] (BF16 when bf16_out, else FP32) of mhc_pre, given dy BF16 / FP32
// [T, H] (of the normed output when norm_weight is given) and d_post / d_comb.
torch::Tensor dsv4_mhc_pre_backward(torch::Tensor dy, torch::Tensor dpost, torch::Tensor dcomb,
                                    torch::Tensor res, torch::Tensor fn, torch::Tensor scale,
                                    torch::Tensor base, torch::Tensor raw, torch::Tensor ss,
                                    double rms_eps, double eps, int64_t iters, double post_mult,
                                    std::optional<torch::Tensor> norm_weight, double norm_eps,
                                    bool bf16_out) {
  int64_t t, h;
  check_res(res, t, h);
  check_grad(dy, {t, h}, "dy");
  check_f32(dpost, {t, kHc}, "d_post");
  check_f32(dcomb, {t, kHc, kHc}, "d_comb");
  check_f32(fn, {24, kHc * h}, "fn");
  check_f32(scale, {3}, "scale");
  check_f32(base, {24}, "base");
  check_f32(raw, {t, 24}, "raw");
  check_f32(ss, {t}, "ss");
  TORCH_CHECK(iters >= 1 && 1 + 2 * (iters - 1) <= kMaxSteps, "sinkhorn iters out of range");
  const __nv_bfloat16* nw = norm_ptr(norm_weight, h);
  const c10::cuda::OptionalCUDAGuard guard(device_of(res));
  return pre_backward_impl<24>(dy, dpost.data_ptr<float>(), dcomb.data_ptr<float>(), res, fn,
                               scale, base, raw, ss, rms_eps, eps, iters, post_mult, nw, norm_eps,
                               bf16_out, t, h);
}

// d_residual [T, 4, H] of hc_head, given dy BF16 / FP32 [T, H].
torch::Tensor dsv4_mhc_head_backward(torch::Tensor dy, torch::Tensor res, torch::Tensor fn,
                                     torch::Tensor scale, torch::Tensor base, torch::Tensor raw,
                                     torch::Tensor ss, double rms_eps, double eps,
                                     std::optional<torch::Tensor> norm_weight, double norm_eps,
                                     bool bf16_out) {
  int64_t t, h;
  check_res(res, t, h);
  check_grad(dy, {t, h}, "dy");
  check_f32(fn, {kHc, kHc * h}, "fn");
  check_f32(scale, {1}, "scale");
  check_f32(base, {kHc}, "base");
  check_f32(raw, {t, kHc}, "raw");
  check_f32(ss, {t}, "ss");
  const __nv_bfloat16* nw = norm_ptr(norm_weight, h);
  const c10::cuda::OptionalCUDAGuard guard(device_of(res));
  return pre_backward_impl<kHc>(dy, nullptr, nullptr, res, fn, scale, base, raw, ss, rms_eps, eps,
                                1, 1.0, nw, norm_eps, bf16_out, t, h);
}
