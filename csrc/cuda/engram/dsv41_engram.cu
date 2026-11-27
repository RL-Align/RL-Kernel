// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// DSv4.1 Engram gate, numeric profile dsv41-engram-v1. Byte-equal (forward) to the golden
// rl_engine.kernels.ops.pytorch.engram.dsv41_engram (its docstring is the arithmetic
// contract); the backward (dX of the streams only: the tables, wkv, q and k are frozen) is
// row-local and deterministic, with no byte contract.
//
// One block of 128 threads per (token, stream). Thread i owns the runs of 8 columns
// [8 i + 1024 j, + 8) of the stream (j = chunk, at most 5 for H <= 5120), as the mHC kernel
// does; each H-long sum is one fmaf chain per thread in ascending (j, column), then the
// warp halving tree and (w0 + w1) + (w2 + w3). Every other FP32 step is a separately
// rounded __f*_rn, in the order of DeepSeek's reference (inference/model.py Engram):
// w = q k, rstd = rx rk, d = (dot rstd) H^-0.5, copysign(sqrt(max(|d|, clamp)), d);
// sigmoid uses libdevice expf.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>
#include <type_traits>

#if defined(__USE_FAST_MATH__)
#error "dsv41_engram.cu must not be built with --use_fast_math (dsv41-engram-v1 is bitwise)"
#endif

namespace {

constexpr int kHc = 4;
constexpr int kRun = 8;
constexpr int kThreads = 128;
constexpr int kSpan = kThreads * kRun;
constexpr int kMaxChunks = 5;  // H <= 5120

__device__ __forceinline__ float tree4(float a0, float a1, float a2, float a3) {
  return __fadd_rn(__fadd_rn(a0, a1), __fadd_rn(a2, a3));
}

__device__ __forceinline__ float warp_halve(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, off));
  return v;
}

// NV sums over the 128 threads: halving within each warp, then (w0 + w1) + (w2 + w3).
template <int NV>
__device__ void slice_reduce(float (&v)[NV], float (&out)[NV], float (*sh)[4]) {
  const int w = threadIdx.x / 32, lane = threadIdx.x % 32;
#pragma unroll
  for (int q = 0; q < NV; ++q) {
    const float r = warp_halve(v[q]);
    if (lane == 0) sh[q][w] = r;
  }
  __syncthreads();
#pragma unroll
  for (int q = 0; q < NV; ++q) out[q] = tree4(sh[q][0], sh[q][1], sh[q][2], sh[q][3]);
  __syncthreads();
}

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

__device__ __forceinline__ float rms_scale(float ss, int h, float eps) {
  return __fdiv_rn(1.f, __fsqrt_rn(__fadd_rn(__fdiv_rn(ss, static_cast<float>(h)), eps)));
}

// The gate scalar of one (token, stream) from its three sums.
struct Gate {
  float rx, rk, dot, d, g;
};

// ch = FP32(H^-0.5), computed on the host in double.
__device__ __forceinline__ Gate gate_of(float ssx, float ssk, float dot, int h, float eps,
                                        float clamp, float ch, bool live) {
  Gate s;
  s.rx = rms_scale(ssx, h, eps);
  s.rk = rms_scale(ssk, h, eps);
  s.dot = dot;
  s.d = __fmul_rn(__fmul_rn(dot, __fmul_rn(s.rx, s.rk)), ch);
  const float gi = copysignf(__fsqrt_rn(fmaxf(fabsf(s.d), clamp)), s.d);
  s.g = live ? __fdiv_rn(1.f, __fadd_rn(1.f, expf(-gi))) : 0.f;
  return s;
}

template <int CH>
__global__ void __launch_bounds__(kThreads)
engram_gate_kernel(const __nv_bfloat16* __restrict__ x, const float* __restrict__ kv,
                   int64_t kv_ld, const __nv_bfloat16* __restrict__ qw,
                   const __nv_bfloat16* __restrict__ kw, const bool* __restrict__ mask, int h,
                   float eps, float clamp, float ch, __nv_bfloat16* __restrict__ out,
                   float* __restrict__ gate_out) {
  __shared__ float sh[3][4];
  const int t = blockIdx.x / kHc, hs = blockIdx.x % kHc, tid = threadIdx.x;
  const __nv_bfloat16* xs = x + (static_cast<int64_t>(t) * kHc + hs) * h;
  const float* key = kv + static_cast<int64_t>(t) * kv_ld + static_cast<int64_t>(hs) * h;
  const float* val = kv + static_cast<int64_t>(t) * kv_ld + static_cast<int64_t>(kHc) * h;
  float xv[CH][kRun];
  float acc[3] = {0.f, 0.f, 0.f};
#pragma unroll
  for (int it = 0; it < CH; ++it) {
    const int c = (it * kThreads + tid) * kRun;
    if (c >= h) continue;
    float kk[kRun], qv[kRun], kwv[kRun];
    load8(xs + c, xv[it]);
    load8(key + c, kk);
    load8(qw + static_cast<int64_t>(hs) * h + c, qv);
    load8(kw + static_cast<int64_t>(hs) * h + c, kwv);
#pragma unroll
    for (int e = 0; e < kRun; ++e) {
      acc[0] = fmaf(xv[it][e], xv[it][e], acc[0]);
      acc[1] = fmaf(kk[e], kk[e], acc[1]);
      acc[2] = fmaf(__fmul_rn(xv[it][e], __fmul_rn(qv[e], kwv[e])), kk[e], acc[2]);
    }
  }
  float tot[3];
  slice_reduce<3>(acc, tot, sh);
  const bool live = mask == nullptr || mask[t];
  const Gate s = gate_of(tot[0], tot[1], tot[2], h, eps, clamp, ch, live);
  if (gate_out != nullptr && tid == 0) gate_out[static_cast<int64_t>(t) * kHc + hs] = s.g;
  __nv_bfloat16* os = out + (static_cast<int64_t>(t) * kHc + hs) * h;
#pragma unroll
  for (int it = 0; it < CH; ++it) {
    const int c = (it * kThreads + tid) * kRun;
    if (c >= h) continue;
    float vv[kRun], o[kRun];
    load8(val + c, vv);
#pragma unroll
    for (int e = 0; e < kRun; ++e) o[e] = __fadd_rn(xv[it][e], __fmul_rn(s.g, vv[e]));
    store8(os + c, o);
  }
}

// dX of the gated update (frozen kv / q / k):
//   dx = dout + G (rx q k key - dot rx^3 x / H),  G = (sum dout v) g (1 - g) dgi/dd ch rk,
//   dgi/dd = 1 / (2 sqrt|d|) where |d| > clamp, else 0.
template <int CH, typename TG, typename TO>
__global__ void __launch_bounds__(kThreads)
engram_gate_bwd_kernel(const TG* __restrict__ dout, const __nv_bfloat16* __restrict__ x,
                       const float* __restrict__ kv, int64_t kv_ld,
                       const __nv_bfloat16* __restrict__ qw, const __nv_bfloat16* __restrict__ kw,
                       const bool* __restrict__ mask, int h, float eps, float clamp, float ch,
                       TO* __restrict__ dx) {
  __shared__ float sh[4][4];
  const int t = blockIdx.x / kHc, hs = blockIdx.x % kHc, tid = threadIdx.x;
  const int64_t row = (static_cast<int64_t>(t) * kHc + hs) * h;
  const float* key = kv + static_cast<int64_t>(t) * kv_ld + static_cast<int64_t>(hs) * h;
  const float* val = kv + static_cast<int64_t>(t) * kv_ld + static_cast<int64_t>(kHc) * h;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};  // ssx, ssk, dot, sum(dout v)
#pragma unroll
  for (int it = 0; it < CH; ++it) {
    const int c = (it * kThreads + tid) * kRun;
    if (c >= h) continue;
    float xv[kRun], kk[kRun], qv[kRun], kwv[kRun], vv[kRun], gv[kRun];
    load8(x + row + c, xv);
    load8(key + c, kk);
    load8(qw + static_cast<int64_t>(hs) * h + c, qv);
    load8(kw + static_cast<int64_t>(hs) * h + c, kwv);
    load8(val + c, vv);
    load8(dout + row + c, gv);
#pragma unroll
    for (int e = 0; e < kRun; ++e) {
      acc[0] = fmaf(xv[e], xv[e], acc[0]);
      acc[1] = fmaf(kk[e], kk[e], acc[1]);
      acc[2] = fmaf(__fmul_rn(xv[e], __fmul_rn(qv[e], kwv[e])), kk[e], acc[2]);
      acc[3] = fmaf(gv[e], vv[e], acc[3]);
    }
  }
  float tot[4];
  slice_reduce<4>(acc, tot, sh);
  const bool live = mask == nullptr || mask[t];
  const Gate s = gate_of(tot[0], tot[1], tot[2], h, eps, clamp, ch, live);
  const float ad = fabsf(s.d);
  const float dgi = (live && ad > clamp) ? 0.5f / sqrtf(ad) : 0.f;
  const float big = tot[3] * (s.g * (1.f - s.g)) * dgi * ch * s.rk;
  const float c2 = s.dot * s.rx * s.rx * s.rx / static_cast<float>(h);
#pragma unroll
  for (int it = 0; it < CH; ++it) {
    const int c = (it * kThreads + tid) * kRun;
    if (c >= h) continue;
    float xv[kRun], kk[kRun], qv[kRun], kwv[kRun], gv[kRun], o[kRun];
    load8(x + row + c, xv);
    load8(key + c, kk);
    load8(qw + static_cast<int64_t>(hs) * h + c, qv);
    load8(kw + static_cast<int64_t>(hs) * h + c, kwv);
    load8(dout + row + c, gv);
#pragma unroll
    for (int e = 0; e < kRun; ++e)
      o[e] = gv[e] + big * (s.rx * qv[e] * kwv[e] * kk[e] - c2 * xv[e]);
    store8(dx + row + c, o);
  }
}

template <typename F>
void with_chunks(int64_t h, F&& f) {
  if (h <= 4 * kSpan)
    f(std::integral_constant<int, 4>{});
  else
    f(std::integral_constant<int, 5>{});
}

void check_inputs(const torch::Tensor& x, const torch::Tensor& kv, const torch::Tensor& qw,
                  const torch::Tensor& kw, const std::optional<torch::Tensor>& mask, int64_t& t,
                  int64_t& h) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kBFloat16 &&
                  x.dim() == 3 && x.size(1) == kHc,
              "engram: streams must be contiguous BF16 [T, 4, H]");
  t = x.size(0);
  h = x.size(2);
  TORCH_CHECK(h % kThreads == 0 && h <= kMaxChunks * kSpan,
              "engram: H must be a multiple of 128 and at most 5120, got ", h);
  TORCH_CHECK(kv.is_cuda() && kv.scalar_type() == torch::kFloat32 && kv.dim() == 2 &&
                  kv.size(0) == t && kv.size(1) == (kHc + 1) * h && kv.stride(1) == 1 &&
                  kv.stride(0) % 4 == 0 && reinterpret_cast<uintptr_t>(kv.data_ptr()) % 16 == 0,
              "engram: kv must be FP32 [T, 5 H] with 16-byte aligned rows");
  for (const torch::Tensor* w : {&qw, &kw})
    TORCH_CHECK(w->is_cuda() && w->is_contiguous() && w->scalar_type() == torch::kBFloat16 &&
                    w->dim() == 2 && w->size(0) == kHc && w->size(1) == h,
                "engram: q / k weights must be contiguous BF16 [4, H]");
  if (mask.has_value())
    TORCH_CHECK(mask->is_cuda() && mask->is_contiguous() && mask->scalar_type() == torch::kBool &&
                    mask->dim() == 1 && mask->size(0) == t,
                "engram: mask must be contiguous bool [T]");
}

// FP32(H^-0.5) as the reference's `self.dim**-0.5` (a double, rounded when it scales FP32).
float inv_sqrt(int64_t h) { return static_cast<float>(std::pow(static_cast<double>(h), -0.5)); }

const __nv_bfloat16* bf(const torch::Tensor& x) {
  return reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
}

}  // namespace

// streams BF16 [T, 4, H], kv FP32 [T, 5 H] (wkv of the looked-up rows: four keys, then the
// value), q / k BF16 [4, H], optional mask bool [T] (false: gate 0) -> (out BF16 [T, 4, H],
// gate FP32 [T, 4]).
std::vector<torch::Tensor> dsv41_engram_gate(torch::Tensor x, torch::Tensor kv, torch::Tensor qw,
                                             torch::Tensor kw, std::optional<torch::Tensor> mask,
                                             double eps, double clamp) {
  int64_t t, h;
  check_inputs(x, kv, qw, kw, mask, t, h);
  const c10::cuda::OptionalCUDAGuard guard(device_of(x));
  auto out = torch::empty_like(x);
  auto gate = torch::empty({t, kHc}, x.options().dtype(torch::kFloat32));
  if (t == 0) return {out, gate};
  with_chunks(h, [&](auto ch) {
    constexpr int CH = decltype(ch)::value;
    engram_gate_kernel<CH><<<static_cast<unsigned>(t * kHc), kThreads, 0,
                             at::cuda::getCurrentCUDAStream()>>>(
        bf(x), kv.data_ptr<float>(), kv.stride(0), bf(qw), bf(kw),
        mask.has_value() ? mask->data_ptr<bool>() : nullptr, static_cast<int>(h),
        static_cast<float>(eps), static_cast<float>(clamp), inv_sqrt(h),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), gate.data_ptr<float>());
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, gate};
}

// dX [T, 4, H] (BF16 when bf16_out, else FP32) of dsv41_engram_gate given dout BF16 / FP32.
torch::Tensor dsv41_engram_gate_backward(torch::Tensor dout, torch::Tensor x, torch::Tensor kv,
                                         torch::Tensor qw, torch::Tensor kw,
                                         std::optional<torch::Tensor> mask, double eps,
                                         double clamp, bool bf16_out) {
  int64_t t, h;
  check_inputs(x, kv, qw, kw, mask, t, h);
  TORCH_CHECK(dout.is_cuda() && dout.is_contiguous() && dout.sizes() == x.sizes() &&
                  (dout.scalar_type() == torch::kBFloat16 || dout.scalar_type() == torch::kFloat32),
              "engram: dout must be contiguous BF16 / FP32 like the streams");
  const c10::cuda::OptionalCUDAGuard guard(device_of(x));
  auto dx = torch::empty(x.sizes(), x.options().dtype(bf16_out ? torch::kBFloat16 : torch::kFloat32));
  if (t == 0) return dx;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  with_chunks(h, [&](auto ch) {
    constexpr int CH = decltype(ch)::value;
    auto go = [&](auto* g, auto* o) {
      using TG = std::remove_const_t<std::remove_pointer_t<decltype(g)>>;
      using TO = std::remove_pointer_t<decltype(o)>;
      engram_gate_bwd_kernel<CH, TG, TO><<<static_cast<unsigned>(t * kHc), kThreads, 0, stream>>>(
          g, bf(x), kv.data_ptr<float>(), kv.stride(0), bf(qw), bf(kw),
          mask.has_value() ? mask->data_ptr<bool>() : nullptr, static_cast<int>(h),
          static_cast<float>(eps), static_cast<float>(clamp), inv_sqrt(h), o);
    };
    auto pick_out = [&](auto* g) {
      if (bf16_out)
        go(g, reinterpret_cast<__nv_bfloat16*>(dx.data_ptr<at::BFloat16>()));
      else
        go(g, dx.data_ptr<float>());
    };
    if (dout.scalar_type() == torch::kBFloat16)
      pick_out(reinterpret_cast<const __nv_bfloat16*>(dout.data_ptr<at::BFloat16>()));
    else
      pick_out(static_cast<const float*>(dout.data_ptr<float>()));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return dx;
}
