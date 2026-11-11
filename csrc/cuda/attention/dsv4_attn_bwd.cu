// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// DSv4 attention backward helpers (rl_engine/models/deepseek_v4/attention_backward.py).
//
// norm_rope_backward: one warp per row of D (512 or 1024) values, in registers:
//   d   = dy[src(row)]                                  (optional row gather)
//   d   = rotate(d, sign * theta(pos(row)))             on the last 64 dims, GPT-J pairs
//                                                       (sign -1: RoPE transpose,
//                                                        +1: inverse-RoPE transpose, 0: none)
//   if norm:  g = d * w;  r = rsqrt(mean(x^2) + eps);  xh = x * r
//             out = r * (g - xh * mean(g * xh))        (RMSNorm backward, weight optional)
//   else:     out = d
// written to row (row / heads) * out_heads + row % heads (so the o-side can write into
// the padded-head layout). FP32 math; row-local, so deterministic. The backward has no
// byte contract (the forward does).
//
// compressor_backward: one block per compressed entry (request b, entry e) of the main KV
// compressor (ratio 4 with overlap, coff 2; ratio 128, coff 1). With span = coff * ratio
// window slots, slot s at in-request position q = e * ratio + ratio - span + s (q >= 0),
// half h = (coff == 2 && s >= ratio), column c in [0, 512):
//   kv[s, c] = score[q, h * 512 + c];  sc[s, c] = score[q, width + h * 512 + c] + ape[s % ratio]
//   p = softmax_s(sc);  x = sum_s p kv;  entry = rope(rmsnorm(x) * w, theta(e * ratio))
// From d_entry: dn = rope_transpose(d_entry); dx = RMSNorm backward(x, dn, w);
//   d kv[s, c] = p dx;  d sc[s, c] = p (kv - x) dx      written in BF16 into d_score at
// the slot's token row (each (token, half) belongs to one entry; uncovered rows stay as
// the caller left them). Slots are split over G thread groups (online softmax, merged in
// shared memory); each group's 128 threads own 4 columns each. FP32 math, deterministic.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>

#include <cstdint>
#include <type_traits>

namespace {

constexpr int kNope = 448;
constexpr int kWarps = 8;

// 16-byte vector loads / stores of N contiguous values (pointers are 16-byte aligned).
template <int N>
__device__ __forceinline__ void load_vec(const float* p, float (&v)[N]) {
#pragma unroll
  for (int i = 0; i < N; i += 4) {
    const float4 a = *reinterpret_cast<const float4*>(p + i);
    v[i] = a.x; v[i + 1] = a.y; v[i + 2] = a.z; v[i + 3] = a.w;
  }
}
template <int N>
__device__ __forceinline__ void load_vec(const __nv_bfloat16* p, float (&v)[N]) {
#pragma unroll
  for (int i = 0; i < N; i += 8) {
    const uint4 a = *reinterpret_cast<const uint4*>(p + i);
    const uint32_t w[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      v[i + 2 * j] = __uint_as_float(w[j] << 16);
      v[i + 2 * j + 1] = __uint_as_float(w[j] & 0xFFFF0000u);
    }
  }
}
template <int N>
__device__ __forceinline__ void store_vec(float* p, const float (&v)[N]) {
#pragma unroll
  for (int i = 0; i < N; i += 4)
    *reinterpret_cast<float4*>(p + i) = make_float4(v[i], v[i + 1], v[i + 2], v[i + 3]);
}
template <int N>
__device__ __forceinline__ void store_vec(__nv_bfloat16* p, const float (&v)[N]) {
#pragma unroll
  for (int i = 0; i < N; i += 8) {
    uint4 a;
    uint32_t* w = reinterpret_cast<uint32_t*>(&a);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const __nv_bfloat162 b = __floats2bfloat162_rn(v[i + 2 * j], v[i + 2 * j + 1]);
      w[j] = *reinterpret_cast<const uint32_t*>(&b);
    }
    *reinterpret_cast<uint4*>(p + i) = a;
  }
}

// 4 contiguous values (8- or 16-byte aligned).
__device__ __forceinline__ void load4(const float* p, float (&v)[4]) {
  const float4 a = *reinterpret_cast<const float4*>(p);
  v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
}
__device__ __forceinline__ void load4(const __nv_bfloat16* p, float (&v)[4]) {
  const uint2 a = *reinterpret_cast<const uint2*>(p);
  v[0] = __uint_as_float(a.x << 16); v[1] = __uint_as_float(a.x & 0xFFFF0000u);
  v[2] = __uint_as_float(a.y << 16); v[3] = __uint_as_float(a.y & 0xFFFF0000u);
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// PER = D / 32 values per lane, contiguous: lane l owns [l * PER, (l + 1) * PER).
template <int PER, typename TD, typename TO>
__global__ void __launch_bounds__(kWarps * 32)
    norm_rope_bwd_kernel(const TD* __restrict__ dy, int64_t dy_ld,
                         const int64_t* __restrict__ src_rows, const __nv_bfloat16* __restrict__ x,
                         int64_t x_ld, const __nv_bfloat16* __restrict__ w,
                         const int64_t* __restrict__ positions, int heads,
                         const float* __restrict__ cos_sin, int64_t cs_ld, int rope_sign,
                         bool norm, float eps, int64_t rows, TO* __restrict__ out, int64_t out_ld,
                         int out_heads) {
  constexpr int D = PER * 32;
  const int lane = threadIdx.x & 31;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarps + (threadIdx.x >> 5);
  if (row >= rows) return;
  const int64_t src = src_rows != nullptr ? src_rows[row] : row;
  const int base = lane * PER;
  float d[PER];
  load_vec<PER>(dy + src * dy_ld + base, d);
  if (rope_sign != 0 && base >= kNope) {
    const int64_t pos = positions[row / heads];
    const float* cs = cos_sin + pos * cs_ld;
    const float sg = static_cast<float>(rope_sign);
#pragma unroll
    for (int i = 0; i < PER; i += 2) {
      const int pair = (base + i - kNope) >> 1;
      const float c = cs[pair], s = cs[32 + pair];
      const float d0 = d[i], d1 = d[i + 1];
      d[i] = d0 * c - sg * d1 * s;
      d[i + 1] = d1 * c + sg * d0 * s;
    }
  }
  if (norm) {
    float xv[PER];
    load_vec<PER>(x + row * x_ld + base, xv);
    float wv[PER];
    if (w != nullptr) load_vec<PER>(w + base, wv);
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < PER; ++i) ss += xv[i] * xv[i];
    const float r = rsqrtf(warp_sum(ss) / D + eps);
    float dot = 0.f;
#pragma unroll
    for (int i = 0; i < PER; ++i) {
      if (w != nullptr) d[i] *= wv[i];
      xv[i] *= r;
      dot += d[i] * xv[i];
    }
    const float m = warp_sum(dot) / D;
#pragma unroll
    for (int i = 0; i < PER; ++i) d[i] = r * (d[i] - xv[i] * m);
  }
  const int64_t orow = (row / heads) * out_heads + row % heads;
  store_vec<PER>(out + orow * out_ld + base, d);
}

template <int PER>
void launch(const torch::Tensor& dy, const int64_t* src, const torch::Tensor& x,
            const torch::Tensor& w, const int64_t* pos, int heads, const torch::Tensor& cs,
            int rope_sign, bool norm, double eps, int64_t rows, torch::Tensor& out,
            int out_heads) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  const unsigned blocks = static_cast<unsigned>((rows + kWarps - 1) / kWarps);
  const __nv_bfloat16* xp =
      norm ? reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()) : nullptr;
  const int64_t x_ld = norm ? x.stride(0) : 0;
  const __nv_bfloat16* wp =
      w.defined() ? reinterpret_cast<const __nv_bfloat16*>(w.data_ptr<at::BFloat16>()) : nullptr;
  const float* csp = rope_sign != 0 ? cs.data_ptr<float>() : nullptr;
  const int64_t cs_ld = rope_sign != 0 ? cs.stride(0) : 0;
  auto go = [&](auto* dyp, auto* op) {
    using TD = std::remove_const_t<std::remove_pointer_t<decltype(dyp)>>;
    using TO = std::remove_pointer_t<decltype(op)>;
    norm_rope_bwd_kernel<PER, TD, TO><<<blocks, kWarps * 32, 0, stream>>>(
        dyp, dy.stride(0), src, xp, x_ld, wp, pos, heads, csp, cs_ld, rope_sign, norm,
        static_cast<float>(eps), rows, op, out.stride(0), out_heads);
  };
  const bool dy_bf16 = dy.scalar_type() == torch::kBFloat16;
  const bool out_bf16 = out.scalar_type() == torch::kBFloat16;
  auto dyb = reinterpret_cast<const __nv_bfloat16*>(dy_bf16 ? dy.data_ptr() : nullptr);
  auto dyf = dy_bf16 ? nullptr : dy.data_ptr<float>();
  auto ob = reinterpret_cast<__nv_bfloat16*>(out_bf16 ? out.data_ptr() : nullptr);
  auto of = out_bf16 ? nullptr : out.data_ptr<float>();
  if (dy_bf16 && out_bf16) go(dyb, ob);
  else if (dy_bf16) go(dyb, of);
  else if (out_bf16) go(dyf, ob);
  else go(dyf, of);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

constexpr int kCD = 512;    // compressor head_dim
constexpr int kCTPG = 128;  // threads per slot group, 4 columns each

template <int G, typename TD>
__global__ void __launch_bounds__(G * kCTPG)
    compressor_bwd_kernel(const float* __restrict__ score, int64_t s_ld,
                          const float* __restrict__ ape, const TD* __restrict__ d_src,
                          int64_t d_ld, const int64_t* __restrict__ meta,
                          const float* __restrict__ norm_w, const float* __restrict__ cos_sin,
                          int64_t cs_ld, int ratio, int coff, float eps,
                          __nv_bfloat16* __restrict__ d_score, int64_t o_ld) {
  extern __shared__ float smem[];  // [3][G][512] partial (max, sum, acc), then [2][G][4]
  float* sm_m = smem;
  float* sm_l = smem + G * kCD;
  float* sm_a = smem + 2 * G * kCD;
  float* sm_r = smem + 3 * G * kCD;
  const int grp = threadIdx.x / kCTPG;
  const int t = threadIdx.x % kCTPG;
  const int warp = t >> 5, lane = threadIdx.x & 31;
  const int c0 = t * 4;
  const int64_t* m3 = meta + static_cast<int64_t>(blockIdx.x) * 3;
  const int64_t src = m3[0], start = m3[1], e = m3[2];
  const int span = coff * ratio, width = coff * kCD;
  const int64_t first = e * ratio + ratio - span;

  // Pass 1: per-group online softmax over its slots.
  float m[4], l[4], acc[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) m[j] = -INFINITY, l[j] = 0.f, acc[j] = 0.f;
  for (int s = grp; s < span; s += G) {
    const int64_t q = first + s;
    if (q < 0) continue;
    const int col = ((coff == 2 && s >= ratio) ? kCD : 0) + c0;
    const float* row = score + (start + q) * s_ld + col;
    const float4 kv = *reinterpret_cast<const float4*>(row);
    const float4 gt = *reinterpret_cast<const float4*>(row + width);
    const float4 ap = *reinterpret_cast<const float4*>(ape + (s % ratio) * width + col);
    const float kvv[4] = {kv.x, kv.y, kv.z, kv.w};
    const float sc[4] = {gt.x + ap.x, gt.y + ap.y, gt.z + ap.z, gt.w + ap.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float mn = fmaxf(m[j], sc[j]);
      const float a = expf(m[j] - mn), b = expf(sc[j] - mn);
      l[j] = l[j] * a + b;
      acc[j] = acc[j] * a + b * kvv[j];
      m[j] = mn;
    }
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    sm_m[grp * kCD + c0 + j] = m[j];
    sm_l[grp * kCD + c0 + j] = l[j];
    sm_a[grp * kCD + c0 + j] = acc[j];
  }
  __syncthreads();
  // Merge the groups (every group computes the same totals for its columns).
  float M[4], L[4], X[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    float mm = -INFINITY;
#pragma unroll
    for (int k = 0; k < G; ++k) mm = fmaxf(mm, sm_m[k * kCD + c0 + j]);
    float ll = 0.f, aa = 0.f;
#pragma unroll
    for (int k = 0; k < G; ++k) {
      const float mk = sm_m[k * kCD + c0 + j];
      if (mk == -INFINITY) continue;  // group had no valid slot
      const float f = expf(mk - mm);
      ll += sm_l[k * kCD + c0 + j] * f;
      aa += sm_a[k * kCD + c0 + j] * f;
    }
    M[j] = mm, L[j] = ll, X[j] = aa / ll;
  }

  // RoPE transpose of the entry's gradient, then the RMSNorm backward.
  float dn[4];
  load4(d_src + src * d_ld + c0, dn);
  if (c0 >= kNope) {
    const float* cs = cos_sin + (e * ratio) * cs_ld;
#pragma unroll
    for (int j = 0; j < 4; j += 2) {
      const int pair = (c0 + j - kNope) >> 1;
      const float c = cs[pair], sn = cs[32 + pair];
      const float d0 = dn[j], d1 = dn[j + 1];
      dn[j] = d0 * c + d1 * sn;
      dn[j + 1] = d1 * c - d0 * sn;
    }
  }
  float wv[4];
  load4(norm_w + c0, wv);
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) ss += X[j] * X[j];
  ss = warp_sum(ss);
  if (lane == 0) sm_r[grp * 4 + warp] = ss;
  __syncthreads();
  ss = sm_r[grp * 4] + sm_r[grp * 4 + 1] + sm_r[grp * 4 + 2] + sm_r[grp * 4 + 3];
  const float r = rsqrtf(ss / kCD + eps);
  float g[4], xh[4], dot = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    g[j] = dn[j] * wv[j];
    xh[j] = X[j] * r;
    dot += g[j] * xh[j];
  }
  dot = warp_sum(dot);
  if (lane == 0) sm_r[4 * G + grp * 4 + warp] = dot;
  __syncthreads();
  dot = sm_r[4 * G + grp * 4] + sm_r[4 * G + grp * 4 + 1] + sm_r[4 * G + grp * 4 + 2] +
        sm_r[4 * G + grp * 4 + 3];
  float dx[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) dx[j] = r * (g[j] - xh[j] * (dot / kCD));

  // Pass 2: per-slot gradients of kv and of the gate score.
  for (int s = grp; s < span; s += G) {
    const int64_t q = first + s;
    if (q < 0) continue;
    const int col = ((coff == 2 && s >= ratio) ? kCD : 0) + c0;
    const float* row = score + (start + q) * s_ld + col;
    const float4 kv = *reinterpret_cast<const float4*>(row);
    const float4 gt = *reinterpret_cast<const float4*>(row + width);
    const float4 ap = *reinterpret_cast<const float4*>(ape + (s % ratio) * width + col);
    const float kvv[4] = {kv.x, kv.y, kv.z, kv.w};
    const float sc[4] = {gt.x + ap.x, gt.y + ap.y, gt.z + ap.z, gt.w + ap.w};
    float dk[4], ds[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float p = expf(sc[j] - M[j]) / L[j];
      dk[j] = p * dx[j];
      ds[j] = p * (kvv[j] - X[j]) * dx[j];
    }
    __nv_bfloat16* out = d_score + (start + q) * o_ld + col;
    const __nv_bfloat162 k0 = __floats2bfloat162_rn(dk[0], dk[1]);
    const __nv_bfloat162 k1 = __floats2bfloat162_rn(dk[2], dk[3]);
    const __nv_bfloat162 s0 = __floats2bfloat162_rn(ds[0], ds[1]);
    const __nv_bfloat162 s1 = __floats2bfloat162_rn(ds[2], ds[3]);
    uint2 kw, sw;
    kw.x = *reinterpret_cast<const uint32_t*>(&k0);
    kw.y = *reinterpret_cast<const uint32_t*>(&k1);
    sw.x = *reinterpret_cast<const uint32_t*>(&s0);
    sw.y = *reinterpret_cast<const uint32_t*>(&s1);
    *reinterpret_cast<uint2*>(out) = kw;
    *reinterpret_cast<uint2*>(out + width) = sw;
  }
}

template <int G>
void launch_compressor(const torch::Tensor& score, const torch::Tensor& ape,
                       const torch::Tensor& d_src, const torch::Tensor& meta,
                       const torch::Tensor& norm_w, const torch::Tensor& cos_sin, int ratio,
                       int coff, double eps, torch::Tensor& d_score) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  const size_t smem = (3 * G * kCD + 8 * G) * sizeof(float);
  auto go = [&](auto* dp) {
    using TD = std::remove_const_t<std::remove_pointer_t<decltype(dp)>>;
    auto kernel = compressor_bwd_kernel<G, TD>;
    if (smem > 48 * 1024)
      C10_CUDA_CHECK(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    kernel<<<static_cast<unsigned>(meta.size(0)), G * kCTPG, smem, stream>>>(
        score.data_ptr<float>(), score.stride(0), ape.data_ptr<float>(), dp, d_src.stride(0),
        meta.data_ptr<int64_t>(), norm_w.data_ptr<float>(), cos_sin.data_ptr<float>(),
        cos_sin.stride(0), ratio, coff, static_cast<float>(eps),
        reinterpret_cast<__nv_bfloat16*>(d_score.data_ptr()), d_score.stride(0));
  };
  if (d_src.scalar_type() == torch::kBFloat16)
    go(reinterpret_cast<const __nv_bfloat16*>(d_src.data_ptr()));
  else
    go(static_cast<const float*>(d_src.data_ptr<float>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

// Main KV compressor backward (see the header comment): writes d_score BF16 [T, 2 * width]
// rows covered by an entry. score: FP32 [T, 2 * width] (wkv | wgate, before ape); ape: FP32
// [ratio, width]; d_src: BF16 / FP32 [*, 512], entry gradients; meta: int64 [E, 3] =
// (d_src row, token row of the request's position 0, entry index e); norm_w: FP32 [512];
// cos_sin: FP32 [max_len, 64]. Full-prefill requests (token row = start + position).
void dsv4_compressor_backward(torch::Tensor score, torch::Tensor ape, torch::Tensor d_src,
                              torch::Tensor meta, torch::Tensor norm_w, torch::Tensor cos_sin,
                              int64_t ratio, double eps, torch::Tensor d_score) {
  TORCH_CHECK(ratio == 4 || ratio == 128, "dsv4_compressor_backward: ratio must be 4 or 128");
  const int coff = ratio == 4 ? 2 : 1;
  const int64_t width = coff * kCD;
  TORCH_CHECK(score.is_cuda() && score.scalar_type() == torch::kFloat32 && score.dim() == 2 &&
                  score.size(1) == 2 * width && score.stride(1) == 1 &&
                  score.stride(0) % 4 == 0,
              "dsv4_compressor_backward: score must be FP32 [T, 2 * width], rows of 16 bytes");
  TORCH_CHECK(d_score.scalar_type() == torch::kBFloat16 && d_score.dim() == 2 &&
                  d_score.size(0) == score.size(0) && d_score.size(1) == 2 * width &&
                  d_score.stride(1) == 1 && d_score.stride(0) % 4 == 0,
              "dsv4_compressor_backward: d_score must be BF16 like score");
  TORCH_CHECK(ape.scalar_type() == torch::kFloat32 && ape.is_contiguous() &&
                  ape.size(0) == ratio && ape.size(1) == width,
              "dsv4_compressor_backward: ape must be FP32 [ratio, width]");
  TORCH_CHECK((d_src.scalar_type() == torch::kBFloat16 || d_src.scalar_type() == torch::kFloat32) &&
                  d_src.dim() == 2 && d_src.size(1) == kCD && d_src.stride(1) == 1 &&
                  (d_src.stride(0) * d_src.element_size()) % 16 == 0,
              "dsv4_compressor_backward: d_src must be BF16 / FP32 [*, 512]");
  TORCH_CHECK(meta.scalar_type() == torch::kInt64 && meta.is_contiguous() && meta.dim() == 2 &&
                  meta.size(1) == 3,
              "dsv4_compressor_backward: meta must be int64 [E, 3]");
  TORCH_CHECK(norm_w.scalar_type() == torch::kFloat32 && norm_w.is_contiguous() &&
                  norm_w.numel() == kCD,
              "dsv4_compressor_backward: norm_w must be FP32 [512]");
  TORCH_CHECK(cos_sin.scalar_type() == torch::kFloat32 && cos_sin.stride(1) == 1,
              "dsv4_compressor_backward: cos_sin must be FP32");
  for (const auto* t : {&ape, &d_src, &meta, &norm_w, &cos_sin, &d_score})
    TORCH_CHECK(t->device() == score.device(), "dsv4_compressor_backward: one device");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(score.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(d_src.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(d_score.data_ptr()) % 8 == 0,
              "dsv4_compressor_backward: misaligned buffers");
  const c10::cuda::CUDAGuard guard(score.device());
  if (meta.size(0) == 0) return;
  if (ratio == 4)
    launch_compressor<4>(score, ape, d_src, meta, norm_w, cos_sin, 4, 2, eps, d_score);
  else
    launch_compressor<8>(score, ape, d_src, meta, norm_w, cos_sin, 128, 1, eps, d_score);
}

// out[(row / heads) * out_heads + row % heads] = norm_backward(rope_transpose(dy[src(row)]))
// for row < rows. dy: BF16 / FP32 [*, D] (row-contiguous); src_rows: optional int64 [rows];
// x: BF16 [rows, D] (when norm); w: optional BF16 [D]; positions: int64 [rows / heads]
// (when rope_sign != 0); out: BF16 / FP32 2-D [*, D] with contiguous rows (any row
// stride), preallocated.
void dsv4_norm_rope_backward(torch::Tensor dy, c10::optional<torch::Tensor> src_rows,
                             c10::optional<torch::Tensor> x, c10::optional<torch::Tensor> w,
                             c10::optional<torch::Tensor> positions, int64_t heads,
                             c10::optional<torch::Tensor> cos_sin, int64_t rope_sign, bool norm,
                             double eps, int64_t rows, torch::Tensor out, int64_t out_heads) {
  const int64_t d = dy.size(-1);
  TORCH_CHECK(d == 512 || d == 1024, "dsv4_norm_rope_backward: D must be 512 or 1024");
  TORCH_CHECK(dy.is_cuda() && dy.dim() == 2 && dy.stride(-1) == 1 && out.is_cuda() &&
                  out.dim() == 2 && out.stride(-1) == 1 && out.size(-1) == d,
              "dsv4_norm_rope_backward: dy / out must be CUDA with contiguous rows of D");
  TORCH_CHECK(rope_sign == 0 || (d == 512 && positions.has_value() && cos_sin.has_value()),
              "dsv4_norm_rope_backward: RoPE needs D = 512, positions and cos_sin");
  TORCH_CHECK(!norm || x.has_value(), "dsv4_norm_rope_backward: norm needs x");
  TORCH_CHECK(heads >= 1 && out_heads >= heads, "dsv4_norm_rope_backward: bad heads");
  auto aligned = [](const torch::Tensor& t) {
    return reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0 &&
           (t.dim() < 2 || (t.stride(0) * t.element_size()) % 16 == 0);
  };
  TORCH_CHECK(aligned(dy) && aligned(out) && (!x.has_value() || aligned(*x)),
              "dsv4_norm_rope_backward: rows must be 16-byte aligned");
  const c10::cuda::CUDAGuard guard(dy.device());
  if (rows == 0) return;
  const int64_t* src = src_rows.has_value() ? src_rows->data_ptr<int64_t>() : nullptr;
  const int64_t* pos = positions.has_value() ? positions->data_ptr<int64_t>() : nullptr;
  torch::Tensor xt = x.has_value() ? *x : torch::Tensor();
  torch::Tensor wt = w.has_value() ? *w : torch::Tensor();
  torch::Tensor cst = cos_sin.has_value() ? *cos_sin : torch::Tensor();
  if (norm) TORCH_CHECK(xt.scalar_type() == torch::kBFloat16 && xt.stride(-1) == 1, "x: BF16");
  if (wt.defined()) TORCH_CHECK(wt.scalar_type() == torch::kBFloat16 && wt.numel() == d, "w");
  if (d == 512)
    launch<16>(dy, src, xt, wt, pos, static_cast<int>(heads), cst, static_cast<int>(rope_sign),
               norm, eps, rows, out, static_cast<int>(out_heads));
  else
    launch<32>(dy, src, xt, wt, pos, static_cast<int>(heads), cst, static_cast<int>(rope_sign),
               norm, eps, rows, out, static_cast<int>(out_heads));
}
