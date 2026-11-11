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

}  // namespace

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
