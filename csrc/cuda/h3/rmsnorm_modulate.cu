// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// MiniMax-H3 RMSNorm and fused AdaLN modulation (RFC #420 `h3_rmsnorm`).
//
//   n[r]   = cast(w * (rstd[r] * x[r]))            rstd = rsqrtf(sum(x^2) / N + eps)
//   out[r] = cast(cast(n[r] * cast(1 + scale[i])) + shift[i])     i = row_index[r % S]
//
// Forward contract h3-rmsnorm-v1: the statistics replay PyTorch's
// vectorized_layer_norm_kernel<T, float, rms_norm> (torch cf30153): one
// (32, 4) block per row, 4-element vectors, thread t sums vectors t, t+128, ...
// in order, a shuffle-down tree (16..1), a cross-warp tree, sum / N, rsqrtf,
// then w * (rstd * x) and one cast. nn.RMSNorm therefore matches bitwise, and
// each modulation step rounds to the tensor dtype exactly where the eager
// `n * (1.0 + scale) + shift` expression does. Rows are independent: the result
// does not depend on batch size, position or the other rows.
//
// Backward (FP32, one cast per output, no atomics):
//   d_n      = g * cast(1 + scale)  (g without modulation)
//   dx       = rstd * w * d_n - x * rstd^3 * sum(w * d_n * x) / N   (row-local)
//   dweight  = sum_r d_n * x * rstd  over fixed 256-row tiles, tiles folded in order
//   dshift   = sum_{r -> i} g,   dscale = sum_{r -> i} g * n         (sorted tiles)

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <utility>
#include <vector>

namespace {

constexpr int kVec = 4;           // PyTorch's layer-norm vec_size
constexpr int kWarps = 4;         // num_threads() / warp_size = 128 / 32
constexpr int kRowTile = 256;     // dweight row tile

__device__ __forceinline__ float to_f(float v) { return v; }
__device__ __forceinline__ float to_f(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float to_f(__half v) { return __half2float(v); }

template <typename T>
__device__ __forceinline__ T from_f(float v);
template <>
__device__ __forceinline__ float from_f<float>(float v) { return v; }
template <>
__device__ __forceinline__ __nv_bfloat16 from_f<__nv_bfloat16>(float v) { return __float2bfloat16(v); }
template <>
__device__ __forceinline__ __half from_f<__half>(float v) { return __float2half(v); }

// Round to T and back: the value an eager T-dtype op would produce.
template <typename T>
__device__ __forceinline__ float round_t(float v) { return to_f(from_f<T>(v)); }

// PyTorch's rms statistics: sequential per thread, shuffle-down, cross-warp tree.
// Returns sum(v^2) / N on every thread.
template <typename T>
__device__ __forceinline__ float row_mean_square(const T* __restrict__ row, int n, float* buf) {
  const int numx = blockDim.x * blockDim.y;
  const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
  float s = 0.0f;
  for (int i = thrx; i < n / kVec; i += numx) {
#pragma unroll
    for (int k = 0; k < kVec; ++k) {
      const float v = to_f(row[i * kVec + k]);
      s = s + v * v;
    }
  }
  for (int off = 16; off > 0; off >>= 1) s = s + __shfl_down_sync(0xffffffffu, s, off);
  for (int off = blockDim.y / 2; off > 0; off /= 2) {
    if (threadIdx.x == 0 && threadIdx.y >= off && threadIdx.y < 2 * off) buf[threadIdx.y - off] = s;
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y < off) s = s + buf[threadIdx.y];
    __syncthreads();
  }
  if (thrx == 0) buf[0] = s / float(n);
  __syncthreads();
  const float out = buf[0];
  __syncthreads();
  return out;
}

// Same tree for an arbitrary per-element FP32 term; returns the row sum.
__device__ __forceinline__ float row_sum_tree(float s, float* buf) {
  for (int off = 16; off > 0; off >>= 1) s = s + __shfl_down_sync(0xffffffffu, s, off);
  for (int off = blockDim.y / 2; off > 0; off /= 2) {
    if (threadIdx.x == 0 && threadIdx.y >= off && threadIdx.y < 2 * off) buf[threadIdx.y - off] = s;
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y < off) s = s + buf[threadIdx.y];
    __syncthreads();
  }
  const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
  if (thrx == 0) buf[0] = s;
  __syncthreads();
  const float out = buf[0];
  __syncthreads();
  return out;
}

struct Modulation {
  const void* shift;       // (R, N) rows with stride `row_stride` (elements), or nullptr
  const void* scale;
  int64_t row_stride;
  const int64_t* index;    // (S,) table row per sequence position
  int64_t seq;             // S: row r of x uses index[r % S]
};

template <typename T>
__global__ void __launch_bounds__(32 * kWarps)
    rmsnorm_modulate_fwd_kernel(const T* __restrict__ x, const T* __restrict__ w, T* __restrict__ y,
                                float* __restrict__ rstd_out, int n, float eps, Modulation mod) {
  __shared__ float buf[kWarps];
  const int64_t r = blockIdx.x;
  const T* xr = x + r * n;
  const float rstd = rsqrtf(row_mean_square(xr, n, buf) + eps);
  const T* shift = nullptr;
  const T* scale = nullptr;
  if (mod.index != nullptr) {
    const int64_t i = mod.index[r % mod.seq];
    shift = static_cast<const T*>(mod.shift) + i * mod.row_stride;
    scale = static_cast<const T*>(mod.scale) + i * mod.row_stride;
  }
  const int numx = blockDim.x * blockDim.y;
  const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
  for (int i = thrx; i < n / kVec; i += numx) {
#pragma unroll
    for (int k = 0; k < kVec; ++k) {
      const int j = i * kVec + k;
      float v = round_t<T>(__fmul_rn(to_f(w[j]), __fmul_rn(rstd, to_f(xr[j]))));
      if (shift != nullptr) {
        // Separately rounded ops (no FMA contraction), as the eager expression.
        const float t1 = round_t<T>(__fadd_rn(1.0f, to_f(scale[j])));
        v = round_t<T>(__fadd_rn(round_t<T>(__fmul_rn(v, t1)), to_f(shift[j])));
      }
      y[r * n + j] = from_f<T>(v);
    }
  }
  if (thrx == 0) rstd_out[r] = rstd;
}

template <typename T>
__device__ __forceinline__ float d_norm_out(const T* g_row, const T* scale, int j) {
  const float g = to_f(g_row[j]);
  return scale == nullptr ? g : g * round_t<T>(1.0f + to_f(scale[j]));
}

template <typename T>
__global__ void __launch_bounds__(32 * kWarps)
    rmsnorm_modulate_dx_kernel(const T* __restrict__ g, const T* __restrict__ x,
                               const T* __restrict__ w, const float* __restrict__ rstd,
                               T* __restrict__ dx, int n, Modulation mod) {
  __shared__ float buf[kWarps];
  const int64_t r = blockIdx.x;
  const T* xr = x + r * n;
  const T* gr = g + r * n;
  const T* scale = mod.index != nullptr
                       ? static_cast<const T*>(mod.scale) + mod.index[r % mod.seq] * mod.row_stride
                       : nullptr;
  const int numx = blockDim.x * blockDim.y;
  const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
  float s = 0.0f;
  for (int i = thrx; i < n / kVec; i += numx) {
#pragma unroll
    for (int k = 0; k < kVec; ++k) {
      const int j = i * kVec + k;
      s = fmaf(to_f(w[j]) * d_norm_out(gr, scale, j), to_f(xr[j]), s);
    }
  }
  const float dot = row_sum_tree(s, buf);
  const float rs = rstd[r];
  const float coef = rs * rs * rs * dot / float(n);
  for (int i = thrx; i < n / kVec; i += numx) {
#pragma unroll
    for (int k = 0; k < kVec; ++k) {
      const int j = i * kVec + k;
      dx[r * n + j] = from_f<T>(rs * to_f(w[j]) * d_norm_out(gr, scale, j) - to_f(xr[j]) * coef);
    }
  }
}

// partial[tile, j] = sum over rows of the tile (ascending) of d_n * x * rstd
template <typename T>
__global__ void rmsnorm_dweight_partial_kernel(const T* __restrict__ g, const T* __restrict__ x,
                                               const float* __restrict__ rstd,
                                               float* __restrict__ partial, int64_t rows, int n,
                                               Modulation mod) {
  const int j = blockIdx.y * blockDim.x + threadIdx.x;
  if (j >= n) return;
  const int64_t r0 = static_cast<int64_t>(blockIdx.x) * kRowTile;
  const int64_t r1 = min(r0 + kRowTile, rows);
  float acc = 0.0f;
  for (int64_t r = r0; r < r1; ++r) {
    const T* scale = mod.index != nullptr ? static_cast<const T*>(mod.scale) +
                                                mod.index[r % mod.seq] * mod.row_stride
                                          : nullptr;
    acc = fmaf(d_norm_out(g + r * n, scale, j), to_f(x[r * n + j]) * rstd[r], acc);
  }
  partial[static_cast<int64_t>(blockIdx.x) * n + j] = acc;
}

template <typename T>
__global__ void fold_tiles_kernel(const float* __restrict__ partial, T* __restrict__ out,
                                  int64_t tiles, int n) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n) return;
  float acc = 0.0f;
  for (int64_t t = 0; t < tiles; ++t) acc += partial[t * n + j];
  out[j] = from_f<T>(acc);
}

// Table gradient tiles: positions sorted by (index, position); columns j < n
// accumulate g (shift), columns j >= n accumulate g * n_out (scale).
template <typename T>
__global__ void modulation_grad_partial_kernel(
    const T* __restrict__ g, const T* __restrict__ x, const T* __restrict__ w,
    const float* __restrict__ rstd, const int64_t* __restrict__ sorted_pos,
    const int64_t* __restrict__ tile_begin, const int64_t* __restrict__ tile_end,
    float* __restrict__ partial, int n) {
  const int64_t tile = blockIdx.x;
  const int j2 = blockIdx.y * blockDim.x + threadIdx.x;
  if (j2 >= 2 * n) return;
  const bool is_scale = j2 >= n;
  const int j = is_scale ? j2 - n : j2;
  const float wj = to_f(w[j]);
  float acc = 0.0f;
  for (int64_t p = tile_begin[tile]; p < tile_end[tile]; ++p) {
    const int64_t r = sorted_pos[p];
    const float gv = to_f(g[r * n + j]);
    if (is_scale) {
      const float nv = round_t<T>(wj * (rstd[r] * to_f(x[r * n + j])));
      acc = fmaf(gv, nv, acc);
    } else {
      acc += gv;
    }
  }
  partial[tile * 2 * n + j2] = acc;
}

__global__ void fold_segments_kernel(const float* __restrict__ partial,
                                     const int64_t* __restrict__ seg_first_tile,
                                     float* __restrict__ out, int width) {
  const int64_t seg = blockIdx.x;
  const int j = blockIdx.y * blockDim.x + threadIdx.x;
  if (j >= width) return;
  float acc = 0.0f;
  for (int64_t t = seg_first_tile[seg]; t < seg_first_tile[seg + 1]; ++t) acc += partial[t * width + j];
  out[seg * width + j] = acc;
}

void check_rows(const torch::Tensor& t, const char* name, const torch::Tensor& like) {
  TORCH_CHECK(t.is_cuda() && t.device() == like.device(), name, " must be on ", like.device());
  TORCH_CHECK(t.dim() == 2 && t.is_contiguous(), name, " must be a contiguous 2-D tensor");
  TORCH_CHECK(t.scalar_type() == like.scalar_type(), name, " must have dtype ", like.scalar_type());
}

Modulation make_modulation(const c10::optional<torch::Tensor>& shift,
                           const c10::optional<torch::Tensor>& scale,
                           const c10::optional<torch::Tensor>& index, const torch::Tensor& x,
                           int64_t n) {
  Modulation mod{nullptr, nullptr, 0, nullptr, 1};
  if (!index.has_value()) {
    TORCH_CHECK(!shift.has_value() && !scale.has_value(), "shift/scale need a row index");
    return mod;
  }
  TORCH_CHECK(shift.has_value() && scale.has_value(), "modulation needs both shift and scale");
  const auto& sh = *shift;
  const auto& sc = *scale;
  const auto& ix = *index;
  for (const auto* t : {&sh, &sc}) {
    TORCH_CHECK(t->is_cuda() && t->device() == x.device() && t->scalar_type() == x.scalar_type(),
                "shift/scale must be CUDA tensors with x's dtype");
    TORCH_CHECK(t->dim() == 2 && t->size(1) == n && t->stride(1) == 1,
                "shift/scale must be (R, N) with unit column stride");
  }
  TORCH_CHECK(sh.size(0) == sc.size(0) && sh.stride(0) == sc.stride(0),
              "shift and scale must be views of the same table layout");
  TORCH_CHECK(ix.is_cuda() && ix.device() == x.device() && ix.scalar_type() == at::kLong &&
                  ix.dim() == 1 && ix.is_contiguous() && ix.numel() > 0,
              "row index must be a non-empty contiguous int64 tensor on ", x.device());
  TORCH_CHECK(x.size(0) % ix.size(0) == 0, "x rows (", x.size(0), ") must be a multiple of S (",
              ix.size(0), ")");
  mod.shift = sh.data_ptr();
  mod.scale = sc.data_ptr();
  mod.row_stride = sh.stride(0);
  mod.index = ix.data_ptr<int64_t>();
  mod.seq = ix.size(0);
  return mod;
}

void check_xw(const torch::Tensor& x, const torch::Tensor& w) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous(), "x must be a contiguous (M, N) CUDA tensor");
  TORCH_CHECK(x.size(0) > 0, "x must have at least one row");
  TORCH_CHECK(w.is_cuda() && w.device() == x.device() && w.dim() == 1 &&
                  w.size(0) == x.size(1) && w.is_contiguous() &&
                  w.scalar_type() == x.scalar_type(),
              "weight must be a contiguous (N,) CUDA tensor with x's dtype and device");
  TORCH_CHECK(x.size(1) > 0, "x must have at least one column");
  TORCH_CHECK(x.size(1) % kVec == 0, "N=", x.size(1), " must be a multiple of ", kVec,
              " (PyTorch's vectorized RMSNorm path)");
}

// float32 / float16 / bfloat16 only (float64 has no PyTorch-matching vector path here).
#define H3_DISPATCH(TYPE, NAME, ...)                                                       \
  [&] {                                                                                    \
    switch (TYPE) {                                                                        \
      case at::kFloat: { using T = float; __VA_ARGS__(); break; }                          \
      case at::kHalf: { using T = __half; __VA_ARGS__(); break; }                          \
      case at::kBFloat16: { using T = __nv_bfloat16; __VA_ARGS__(); break; }               \
      default: TORCH_CHECK(false, NAME, ": unsupported dtype ", TYPE);                     \
    }                                                                                      \
  }()

}  // namespace

// x (M, N); optional shift/scale (R, N) strided views and index (S,), M % S == 0.
std::vector<torch::Tensor> h3_rmsnorm_forward(torch::Tensor x, torch::Tensor weight, double eps,
                                              c10::optional<torch::Tensor> shift,
                                              c10::optional<torch::Tensor> scale,
                                              c10::optional<torch::Tensor> index) {
  check_xw(x, weight);
  TORCH_CHECK(x.scalar_type() != at::kDouble, "float64 is not supported");
  const int64_t n = x.size(1);
  const Modulation mod = make_modulation(shift, scale, index, x, n);
  const c10::cuda::CUDAGuard guard(x.device());
  auto y = torch::empty_like(x);
  auto rstd = torch::empty({x.size(0)}, x.options().dtype(at::kFloat));
  auto stream = at::cuda::getCurrentCUDAStream();
  H3_DISPATCH(x.scalar_type(), "h3_rmsnorm_forward", [&] {
    rmsnorm_modulate_fwd_kernel<T><<<static_cast<unsigned>(x.size(0)), dim3(32, kWarps), 0, stream>>>(
        reinterpret_cast<const T*>(x.data_ptr()), reinterpret_cast<const T*>(weight.data_ptr()),
        reinterpret_cast<T*>(y.data_ptr()), rstd.data_ptr<float>(), static_cast<int>(n),
        static_cast<float>(eps), mod);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y, rstd};
}

// Returns {dx, dweight} and, with modulation, {d_shift, d_scale} as FP32 (R, N).
std::vector<torch::Tensor> h3_rmsnorm_backward(
    torch::Tensor grad, torch::Tensor x, torch::Tensor weight, torch::Tensor rstd,
    c10::optional<torch::Tensor> shift, c10::optional<torch::Tensor> scale,
    c10::optional<torch::Tensor> index, c10::optional<torch::Tensor> sorted_pos,
    c10::optional<torch::Tensor> tile_begin, c10::optional<torch::Tensor> tile_end,
    c10::optional<torch::Tensor> seg_first_tile) {
  check_xw(x, weight);
  check_rows(grad, "grad", x);
  TORCH_CHECK(grad.sizes() == x.sizes(), "grad must match x");
  TORCH_CHECK(rstd.is_cuda() && rstd.device() == x.device() &&
                  rstd.scalar_type() == at::kFloat && rstd.dim() == 1 &&
                  rstd.is_contiguous() && rstd.size(0) == x.size(0),
              "rstd must be contiguous float32 (M,) statistics on x's device");
  const int64_t rows = x.size(0);
  const int64_t n = x.size(1);
  const Modulation mod = make_modulation(shift, scale, index, x, n);
  if (mod.index != nullptr) {
    TORCH_CHECK(sorted_pos.has_value() && tile_begin.has_value() && tile_end.has_value() &&
                    seg_first_tile.has_value(),
                "modulated backward needs the sorted segment tiles");
    const std::pair<const torch::Tensor*, const char*> tiles[] = {{&*sorted_pos, "sorted_pos"},
                                                                  {&*tile_begin, "tile_begin"},
                                                                  {&*tile_end, "tile_end"},
                                                                  {&*seg_first_tile, "seg_first_tile"}};
    for (const auto& [t, name] : tiles) {
      TORCH_CHECK(t->is_cuda() && t->device() == x.device() && t->scalar_type() == at::kLong &&
                      t->dim() == 1 && t->is_contiguous(),
                  name, ": tile metadata must be contiguous int64 tensors on ", x.device());
    }
    TORCH_CHECK(sorted_pos->size(0) == rows, "sorted_pos must have M entries");
    TORCH_CHECK(seg_first_tile->size(0) == shift->size(0) + 1,
                "seg_first_tile must have R + 1 entries");
    TORCH_CHECK(tile_begin->size(0) == tile_end->size(0),
                "tile_begin and tile_end must have the same number of entries");
  } else {
    TORCH_CHECK(!sorted_pos.has_value() && !tile_begin.has_value() && !tile_end.has_value() &&
                    !seg_first_tile.has_value(),
                "sorted segment tiles require modulation");
  }
  const c10::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto dx = torch::empty_like(x);
  const int64_t tiles = (rows + kRowTile - 1) / kRowTile;
  auto partial = torch::empty({tiles, n}, x.options().dtype(at::kFloat));
  auto dweight = torch::empty_like(weight);
  const int threads = 256;
  const unsigned col_blocks = static_cast<unsigned>((n + threads - 1) / threads);
  H3_DISPATCH(x.scalar_type(), "h3_rmsnorm_backward", [&] {
    const T* g = reinterpret_cast<const T*>(grad.data_ptr());
    const T* xp = reinterpret_cast<const T*>(x.data_ptr());
    const T* wp = reinterpret_cast<const T*>(weight.data_ptr());
    rmsnorm_modulate_dx_kernel<T><<<static_cast<unsigned>(rows), dim3(32, kWarps), 0, stream>>>(
        g, xp, wp, rstd.data_ptr<float>(), reinterpret_cast<T*>(dx.data_ptr()),
        static_cast<int>(n), mod);
    rmsnorm_dweight_partial_kernel<T><<<dim3(static_cast<unsigned>(tiles), col_blocks), threads, 0,
                                        stream>>>(g, xp, rstd.data_ptr<float>(),
                                                  partial.data_ptr<float>(), rows,
                                                  static_cast<int>(n), mod);
    fold_tiles_kernel<T><<<col_blocks, threads, 0, stream>>>(
        partial.data_ptr<float>(), reinterpret_cast<T*>(dweight.data_ptr()), tiles,
        static_cast<int>(n));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (mod.index == nullptr) return {dx, dweight};

  const int64_t segments = seg_first_tile->numel() - 1;
  const int64_t seg_tiles = tile_begin->numel();
  const int64_t width = 2 * n;
  auto seg_partial = torch::empty({std::max<int64_t>(seg_tiles, 1), width}, x.options().dtype(at::kFloat));
  auto table_grad = torch::empty({segments, width}, x.options().dtype(at::kFloat));
  const unsigned wide_blocks = static_cast<unsigned>((width + threads - 1) / threads);
  if (seg_tiles > 0) {
    H3_DISPATCH(x.scalar_type(), "h3_rmsnorm_modulation_grad", [&] {
      modulation_grad_partial_kernel<T>
          <<<dim3(static_cast<unsigned>(seg_tiles), wide_blocks), threads, 0, stream>>>(
              reinterpret_cast<const T*>(grad.data_ptr()), reinterpret_cast<const T*>(x.data_ptr()),
              reinterpret_cast<const T*>(weight.data_ptr()), rstd.data_ptr<float>(),
              sorted_pos->data_ptr<int64_t>(), tile_begin->data_ptr<int64_t>(),
              tile_end->data_ptr<int64_t>(), seg_partial.data_ptr<float>(), static_cast<int>(n));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  fold_segments_kernel<<<dim3(static_cast<unsigned>(segments), wide_blocks), threads, 0, stream>>>(
      seg_partial.data_ptr<float>(), seg_first_tile->data_ptr<int64_t>(),
      table_grad.data_ptr<float>(), static_cast<int>(width));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dx, dweight, table_grad.narrow(1, 0, n), table_grad.narrow(1, n, n)};
}
