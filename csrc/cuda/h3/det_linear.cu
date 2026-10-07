// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// Deterministic, batch-invariant row-wise linear layers for the MiniMax-H3
// conditioning path (RFC #420: `timestep_mlp_fp32`, `adaln_projection_3mod`).
//
// These layers run on a handful of rows (one per distinct timestep) against
// large weights, so they are GEMVs bound by weight bandwidth. The reduction
// order is fixed and depends only on K (and N for the input gradient), never
// on the number of rows or their position.
//
// Reduction contract h3-det-linear-v1
// -----------------------------------
// forward  y[t, n] = act(bias[n] + dot(x[t, :], W[n, :]))
//   * one warp per output column n; lane l owns the 16-byte K chunks
//     c = l, l + 32, l + 64, ... (VEC = 16 / sizeof(w) elements each);
//   * each lane accumulates its chunks in ascending order and the elements of
//     a chunk in ascending order with fmaf into an FP32 accumulator from 0;
//   * the 32 lane sums are combined with an xor butterfly (16, 8, 4, 2, 1),
//     whose result is identical on every lane (FP add is commutative);
//   * the bias is added once after the butterfly; the activation runs in FP32
//     (SiLU as v / (1 + expf(-v)), PyTorch's formula); one cast at the store.
//   * this FP32 forward is used for FP32 weights; BF16 weights take the
//     tensor-core forward of contract h3-det-linear-bf16-mma-v1 below.
// d_input  dx[t, k] = sum_n g[t, n] * W[n, k]
//   * N is split into fixed chunks of 64 rows; each chunk is an ascending
//     fmaf chain into FP32 from 0; the chunk partials are then left-folded in
//     ascending chunk order and cast once.
// d_weight dW[n, k] = sum_t g[t, n] * x[t, k];  d_bias db[n] = sum_t g[t, n]
//   * an ascending-t fmaf chain into FP32 from 0, cast once. This is the one
//     cross-row reduction; its order is the logical row order.
//
// Reduction contract h3-det-linear-bf16-mma-v1 (BF16 x and W, FP32 accumulate)
// -----------------------------------------------------------------------------
// forward  y[t, n] = act(bias[n] + dot(x[t, :], W[n, :]))
//   * a warp owns 16 output columns (MMA rows) and 8 input rows (MMA columns;
//     rows past T are zero), so every launch runs the same instruction
//     sequence for every T <= 8, and an MMA column never sees another's data;
//   * K is visited in groups of 16 in ascending order. Within a 32-wide step,
//     quad q supplies k = 32s + 8q + {0..3} to the first mma.sync m16n8k16 and
//     k = 32s + 8q + {4..7} to the second (the same k map for W and x);
//   * every mma.sync starts from a zero accumulator, so the tensor core only
//     sums 16 products; its FP32 result is added to the running FP32 sum with
//     an IEEE add, group by group in ascending k;
//   * then bias, FP32 activation and one cast, as above.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <vector>

namespace {

#ifndef H3_FWD_F32_COLS
#define H3_FWD_F32_COLS 2
#define H3_FWD_F32_AHEAD 6
#endif
#ifndef H3_FWD_CONFIGS
#define H3_FWD_CONFIGS H3_FWD_CASE(H3_FWD_F32_COLS, H3_FWD_F32_AHEAD)
#endif

constexpr int kWarp = 32;
constexpr int kWarpsPerBlock = 8;
constexpr int kRowTile = 4;
constexpr int kDInputChunk = 64;

enum Activation : int64_t { kActNone = 0, kActSilu = 1 };

__device__ __forceinline__ float to_float(float v) { return v; }
__device__ __forceinline__ float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T>
__device__ __forceinline__ T from_float(float v);
template <>
__device__ __forceinline__ float from_float<float>(float v) { return v; }
template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) {
  return __float2bfloat16(v);  // round to nearest even, like at::BFloat16
}

// 16-byte vector load converted to FP32.
template <typename T>
struct Vec16 {
  static constexpr int kN = 16 / sizeof(T);
  __device__ __forceinline__ static void load(const T* ptr, float (&out)[kN]) {
    unpack(load_raw(ptr), out);
  }
  __device__ __forceinline__ static uint4 load_raw(const T* ptr) {
    return *reinterpret_cast<const uint4*>(ptr);
  }
  // Streaming weights are read once: bypass L1 allocation.
  __device__ __forceinline__ static uint4 load_stream(const T* ptr) {
    return __ldcs(reinterpret_cast<const uint4*>(ptr));
  }
  __device__ __forceinline__ static void unpack(const uint4& raw, float (&out)[kN]) {
    const T* vals = reinterpret_cast<const T*>(&raw);
#pragma unroll
    for (int i = 0; i < kN; ++i) out[i] = to_float(vals[i]);
  }
};

__device__ __forceinline__ float warp_butterfly_sum(float v) {
#pragma unroll
  for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, offset);
  }
  return v;
}

// Each warp owns kCols adjacent output columns so that kLoadAhead x kCols
// 16-byte weight loads are in flight per lane. Grouping columns and chunks
// changes only when loads are issued: every column still accumulates its own
// chunks in ascending order, exactly as a one-column-per-warp kernel would.
template <typename x_t, typename w_t, typename out_t, int kRows, int kCols, int kLoadAhead>
__global__ void __launch_bounds__(kWarpsPerBlock * kWarp)
    det_linear_forward_kernel(const x_t* __restrict__ x, const w_t* __restrict__ w,
                              const w_t* __restrict__ bias, out_t* __restrict__ out,
                              float* __restrict__ pre_act, int64_t rows, int64_t n_out,
                              int64_t k_in, int64_t activation) {
  using WV = Vec16<w_t>;
  using XV = Vec16<x_t>;
  constexpr int kVec = WV::kN;
  static_assert(kCols * kRows <= kWarp, "one warp cannot store more than kWarp outputs per tile");
  static_assert(XV::kN == kVec, "x and weight share a dtype");
  constexpr int64_t kStride = static_cast<int64_t>(kWarp) * kVec;

  const int lane = threadIdx.x % kWarp;
  const int64_t warp_global =
      static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + threadIdx.x / kWarp;
  const int64_t warp_stride = static_cast<int64_t>(gridDim.x) * kWarpsPerBlock;

  for (int64_t n0 = warp_global * kCols; n0 < n_out; n0 += warp_stride * kCols) {
    for (int64_t t0 = 0; t0 < rows; t0 += kRows) {
      float acc[kCols][kRows];
#pragma unroll
      for (int c = 0; c < kCols; ++c) {
#pragma unroll
        for (int r = 0; r < kRows; ++r) acc[c][r] = 0.0f;
      }
      for (int64_t g0 = static_cast<int64_t>(lane) * kVec; g0 < k_in;
           g0 += kStride * kLoadAhead) {
        uint4 raw[kLoadAhead][kCols];
#pragma unroll
        for (int u = 0; u < kLoadAhead; ++u) {
          const int64_t k0 = g0 + u * kStride;
#pragma unroll
          for (int c = 0; c < kCols; ++c) {
            if (k0 < k_in && n0 + c < n_out) raw[u][c] = WV::load_stream(w + (n0 + c) * k_in + k0);
          }
        }
#pragma unroll
        for (int u = 0; u < kLoadAhead; ++u) {
          const int64_t k0 = g0 + u * kStride;
          if (k0 >= k_in) break;
#pragma unroll
          for (int r = 0; r < kRows; ++r) {
            if (t0 + r < rows) {
              float xv[kVec];
              XV::load(x + (t0 + r) * k_in + k0, xv);
#pragma unroll
              for (int c = 0; c < kCols; ++c) {
                float wv[kVec];
                WV::unpack(raw[u][c], wv);
#pragma unroll
                for (int j = 0; j < kVec; ++j) acc[c][r] = fmaf(xv[j], wv[j], acc[c][r]);
              }
            }
          }
        }
      }
#pragma unroll
      for (int c = 0; c < kCols; ++c) {
        const int64_t n = n0 + c;
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
          const int64_t t = t0 + r;
          const float sum = warp_butterfly_sum(acc[c][r]);
          if (lane == c * kRows + r && t < rows && n < n_out) {
            float v = sum + (bias != nullptr ? to_float(bias[n]) : 0.0f);
            if (pre_act != nullptr) pre_act[t * n_out + n] = v;
            if (activation == kActSilu) v = v / (1.0f + expf(-v));
            out[t * n_out + n] = from_float<out_t>(v);
          }
        }
      }
    }
  }
}

__device__ __forceinline__ void mma_m16n8k16_bf16(float (&c)[4], const uint32_t (&a)[4],
                                                  uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  // BF16 mma.sync needs SM80+; the host refuses to launch below that.
  __trap();
#else
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

constexpr int kMmaCols = 16;    // output columns per warp (MMA M)
constexpr int kMmaRows = 8;     // input rows per pass (MMA N)
constexpr int kMmaAhead = 4;    // 32-wide K steps loaded before use

__device__ __forceinline__ void mma_group_accumulate(float (&acc)[4], const uint32_t (&a)[4],
                                                     uint32_t b0, uint32_t b1) {
  float part[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  mma_m16n8k16_bf16(part, a, b0, b1);
#pragma unroll
  for (int i = 0; i < 4; ++i) acc[i] += part[i];
}

// Contract h3-det-linear-bf16-mma-v1. Lane = 4 * g + q: g picks the MMA row
// (output columns n0 + g and n0 + g + 8) and the MMA column (input row t0 + g).
__global__ void __launch_bounds__(kWarpsPerBlock * kWarp)
    det_linear_forward_bf16_mma_kernel(const __nv_bfloat16* __restrict__ x,
                                       const __nv_bfloat16* __restrict__ w,
                                       const __nv_bfloat16* __restrict__ bias,
                                       __nv_bfloat16* __restrict__ out, float* __restrict__ pre_act,
                                       int64_t rows, int64_t n_out, int64_t k_in,
                                       int64_t activation) {
  const int lane = threadIdx.x % kWarp;
  const int g = lane >> 2;
  const int q = lane & 3;
  const int64_t warp_global =
      static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + threadIdx.x / kWarp;
  const int64_t warp_stride = static_cast<int64_t>(gridDim.x) * kWarpsPerBlock;
  const uint4 zero = make_uint4(0u, 0u, 0u, 0u);

  for (int64_t n0 = warp_global * kMmaCols; n0 < n_out; n0 += warp_stride * kMmaCols) {
    const bool lo_ok = n0 + g < n_out;
    const bool hi_ok = n0 + g + 8 < n_out;
    const __nv_bfloat16* w_lo = w + (n0 + g) * k_in;
    const __nv_bfloat16* w_hi = w + (n0 + g + 8) * k_in;
    for (int64_t t0 = 0; t0 < rows; t0 += kMmaRows) {
      const bool x_ok = t0 + g < rows;
      const __nv_bfloat16* x_row = x + (t0 + g) * k_in;
      float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      for (int64_t kb = 0; kb < k_in; kb += 32 * kMmaAhead) {
        uint4 a_lo[kMmaAhead], a_hi[kMmaAhead], bx[kMmaAhead];
#pragma unroll
        for (int u = 0; u < kMmaAhead; ++u) {
          const int64_t k = kb + 32 * u + 8 * q;
          const bool k_ok = k < k_in;
          a_lo[u] = (k_ok && lo_ok) ? __ldcs(reinterpret_cast<const uint4*>(w_lo + k)) : zero;
          a_hi[u] = (k_ok && hi_ok) ? __ldcs(reinterpret_cast<const uint4*>(w_hi + k)) : zero;
          bx[u] = (k_ok && x_ok) ? *reinterpret_cast<const uint4*>(x_row + k) : zero;
        }
#pragma unroll
        for (int u = 0; u < kMmaAhead; ++u) {
          if (kb + 32 * u >= k_in) break;
          const uint32_t first[4] = {a_lo[u].x, a_hi[u].x, a_lo[u].y, a_hi[u].y};
          const uint32_t second[4] = {a_lo[u].z, a_hi[u].z, a_lo[u].w, a_hi[u].w};
          mma_group_accumulate(acc, first, bx[u].x, bx[u].y);
          mma_group_accumulate(acc, second, bx[u].z, bx[u].w);
        }
      }
      // acc = {(n0+g, t0+2q), (n0+g, t0+2q+1), (n0+g+8, t0+2q), (n0+g+8, t0+2q+1)}
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int64_t n = n0 + g + (i >= 2 ? 8 : 0);
        const int64_t t = t0 + 2 * q + (i & 1);
        if (n < n_out && t < rows) {
          float v = acc[i] + (bias != nullptr ? __bfloat162float(bias[n]) : 0.0f);
          if (pre_act != nullptr) pre_act[t * n_out + n] = v;
          if (activation == kActSilu) v = v / (1.0f + expf(-v));
          out[t * n_out + n] = __float2bfloat16(v);
        }
      }
    }
  }
}

// partial[c, t, k] = sum_{n in chunk c} g[t, n] * w[n, k]
template <typename g_t, typename w_t>
__global__ void det_linear_dinput_partial_kernel(const g_t* __restrict__ grad,
                                                 const w_t* __restrict__ w,
                                                 float* __restrict__ partial, int64_t rows,
                                                 int64_t n_out, int64_t k_in) {
  const int64_t k = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t chunk = blockIdx.y;
  if (k >= k_in) return;
  const int64_t n_begin = chunk * kDInputChunk;
  const int64_t n_end = min(n_begin + kDInputChunk, n_out);
  for (int64_t t0 = 0; t0 < rows; t0 += kRowTile) {
    float acc[kRowTile];
#pragma unroll
    for (int r = 0; r < kRowTile; ++r) acc[r] = 0.0f;
    for (int64_t n = n_begin; n < n_end; ++n) {
      const float wv = to_float(w[n * k_in + k]);
#pragma unroll
      for (int r = 0; r < kRowTile; ++r) {
        const int64_t t = t0 + r;
        if (t < rows) acc[r] = fmaf(to_float(grad[t * n_out + n]), wv, acc[r]);
      }
    }
#pragma unroll
    for (int r = 0; r < kRowTile; ++r) {
      const int64_t t = t0 + r;
      if (t < rows) partial[(chunk * rows + t) * k_in + k] = acc[r];
    }
  }
}

template <typename out_t>
__global__ void det_fold_chunks_kernel(const float* __restrict__ partial, out_t* __restrict__ out,
                                       int64_t chunks, int64_t elems) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= elems) return;
  float acc = partial[i];
  for (int64_t c = 1; c < chunks; ++c) acc += partial[c * elems + i];
  out[i] = from_float<out_t>(acc);
}

// dW[n, k] = sum_t g[t, n] * x[t, k] ; ascending t, FP32 from 0, one cast.
template <typename g_t, typename x_t, typename w_t>
__global__ void det_linear_dweight_kernel(const g_t* __restrict__ grad, const x_t* __restrict__ x,
                                          w_t* __restrict__ dw, int64_t rows, int64_t n_out,
                                          int64_t k_in) {
  const int64_t total = n_out * k_in;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t n = i / k_in;
    const int64_t k = i - n * k_in;
    float acc = 0.0f;
    for (int64_t t = 0; t < rows; ++t) {
      acc = fmaf(to_float(grad[t * n_out + n]), to_float(x[t * k_in + k]), acc);
    }
    dw[i] = from_float<w_t>(acc);
  }
}

template <typename g_t, typename w_t>
__global__ void det_linear_dbias_kernel(const g_t* __restrict__ grad, w_t* __restrict__ db,
                                        int64_t rows, int64_t n_out) {
  const int64_t n = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (n >= n_out) return;
  float acc = 0.0f;
  for (int64_t t = 0; t < rows; ++t) acc += to_float(grad[t * n_out + n]);
  db[n] = from_float<w_t>(acc);
}

template <typename T>
struct CudaType {
  using type = T;
};
template <>
struct CudaType<at::BFloat16> {
  using type = __nv_bfloat16;
};

template <typename T>
const typename CudaType<T>::type* cptr(const torch::Tensor& t) {
  return reinterpret_cast<const typename CudaType<T>::type*>(t.data_ptr<T>());
}
template <typename T>
typename CudaType<T>::type* mptr(torch::Tensor& t) {
  return reinterpret_cast<typename CudaType<T>::type*>(t.data_ptr<T>());
}

void check_matrix(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.dim() == 2, name, " must be 2-D");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.scalar_type() == at::kFloat || t.scalar_type() == at::kBFloat16, name,
              " must be float32 or bfloat16, got ", t.scalar_type());
  TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, name,
              " must be 16-byte aligned");
}

int64_t grid_for(int64_t work, int64_t per_block) {
  const int64_t blocks = (work + per_block - 1) / per_block;
  return std::max<int64_t>(1, std::min<int64_t>(blocks, 1 << 20));
}

// Launch shape only: the row tile, the columns per warp and the load depth
// change which loads are in flight, never a column's accumulation order.
struct ForwardConfig {
  int rows;
  int cols;
  int ahead;
};

ForwardConfig pick_forward_config(int64_t rows) {
  const int row_tile = rows <= 1 ? 1 : (rows <= 2 ? 2 : 4);
  return {row_tile, H3_FWD_F32_COLS, H3_FWD_F32_AHEAD};
}

template <typename T, int kRows, int kCols, int kAhead>
void launch_forward_impl(const T* x, const T* w, const T* bias, T* out, float* pre,
                         int64_t rows, int64_t n_out, int64_t k_in, int64_t activation,
                         cudaStream_t stream) {
  const int64_t blocks = grid_for((n_out + kCols - 1) / kCols, kWarpsPerBlock);
  det_linear_forward_kernel<T, T, T, kRows, kCols, kAhead>
      <<<static_cast<unsigned>(blocks), kWarpsPerBlock * kWarp, 0, stream>>>(
          x, w, bias, out, pre, rows, n_out, k_in, activation);
}

template <typename T, int kRows>
void launch_forward_rows(const ForwardConfig& cfg, const T* x, const T* w, const T* bias, T* out,
                         float* pre, int64_t rows, int64_t n_out, int64_t k_in,
                         int64_t activation, cudaStream_t stream) {
#define H3_FWD_CASE(C, A)                                                                  \
  if (cfg.cols == C && cfg.ahead == A) {                                                   \
    launch_forward_impl<T, kRows, C, A>(x, w, bias, out, pre, rows, n_out, k_in, activation, \
                                        stream);                                           \
    return;                                                                                \
  }
  H3_FWD_CONFIGS
#undef H3_FWD_CASE
  TORCH_CHECK(false, "unsupported det_linear forward config cols=", cfg.cols,
              " ahead=", cfg.ahead);
}

template <typename T>
void launch_forward(const ForwardConfig& cfg, const T* x, const T* w, const T* bias, T* out,
                    float* pre, int64_t rows, int64_t n_out, int64_t k_in, int64_t activation,
                    cudaStream_t stream) {
  if (cfg.rows == 1) {
    launch_forward_rows<T, 1>(cfg, x, w, bias, out, pre, rows, n_out, k_in, activation, stream);
  } else if (cfg.rows == 2) {
    launch_forward_rows<T, 2>(cfg, x, w, bias, out, pre, rows, n_out, k_in, activation, stream);
  } else {
    launch_forward_rows<T, 4>(cfg, x, w, bias, out, pre, rows, n_out, k_in, activation, stream);
  }
}

}  // namespace

std::vector<torch::Tensor> h3_det_linear_forward(torch::Tensor x, torch::Tensor weight,
                                                 c10::optional<torch::Tensor> bias,
                                                 int64_t activation, bool save_pre_activation) {
  check_matrix(x, "x");
  check_matrix(weight, "weight");
  TORCH_CHECK(x.device() == weight.device(), "x and weight must be on the same device");
  TORCH_CHECK(x.scalar_type() == weight.scalar_type(),
              "x and weight must share a dtype (cast at the declared boundary first), got ",
              x.scalar_type(), " and ", weight.scalar_type());
  TORCH_CHECK(activation == kActNone || activation == kActSilu, "unknown activation ",
              activation);
  const int64_t rows = x.size(0);
  const int64_t k_in = x.size(1);
  const int64_t n_out = weight.size(0);
  TORCH_CHECK(rows > 0, "x must have at least one row");
  TORCH_CHECK(weight.size(1) == k_in, "weight is [", n_out, ", ", weight.size(1),
              "] but x has K=", k_in);
  const int64_t vec = 16 / x.element_size();
  TORCH_CHECK(k_in % vec == 0, "K=", k_in, " must be a multiple of ", vec,
              " for 16-byte vector loads");
  if (bias.has_value()) {
    TORCH_CHECK(bias->is_cuda() && bias->dim() == 1 && bias->size(0) == n_out &&
                    bias->is_contiguous() && bias->scalar_type() == weight.scalar_type(),
                "bias must be a contiguous [N] CUDA tensor with the weight dtype");
    TORCH_CHECK(bias->device() == weight.device(),
                "bias and weight must be on the same device");
  }

  const c10::cuda::CUDAGuard device_guard(x.device());
  auto out = torch::empty({rows, n_out}, x.options());
  torch::Tensor pre;
  if (save_pre_activation) pre = torch::empty({rows, n_out}, x.options().dtype(at::kFloat));
  auto stream = at::cuda::getCurrentCUDAStream();
  if (x.scalar_type() == at::kFloat) {
    const ForwardConfig cfg = pick_forward_config(rows);
    launch_forward<float>(cfg, cptr<float>(x), cptr<float>(weight),
                          bias.has_value() ? cptr<float>(*bias) : nullptr, mptr<float>(out),
                          save_pre_activation ? pre.data_ptr<float>() : nullptr, rows, n_out,
                          k_in, activation, stream);
  } else {
    TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major >= 8,
                "the BF16 h3 det_linear forward uses mma.sync and needs SM80 or newer");
    const int64_t blocks = grid_for((n_out + kMmaCols - 1) / kMmaCols, kWarpsPerBlock);
    det_linear_forward_bf16_mma_kernel<<<static_cast<unsigned>(blocks), kWarpsPerBlock * kWarp,
                                         0, stream>>>(
        cptr<at::BFloat16>(x), cptr<at::BFloat16>(weight),
        bias.has_value() ? cptr<at::BFloat16>(*bias) : nullptr, mptr<at::BFloat16>(out),
        save_pre_activation ? pre.data_ptr<float>() : nullptr, rows, n_out, k_in, activation);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (save_pre_activation) return {out, pre};
  return {out};
}

// grad [T, N] (float32), weight [N, K] -> grad_input [T, K] in out_dtype.
torch::Tensor h3_det_linear_backward_input(torch::Tensor grad, torch::Tensor weight,
                                           c10::ScalarType out_dtype) {
  TORCH_CHECK(grad.is_cuda() && grad.dim() == 2 && grad.is_contiguous() &&
                  grad.scalar_type() == at::kFloat,
              "grad must be a contiguous 2-D float32 CUDA tensor");
  check_matrix(weight, "weight");
  TORCH_CHECK(grad.device() == weight.device(), "grad and weight must be on the same device");
  TORCH_CHECK(grad.size(1) == weight.size(0), "grad N != weight N");
  TORCH_CHECK(out_dtype == at::kFloat || out_dtype == at::kBFloat16,
              "out_dtype must be float32 or bfloat16");
  const int64_t rows = grad.size(0);
  TORCH_CHECK(rows > 0, "grad must have at least one row");
  const int64_t n_out = weight.size(0);
  const int64_t k_in = weight.size(1);
  const c10::cuda::CUDAGuard device_guard(grad.device());
  const int64_t chunks = (n_out + kDInputChunk - 1) / kDInputChunk;
  auto partial = torch::empty({chunks, rows, k_in}, grad.options());
  auto out = torch::empty({rows, k_in}, grad.options().dtype(out_dtype));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int threads = 256;
  dim3 grid(static_cast<unsigned>((k_in + threads - 1) / threads), static_cast<unsigned>(chunks));
  if (weight.scalar_type() == at::kFloat) {
    det_linear_dinput_partial_kernel<float, float><<<grid, threads, 0, stream>>>(
        grad.data_ptr<float>(), cptr<float>(weight), partial.data_ptr<float>(), rows, n_out,
        k_in);
  } else {
    det_linear_dinput_partial_kernel<float, __nv_bfloat16><<<grid, threads, 0, stream>>>(
        grad.data_ptr<float>(), cptr<at::BFloat16>(weight), partial.data_ptr<float>(), rows,
        n_out, k_in);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int64_t elems = rows * k_in;
  const unsigned fold_blocks = static_cast<unsigned>((elems + threads - 1) / threads);
  if (out_dtype == at::kFloat) {
    det_fold_chunks_kernel<float><<<fold_blocks, threads, 0, stream>>>(
        partial.data_ptr<float>(), mptr<float>(out), chunks, elems);
  } else {
    det_fold_chunks_kernel<__nv_bfloat16><<<fold_blocks, threads, 0, stream>>>(
        partial.data_ptr<float>(), mptr<at::BFloat16>(out), chunks, elems);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// grad [T, N] float32, x [T, K] (float32 or bfloat16) -> (dW [N, K], dbias [N]) in w_dtype.
std::vector<torch::Tensor> h3_det_linear_backward_weight(torch::Tensor grad, torch::Tensor x,
                                                         c10::ScalarType w_dtype,
                                                         bool with_bias) {
  TORCH_CHECK(grad.is_cuda() && grad.dim() == 2 && grad.is_contiguous() &&
                  grad.scalar_type() == at::kFloat,
              "grad must be a contiguous 2-D float32 CUDA tensor");
  check_matrix(x, "x");
  TORCH_CHECK(grad.device() == x.device(), "grad and x must be on the same device");
  TORCH_CHECK(grad.size(0) == x.size(0), "grad rows != x rows");
  TORCH_CHECK(w_dtype == at::kFloat || w_dtype == at::kBFloat16,
              "w_dtype must be float32 or bfloat16");
  const int64_t rows = grad.size(0);
  TORCH_CHECK(rows > 0, "grad must have at least one row");
  const int64_t n_out = grad.size(1);
  const int64_t k_in = x.size(1);
  const c10::cuda::CUDAGuard device_guard(grad.device());
  auto dw = torch::empty({n_out, k_in}, grad.options().dtype(w_dtype));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int threads = 256;
  const int64_t total = n_out * k_in;
  const unsigned blocks = static_cast<unsigned>(grid_for(total, threads));
  const float* g = grad.data_ptr<float>();
#define H3_DW_LAUNCH(XT, XC, WT, WC)                                                     \
  det_linear_dweight_kernel<float, XC, WC><<<blocks, threads, 0, stream>>>(             \
      g, cptr<XT>(x), mptr<WT>(dw), rows, n_out, k_in)
  if (x.scalar_type() == at::kFloat && w_dtype == at::kFloat) {
    H3_DW_LAUNCH(float, float, float, float);
  } else if (x.scalar_type() == at::kFloat) {
    H3_DW_LAUNCH(float, float, at::BFloat16, __nv_bfloat16);
  } else if (w_dtype == at::kFloat) {
    H3_DW_LAUNCH(at::BFloat16, __nv_bfloat16, float, float);
  } else {
    H3_DW_LAUNCH(at::BFloat16, __nv_bfloat16, at::BFloat16, __nv_bfloat16);
  }
#undef H3_DW_LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (!with_bias) return {dw};
  auto db = torch::empty({n_out}, grad.options().dtype(w_dtype));
  const unsigned bias_blocks = static_cast<unsigned>((n_out + threads - 1) / threads);
  if (w_dtype == at::kFloat) {
    det_linear_dbias_kernel<float, float><<<bias_blocks, threads, 0, stream>>>(
        g, mptr<float>(db), rows, n_out);
  } else {
    det_linear_dbias_kernel<float, __nv_bfloat16><<<bias_blocks, threads, 0, stream>>>(
        g, mptr<at::BFloat16>(db), rows, n_out);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dw, db};
}
