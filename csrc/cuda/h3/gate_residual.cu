// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// MiniMax-H3 gated residual (RFC #420 `adaln_gate_residual`).
//
//   out[r, j] = cast(residual[r, j] + cast(gate[i, j] * y[r, j]))     i = index[r % S]
//
// The gate row is read from a strided view of the AdaLN table (gate_msa or
// gate_mlp) inside the kernel, and the two roundings sit exactly where the
// eager `residual + gate.index_select(0, i) * y` rounds (__fmul_rn/__fadd_rn
// keep FP32 from contracting them into one FMA), so the result is
// bitwise equal to diffusers. Elementwise: every output depends only on its
// own inputs.
//
// Backward: d_residual = grad; d_y = cast(grad * gate) (the product of two
// 16-bit values is exact in FP32, so this equals the eager VJP bitwise);
// d_gate[i] = sum over positions mapped to i of grad * y, an FP32 segmented
// sum over positions sorted stably by row, in fixed tiles folded in order.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <vector>

namespace {

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

template <typename T>
__device__ __forceinline__ float round_t(float v) { return to_f(from_f<T>(v)); }

struct Gate {
  const void* rows;      // (R, N) view, row stride `stride`
  int64_t stride;
  const int64_t* index;  // (S,)
  int64_t seq;
};

// One block per row (grid-strided); 16-byte vectors when the row allows it.
// `kVecElems` elements are loaded together; each is computed independently.
template <typename T, bool kDy>
__global__ void gate_residual_rows_kernel(const T* __restrict__ a, const T* __restrict__ b,
                                          T* __restrict__ out, int64_t rows, int64_t n,
                                          Gate gate, bool vectorized) {
  constexpr int kVecElems = 16 / sizeof(T);
  for (int64_t r = blockIdx.x; r < rows; r += gridDim.x) {
    const T* g = static_cast<const T*>(gate.rows) + gate.index[r % gate.seq] * gate.stride;
    const T* ar = a + r * n;
    const T* br = kDy ? nullptr : b + r * n;
    T* orow = out + r * n;
    if (vectorized) {
      for (int64_t v = threadIdx.x; v < n / kVecElems; v += blockDim.x) {
        const uint4 av = reinterpret_cast<const uint4*>(ar)[v];
        const uint4 gv = reinterpret_cast<const uint4*>(g)[v];
        uint4 bv = av;
        if (!kDy) bv = reinterpret_cast<const uint4*>(br)[v];
        const T* ae = reinterpret_cast<const T*>(&av);
        const T* ge = reinterpret_cast<const T*>(&gv);
        const T* be = reinterpret_cast<const T*>(&bv);
        uint4 ov;
        T* oe = reinterpret_cast<T*>(&ov);
#pragma unroll
        for (int k = 0; k < kVecElems; ++k) {
          if (kDy) {
            oe[k] = from_f<T>(to_f(ae[k]) * to_f(ge[k]));  // grad * gate
          } else {
            const float p = round_t<T>(__fmul_rn(to_f(ge[k]), to_f(be[k])));  // gate * y
            oe[k] = from_f<T>(__fadd_rn(to_f(ae[k]), p));           // residual + p
          }
        }
        reinterpret_cast<uint4*>(orow)[v] = ov;
      }
    } else {
      for (int64_t j = threadIdx.x; j < n; j += blockDim.x) {
        if (kDy) {
          orow[j] = from_f<T>(to_f(ar[j]) * to_f(g[j]));
        } else {
          const float p = round_t<T>(__fmul_rn(to_f(g[j]), to_f(br[j])));
          orow[j] = from_f<T>(__fadd_rn(to_f(ar[j]), p));
        }
      }
    }
  }
}

// partial[tile, j] = sum over sorted positions of the tile (ascending) of grad * y
template <typename T>
__global__ void gate_grad_partial_kernel(const T* __restrict__ grad, const T* __restrict__ y,
                                         const int64_t* __restrict__ sorted_pos,
                                         const int64_t* __restrict__ tile_begin,
                                         const int64_t* __restrict__ tile_end,
                                         float* __restrict__ partial, int64_t n) {
  const int64_t tile = blockIdx.x;
  const int64_t j = static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
  if (j >= n) return;
  float acc = 0.0f;
  for (int64_t p = tile_begin[tile]; p < tile_end[tile]; ++p) {
    const int64_t e = sorted_pos[p] * n + j;
    acc = fmaf(to_f(grad[e]), to_f(y[e]), acc);
  }
  partial[tile * n + j] = acc;
}

template <typename T>
__global__ void gate_grad_fold_kernel(const float* __restrict__ partial,
                                      const int64_t* __restrict__ seg_first_tile,
                                      T* __restrict__ out, int64_t n) {
  const int64_t seg = blockIdx.x;
  const int64_t j = static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
  if (j >= n) return;
  float acc = 0.0f;
  for (int64_t t = seg_first_tile[seg]; t < seg_first_tile[seg + 1]; ++t) acc += partial[t * n + j];
  out[seg * n + j] = from_f<T>(acc);
}

#define H3_DISPATCH(TYPE, NAME, ...)                                                       \
  [&] {                                                                                    \
    switch (TYPE) {                                                                        \
      case at::kFloat: { using T = float; __VA_ARGS__(); break; }                          \
      case at::kHalf: { using T = __half; __VA_ARGS__(); break; }                          \
      case at::kBFloat16: { using T = __nv_bfloat16; __VA_ARGS__(); break; }               \
      default: TORCH_CHECK(false, NAME, ": unsupported dtype ", TYPE);                     \
    }                                                                                      \
  }()

Gate make_gate(const torch::Tensor& gate, const torch::Tensor& index, const torch::Tensor& x) {
  TORCH_CHECK(gate.is_cuda() && gate.device() == x.device() && gate.scalar_type() == x.scalar_type(),
              "gate must be a CUDA tensor with the activations' dtype");
  TORCH_CHECK(gate.dim() == 2 && gate.size(1) == x.size(1) && gate.stride(1) == 1,
              "gate must be (R, N) with unit column stride");
  TORCH_CHECK(index.is_cuda() && index.device() == x.device() && index.scalar_type() == at::kLong &&
                  index.dim() == 1 && index.is_contiguous() && index.numel() > 0,
              "index must be a non-empty contiguous int64 tensor on ", x.device());
  TORCH_CHECK(x.size(0) % index.size(0) == 0, "rows must be a multiple of S");
  return Gate{gate.data_ptr(), gate.stride(0), index.data_ptr<int64_t>(), index.size(0)};
}

void check_act(const torch::Tensor& t, const char* name, const torch::Tensor& like) {
  TORCH_CHECK(t.is_cuda() && t.device() == like.device() && t.scalar_type() == like.scalar_type(),
              name, " must match the residual's device and dtype");
  TORCH_CHECK(t.sizes() == like.sizes() && t.is_contiguous(), name,
              " must be contiguous with the residual's shape");
}

unsigned row_blocks(int64_t rows) {
  return static_cast<unsigned>(std::min<int64_t>(rows, 1 << 30));
}

bool can_vectorize(const torch::Tensor& gate, int64_t n, std::initializer_list<const torch::Tensor*> acts) {
  const int64_t vec = 16 / gate.element_size();
  if (n % vec != 0 || (gate.stride(0) * gate.element_size()) % 16 != 0) return false;
  if (reinterpret_cast<uintptr_t>(gate.data_ptr()) % 16 != 0) return false;
  for (const auto* t : acts) {
    if (reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 != 0) return false;
  }
  return true;
}

}  // namespace

// residual, y: (M, N); gate: (R, N) view; index: (S,), M % S == 0.
torch::Tensor h3_gate_residual_forward(torch::Tensor residual, torch::Tensor y, torch::Tensor gate,
                                       torch::Tensor index) {
  TORCH_CHECK(residual.is_cuda() && residual.dim() == 2 && residual.is_contiguous() &&
                  residual.numel() > 0,
              "residual must be a non-empty contiguous (M, N) CUDA tensor");
  check_act(y, "y", residual);
  const Gate g = make_gate(gate, index, residual);
  const c10::cuda::CUDAGuard guard(residual.device());
  auto out = torch::empty_like(residual);
  auto stream = at::cuda::getCurrentCUDAStream();
  const bool vec = can_vectorize(gate, residual.size(1), {&residual, &y, &out});
  H3_DISPATCH(residual.scalar_type(), "h3_gate_residual_forward", [&] {
    gate_residual_rows_kernel<T, false><<<row_blocks(residual.size(0)), 256, 0, stream>>>(
        reinterpret_cast<const T*>(residual.data_ptr()), reinterpret_cast<const T*>(y.data_ptr()),
        reinterpret_cast<T*>(out.data_ptr()), residual.size(0), residual.size(1), g, vec);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// Returns {d_y, d_gate (R, N) in the gate dtype}. d_residual is the incoming grad.
std::vector<torch::Tensor> h3_gate_residual_backward(torch::Tensor grad, torch::Tensor y,
                                                     torch::Tensor gate, torch::Tensor index,
                                                     torch::Tensor sorted_pos,
                                                     torch::Tensor tile_begin,
                                                     torch::Tensor tile_end,
                                                     torch::Tensor seg_first_tile) {
  TORCH_CHECK(grad.is_cuda() && grad.dim() == 2 && grad.is_contiguous(),
              "grad must be a contiguous (M, N) CUDA tensor");
  check_act(y, "y", grad);
  const Gate g = make_gate(gate, index, grad);
  for (const auto* t : {&sorted_pos, &tile_begin, &tile_end, &seg_first_tile}) {
    TORCH_CHECK(t->is_cuda() && t->device() == grad.device() && t->scalar_type() == at::kLong &&
                    t->dim() == 1 && t->is_contiguous(),
                "tile metadata must be contiguous int64 tensors on ", grad.device());
  }
  const int64_t rows = grad.size(0);
  const int64_t n = grad.size(1);
  const int64_t segments = seg_first_tile.numel() - 1;
  TORCH_CHECK(segments == gate.size(0), "segments must equal the gate rows");
  const int64_t tiles = tile_begin.numel();
  const c10::cuda::CUDAGuard guard(grad.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto dy = torch::empty_like(y);
  auto partial = torch::empty({std::max<int64_t>(tiles, 1), n}, grad.options().dtype(at::kFloat));
  auto dgate = torch::empty({segments, n}, grad.options());
  const int threads = 256;
  const unsigned col_blocks = static_cast<unsigned>((n + threads - 1) / threads);
  H3_DISPATCH(grad.scalar_type(), "h3_gate_residual_backward", [&] {
    const T* gp = reinterpret_cast<const T*>(grad.data_ptr());
    const T* yp = reinterpret_cast<const T*>(y.data_ptr());
    const bool vec = can_vectorize(gate, n, {&grad, &dy});
    gate_residual_rows_kernel<T, true><<<row_blocks(rows), threads, 0, stream>>>(
        gp, nullptr, reinterpret_cast<T*>(dy.data_ptr()), rows, n, g, vec);
    if (tiles > 0) {
      gate_grad_partial_kernel<T><<<dim3(static_cast<unsigned>(tiles), col_blocks), threads, 0, stream>>>(
          gp, yp, sorted_pos.data_ptr<int64_t>(), tile_begin.data_ptr<int64_t>(),
          tile_end.data_ptr<int64_t>(), partial.data_ptr<float>(), n);
    }
    gate_grad_fold_kernel<T><<<dim3(static_cast<unsigned>(segments), col_blocks), threads, 0, stream>>>(
        partial.data_ptr<float>(), seg_first_tile.data_ptr<int64_t>(),
        reinterpret_cast<T*>(dgate.data_ptr()), n);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dy, dgate};
}
