// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// MiniMax-H3 AdaLN row gather (RFC #420 `adaln_row_gather`).
//
// rows: (3T, C * H) modulation rows, row r = timestep * 3 + modality and
// column block c = shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp,
// gate_mlp. For every packed-sequence position s:
//
//   r[s]          = timestep_indices[s] * modality_num + token_tags[s]
//   out[c, s, :]  = rows[r[s], c * H : (c + 1) * H]
//
// Forward is a pure copy: bitwise equal to C index_select calls, one launch,
// independent of S, packing order and repeats.
//
// Backward d_rows[r, j] = sum_{s : r[s] = r} grad[c(j), s, h(j)] is the one
// cross-row reduction. Positions are visited in a fixed order: the caller
// sorts them by (r, s) with a stable sort and cuts every segment into tiles of
// at most kTile positions. Each tile is an ascending FP32 chain from 0; the
// tiles of a segment are then left-folded in ascending order and cast once.
// No atomics: the result is a function of (r[s], grad) only.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>

#include <cstdint>

namespace {

__device__ __forceinline__ float to_float(float v) { return v; }
__device__ __forceinline__ float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float to_float(__half v) { return __half2float(v); }
__device__ __forceinline__ float to_float(double v) { return static_cast<float>(v); }

template <typename T>
__device__ __forceinline__ T from_float(float v);
template <>
__device__ __forceinline__ float from_float<float>(float v) { return v; }
template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}
template <>
__device__ __forceinline__ __half from_float<__half>(float v) { return __float2half(v); }
template <>
__device__ __forceinline__ double from_float<double>(float v) { return v; }

template <typename T>
struct CudaType {
  using type = T;
};
template <>
struct CudaType<at::BFloat16> {
  using type = __nv_bfloat16;
};
template <>
struct CudaType<at::Half> {
  using type = __half;
};

template <typename idx_t>
__device__ __forceinline__ int64_t adaln_row(const idx_t* ti, const idx_t* tags, int64_t s,
                                             int64_t modality_num, int64_t num_timesteps) {
  const int64_t timestep = static_cast<int64_t>(ti[s]);
  const int64_t tag = static_cast<int64_t>(tags[s]);
  // Check each semantic index before multiplication, including values that overflow int64.
  CUDA_KERNEL_ASSERT(timestep >= 0 && timestep < num_timesteps);
  CUDA_KERNEL_ASSERT(tag >= 0 && tag < modality_num);
  return timestep * modality_num + tag;
}

// One block per (s, c); 16-byte copies when every row start is 16-byte aligned.
template <typename T, typename idx_t, bool kVector>
__global__ void adaln_row_gather_kernel(const T* __restrict__ rows, int64_t row_stride,
                                        const idx_t* __restrict__ ti,
                                        const idx_t* __restrict__ tags, T* __restrict__ out,
                                        int64_t seq, int64_t hidden, int64_t chunks,
                                        int64_t modality_num, int64_t num_timesteps) {
  for (int64_t s = blockIdx.x; s < seq; s += gridDim.x) {
    const int64_t c = blockIdx.y;
    const int64_t r = adaln_row(ti, tags, s, modality_num, num_timesteps);
    const T* src = rows + r * row_stride + c * hidden;
    T* dst = out + (c * seq + s) * hidden;
    if constexpr (kVector) {
      constexpr int kVec = 16 / sizeof(T);
      const uint4* src4 = reinterpret_cast<const uint4*>(src);
      uint4* dst4 = reinterpret_cast<uint4*>(dst);
      for (int64_t i = threadIdx.x; i < hidden / kVec; i += blockDim.x) dst4[i] = src4[i];
    } else {
      for (int64_t i = threadIdx.x; i < hidden; i += blockDim.x) dst[i] = src[i];
    }
  }
}

// partial[tile, j] = sum over sorted positions p in tile (ascending) of grad[c, s_p, h]
template <typename g_t>
__global__ void adaln_row_gather_partial_kernel(const g_t* __restrict__ grad,
                                                const int64_t* __restrict__ sorted_pos,
                                                const int64_t* __restrict__ tile_begin,
                                                const int64_t* __restrict__ tile_end,
                                                float* __restrict__ partial, int64_t seq,
                                                int64_t hidden, int64_t width) {
  const int64_t tile = blockIdx.x;
  const int64_t j = static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
  if (j >= width) return;
  const int64_t c = j / hidden;
  const int64_t h = j - c * hidden;
  const g_t* g = grad + c * seq * hidden + h;
  float acc = 0.0f;
  for (int64_t p = tile_begin[tile]; p < tile_end[tile]; ++p) {
    acc += to_float(g[sorted_pos[p] * hidden]);
  }
  partial[tile * width + j] = acc;
}

template <typename out_t>
__global__ void adaln_row_gather_fold_kernel(const float* __restrict__ partial,
                                             const int64_t* __restrict__ seg_first_tile,
                                             out_t* __restrict__ out, int64_t width) {
  const int64_t r = blockIdx.x;
  const int64_t j = static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
  if (j >= width) return;
  float acc = 0.0f;
  for (int64_t tile = seg_first_tile[r]; tile < seg_first_tile[r + 1]; ++tile) {
    acc += partial[tile * width + j];
  }
  out[r * width + j] = from_float<out_t>(acc);
}

void check_index(const torch::Tensor& t, const char* name, int64_t seq,
                 const torch::Tensor& like) {
  TORCH_CHECK(t.is_cuda() && t.device() == like.device(), name, " must be on ", like.device());
  TORCH_CHECK(t.dim() == 1 && t.size(0) == seq, name, " must be (", seq, ",)");
  TORCH_CHECK(t.scalar_type() == at::kLong || t.scalar_type() == at::kInt, name,
              " must be int64 or int32");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

}  // namespace

// rows (R, C * H) with unit column stride -> out (C, S, H) contiguous.
torch::Tensor h3_adaln_row_gather_forward(torch::Tensor rows, torch::Tensor timestep_indices,
                                          torch::Tensor token_tags, int64_t chunks,
                                          int64_t modality_num) {
  TORCH_CHECK(rows.is_cuda() && rows.dim() == 2 && rows.stride(1) == 1,
              "rows must be a 2-D CUDA tensor with unit column stride");
  TORCH_CHECK(chunks > 0 && rows.size(1) % chunks == 0, "rows width ", rows.size(1),
              " is not a multiple of ", chunks);
  TORCH_CHECK(modality_num > 0 && rows.size(0) % modality_num == 0, "rows count ",
              rows.size(0), " is not a multiple of ", modality_num);
  const int64_t num_timesteps = rows.size(0) / modality_num;
  const int64_t seq = timestep_indices.numel();
  TORCH_CHECK(seq > 0, "the packed sequence must not be empty");
  check_index(timestep_indices, "timestep_indices", seq, rows);
  check_index(token_tags, "token_tags", seq, rows);
  TORCH_CHECK(timestep_indices.scalar_type() == token_tags.scalar_type(),
              "timestep_indices and token_tags must share an integer dtype");
  const int64_t hidden = rows.size(1) / chunks;
  const c10::cuda::CUDAGuard device_guard(rows.device());
  auto out = torch::empty({chunks, seq, hidden}, rows.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(static_cast<unsigned>(std::min<int64_t>(seq, 1 << 20)),
                  static_cast<unsigned>(chunks));
  const int threads = 128;
  const int64_t elem = rows.element_size();
  const bool vector = (hidden * elem) % 16 == 0 && (rows.stride(0) * elem) % 16 == 0 &&
                      reinterpret_cast<uintptr_t>(rows.data_ptr()) % 16 == 0;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, rows.scalar_type(), "h3_adaln_row_gather_forward", [&] {
        using T = typename CudaType<scalar_t>::type;
        const T* src = reinterpret_cast<const T*>(rows.data_ptr<scalar_t>());
        T* dst = reinterpret_cast<T*>(out.data_ptr<scalar_t>());
        auto launch = [&](auto idx_tag) {
          using idx_t = decltype(idx_tag);
          const idx_t* ti = timestep_indices.data_ptr<idx_t>();
          const idx_t* tags = token_tags.data_ptr<idx_t>();
          if (vector) {
            adaln_row_gather_kernel<T, idx_t, true><<<grid, threads, 0, stream>>>(
                src, rows.stride(0), ti, tags, dst, seq, hidden, chunks, modality_num,
                num_timesteps);
          } else {
            adaln_row_gather_kernel<T, idx_t, false><<<grid, threads, 0, stream>>>(
                src, rows.stride(0), ti, tags, dst, seq, hidden, chunks, modality_num,
                num_timesteps);
          }
        };
        if (timestep_indices.scalar_type() == at::kLong) {
          launch(int64_t{0});
        } else {
          launch(int32_t{0});
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// grad (C, S, H) contiguous; sorted_pos (S,) positions sorted stably by row;
// tiles [tile_begin, tile_end) in sorted order; seg_first_tile (R + 1,).
torch::Tensor h3_adaln_row_gather_backward(torch::Tensor grad, torch::Tensor sorted_pos,
                                           torch::Tensor tile_begin, torch::Tensor tile_end,
                                           torch::Tensor seg_first_tile,
                                           c10::ScalarType out_dtype) {
  TORCH_CHECK(grad.is_cuda() && grad.dim() == 3 && grad.is_contiguous(),
              "grad must be a contiguous (C, S, H) CUDA tensor");
  for (const auto* t : {&sorted_pos, &tile_begin, &tile_end, &seg_first_tile}) {
    TORCH_CHECK(t->is_cuda() && t->device() == grad.device() && t->dim() == 1 && t->is_contiguous() &&
                    t->scalar_type() == at::kLong,
                "tile metadata must be contiguous int64 tensors on ", grad.device());
  }
  TORCH_CHECK(tile_begin.numel() == tile_end.numel(), "tile_begin/tile_end length mismatch");
  const int64_t chunks = grad.size(0);
  const int64_t seq = grad.size(1);
  const int64_t hidden = grad.size(2);
  const int64_t width = chunks * hidden;
  const int64_t num_rows = seg_first_tile.numel() - 1;
  const int64_t tiles = tile_begin.numel();
  TORCH_CHECK(sorted_pos.numel() == seq, "sorted_pos must have S entries");
  const c10::cuda::CUDAGuard device_guard(grad.device());
  auto out = torch::empty({num_rows, width}, grad.options().dtype(out_dtype));
  auto partial = torch::empty({std::max<int64_t>(tiles, 1), width},
                              grad.options().dtype(at::kFloat));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int threads = 256;
  const unsigned col_blocks = static_cast<unsigned>((width + threads - 1) / threads);
  if (tiles > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, grad.scalar_type(), "h3_adaln_row_gather_partial", [&] {
          using G = typename CudaType<scalar_t>::type;
          adaln_row_gather_partial_kernel<G>
              <<<dim3(static_cast<unsigned>(tiles), col_blocks), threads, 0, stream>>>(
                  reinterpret_cast<const G*>(grad.data_ptr<scalar_t>()),
                  sorted_pos.data_ptr<int64_t>(), tile_begin.data_ptr<int64_t>(),
                  tile_end.data_ptr<int64_t>(), partial.data_ptr<float>(), seq, hidden, width);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, out_dtype, "h3_adaln_row_gather_fold", [&] {
        using O = typename CudaType<scalar_t>::type;
        adaln_row_gather_fold_kernel<O>
            <<<dim3(static_cast<unsigned>(num_rows), col_blocks), threads, 0, stream>>>(
                partial.data_ptr<float>(), seg_first_tile.data_ptr<int64_t>(),
                reinterpret_cast<O*>(out.data_ptr<scalar_t>()), width);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
