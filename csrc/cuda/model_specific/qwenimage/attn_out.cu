// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// Qwen-Image attention output: tiled SIMT GEMM with an explicit FP32 tree.
// A leaf is 32 ascending FMA steps. For K=3072/TP, the leaf count is
// 3 * 2^L. Merge each consecutive triple as a + (b + c), then merge triples
// with a balanced tree. This is the midpoint tree for TP1/2/4/8.
// dW retains an ascending-row FMA fold. Shared-memory tiles change data
// movement only; they do not change any accumulator's arithmetic order.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <climits>

namespace {

constexpr int kRows = 16;
constexpr int kCols = 64;
constexpr int kLeaf = 32;
constexpr int kThreads = 256;
constexpr int kValues = 4;
constexpr int kWideCols = 128;
// Small calls retain the original single-row layout and 32-row dW staging.
constexpr int kLargeRows = 1024;

template <int Groups, int Levels, bool TwoRows>
__global__ void tree_gemm_kernel(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ output, int rows, int cols, int reduction) {
  constexpr int tile_rows = TwoRows ? 32 : kRows;
  constexpr int values = TwoRows ? 8 : kValues;
  // TwoRows stages K-major operands for adjacent 2x4 output microtiles.
  // The original path stages output-major operands with bank-separating padding.
  __shared__ float tile_a[TwoRows ? kLeaf : kRows][kLeaf + 1];
  __shared__ __align__(16) float tile_b[TwoRows ? kLeaf : kCols][TwoRows ? 68 : kLeaf + 1];
  const int tid = threadIdx.x;
  const int row = blockIdx.y * tile_rows + (tid / 16) * (TwoRows ? 2 : 1);
  const int col = blockIdx.x * kCols + (tid % 16) * (TwoRows ? 4 : 1);
  float stack[Levels + 1][values];
  for (int group = 0; group < Groups; ++group) {
    float first[values], second[values], acc[values];
    for (int branch = 0; branch < 3; ++branch) {
      const int start = (group * 3 + branch) * kLeaf;
      for (int i = tid; i < tile_rows * kLeaf; i += kThreads) {
        const int r = i / kLeaf, k = i % kLeaf;
        const int global_row = blockIdx.y * tile_rows + r;
        const float value = global_row < rows
            ? a[static_cast<int64_t>(global_row) * reduction + start + k] : 0.0f;
        if constexpr (TwoRows) tile_a[k][r] = value;
        else tile_a[r][k] = value;
      }
      for (int i = tid; i < kCols * kLeaf; i += kThreads) {
        const int n = i / kLeaf, k = i % kLeaf;
        const int global_col = blockIdx.x * kCols + n;
        const float value = global_col < cols
            ? b[static_cast<int64_t>(global_col) * reduction + start + k] : 0.0f;
        if constexpr (TwoRows) tile_b[k][n] = value;
        else tile_b[n][k] = value;
      }
      __syncthreads();
      #pragma unroll
      for (int j = 0; j < values; ++j) acc[j] = 0.0f;
      #pragma unroll
      for (int k = 0; k < kLeaf; ++k) {
        if constexpr (TwoRows) {
          const float av0 = tile_a[k][(tid / 16) * 2];
          const float av1 = tile_a[k][(tid / 16) * 2 + 1];
          const float4 bv = *reinterpret_cast<const float4*>(&tile_b[k][(tid % 16) * 4]);
          const float weights[4] = {bv.x, bv.y, bv.z, bv.w};
          #pragma unroll
          for (int j = 0; j < 4; ++j) {
            acc[j] = __fmaf_rn(av0, weights[j], acc[j]);
            acc[j + 4] = __fmaf_rn(av1, weights[j], acc[j + 4]);
          }
        } else {
          const float av = tile_a[tid / 16][k];
          #pragma unroll
          for (int j = 0; j < kValues; ++j) {
            acc[j] = __fmaf_rn(av, tile_b[tid % 16 + j * 16][k], acc[j]);
          }
        }
      }
      __syncthreads();
      #pragma unroll
      for (int j = 0; j < values; ++j) {
        if (branch == 0) first[j] = acc[j];
        else if (branch == 1) second[j] = acc[j];
        else acc[j] = __fadd_rn(first[j], __fadd_rn(second[j], acc[j]));
      }
    }
    // Compile-time stack indices keep the partial roots in registers.
    bool carry = true;
    #pragma unroll
    for (int level = 0; level <= Levels; ++level) {
      if (carry) {
        if ((group >> level) & 1) {
          #pragma unroll
          for (int j = 0; j < values; ++j) acc[j] = __fadd_rn(stack[level][j], acc[j]);
        } else {
          #pragma unroll
          for (int j = 0; j < values; ++j) stack[level][j] = acc[j];
          carry = false;
        }
      }
    }
  }
  #pragma unroll
  for (int j = 0; j < values; ++j) {
    const int output_row = row + (TwoRows ? j / 4 : 0);
    const int output_col = col + (TwoRows ? j % 4 : j * 16);
    if (output_row < rows && output_col < cols) {
      output[static_cast<int64_t>(output_row) * cols + output_col] = stack[Levels][j];
    }
  }
}

template <bool Wide>
__global__ void weight_grad_kernel(
    const float* __restrict__ grad, const float* __restrict__ x,
    float* __restrict__ output, int rows, int out_dim, int in_dim) {
  constexpr int row_chunk = Wide ? 64 : kLeaf;
  constexpr int tile_cols = Wide ? kWideCols : kCols;
  constexpr int values = Wide ? 8 : kValues;
  __shared__ float tile_g[row_chunk][kRows + 1];
  __shared__ float tile_x[row_chunk][tile_cols + 1];
  const int tid = threadIdx.x;
  const int n = blockIdx.y * kRows + tid / 16;
  const int k = blockIdx.x * tile_cols + tid % 16;
  float acc[values] = {};
  for (int start = 0; start < rows; start += row_chunk) {
    for (int i = tid; i < row_chunk * kRows; i += kThreads) {
      const int s = i / kRows, col = i % kRows;
      const int gn = blockIdx.y * kRows + col;
      tile_g[s][col] = start + s < rows && gn < out_dim
          ? grad[static_cast<int64_t>(start + s) * out_dim + gn] : 0.0f;
    }
    for (int i = tid; i < row_chunk * tile_cols; i += kThreads) {
      const int s = i / tile_cols, col = i % tile_cols;
      const int gk = blockIdx.x * tile_cols + col;
      tile_x[s][col] = start + s < rows && gk < in_dim
          ? x[static_cast<int64_t>(start + s) * in_dim + gk] : 0.0f;
    }
    __syncthreads();
    // Do not execute padded FMA steps: even adding zero can change -0 bytes.
    for (int s = 0; s < min(row_chunk, rows - start); ++s) {
      const float gv = tile_g[s][tid / 16];
      #pragma unroll
      for (int j = 0; j < values; ++j) {
        acc[j] = __fmaf_rn(gv, tile_x[s][tid % 16 + j * 16], acc[j]);
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int j = 0; j < values; ++j) {
    if (n < out_dim && k + j * 16 < in_dim) {
      output[static_cast<int64_t>(n) * in_dim + k + j * 16] = acc[j];
    }
  }
}

void check_matrix(const torch::Tensor& value) {
  TORCH_CHECK(value.is_cuda() && value.scalar_type() == at::kFloat,
              "attention output matrices must be CUDA float32 tensors");
  TORCH_CHECK(value.dim() == 2 && value.is_contiguous(),
              "attention output matrices must be contiguous and two-dimensional");
  TORCH_CHECK(value.size(0) > 0 && value.size(0) <= INT_MAX - kCols && value.size(1) > 0 &&
              value.size(1) <= INT_MAX - kCols, "attention output matrix dimensions exceed the supported range");
}

template <bool TwoRows>
void launch_tree(const float* a, const float* b, float* output,
                 int rows, int cols, int reduction, cudaStream_t stream) {
  constexpr int tile_rows = TwoRows ? 32 : kRows;
  const dim3 grid((cols + kCols - 1) / kCols, (rows + tile_rows - 1) / tile_rows);
  switch (reduction) {
    case 3072: tree_gemm_kernel<32, 5, TwoRows><<<grid, kThreads, 0, stream>>>(
        a, b, output, rows, cols, reduction); break;
    case 1536: tree_gemm_kernel<16, 4, TwoRows><<<grid, kThreads, 0, stream>>>(
        a, b, output, rows, cols, reduction); break;
    case 768: tree_gemm_kernel<8, 3, TwoRows><<<grid, kThreads, 0, stream>>>(
        a, b, output, rows, cols, reduction); break;
    case 384: tree_gemm_kernel<4, 2, TwoRows><<<grid, kThreads, 0, stream>>>(
        a, b, output, rows, cols, reduction); break;
    default: TORCH_CHECK(false, "attention output reduction must be 3072, 1536, 768 or 384");
  }
}

}  // namespace

torch::Tensor qwenimage_attn_out_tree_gemm(torch::Tensor a, torch::Tensor b) {
  check_matrix(a);
  check_matrix(b);
  TORCH_CHECK(a.device() == b.device() && a.size(1) == b.size(1),
              "attention output GEMM inputs must share a device and reduction dimension");
  TORCH_CHECK(a.size(0) <= static_cast<int64_t>(65535) * kRows,
              "attention output GEMM exceeds the CUDA row grid limit");
  const c10::cuda::CUDAGuard guard(a.device());
  const int rows = a.size(0), cols = b.size(0), reduction = a.size(1);
  auto result = torch::empty({rows, cols}, a.options());
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (rows < kLargeRows) {
    launch_tree<false>(a.data_ptr<float>(), b.data_ptr<float>(), result.data_ptr<float>(),
                       rows, cols, reduction, stream);
  } else {
    launch_tree<true>(a.data_ptr<float>(), b.data_ptr<float>(), result.data_ptr<float>(),
                      rows, cols, reduction, stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return result;
}

torch::Tensor qwenimage_attn_out_weight_grad(torch::Tensor grad, torch::Tensor x) {
  check_matrix(grad);
  check_matrix(x);
  TORCH_CHECK(grad.device() == x.device() && grad.size(0) == x.size(0),
              "attention output weight-gradient inputs must share a device and row count");
  TORCH_CHECK(grad.size(1) == 3072 &&
              (x.size(1) == 3072 || x.size(1) == 1536 || x.size(1) == 768 || x.size(1) == 384),
              "attention output weight-gradient dimensions do not match TP1/2/4/8");
  const c10::cuda::CUDAGuard guard(x.device());
  const int rows = x.size(0), out_dim = grad.size(1), in_dim = x.size(1);
  auto result = torch::empty({out_dim, in_dim}, x.options());
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (rows < kLargeRows) {
    const dim3 grid((in_dim + kCols - 1) / kCols, (out_dim + kRows - 1) / kRows);
    weight_grad_kernel<false><<<grid, kThreads, 0, stream>>>(
        grad.data_ptr<float>(), x.data_ptr<float>(), result.data_ptr<float>(), rows, out_dim, in_dim);
  } else {
    const dim3 grid((in_dim + kWideCols - 1) / kWideCols, (out_dim + kRows - 1) / kRows);
    weight_grad_kernel<true><<<grid, kThreads, 0, stream>>>(
        grad.data_ptr<float>(), x.data_ptr<float>(), result.data_ptr<float>(), rows, out_dim, in_dim);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return result;
}
