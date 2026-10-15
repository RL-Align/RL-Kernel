// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors

// Reuse CUTLASS device::Gemm at revision 56ca28adb32ffadd097566545b8f7d83b652e998.
// The dependency's BSD-3-Clause notice is packaged in the Python wrapper.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <climits>

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_array.h>

using Gemm = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::arch::OpClassSimt, cutlass::arch::Sm50,
    cutlass::gemm::GemmShape<64, 64, 8>,
    cutlass::gemm::GemmShape<32, 32, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2, 1, 1, false, cutlass::arch::OpMultiplyAdd>;

constexpr int kTreeLeaf = 32;

at::Tensor cutlass_gemm_fp32_tn(at::Tensor a, at::Tensor b_transposed) {
  TORCH_CHECK(a.is_cuda() && b_transposed.is_cuda(), "CUTLASS GEMM requires CUDA tensors");
  TORCH_CHECK(a.device() == b_transposed.device(), "CUTLASS GEMM devices must match");
  TORCH_CHECK(a.scalar_type() == at::kFloat && b_transposed.scalar_type() == at::kFloat,
              "CUTLASS GEMM requires FP32");
  TORCH_CHECK(a.dim() == 2 && b_transposed.dim() == 2,
              "CUTLASS GEMM requires rank-two matrices");
  TORCH_CHECK(a.is_contiguous() && b_transposed.is_contiguous(),
              "CUTLASS GEMM matrices must be contiguous");
  TORCH_CHECK(a.size(1) == b_transposed.size(1), "CUTLASS GEMM K dimensions must match");
  const c10::cuda::CUDAGuard guard(a.device());
  const auto* properties = at::cuda::getDeviceProperties(a.get_device());
  TORCH_CHECK(properties->major == 9 && properties->minor == 0,
              "CUTLASS video projection requires CUDA SM90");
  const auto m64 = a.size(0), n64 = b_transposed.size(0), k64 = a.size(1);
  TORCH_CHECK(m64 > 0 && n64 > 0 && k64 > 0 &&
              m64 <= INT_MAX && n64 <= INT_MAX && k64 <= INT_MAX,
              "CUTLASS GEMM dimensions must be positive int32 values");
  const int m = static_cast<int>(m64), n = static_cast<int>(n64), k = static_cast<int>(k64);
  auto output = at::empty({m, n}, a.options());
  Gemm::Arguments args({m, n, k}, {a.data_ptr<float>(), k},
                       {b_transposed.data_ptr<float>(), k},
                       {output.data_ptr<float>(), n}, {output.data_ptr<float>(), n},
                       {1.0f, 0.0f}, 1);
  Gemm gemm;
  auto status = gemm.can_implement(args);
  TORCH_CHECK(status == cutlass::Status::kSuccess, cutlassGetStatusString(status));
  status = gemm(args, nullptr, c10::cuda::getCurrentCUDAStream(a.get_device()).stream());
  TORCH_CHECK(status == cutlass::Status::kSuccess, cutlassGetStatusString(status));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

int cutlass_cuda_version() { return CUDART_VERSION; }

template <typename LayoutA, typename LayoutB>
using StridedLeafGemm = cutlass::gemm::device::GemmArray<
    float, LayoutA, float, LayoutB, float, cutlass::layout::RowMajor,
    float, cutlass::arch::OpClassSimt, cutlass::arch::Sm50,
    cutlass::gemm::GemmShape<64, 64, 8>,
    cutlass::gemm::GemmShape<32, 32, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    2, 1, 1, cutlass::arch::OpMultiplyAdd>;

__global__ void tree_leaf_pointers(
    const float* a, const float* bt, float* workspace, int64_t* pointers,
    const int32_t* starts, const int32_t* nodes, const int32_t* leaf_ids,
    int64_t count, int64_t stride_ak, int64_t stride_btk, int64_t elements) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const int leaf = leaf_ids[index];
  const int64_t start = starts[leaf];
  pointers[index] = reinterpret_cast<int64_t>(a + start * stride_ak);
  pointers[count + index] = reinterpret_cast<int64_t>(bt + start * stride_btk);
  pointers[2 * count + index] = reinterpret_cast<int64_t>(
      workspace + static_cast<int64_t>(nodes[leaf]) * elements);
}

template <typename LayoutA, typename LayoutB>
void launch_strided_leaves(
    const at::Tensor& a, const at::Tensor& bt, const at::Tensor& pointers,
    int64_t count, int length, int64_t lda, int64_t ldb, cudaStream_t stream) {
  using Leaf = StridedLeafGemm<LayoutA, LayoutB>;
  const auto* addresses = pointers.data_ptr<int64_t>();
  typename Leaf::Arguments args(
      {static_cast<int>(a.size(0)), static_cast<int>(bt.size(0)), length},
      reinterpret_cast<const float* const*>(addresses), LayoutA(lda),
      reinterpret_cast<const float* const*>(addresses + count), LayoutB(ldb),
      reinterpret_cast<const float* const*>(addresses + 2 * count),
      cutlass::layout::RowMajor(bt.size(0)),
      reinterpret_cast<float* const*>(addresses + 2 * count),
      cutlass::layout::RowMajor(bt.size(0)), {1.0f, 0.0f}, static_cast<int>(count));
  Leaf gemm;
  auto status = gemm.can_implement(args);
  TORCH_CHECK(status == cutlass::Status::kSuccess, cutlassGetStatusString(status));
  status = gemm(args, nullptr, stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, cutlassGetStatusString(status));
}

void cutlass_gemm_fp32_tree_leaves_strided(
    at::Tensor a, at::Tensor bt, at::Tensor workspace, at::Tensor starts,
    at::Tensor nodes, at::Tensor leaf_ids, int64_t length) {
  TORCH_CHECK(a.is_cuda() && a.scalar_type() == at::kFloat && a.dim() == 2,
              "CUTLASS leaves require rank-two CUDA FP32 A");
  const c10::cuda::CUDAGuard guard(a.device());
  const auto* properties = at::cuda::getDeviceProperties(a.get_device());
  TORCH_CHECK(properties->major == 9 && properties->minor == 0,
              "CUTLASS video projection requires CUDA SM90");
  TORCH_CHECK(bt.device() == a.device() && bt.scalar_type() == at::kFloat &&
              bt.dim() == 2 && a.size(1) == bt.size(1),
              "CUTLASS leaf dimensions, dtypes and devices must match");
  TORCH_CHECK(a.size(0) > 0 && bt.size(0) > 0 && length > 0 && length <= kTreeLeaf &&
              a.size(0) <= INT_MAX && bt.size(0) <= INT_MAX,
              "CUTLASS leaf sizes exceed the supported dimensions");
  TORCH_CHECK(workspace.device() == a.device() && workspace.scalar_type() == at::kFloat &&
              workspace.dim() == 3 && workspace.is_contiguous() &&
              workspace.size(1) == a.size(0) && workspace.size(2) == bt.size(0),
              "CUTLASS leaves require contiguous FP32 tree workspace");
  for (const auto& metadata : {starts, nodes, leaf_ids}) {
    TORCH_CHECK(metadata.device() == a.device() && metadata.scalar_type() == at::kInt &&
                metadata.dim() == 1 && metadata.is_contiguous(),
                "CUTLASS leaf metadata must be contiguous CUDA int32 vectors");
  }
  TORCH_CHECK(starts.numel() == nodes.numel() && leaf_ids.numel() > 0 &&
              leaf_ids.numel() <= 65535,
              "CUTLASS leaf metadata sizes are inconsistent");
  const bool a_row = a.stride(1) == 1;
  const bool b_column = bt.stride(1) == 1;
  TORCH_CHECK((a_row || a.stride(0) == 1) &&
              (b_column || bt.stride(0) == 1) &&
              a.stride(0) > 0 && a.stride(1) > 0 &&
              bt.stride(0) > 0 && bt.stride(1) > 0,
              "CUTLASS leaves require a row-major or column-major view");
  const int64_t lda = a_row ? a.stride(0) : a.stride(1);
  const int64_t ldb = b_column ? bt.stride(0) : bt.stride(1);
  TORCH_CHECK(lda <= INT_MAX && ldb <= INT_MAX,
              "CUTLASS leaf leading dimensions exceed int32");
  const int64_t count = leaf_ids.numel();
  auto pointers = at::empty({3, count}, a.options().dtype(at::kLong));
  const auto stream = c10::cuda::getCurrentCUDAStream(a.get_device()).stream();
  tree_leaf_pointers<<<(count + 255) / 256, 256, 0, stream>>>(
      a.data_ptr<float>(), bt.data_ptr<float>(), workspace.data_ptr<float>(),
      pointers.data_ptr<int64_t>(), starts.data_ptr<int32_t>(), nodes.data_ptr<int32_t>(),
      leaf_ids.data_ptr<int32_t>(), count, a.stride(1), bt.stride(1),
      a.size(0) * bt.size(0));
  using Row = cutlass::layout::RowMajor;
  using Column = cutlass::layout::ColumnMajor;
  if (a_row && b_column) {
    launch_strided_leaves<Row, Column>(a, bt, pointers, count, length, lda, ldb, stream);
  } else if (a_row) {
    launch_strided_leaves<Row, Row>(a, bt, pointers, count, length, lda, ldb, stream);
  } else if (b_column) {
    launch_strided_leaves<Column, Column>(a, bt, pointers, count, length, lda, ldb, stream);
  } else {
    launch_strided_leaves<Column, Row>(a, bt, pointers, count, length, lda, ldb, stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
