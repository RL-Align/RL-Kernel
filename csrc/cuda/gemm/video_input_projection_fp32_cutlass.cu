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

// Match the shared tree plan's 32-term leaves. One batched launch writes all
// leaves into their existing workspace nodes; Triton then merges each level.
constexpr int kTreeLeaf = 32;
using LeafGemm = cutlass::gemm::device::GemmArray<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::arch::OpClassSimt, cutlass::arch::Sm50,
    cutlass::gemm::GemmShape<64, 64, 8>,
    cutlass::gemm::GemmShape<32, 32, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    2, 1, 1, cutlass::arch::OpMultiplyAdd>;

template <bool WriteOutputPointers>
__global__ void pack_tree_operand(
    const float* input, float* packed, int64_t* pointers,
    const int32_t* starts, const int32_t* lengths, const int32_t* nodes,
    int64_t rows, int64_t stride_row, int64_t stride_k, int64_t total,
    float* workspace, int64_t* output_pointers, int64_t output_elements) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) return;
  const int64_t leaf_elements = rows * kTreeLeaf;
  const int leaf = static_cast<int>(index / leaf_elements);
  const int64_t local = index % leaf_elements;
  const int64_t row = local / kTreeLeaf;
  const int k = static_cast<int>(local % kTreeLeaf);
  packed[index] = k < lengths[leaf]
      ? input[row * stride_row + (static_cast<int64_t>(starts[leaf]) + k) * stride_k]
      : 0.0f;
  if (local == 0) {
    pointers[leaf] = reinterpret_cast<int64_t>(packed + leaf * leaf_elements);
    if constexpr (WriteOutputPointers) {
      output_pointers[leaf] = reinterpret_cast<int64_t>(
          workspace + static_cast<int64_t>(nodes[leaf]) * output_elements);
    }
  }
}

void cutlass_gemm_fp32_tree_leaves_tn(
    at::Tensor a, at::Tensor bt, at::Tensor workspace,
    at::Tensor starts, at::Tensor lengths, at::Tensor nodes) {
  TORCH_CHECK(a.is_cuda() && a.scalar_type() == at::kFloat && a.dim() == 2,
              "CUTLASS leaves require rank-two CUDA FP32 A");
  const c10::cuda::CUDAGuard guard(a.device());
  const auto* properties = at::cuda::getDeviceProperties(a.get_device());
  TORCH_CHECK(properties->major == 9 && properties->minor == 0,
              "CUTLASS video projection requires CUDA SM90");
  TORCH_CHECK(bt.device() == a.device() && bt.scalar_type() == at::kFloat && bt.dim() == 2,
              "CUTLASS leaves require rank-two FP32 Bt on the same device");
  TORCH_CHECK(a.size(1) == bt.size(1) && a.size(0) > 0 && bt.size(0) > 0,
              "CUTLASS leaf dimensions must match and be non-empty");
  TORCH_CHECK(a.stride(0) > 0 && a.stride(1) > 0 && bt.stride(0) > 0 && bt.stride(1) > 0,
              "CUTLASS leaf operands require positive strides");
  TORCH_CHECK(workspace.device() == a.device() && workspace.scalar_type() == at::kFloat &&
              workspace.dim() == 3 && workspace.is_contiguous() &&
              workspace.size(1) == a.size(0) && workspace.size(2) == bt.size(0),
              "CUTLASS leaves require contiguous FP32 tree workspace");
  for (const auto& metadata : {starts, lengths, nodes}) {
    TORCH_CHECK(metadata.device() == a.device() && metadata.scalar_type() == at::kInt &&
                metadata.dim() == 1 && metadata.is_contiguous() &&
                metadata.numel() == starts.numel(),
                "CUTLASS tree metadata must be matching contiguous CUDA int32 vectors");
  }
  const int64_t leaves = starts.numel(), m = a.size(0), n = bt.size(0);
  TORCH_CHECK(leaves > 0 && leaves <= 65535 && m <= INT_MAX && n <= INT_MAX,
              "CUTLASS batched leaf grid exceeds its supported dimensions");
  auto packed_a = at::empty({leaves, m, kTreeLeaf}, a.options());
  auto packed_b = at::empty({leaves, n, kTreeLeaf}, a.options());
  auto pointers = at::empty({3, leaves}, a.options().dtype(at::kLong));
  auto* ptr_a = pointers.data_ptr<int64_t>();
  auto* ptr_b = ptr_a + leaves;
  auto* ptr_output = ptr_b + leaves;
  const auto stream = c10::cuda::getCurrentCUDAStream(a.get_device()).stream();
  constexpr int threads = 256;
  pack_tree_operand<false><<<(packed_a.numel() + threads - 1) / threads, threads, 0, stream>>>(
      a.data_ptr<float>(), packed_a.data_ptr<float>(), ptr_a,
      starts.data_ptr<int32_t>(), lengths.data_ptr<int32_t>(), nodes.data_ptr<int32_t>(),
      m, a.stride(0), a.stride(1), packed_a.numel(), nullptr, nullptr, m * n);
  pack_tree_operand<true><<<(packed_b.numel() + threads - 1) / threads, threads, 0, stream>>>(
      bt.data_ptr<float>(), packed_b.data_ptr<float>(), ptr_b,
      starts.data_ptr<int32_t>(), lengths.data_ptr<int32_t>(), nodes.data_ptr<int32_t>(),
      n, bt.stride(0), bt.stride(1), packed_b.numel(), workspace.data_ptr<float>(),
      ptr_output, m * n);
  LeafGemm::Arguments args(
      {static_cast<int>(m), static_cast<int>(n), kTreeLeaf},
      reinterpret_cast<const float* const*>(ptr_a), cutlass::layout::RowMajor(kTreeLeaf),
      reinterpret_cast<const float* const*>(ptr_b), cutlass::layout::ColumnMajor(kTreeLeaf),
      reinterpret_cast<const float* const*>(ptr_output), cutlass::layout::RowMajor(n),
      reinterpret_cast<float* const*>(ptr_output), cutlass::layout::RowMajor(n),
      {1.0f, 0.0f}, static_cast<int>(leaves));
  LeafGemm gemm;
  auto status = gemm.can_implement(args);
  TORCH_CHECK(status == cutlass::Status::kSuccess, cutlassGetStatusString(status));
  status = gemm(args, nullptr, stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, cutlassGetStatusString(status));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

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
