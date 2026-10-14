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
