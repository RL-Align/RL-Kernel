// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors

// WS1 - Batch-invariant deterministic GEMM (hand-written, no CUTLASS).
//
// First-milestone naive implementation: one thread computes one output element,
// walking the whole K dimension in a fixed loop order with FP32 accumulation.
// NO split-K, NO shape-based kernel selection -> a row's reduction order is
// independent of the batch (M) dimension, so the output is bitwise-invariant to
// batch size, chunked-prefill splitting, and padding layout.
//
// This is intentionally slow (correctness + invariance first, per #146). A
// tensor-core (mma.sync / ldmatrix) optimization, matching the
// prefix_shared_attention.cu style, is a follow-up within this same file.
//
//   fwd: C = A @ B   |   dA = dC @ B^T   |   dB = A^T @ dC
// Backward reuses the same kernel on transposed operands, so the gradients
// inherit the same invariance.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

namespace {

using nv_bf16 = __nv_bfloat16;

constexpr int TILE = 16;  // 16x16 thread block

__host__ __device__ constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

// C[M,N] = A[M,K] @ B[K,N], all row-major, BF16 in / FP32 accumulate / BF16 out.
// Each thread owns one C[row, col]; the K loop order is fixed and identical for
// every (row, col) regardless of M -> batch-invariant.
__global__ void det_gemm_naive(const nv_bf16* __restrict__ A,
                               const nv_bf16* __restrict__ B,
                               nv_bf16* __restrict__ C,
                               int M, int N, int K) {
  const int row = blockIdx.y * TILE + threadIdx.y;
  const int col = blockIdx.x * TILE + threadIdx.x;
  if (row >= M || col >= N) return;

  float acc = 0.0f;  // FP32 accumulation
  // Fixed ascending K order, no split-K, no atomics. Deterministic.
  for (int k = 0; k < K; ++k) {
    float a = __bfloat162float(A[row * K + k]);
    float b = __bfloat162float(B[k * N + col]);
    acc += a * b;
  }
  C[row * N + col] = __float2bfloat16(acc);
}

void launch_naive(const nv_bf16* A, const nv_bf16* B, nv_bf16* C,
                  int M, int N, int K, cudaStream_t stream) {
  dim3 block(TILE, TILE);
  dim3 grid(cdiv(N, TILE), cdiv(M, TILE));
  det_gemm_naive<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

inline const nv_bf16* bf16(const torch::Tensor& t) {
  return reinterpret_cast<const nv_bf16*>(t.data_ptr<at::BFloat16>());
}
inline nv_bf16* bf16o(torch::Tensor& t) {
  return reinterpret_cast<nv_bf16*>(t.data_ptr<at::BFloat16>());
}
void check_in(const torch::Tensor& t, const char* n) {
  TORCH_CHECK(t.is_cuda(), n, " must be CUDA");
  TORCH_CHECK(t.scalar_type() == torch::kBFloat16, n, " must be bf16");
}

}

// fwd: C = A @ B
torch::Tensor det_gemm_fwd(torch::Tensor a, torch::Tensor b) {
  check_in(a, "A"); check_in(b, "B");
  a = a.contiguous(); b = b.contiguous();
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "det_gemm_fwd: expect 2D [M,K]@[K,N]");
  const int M = a.size(0), K = a.size(1);
  TORCH_CHECK(b.size(0) == K, "det_gemm_fwd: K mismatch");
  const int N = b.size(1);
  auto c = torch::empty({M, N}, a.options());
  launch_naive(bf16(a), bf16(b), bf16o(c), M, N, K,
               at::cuda::getCurrentCUDAStream());
  return c;
}

// dA = dC @ B^T  -> forward GEMM on materialized transpose of B
torch::Tensor det_gemm_da(torch::Tensor dc, torch::Tensor b) {
  check_in(dc, "dC"); check_in(b, "B");
  dc = dc.contiguous();
  auto bt = b.t().contiguous();              // [N, K]
  const int M = dc.size(0), N = dc.size(1), K = bt.size(1);
  TORCH_CHECK(bt.size(0) == N, "det_gemm_da: N mismatch");
  auto da = torch::empty({M, K}, dc.options());  // [M, K]
  launch_naive(bf16(dc), bf16(bt), bf16o(da), M, K, N,
               at::cuda::getCurrentCUDAStream());
  return da;
}

// dB = A^T @ dC  -> forward GEMM on materialized transpose of A
torch::Tensor det_gemm_db(torch::Tensor a, torch::Tensor dc) {
  check_in(a, "A"); check_in(dc, "dC");
  dc = dc.contiguous();
  auto at = a.t().contiguous();              // [K, M]
  const int K = at.size(0), M = at.size(1), N = dc.size(1);
  TORCH_CHECK(dc.size(0) == M, "det_gemm_db: M mismatch");
  auto db = torch::empty({K, N}, a.options());   // [K, N]
  launch_naive(bf16(at), bf16(dc), bf16o(db), K, N, M,
               at::cuda::getCurrentCUDAStream());
  return db;
}
