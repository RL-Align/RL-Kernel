// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
// csrc/cuda/gemm/det_gemm_kernel.cu
//
// WS1 Batch-invariant deterministic GEMM (hand-written, no CUTLASS).
//
//   SM90 path : TMA load + mma.sync (m16n8k16), FP32 accum, single-CTA-per-tile,
//               fixed K order, NO split-K -> batch-invariant.
//   Fallback  : naive FP32 scalar kernel (also the correctness ground truth).
//
// Both: BF16 in / FP32 accum / no TF32 / no split-K.
//   fwd: C = A @ B   |   dA = dC @ B^T   |   dB = A^T @ dC
// Backward reuses the forward kernel on transposed operands.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

#if defined(RL_KERNEL_ENABLE_SM90)
#include "det_gemm_tma.cuh"
#endif

namespace {

using nv_bf16 = __nv_bfloat16;

__host__ __device__ constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

// Naive FP32 scalar kernel (fallback + ground truth). Batch-invariant by
// construction: one thread = one output element, fixed ascending K loop.
constexpr int NAIVE_TILE = 16;

__global__ void det_gemm_naive(const nv_bf16* __restrict__ A,
                               const nv_bf16* __restrict__ B,
                               nv_bf16* __restrict__ C,
                               int M, int N, int K) {
  const int row = blockIdx.y * NAIVE_TILE + threadIdx.y;
  const int col = blockIdx.x * NAIVE_TILE + threadIdx.x;
  if (row >= M || col >= N) return;
  float acc = 0.0f;
  for (int k = 0; k < K; ++k)
    acc += __bfloat162float(A[row * K + k]) * __bfloat162float(B[k * N + col]);
  C[row * N + col] = __float2bfloat16(acc);
}

void launch_naive(const nv_bf16* A, const nv_bf16* B, nv_bf16* C,
                  int M, int N, int K, cudaStream_t stream) {
  dim3 block(NAIVE_TILE, NAIVE_TILE);
  dim3 grid(cdiv(N, NAIVE_TILE), cdiv(M, NAIVE_TILE));
  det_gemm_naive<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

#if defined(RL_KERNEL_ENABLE_SM90)
// SM90 path: TMA load + mma.sync. C[M,N] = A[M,K] @ B[K,N].
// Each CTA owns one [BM,BN] output tile, walks full K in fixed order (no
// split-K). A tile [BM,BK] row-major; B operand col-major [n,k] supplied by
// passing B^T ([N,K] row-major) so the B smem tile is [BN,BK] (row=n,col=k),
// matching the validated logp ldmatrix addressing.
constexpr int BM = 128, BN = 64, BK = 32;
constexpr int WARPS = 4;
constexpr int WG_THREADS = WARPS * 32;  // 128
constexpr int STAGES = 2;

constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16;
constexpr int WARP_M = BM / WARPS;        // 16
constexpr int M_TILES = WARP_M / MMA_M;   // 1
constexpr int N_TILES = BN / MMA_N;       // 8
constexpr int K_TILES = BK / MMA_K;       // 2
constexpr int KK_GROUPS = BK / 32;        // 1

__device__ __forceinline__ void ldmatrix_x4(uint32_t regs[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
               : "r"(addr));
}
__device__ __forceinline__ void mma_m16n8k16(const uint32_t A[4], const uint32_t B[2], float D[4]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
               : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
               : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
                 "f"(D[0]), "f"(D[1]), "f"(D[2]), "f"(D[3]));
}

__global__ void det_gemm_sm90_kernel(const __grid_constant__ CUtensorMap a_tmap,
                                     const __grid_constant__ CUtensorMap bt_tmap,
                                     nv_bf16* __restrict__ C,
                                     int M, int N, int K) {
  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int row_base = blockIdx.y * BM;
  const int col_base = blockIdx.x * BN;
  const int kd = K / BK;

  extern __shared__ __align__(1024) char smem[];
  nv_bf16* sA = reinterpret_cast<nv_bf16*>(smem);
  nv_bf16* sB = reinterpret_cast<nv_bf16*>(sA + STAGES * BM * BK);
  int* mbar_base = reinterpret_cast<int*>(sB + STAGES * BN * BK);

  const uint32_t sA_base = static_cast<uint32_t>(__cvta_generic_to_shared(sA));
  const uint32_t sB_base = static_cast<uint32_t>(__cvta_generic_to_shared(sB));
  uint32_t mbar[STAGES];
#pragma unroll
  for (int s = 0; s < STAGES; ++s)
    mbar[s] = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_base + 2 * s));

  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < STAGES; ++s) det_gemm::mbar_init(mbar[s], 1);
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  __syncthreads();

  const uint32_t tile_bytes = (BM * BK + BN * BK) * sizeof(nv_bf16);

  auto issue_load = [&](int k) {
    const int buf = k % STAGES;
    const int koff = k * BK;
    det_gemm::tma_2d_g2s(sA_base + buf * BM * BK * sizeof(nv_bf16), &a_tmap, koff, row_base, mbar[buf]);
    det_gemm::tma_2d_g2s(sB_base + buf * BN * BK * sizeof(nv_bf16), &bt_tmap, koff, col_base, mbar[buf]);
    det_gemm::mbar_arrive_expect_tx(mbar[buf], tile_bytes);
  };

  int phase[STAGES];
#pragma unroll
  for (int s = 0; s < STAGES; ++s) phase[s] = 0;

  float acc[M_TILES][N_TILES][4];
#pragma unroll
  for (int mi = 0; mi < M_TILES; ++mi)
#pragma unroll
    for (int n = 0; n < N_TILES; ++n)
      acc[mi][n][0] = acc[mi][n][1] = acc[mi][n][2] = acc[mi][n][3] = 0.0f;

  if (tid == 0)
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s)
      if (s < kd) issue_load(s);

  for (int k = 0; k < kd; ++k) {       // fixed ascending K order, NO split-K
    const int buf = k % STAGES;
    if (tid == 0 && k + (STAGES - 1) < kd) issue_load(k + (STAGES - 1));
    det_gemm::mbar_wait(mbar[buf], phase[buf]);
    phase[buf] ^= 1;
    __syncthreads();

    const uint32_t sA_buf = sA_base + buf * BM * BK * sizeof(nv_bf16);
    const uint32_t sB_buf = sB_base + buf * BN * BK * sizeof(nv_bf16);

    uint32_t A[M_TILES][K_TILES][4];
#pragma unroll
    for (int mi = 0; mi < M_TILES; ++mi) {
      const int row0 = warp * WARP_M + mi * MMA_M + (lane % 16);
#pragma unroll
      for (int kt = 0; kt < K_TILES; ++kt) {
        const uint32_t a_addr =
            sA_buf + (row0 * BK + (lane / 16) * 8 + kt * MMA_K) * sizeof(nv_bf16);
        ldmatrix_x4(A[mi][kt], a_addr);
      }
    }

#pragma unroll
    for (int n = 0; n < N_TILES; ++n) {
#pragma unroll
      for (int kk = 0; kk < KK_GROUPS; ++kk) {
        uint32_t b4[4];
        const uint32_t b_addr =
            sB_buf + ((n * MMA_N + (lane % 8)) * BK + (lane / 8) * 8 + kk * 32) * sizeof(nv_bf16);
        ldmatrix_x4(b4, b_addr);
        const uint32_t B0[2] = {b4[0], b4[1]};
        const uint32_t B1[2] = {b4[2], b4[3]};
#pragma unroll
        for (int mi = 0; mi < M_TILES; ++mi) {
          mma_m16n8k16(A[mi][2 * kk + 0], B0, acc[mi][n]);
          mma_m16n8k16(A[mi][2 * kk + 1], B1, acc[mi][n]);
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int mi = 0; mi < M_TILES; ++mi) {
    const int row = row_base + warp * WARP_M + mi * MMA_M + lane / 4;
#pragma unroll
    for (int n = 0; n < N_TILES; ++n) {
      const int col = col_base + n * MMA_N + (lane % 4) * 2;
      if (row < M && col + 1 < N) {
        C[row * N + col + 0] = __float2bfloat16(acc[mi][n][0]);
        C[row * N + col + 1] = __float2bfloat16(acc[mi][n][1]);
      }
      if (row + 8 < M && col + 1 < N) {
        C[(row + 8) * N + col + 0] = __float2bfloat16(acc[mi][n][2]);
        C[(row + 8) * N + col + 1] = __float2bfloat16(acc[mi][n][3]);
      }
    }
  }
}

bool launch_sm90(const nv_bf16* A, const nv_bf16* Bt, nv_bf16* C,
                 int M, int N, int K, cudaStream_t stream) {
  if (M % BM != 0 || N % BN != 0 || K % BK != 0) return false;  // fall back

  CUtensorMap a_tmap, bt_tmap;
  det_gemm::init_tmap_noswizzle(&a_tmap, A, M, K, BM, BK);
  det_gemm::init_tmap_noswizzle(&bt_tmap, Bt, N, K, BN, BK);

  const int smem = STAGES * (BM * BK + BN * BK) * sizeof(nv_bf16) + STAGES * 8;
  if (smem > 48 * 1024)
    cudaFuncSetAttribute(det_gemm_sm90_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

  dim3 grid(cdiv(N, BN), cdiv(M, BM));
  det_gemm_sm90_kernel<<<grid, WG_THREADS, smem, stream>>>(a_tmap, bt_tmap, C, M, N, K);
  return true;
}
#endif  // RL_KERNEL_ENABLE_SM90

int sm_major() {
  int dev = 0; cudaGetDevice(&dev);
  cudaDeviceProp p{}; cudaGetDeviceProperties(&p, dev);
  return p.major;
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

torch::Tensor gemm_dispatch(const torch::Tensor& a, const torch::Tensor& b) {
  const int M = a.size(0), K = a.size(1), N = b.size(1);
  auto c = torch::empty({M, N}, a.options());
  auto stream = at::cuda::getCurrentCUDAStream();

#if defined(RL_KERNEL_ENABLE_SM90)
  // Tensor-core path requires N,K tile-aligned. M is padded up to a multiple of
  // BM so that EVERY M (including M=1 and non-aligned M) takes the SAME kernel.
  // Selecting a different kernel based on M would itself break batch-invariance
  // because M is the batch dimension.
  if (sm_major() >= 9 && K % BK == 0 && N % BN == 0) {
    const int Mp = cdiv(M, BM) * BM;
    torch::Tensor a_use = a;
    if (Mp != M) {
      a_use = torch::zeros({Mp, K}, a.options());
      a_use.narrow(0, 0, M).copy_(a);
    }
    torch::Tensor c_use = (Mp != M) ? torch::empty({Mp, N}, a.options()) : c;
    auto bt = b.t().contiguous();  // [N,K]
    if (launch_sm90(bf16(a_use), bf16(bt), bf16o(c_use), Mp, N, K, stream)) {
      if (Mp != M) c.copy_(c_use.narrow(0, 0, M));
      return c;
    }
  }
#endif
  launch_naive(bf16(a), bf16(b), bf16o(c), M, N, K, stream);
  return c;
}

}  // anonymous namespace

torch::Tensor det_gemm_fwd(torch::Tensor a, torch::Tensor b) {
  check_in(a, "A"); check_in(b, "B");
  a = a.contiguous(); b = b.contiguous();
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "det_gemm_fwd: expect 2D [M,K]@[K,N]");
  TORCH_CHECK(b.size(0) == a.size(1), "det_gemm_fwd: K mismatch");
  return gemm_dispatch(a, b);
}

torch::Tensor det_gemm_da(torch::Tensor dc, torch::Tensor b) {
  check_in(dc, "dC"); check_in(b, "B");
  dc = dc.contiguous();
  auto bt = b.t().contiguous();
  TORCH_CHECK(bt.size(0) == dc.size(1), "det_gemm_da: N mismatch");
  return gemm_dispatch(dc, bt);
}

torch::Tensor det_gemm_db(torch::Tensor a, torch::Tensor dc) {
  check_in(a, "A"); check_in(dc, "dC");
  dc = dc.contiguous();
  auto at = a.t().contiguous();
  TORCH_CHECK(dc.size(0) == at.size(1), "det_gemm_db: M mismatch");
  return gemm_dispatch(at, dc);
}
