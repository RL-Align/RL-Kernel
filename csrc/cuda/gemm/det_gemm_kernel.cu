// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
// csrc/cuda/gemm/det_gemm_kernel.cu
//
// WS1 - Batch-invariant deterministic GEMM (hand-written, no CUTLASS).
//
//   SM90 path: TMA load + mma.sync (m16n8k16) tensor cores, single-CTA-per-tile,
//              fixed K-accumulation order, NO split-K -> batch-invariant.
//   Fallback : naive FP32 scalar kernel (also the correctness ground truth).
//
// Both: BF16 in / FP32 accum / no TF32 / no split-K.
//   fwd: C = A @ B   |   dA = dC @ B^T   |   dB = A^T @ dC
// Backward reuses the forward kernel on transposed operands.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

#if defined(RL_KERNEL_ENABLE_SM90)
#include "../utils/tma_utils.cuh"
#include <cudaTypedefs.h>
#endif

namespace {

using nv_bf16 = __nv_bfloat16;

__host__ __device__ constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

// Naive FP32 scalar kernel (SM80 fallback + ground truth). Batch-invariant by
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
// ============================================================================
// SM90 path: TMA load + mma.sync. C[M,N] = A[M,K] @ B[K,N].
//
// mma.sync.m16n8k16.row.col needs A row-major [m16,k16] and B col-major
// [n8,k16] (i.e. B operand indexed as [n, k]). Our B is row-major [K,N], so we
// load a B tile of shape [BK, BN] but feed the mma the (n,k) operand by
// addressing smem as B[k][n] -> we ldmatrix B with the same trick as the logp
// kernel (which loads W[V,D] = [n, k] directly). To match that validated path,
// A tile is [BM, BK] (row=token, col=k), B tile is [BN, BK] (row=n, col=k),
// which is exactly B^T. So the SM90 kernel computes C = A @ (Bt)^T where Bt is
// the [BN,BK] tile = B[k,n] transposed. We materialize that by giving TMA a
// descriptor over B viewed as [N,K]... but B is [K,N] row-major.
//
// To keep it simple and provably correct, the SM90 forward kernel REQUIRES its
// B operand already in [N,K] layout (row=n, col=k). The host wrapper passes
// B^T (contiguous) so the kernel sees Bt[N,K]; mathematically
// C = A[M,K] @ B[K,N] = A @ (Bt[N,K])^T, and the per-tile mma contracts over K
// in fixed order. This mirrors the logp kernel's W[V,D] @ hidden[N,D] pattern.
// ============================================================================
constexpr int BM = 64;     // rows (M) per CTA tile
constexpr int BN = 64;     // cols (N) per CTA tile
constexpr int BK = 32;     // K slice streamed per TMA load
constexpr int WARPS = 4;
constexpr int WG_THREADS = WARPS * 32;  // 128
constexpr int STAGES = 2;

constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16;
constexpr int WARP_M = BM / WARPS;        // 16 -> 1 m-tile per warp
constexpr int M_TILES = WARP_M / MMA_M;   // 1
constexpr int N_TILES = BN / MMA_N;       // 8
constexpr int K_TILES = BK / MMA_K;       // 2
constexpr int KK_GROUPS = BK / 32;        // 1 (ldmatrix.x4 spans 32 cols)

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

// A: row-major [M,K] via TMA tile [BM,BK].  Bt: row-major [N,K] via TMA tile
// [BN,BK].  C: row-major [M,N].  Each CTA owns one [BM,BN] output tile and
// walks the full K in fixed order (no split-K).
__global__ void det_gemm_sm90_kernel(const __grid_constant__ CUtensorMap a_tmap,
                                     const __grid_constant__ CUtensorMap bt_tmap,
                                     nv_bf16* __restrict__ C,
                                     int M, int N, int K) {
  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int row_base = blockIdx.y * BM;
  const int col_base = blockIdx.x * BN;
  const int kd = K / BK;  // K validated multiple of BK on host

  extern __shared__ __align__(1024) char smem[];
  nv_bf16* sA = reinterpret_cast<nv_bf16*>(smem);
  nv_bf16* sB = reinterpret_cast<nv_bf16*>(sA + STAGES * BM * BK);
  int* mbar_base = reinterpret_cast<int*>(sB + STAGES * BN * BK);

  const uint32_t sA_base = static_cast<uint32_t>(__cvta_generic_to_shared(sA));
  const uint32_t sB_base = static_cast<uint32_t>(__cvta_generic_to_shared(sB));
  int mbar[STAGES];
#pragma unroll
  for (int s = 0; s < STAGES; ++s)
    mbar[s] = static_cast<int>(__cvta_generic_to_shared(mbar_base + 2 * s));

  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < STAGES; ++s) mbarrier_init(mbar[s], 1);
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  __syncthreads();

  const uint32_t tile_bytes = (BM * BK + BN * BK) * sizeof(nv_bf16);

  auto issue_load = [&](int k) {
    const int buf = k % STAGES;
    const int k_off = k * BK;
    tma_2d_g2s(static_cast<int>(sA_base + buf * BM * BK * sizeof(nv_bf16)),
               &a_tmap, k_off, row_base, mbar[buf]);
    tma_2d_g2s(static_cast<int>(sB_base + buf * BN * BK * sizeof(nv_bf16)),
               &bt_tmap, k_off, col_base, mbar[buf]);
    mbarrier_arrive_expect_tx(mbar[buf], tile_bytes);
  };

  int phase[STAGES];
#pragma unroll
  for (int s = 0; s < STAGES; ++s) phase[s] = 0;

  // accumulators: this warp's M_TILES m-tiles x N_TILES n-tiles
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
    mbarrier_wait(mbar[buf], phase[buf]);
    phase[buf] ^= 1;
    __syncthreads();

    const uint32_t sA_buf = sA_base + buf * BM * BK * sizeof(nv_bf16);
    const uint32_t sB_buf = sB_base + buf * BN * BK * sizeof(nv_bf16);

    // Load A operand (this warp's rows), all K-steps. Same addressing as logp.
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

    // Load B operand (all n-tiles) and contract. Same addressing as logp's W.
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

  // Epilogue: write acc to C (row-major [M,N]). mma m16n8k16 output layout.
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

// noswizzle TMA descriptor (kernel uses plain row-major ldmatrix addressing).
inline void init_tmap_noswizzle(CUtensorMap* tmap, const nv_bf16* gmem,
                                uint64_t height, uint64_t width,
                                uint32_t box_h, uint32_t box_w) {
  uint64_t size[2] = {width, height};
  uint64_t stride[1] = {width * sizeof(nv_bf16)};
  uint32_t box[2] = {box_w, box_h};
  uint32_t estride[2] = {1, 1};
  CUresult res = cuTensorMapEncodeTiled(
      tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)gmem, size, stride, box, estride,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(res == CUDA_SUCCESS, "det_gemm: cuTensorMapEncodeTiled failed");
}

// Launch SM90 GEMM. A:[M,K] row-major, Bt:[N,K] row-major (= B transposed).
// Requires M%BM==0, N%BN==0, K%BK==0 (host pads/falls back otherwise).
bool launch_sm90(const nv_bf16* A, const nv_bf16* Bt, nv_bf16* C,
                 int M, int N, int K, cudaStream_t stream) {
  if (M % BM != 0 || N % BN != 0 || K % BK != 0) return false;  // fall back

  CUtensorMap a_tmap, bt_tmap;
  init_tmap_noswizzle(&a_tmap, A, M, K, BM, BK);
  init_tmap_noswizzle(&bt_tmap, Bt, N, K, BN, BK);

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

// Core dispatch: C = A[M,K] @ B[K,N]. SM90 kernel needs B^T ([N,K]); it is
// materialized contiguous here. Falls back to naive on SM80 or odd shapes.
torch::Tensor gemm_dispatch(const torch::Tensor& a, const torch::Tensor& b) {
  const int M = a.size(0), K = a.size(1), N = b.size(1);
  auto c = torch::empty({M, N}, a.options());
  auto stream = at::cuda::getCurrentCUDAStream();

#if defined(RL_KERNEL_ENABLE_SM90)
  if (sm_major() >= 9) {
    auto bt = b.t().contiguous();  // [N,K]
    if (launch_sm90(bf16(a), bf16(bt), bf16o(c), M, N, K, stream)) return c;
    // else fall through to naive (shape not tile-aligned)
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
  auto bt = b.t().contiguous();  // [N,K]; dA[M,K] = dC[M,N] @ bt[N,K]
  TORCH_CHECK(bt.size(0) == dc.size(1), "det_gemm_da: N mismatch");
  return gemm_dispatch(dc, bt);
}

torch::Tensor det_gemm_db(torch::Tensor a, torch::Tensor dc) {
  check_in(a, "A"); check_in(dc, "dC");
  dc = dc.contiguous();
  auto at = a.t().contiguous();  // [K,M]; dB[K,N] = at[K,M] @ dC[M,N]
  TORCH_CHECK(dc.size(0) == at.size(1), "det_gemm_db: M mismatch");
  return gemm_dispatch(at, dc);
}
