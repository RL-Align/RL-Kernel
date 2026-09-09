// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// P5 CUDA interface skeleton (P5-1 / P5-4).
//
// Implementation steps (strict path):
//   1. Quantize each input row in independent 32-element blocks and write
//      separate E4M3 code bytes and E8M0 scale bytes.
//   2. Keep W packed as E2M1 nibbles; decode only in registers/shared memory.
//   3. Route contiguous row ranges using expert_offsets and skip empty groups.
//   4. Accumulate each 32-wide partial in ascending K order with explicit
//      FP32 multiply/add, then apply (partial * scale_a) * scale_w.
//   5. Implement FC1 and FC2 through the same aligned [M,K] x [E,N,K]
//      contract; backward writes only FP32 dX and never dW.
//
// The strict P5-4 grouped-GEMM kernels (forward + dX-only backward) are
// implemented below. The P5-1 activation-quantization entry point is still
// fail-closed, so a provider cannot silently claim full P5 support.
//
// ABI and address contract:
//   P5-1 input_ptr:              BF16/FP32 [M,K], row-major, K contiguous.
//   P5-1 activation_codes_ptr:   uint8 E4M3 [M,K], row-major, K contiguous.
//   P5-1 activation_scales_ptr:  uint8 E8M0 [M,K/32].
//       activation_scales_ptr[m*(K/32) + (k/32)] scales
//       activation_codes_ptr[m*K + k].
//   P5-4 packed_weight_codes_ptr: uint8 E2M1 [E,N,K/2], K contiguous;
//       low nibble is even k and high nibble is odd k.
//   P5-4 weight_scales_ptr:      uint8 E8M0 [E,N,K/32].
//   P5-4 expert_offsets_ptr:     int32 [E+1], non-decreasing, first 0, last M.
//   codes_ptr and scales_ptr are separate allocations/pointers. Never derive
//   one address from the other, and never write dequantized W to global memory.
//
// Aligned call sites:
//   FC1: A [M,hidden], W1 [E,2*ffn,hidden], Y1 FP32 [M,2*ffn].
//        Y1[:, :ffn] is gate; Y1[:, ffn:] is up.
//   FC2: A [M,ffn], W2 [E,hidden,ffn], Y2 FP32 [M,hidden].
//   Both require K % 32 == 0. M may be 1; empty expert groups are valid.
//
// Numeric profiles:
//   strict: one-row/unpadded geometry, ascending serial reduction, no split-K,
//           no atomics, no fused multiply-add reassociation.
//   WGMMA is deliberately not implemented in this skeleton. A future
//   performance path must use a separate numeric profile and provenance.


#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace {

// Namespace-local names avoid collisions with existing FUSED_LOGP_* macros and
// the unrelated K_TREE_LEAF constant in det_gemm_kernel.cu.
constexpr int kMoeMxBlockSize = 32;
constexpr int kMoeE8m0Bias = 127;
constexpr int kMoeE4m3ExponentMax = 8;
constexpr int kMoeE2m1ExponentMax = 2;
constexpr int kMoeE4m3MaxFinite = 448;
constexpr int kMoeE2m1MaxFinite = 6;

// Single source of truth for the strict numeric-profile fingerprint. Must match
// CudaP5GemmProvider.provenance()["kernel_fingerprint"] in
// rl_engine/moe/provider.py character-for-character (manually synced).
constexpr const char* kMoeGroupedGemmKernelFingerprint =
    "mxfp8-mxfp4-grouped-gemm-strict-v1";

/****************************** launch descriptor structs***********/

struct MoeQuantizeArgs {
  const void* input_ptr;                 // BF16/FP32 [M,K]
  std::uint8_t* activation_codes_ptr;   // E4M3 uint8 [M,K]
  std::uint8_t* activation_scales_ptr;  // E8M0 uint8 [M,K/32]
  int rows_m;
  int input_k;
};

// These descriptors mirror the vLLM grouped-pointer preparation pattern, but
// retain P5's independent code/scale allocations and [E+1] interval offsets.
// They are written on device by the prep kernels and consumed by the strict
// GEMM kernels (the strict path is fully device-side).
struct MoeGroupedGemmExpert {
  int expert_id;
  int row_begin;
  int row_count;
  int output_n;
  int input_k;
  int block_count;
  const std::uint8_t* activation_codes_ptr;   // row_begin * K bytes
  const std::uint8_t* activation_scales_ptr;  // row_begin * (K/32) bytes
  const std::uint8_t* packed_weight_codes_ptr; // e * N * (K/2) bytes
  const std::uint8_t* weight_scales_ptr;      // e * N * (K/32) bytes
  float* output_ptr;                           // row_begin * N floats
  int64_t activation_row_stride;
  int64_t activation_scale_row_stride;
  // Reserved for the future WGMMA per-expert grouped-GEMM path, which strides
  // experts by N*(K/2) / N*(K/32). Unread by the strict one-row kernel — do not
  // delete.
  int64_t weight_expert_stride;
  int64_t weight_scale_expert_stride;
  int64_t output_row_stride;
};

// Backward descriptor: mirrors MoeGroupedGemmExpert's grouped-pointer layout,
// but the backward reads dy/w and writes dX only (no output, no dW).
struct MoeGroupedGemmBwdExpert {
  int expert_id;
  int row_begin;
  int row_count;
  int output_n;
  int input_k;
  int block_count;
  const __nv_bfloat16* dy_ptr;                 // row_begin * N BF16
  const std::uint8_t* packed_weight_codes_ptr; // e * N * (K/2) bytes
  const std::uint8_t* weight_scales_ptr;       // e * N * (K/32) bytes
  float* dx_ptr;                               // row_begin * K floats
  int64_t dy_row_stride;
  int64_t dx_row_stride;
  // Reserved for the future WGMMA per-expert grouped-GEMM path, which strides
  // experts by N*(K/2) / N*(K/32). Unread by the strict one-row kernel — do not
  // delete.
  int64_t weight_expert_stride;
  int64_t weight_scale_expert_stride;
};


//**********************check tools*********************** */
void moe_check_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void moe_check_same_device(const torch::Tensor& first,
                          const torch::Tensor& other,
                          const char* name) {
  TORCH_CHECK(other.device() == first.device(), name,
              " must be on the same device as the first tensor");
}

// P5 offsets live on CUDA. Copying this small metadata vector to CPU is only
// wrapper validation; future launches must keep all data and dequantization on
// device and must not use this path as a numerical implementation.
std::vector<std::int32_t> moe_copy_offsets(const torch::Tensor& offsets) {
  auto host = offsets.to(torch::kCPU).contiguous();
  const auto* data = host.data_ptr<std::int32_t>();
  return std::vector<std::int32_t>(data, data + host.numel());
}

void moe_check_h100_sm90(const torch::Tensor& tensor) {
#if defined(RL_KERNEL_ENABLE_SM90) || defined(KERNEL_ALIGN_WITH_SM90)
  cudaDeviceProp properties{};
  TORCH_CHECK(cudaGetDeviceProperties(&properties, tensor.get_device()) == cudaSuccess,
              "P5-4 failed to query CUDA device properties");
  TORCH_CHECK(properties.major == 9,
              "P5-4 strict wrapper requires an H100/SM90 CUDA device, got compute capability ",
              properties.major, ".", properties.minor, ".");
#else
  (void)tensor;
  TORCH_CHECK(false,
              "P5-4 strict wrapper is not compiled; build the H100/SM90 path first");
#endif
}

// ---------------------------------------------------------------------------
// Device-side expert descriptor preparation. Each expert's per-group pointers
// and strides are computed on device from expert_offsets (read directly, never
// copied to CPU) and written into a POD descriptor array [E]. The strict GEMM
// kernels consume these descriptors instead of re-deriving base pointers.
// ---------------------------------------------------------------------------

constexpr int kMoePrepBlock = 128;

// Forward prep: thread e builds MoeGroupedGemmExpert[e]. The activation/weight
// base pointers sit at row_begin / e in the flat tensors; every offset is
// promoted to int64 to avoid 32-bit overflow.
__global__ void moe_prepare_experts_fwd_device(
    MoeGroupedGemmExpert* __restrict__ experts,      // [E]
    const std::uint8_t* __restrict__ a_codes,        // [M,K]
    const std::uint8_t* __restrict__ a_scales,       // [M,K/32]
    const std::uint8_t* __restrict__ w_codes,        // [E,N,K/2]
    const std::uint8_t* __restrict__ w_scales,       // [E,N,K/32]
    float* __restrict__ out,                         // [M,N]
    const std::int32_t* __restrict__ expert_offsets, // [E+1]
    int M, int N, int K, int E) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= E) return;
  const int row_begin = expert_offsets[e];
  const int row_count = expert_offsets[e + 1] - row_begin;
  const int blocks = K >> 5;
  experts[e] = {
      e,
      row_begin,
      row_count,
      N,
      K,
      blocks,
      a_codes + static_cast<std::int64_t>(row_begin) * K,
      a_scales + static_cast<std::int64_t>(row_begin) * blocks,
      w_codes + static_cast<std::int64_t>(e) * N * (K >> 1),
      w_scales + static_cast<std::int64_t>(e) * N * blocks,
      out + static_cast<std::int64_t>(row_begin) * N,
      static_cast<std::int64_t>(K),
      static_cast<std::int64_t>(blocks),
      static_cast<std::int64_t>(N) * (K >> 1),
      static_cast<std::int64_t>(N) * blocks,
      static_cast<std::int64_t>(N),
  };
  (void)M;  //for wgmma
}

// Backward prep: same layout, but dy/dx replace activation/output. No output
// pointer and no dW (frozen base); the bwd kernel reads dy/w and writes dX.
__global__ void moe_prepare_experts_bwd_device(
    MoeGroupedGemmBwdExpert* __restrict__ experts,   // [E]
    const __nv_bfloat16* __restrict__ dy,            // [M,N]
    const std::uint8_t* __restrict__ w_codes,        // [E,N,K/2]
    const std::uint8_t* __restrict__ w_scales,       // [E,N,K/32]
    float* __restrict__ dx,                          // [M,K]
    const std::int32_t* __restrict__ expert_offsets, // [E+1]
    int M, int N, int K, int E) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= E) return;
  const int row_begin = expert_offsets[e];
  const int row_count = expert_offsets[e + 1] - row_begin;
  const int blocks = K >> 5;
  experts[e] = {
      e,
      row_begin,
      row_count,
      N,
      K,
      blocks,
      dy + static_cast<std::int64_t>(row_begin) * N,
      w_codes + static_cast<std::int64_t>(e) * N * (K >> 1),
      w_scales + static_cast<std::int64_t>(e) * N * blocks,
      dx + static_cast<std::int64_t>(row_begin) * K,
      static_cast<std::int64_t>(N),
      static_cast<std::int64_t>(K),
      static_cast<std::int64_t>(N) * (K >> 1),
      static_cast<std::int64_t>(N) * blocks,
  };
  (void)M;  //for wgmma
}

// Host entry for prep: device guard + current stream + launch the prep kernel.
// Mirrors vLLM's run_get_group_gemm_starts (no alpha / cute layout / TMA assert
// for P5). The caller owns experts_buf (a byte tensor reinterpreted as the
// descriptor type).
void moe_run_prepare_experts_fwd(
    MoeGroupedGemmExpert* experts,
    const torch::Tensor& activation_codes,
    const torch::Tensor& activation_scales,
    const torch::Tensor& packed_weight_codes,
    const torch::Tensor& weight_scales,
    torch::Tensor& output,
    const torch::Tensor& expert_offsets,
    int M, int N, int K, int E) {
  const c10::cuda::OptionalCUDAGuard device_guard(device_of(activation_codes));
  moe_check_h100_sm90(activation_codes);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int grid = (E + kMoePrepBlock - 1) / kMoePrepBlock;
  moe_prepare_experts_fwd_device<<<grid, kMoePrepBlock, 0, stream>>>(
      experts,
      activation_codes.data_ptr<std::uint8_t>(),
      activation_scales.data_ptr<std::uint8_t>(),
      packed_weight_codes.data_ptr<std::uint8_t>(),
      weight_scales.data_ptr<std::uint8_t>(),
      output.data_ptr<float>(),
      expert_offsets.data_ptr<std::int32_t>(),
      M, N, K, E);
}

void moe_run_prepare_experts_bwd(
    MoeGroupedGemmBwdExpert* experts,
    const torch::Tensor& dy,
    const torch::Tensor& packed_weight_codes,
    const torch::Tensor& weight_scales,
    torch::Tensor& dx,
    const torch::Tensor& expert_offsets,
    int M, int N, int K, int E) {
  const c10::cuda::OptionalCUDAGuard device_guard(device_of(dy));
  moe_check_h100_sm90(dy);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int grid = (E + kMoePrepBlock - 1) / kMoePrepBlock;
  moe_prepare_experts_bwd_device<<<grid, kMoePrepBlock, 0, stream>>>(
      experts,
      reinterpret_cast<const __nv_bfloat16*>(dy.data_ptr<at::BFloat16>()),
      packed_weight_codes.data_ptr<std::uint8_t>(),
      weight_scales.data_ptr<std::uint8_t>(),
      dx.data_ptr<float>(),
      expert_offsets.data_ptr<std::int32_t>(),
      M, N, K, E);
}

/*********MXFP8/4 turn fp32 and find expert for per token*********** */

// ---------------------------------------------------------------------------
// P5-4 strict numeric path (bit-exact vs rl_engine/moe/mx_format.py + oracle).
// ---------------------------------------------------------------------------

// OCP E4M3 ("float8_e4m3fn"): 1 sign (bit 7), 4 exp (bits 6..3), 3 mant
// (bits 2..0), bias 7. exp == 0 is the subnormal 2^-6 * mant/8; exp in 1..15
// is 2^(exp-7) * (1 + mant/8). exp == 15 with mant == 7 is NaN, which the
// oracle's satfinite encode never produces, so it is not handled here.
__device__ __forceinline__ float moe_e4m3_to_f32(std::uint8_t code) {
  const std::uint32_t sign = code >> 7;
  const std::uint32_t exp = (code >> 3) & 0xF;
  const std::uint32_t mant = code & 0x7;
  const float frac = static_cast<float>(mant) * (1.0f / 8.0f);
  if (exp == 0) {
    return sign ? -ldexpf(frac, -6) : ldexpf(frac, -6);
  }
  const float value = ldexpf(1.0f + frac, static_cast<int>(exp) - 7);
  return sign ? -value : value;
}

// OCP E2M1 nibble (0..15): magnitude in {0, 0.5, 1, 1.5, 2, 3, 4, 6}, sign in
// bit 3. The decoded value is exact in FP32.
__device__ __forceinline__ float moe_e2m1_nibble_to_f32(std::uint8_t nibble) {
  constexpr float kMag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float mag = kMag[nibble & 0x7];
  return (nibble & 0x8) ? -mag : mag;
}

// OCP E8M0 scale: 2^(code - 127). Code 255 (NaN) is rejected by the wrapper.
__device__ __forceinline__ float moe_e8m0_to_f32(std::uint8_t code) {
  return ldexpf(1.0f, static_cast<int>(code) - 127);
}

// Rows are pre-sorted by expert; expert_offsets is non-decreasing with first 0
// and last M. Returns the expert e such that offsets[e] <= m < offsets[e+1].
// Empty groups are skipped automatically because no row m lands inside them.
__device__ __forceinline__ int moe_find_expert(int m,
                                              const std::int32_t* __restrict__ offsets,
                                              int E) {
  int lo = 0, hi = E - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (offsets[mid] <= m) lo = mid; else hi = mid - 1;
  }
  return lo;
}

/******************fwd and bwd strict compute kernel************/


// Forward: C[m,n] = sum over 32-blocks j of ((sum_k â_k * ŵ_k) * sa_j) * sw_j.
// k ascends within each block; every multiply/add rounds separately via
// __fmul_rn/__fadd_rn (no FMA), matching oracle._block_scaled_dot byte-for-byte.
__global__ void moe_grouped_gemm_fwd_strict(
    const MoeGroupedGemmExpert* __restrict__ experts,  // [E]
    const std::int32_t* __restrict__ expert_offsets,   // [E+1]
    int M, int N, int K, int E) {
  const std::int64_t total = static_cast<std::int64_t>(M) * N;
  const std::int64_t idx =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int m = static_cast<int>(idx / N);
  const int n = static_cast<int>(idx % N);
  const int e = moe_find_expert(m, expert_offsets, E);
  const MoeGroupedGemmExpert& ex = experts[e];
  const int dr = m - ex.row_begin;

  const int blocks = K >> 5;
  const std::uint8_t* a_row =
      ex.activation_codes_ptr + dr * ex.activation_row_stride;
  const std::uint8_t* a_srow =
      ex.activation_scales_ptr + dr * ex.activation_scale_row_stride;
  const std::uint8_t* w_row =
      ex.packed_weight_codes_ptr + static_cast<std::int64_t>(n) * (K >> 1);
  const std::uint8_t* w_srow =
      ex.weight_scales_ptr + static_cast<std::int64_t>(n) * ex.block_count;

  float acc = 0.0f;
  for (int j = 0; j < blocks; ++j) {
    float partial = 0.0f;
#pragma unroll
    for (int kk = 0; kk < 32; ++kk) {
      const int k = (j << 5) + kk;
      const float av = moe_e4m3_to_f32(a_row[k]);
      const std::uint8_t nib = (w_row[k >> 1] >> ((k & 1) << 2)) & 0xF;
      partial = __fadd_rn(partial, __fmul_rn(av, moe_e2m1_nibble_to_f32(nib)));
    }
    const float sc = __fmul_rn(__fmul_rn(partial, moe_e8m0_to_f32(a_srow[j])),
                               moe_e8m0_to_f32(w_srow[j]));
    acc = __fadd_rn(acc, sc);
  }
  ex.output_ptr[dr * ex.output_row_stride + n] = acc;
}

// Backward: dX[m,k] = sum_n dy[m,n] * bf16(w_decoded[e,n,k]), n ascending,
// FP32 accumulate. W is dequantized to BF16 exactly as oracle w_full.to(bf16).
__global__ void moe_grouped_gemm_bwd_strict(
    const MoeGroupedGemmBwdExpert* __restrict__ experts,  // [E]
    const std::int32_t* __restrict__ expert_offsets,      // [E+1]
    int M, int N, int K, int E) {
  const std::int64_t total = static_cast<std::int64_t>(M) * K;
  const std::int64_t idx =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int m = static_cast<int>(idx / K);
  const int k = static_cast<int>(idx % K);
  const int e = moe_find_expert(m, expert_offsets, E);
  const MoeGroupedGemmBwdExpert& ex = experts[e];
  const int dr = m - ex.row_begin;

  const int blocks = K >> 5;
  const int b = k >> 5;
  const int half_k = k >> 1;
  const int nib_shift = (k & 1) << 2;
  const std::uint8_t* w_codes_e = ex.packed_weight_codes_ptr;
  const std::uint8_t* w_scales_e = ex.weight_scales_ptr;
  const __nv_bfloat16* dy_row = ex.dy_ptr + dr * ex.dy_row_stride;

  float acc = 0.0f;
  for (int n = 0; n < N; ++n) {
    const std::uint8_t nib =
        (w_codes_e[static_cast<std::int64_t>(n) * (K >> 1) + half_k] >> nib_shift) & 0xF;
    const float wfull =
        __fmul_rn(moe_e2m1_nibble_to_f32(nib),
                  moe_e8m0_to_f32(w_scales_e[static_cast<std::int64_t>(n) * blocks + b]));
    const float wbf = __bfloat162float(__float2bfloat16(wfull));
    acc = __fadd_rn(acc, __fmul_rn(__bfloat162float(dy_row[n]), wbf));
  }
  ex.dx_ptr[dr * ex.dx_row_stride + k] = acc;
}

constexpr int kMoeStrictBlock = 128;

/*****************launch fwd and bwd strict compute kernel*********/

void moe_launch_grouped_gemm_fwd_strict(
    const MoeGroupedGemmExpert* experts, const std::int32_t* expert_offsets,
    int M, int N, int K, int E, cudaStream_t stream) {
  const std::int64_t total = static_cast<std::int64_t>(M) * N;
  if (total == 0) return;
  const int grid = static_cast<int>((total + kMoeStrictBlock - 1) / kMoeStrictBlock);
  moe_grouped_gemm_fwd_strict<<<grid, kMoeStrictBlock, 0, stream>>>(
      experts, expert_offsets, M, N, K, E);
}

void moe_launch_grouped_gemm_bwd_strict(
    const MoeGroupedGemmBwdExpert* experts, const std::int32_t* expert_offsets,
    int M, int N, int K, int E, cudaStream_t stream) {
  const std::int64_t total = static_cast<std::int64_t>(M) * K;
  if (total == 0) return;
  const int grid = static_cast<int>((total + kMoeStrictBlock - 1) / kMoeStrictBlock);
  moe_grouped_gemm_bwd_strict<<<grid, kMoeStrictBlock, 0, stream>>>(
      experts, expert_offsets, M, N, K, E);
}

}  // namespace



torch::Tensor moe_mxfp8_mxfp4_grouped_gemm_forward(
    torch::Tensor activation_codes,
    torch::Tensor activation_scales,
    torch::Tensor packed_weight_codes,
    torch::Tensor weight_scales,
    torch::Tensor expert_offsets) {
  moe_check_cuda_contiguous(activation_codes, "P5-4 activation_codes");
  moe_check_cuda_contiguous(activation_scales, "P5-4 activation_scales");
  moe_check_cuda_contiguous(packed_weight_codes, "P5-4 packed_weight_codes");
  moe_check_cuda_contiguous(weight_scales, "P5-4 weight_scales");
  moe_check_cuda_contiguous(expert_offsets, "P5-4 expert_offsets");
  moe_check_same_device(activation_codes, activation_scales, "activation_scales");
  moe_check_same_device(activation_codes, packed_weight_codes, "packed_weight_codes");
  moe_check_same_device(activation_codes, weight_scales, "weight_scales");
  moe_check_same_device(activation_codes, expert_offsets, "expert_offsets");
  TORCH_CHECK(activation_codes.scalar_type() == torch::kUInt8,
              "P5-4 activation_codes must be uint8 E4M3 codes");
  TORCH_CHECK(activation_scales.scalar_type() == torch::kUInt8,
              "P5-4 activation_scales must be uint8 E8M0 codes");
  TORCH_CHECK(packed_weight_codes.scalar_type() == torch::kUInt8,
              "P5-4 packed_weight_codes must be uint8 E2M1x2 codes");
  TORCH_CHECK(weight_scales.scalar_type() == torch::kUInt8,
              "P5-4 weight_scales must be uint8 E8M0 codes");
  TORCH_CHECK(expert_offsets.scalar_type() == torch::kInt32,
              "P5-4 expert_offsets must be int32");
  TORCH_CHECK(activation_codes.dim() == 2,
              "P5-4 activation_codes must be [M,K]");
  TORCH_CHECK(activation_scales.dim() == 2,
              "P5-4 activation_scales must be [M,K/32]");
  TORCH_CHECK(packed_weight_codes.dim() == 3,
              "P5-4 packed_weight_codes must be [E,N,K/2]");
  TORCH_CHECK(weight_scales.dim() == 3,
              "P5-4 weight_scales must be [E,N,K/32]");
  TORCH_CHECK(expert_offsets.dim() == 1,
              "P5-4 expert_offsets must be [E+1]");

  const auto M = activation_codes.size(0);
  const auto K = activation_codes.size(1);
  const auto E = packed_weight_codes.size(0);
  const auto N = packed_weight_codes.size(1);
  TORCH_CHECK(K > 0 && K % kMoeMxBlockSize == 0,
              "P5-4 K must be positive and divisible by 32");
  TORCH_CHECK(activation_scales.size(0) == M &&
                  activation_scales.size(1) == K / kMoeMxBlockSize,
              "P5-4 activation_scales must have shape [M,K/32]");
  TORCH_CHECK(packed_weight_codes.size(2) == K / 2,
              "P5-4 packed_weight_codes must have shape [E,N,K/2]");
  TORCH_CHECK(weight_scales.size(0) == E && weight_scales.size(1) == N &&
                  weight_scales.size(2) == K / kMoeMxBlockSize,
              "P5-4 weight_scales must have shape [E,N,K/32]");
  TORCH_CHECK(expert_offsets.size(0) == E + 1,
              "P5-4 expert_offsets must have E+1 entries");

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(activation_codes));
  const auto offsets = moe_copy_offsets(expert_offsets);
  TORCH_CHECK(offsets.front() == 0,
              "P5-4 expert_offsets must start at 0");
  TORCH_CHECK(offsets.back() == M,
              "P5-4 expert_offsets must end at M");
  for (int64_t e = 0; e < E; ++e) {
    TORCH_CHECK(offsets[e] <= offsets[e + 1],
                "P5-4 expert_offsets must be non-decreasing");
  }

  auto output = torch::empty({M, N}, activation_codes.options().dtype(torch::kFloat32));

  // Allocate the device descriptor array [E], fill it on device via the prep
  // kernel, then hand it (plus expert_offsets) to the strict GEMM launch.
  const auto experts_bytes =
      static_cast<int64_t>(E) * static_cast<int64_t>(sizeof(MoeGroupedGemmExpert));
  auto experts_buf =
      torch::empty({experts_bytes}, activation_codes.options().dtype(torch::kByte));
  auto* experts =
      reinterpret_cast<MoeGroupedGemmExpert*>(experts_buf.data_ptr<std::uint8_t>());

  moe_run_prepare_experts_fwd(
      experts, activation_codes, activation_scales, packed_weight_codes,
      weight_scales, output, expert_offsets,
      static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
      static_cast<int>(E));

  auto stream = at::cuda::getCurrentCUDAStream();
  moe_launch_grouped_gemm_fwd_strict(
      experts, expert_offsets.data_ptr<std::int32_t>(),
      static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
      static_cast<int>(E), stream);
  return output;
}

torch::Tensor moe_mxfp8_mxfp4_grouped_gemm_backward(
    torch::Tensor dy,
    torch::Tensor packed_weight_codes,
    torch::Tensor weight_scales,
    torch::Tensor expert_offsets) {
  moe_check_cuda_contiguous(dy, "P5-4 dy");
  moe_check_cuda_contiguous(packed_weight_codes, "P5-4 packed_weight_codes");
  moe_check_cuda_contiguous(weight_scales, "P5-4 weight_scales");
  moe_check_cuda_contiguous(expert_offsets, "P5-4 expert_offsets");
  moe_check_same_device(dy, packed_weight_codes, "packed_weight_codes");
  moe_check_same_device(dy, weight_scales, "weight_scales");
  moe_check_same_device(dy, expert_offsets, "expert_offsets");
  TORCH_CHECK(dy.scalar_type() == torch::kBFloat16,
              "P5-4 dy must be BF16 [M,N]");
  TORCH_CHECK(packed_weight_codes.scalar_type() == torch::kUInt8,
              "P5-4 packed_weight_codes must be uint8 E2M1x2 codes");
  TORCH_CHECK(weight_scales.scalar_type() == torch::kUInt8,
              "P5-4 weight_scales must be uint8 E8M0 codes");
  TORCH_CHECK(expert_offsets.scalar_type() == torch::kInt32,
              "P5-4 expert_offsets must be int32");
  TORCH_CHECK(dy.dim() == 2, "P5-4 dy must be [M,N]");
  TORCH_CHECK(packed_weight_codes.dim() == 3,
              "P5-4 packed_weight_codes must be [E,N,K/2]");
  TORCH_CHECK(weight_scales.dim() == 3,
              "P5-4 weight_scales must be [E,N,K/32]");
  TORCH_CHECK(expert_offsets.dim() == 1,
              "P5-4 expert_offsets must be [E+1]");

  const auto M = dy.size(0);
  const auto N = dy.size(1);
  const auto E = packed_weight_codes.size(0);
  const auto packed_k = packed_weight_codes.size(2);
  const auto K = packed_k * 2;
  TORCH_CHECK(K > 0 && K % kMoeMxBlockSize == 0,
              "P5-4 K must be positive and divisible by 32");
  TORCH_CHECK(packed_weight_codes.size(1) == N,
              "P5-4 dy N must match packed weight N");
  TORCH_CHECK(weight_scales.size(0) == E && weight_scales.size(1) == N &&
                  weight_scales.size(2) == K / kMoeMxBlockSize,
              "P5-4 weight_scales must have shape [E,N,K/32]");
  TORCH_CHECK(expert_offsets.size(0) == E + 1,
              "P5-4 expert_offsets must have E+1 entries");

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(dy));
  const auto offsets = moe_copy_offsets(expert_offsets);
  TORCH_CHECK(offsets.front() == 0,
              "P5-4 expert_offsets must start at 0");
  TORCH_CHECK(offsets.back() == M,
              "P5-4 expert_offsets must end at M");
  for (int64_t e = 0; e < E; ++e) {
    TORCH_CHECK(offsets[e] <= offsets[e + 1],
                "P5-4 expert_offsets must be non-decreasing");
  }

  auto dx = torch::empty({M, K}, dy.options().dtype(torch::kFloat32));

  // Allocate the device descriptor array [E], fill it on device via the prep
  // kernel, then hand it (plus expert_offsets) to the strict GEMM launch.
  const auto experts_bytes =
      static_cast<int64_t>(E) * static_cast<int64_t>(sizeof(MoeGroupedGemmBwdExpert));
  auto experts_buf =
      torch::empty({experts_bytes}, dy.options().dtype(torch::kByte));
  auto* experts =
      reinterpret_cast<MoeGroupedGemmBwdExpert*>(experts_buf.data_ptr<std::uint8_t>());

  moe_run_prepare_experts_bwd(
      experts, dy, packed_weight_codes, weight_scales, dx, expert_offsets,
      static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
      static_cast<int>(E));

  auto stream = at::cuda::getCurrentCUDAStream();
  moe_launch_grouped_gemm_bwd_strict(
      experts, expert_offsets.data_ptr<std::int32_t>(),
      static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
      static_cast<int>(E), stream);
  return dx;
}
