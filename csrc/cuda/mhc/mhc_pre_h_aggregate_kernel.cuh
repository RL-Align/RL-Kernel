#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace rl_kernel::mhc {

constexpr int64_t kMhcPreHcMult = 4;
constexpr int64_t kMhcPreHiddenSize = 4096;
constexpr int kMhcWarpSize = 32;
constexpr int kMhcPreHAggregateDecodeThreads = 1024;
constexpr int kMhcPreHAggregateBatchThreads = 512;
constexpr int kMhcPreHAggregateBackwardThreads = 256;
constexpr int kMhcPreHAggregateBackwardWarps =
    kMhcPreHAggregateBackwardThreads / kMhcWarpSize;

static_assert(kMhcPreHAggregateBackwardThreads % kMhcWarpSize == 0,
              "MHC H Aggregate backward requires complete warps");
static_assert(kMhcPreHiddenSize % kMhcPreHAggregateBackwardThreads == 0,
              "each backward thread must reduce a fixed number of elements");
static_assert(kMhcPreHAggregateBackwardThreads <= 1024,
              "MHC H Aggregate backward exceeds the CUDA block limit");
static_assert(kMhcPreHAggregateBackwardWarps <= kMhcWarpSize,
              "warp 0 must be able to reduce all per-warp partial sums");

__global__ void mhc_pre_h_aggregate_kernel(__nv_bfloat16 const* residual,
                                           float const* pre,
                                           __nv_bfloat16* output,
                                           int64_t hidden_size) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaGridDependencySynchronize();
#endif

  __shared__ float weights[4];
  if (threadIdx.x < 4) {
    weights[threadIdx.x] =
        pre[static_cast<int64_t>(blockIdx.x) * 4 + threadIdx.x];
  }
  __syncthreads();

  int64_t const token_offset =
      static_cast<int64_t>(blockIdx.x) * 4 * hidden_size;
  int64_t const output_offset = static_cast<int64_t>(blockIdx.x) * hidden_size;
  if ((hidden_size & 1) == 0) {
    auto const* residual_pairs =
        reinterpret_cast<__nv_bfloat162 const*>(residual + token_offset);
    auto* output_pairs =
        reinterpret_cast<__nv_bfloat162*>(output + output_offset);
    int64_t const pair_count = hidden_size / 2;
    for (int64_t hidden_pair = threadIdx.x; hidden_pair < pair_count;
         hidden_pair += blockDim.x) {
      // store two bf16 to a 32-bit reg
      float2 const value_0 = __bfloat1622float2(residual_pairs[hidden_pair]);
      float2 const value_1 =
          __bfloat1622float2(residual_pairs[pair_count + hidden_pair]);
      float2 const value_2 =
          __bfloat1622float2(residual_pairs[2 * pair_count + hidden_pair]);
      float2 const value_3 =
          __bfloat1622float2(residual_pairs[3 * pair_count + hidden_pair]);
      float2 result;
      float const left_x = __fadd_rn(__fmul_rn(weights[0], value_0.x),
                                     __fmul_rn(weights[1], value_1.x));
      float const right_x = __fadd_rn(__fmul_rn(weights[2], value_2.x),
                                      __fmul_rn(weights[3], value_3.x));
      result.x = __fadd_rn(left_x, right_x);
      float const left_y = __fadd_rn(__fmul_rn(weights[0], value_0.y),
                                     __fmul_rn(weights[1], value_1.y));
      float const right_y = __fadd_rn(__fmul_rn(weights[2], value_2.y),
                                      __fmul_rn(weights[3], value_3.y));
      result.y = __fadd_rn(left_y, right_y);
      output_pairs[hidden_pair] = __floats2bfloat162_rn(result.x, result.y);
    }
  } else {
    for (int64_t hidden = threadIdx.x; hidden < hidden_size;
         hidden += blockDim.x) {
      float const product_0 = __fmul_rn(
          weights[0], __bfloat162float(residual[token_offset + hidden]));
      float const product_1 = __fmul_rn(
          weights[1],
          __bfloat162float(residual[token_offset + hidden_size + hidden]));
      float const product_2 = __fmul_rn(
          weights[2], __bfloat162float(
                          residual[token_offset + 2 * hidden_size + hidden]));
      float const product_3 = __fmul_rn(
          weights[3], __bfloat162float(
                          residual[token_offset + 3 * hidden_size + hidden]));
      float const left = __fadd_rn(product_0, product_1);
      float const right = __fadd_rn(product_2, product_3);
      output[output_offset + hidden] = __float2bfloat16_rn(__fadd_rn(left, right));
    }
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

__device__ __forceinline__ float mhc_warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = __fadd_rn(value, __shfl_down_sync(0xffffffff, value, offset));
  }
  return value;
}

__global__ void mhc_pre_h_aggregate_backward_kernel(
    __nv_bfloat16 const* grad_output, __nv_bfloat16 const* residual,
    float const* pre, float* grad_residual, float* grad_pre,
    int64_t hidden_size) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaGridDependencySynchronize();
#endif

  __shared__ float
      warp_sums[kMhcPreHcMult][kMhcPreHAggregateBackwardWarps];

  int64_t const token = static_cast<int64_t>(blockIdx.x);
  int64_t const output_offset = token * hidden_size;
  int64_t const residual_offset = token * 4 * hidden_size;
  float const weight_0 = pre[token * 4];
  float const weight_1 = pre[token * 4 + 1];
  float const weight_2 = pre[token * 4 + 2];
  float const weight_3 = pre[token * 4 + 3];
  float sum_0 = 0.0f;
  float sum_1 = 0.0f;
  float sum_2 = 0.0f;
  float sum_3 = 0.0f;

  for (int64_t hidden = threadIdx.x; hidden < hidden_size;
       hidden += blockDim.x) {
    float const dy = __bfloat162float(grad_output[output_offset + hidden]);
    float const residual_0 =
        __bfloat162float(residual[residual_offset + hidden]);
    float const residual_1 =
        __bfloat162float(residual[residual_offset + hidden_size + hidden]);
    float const residual_2 = __bfloat162float(
        residual[residual_offset + 2 * hidden_size + hidden]);
    float const residual_3 = __bfloat162float(
        residual[residual_offset + 3 * hidden_size + hidden]);

    sum_0 = __fadd_rn(sum_0, __fmul_rn(dy, residual_0));
    sum_1 = __fadd_rn(sum_1, __fmul_rn(dy, residual_1));
    sum_2 = __fadd_rn(sum_2, __fmul_rn(dy, residual_2));
    sum_3 = __fadd_rn(sum_3, __fmul_rn(dy, residual_3));

    grad_residual[residual_offset + hidden] = __fmul_rn(dy, weight_0);
    grad_residual[residual_offset + hidden_size + hidden] =
        __fmul_rn(dy, weight_1);
    grad_residual[residual_offset + 2 * hidden_size + hidden] =
        __fmul_rn(dy, weight_2);
    grad_residual[residual_offset + 3 * hidden_size + hidden] =
        __fmul_rn(dy, weight_3);
  }

  int const lane = threadIdx.x & (kMhcWarpSize - 1);
  int const warp = threadIdx.x / kMhcWarpSize;
  sum_0 = mhc_warp_sum(sum_0);
  sum_1 = mhc_warp_sum(sum_1);
  sum_2 = mhc_warp_sum(sum_2);
  sum_3 = mhc_warp_sum(sum_3);
  if (lane == 0) {
    warp_sums[0][warp] = sum_0;
    warp_sums[1][warp] = sum_1;
    warp_sums[2][warp] = sum_2;
    warp_sums[3][warp] = sum_3;
  }
  __syncthreads();

  if (warp == 0) {
    sum_0 = lane < kMhcPreHAggregateBackwardWarps ? warp_sums[0][lane]
                                                   : 0.0f;
    sum_1 = lane < kMhcPreHAggregateBackwardWarps ? warp_sums[1][lane]
                                                   : 0.0f;
    sum_2 = lane < kMhcPreHAggregateBackwardWarps ? warp_sums[2][lane]
                                                   : 0.0f;
    sum_3 = lane < kMhcPreHAggregateBackwardWarps ? warp_sums[3][lane]
                                                   : 0.0f;
    sum_0 = mhc_warp_sum(sum_0);
    sum_1 = mhc_warp_sum(sum_1);
    sum_2 = mhc_warp_sum(sum_2);
    sum_3 = mhc_warp_sum(sum_3);
    if (lane == 0) {
      grad_pre[token * 4] = sum_0;
      grad_pre[token * 4 + 1] = sum_1;
      grad_pre[token * 4 + 2] = sum_2;
      grad_pre[token * 4 + 3] = sum_3;
    }
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

inline cudaError_t launch_mhc_pre_h_aggregate(
    __nv_bfloat16 const* residual, float const* pre, __nv_bfloat16* output,
    int64_t num_tokens, int64_t hidden_size, cudaStream_t stream,
    bool enable_pdl) {
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(static_cast<unsigned int>(num_tokens));
  config.blockDim = dim3(num_tokens <= 128 ? kMhcPreHAggregateDecodeThreads
                                           : kMhcPreHAggregateBatchThreads);
  config.stream = stream;

  cudaLaunchAttribute attribute{};
  if (enable_pdl) {
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = 1;
    config.attrs = &attribute;
    config.numAttrs = 1;
  }

  return cudaLaunchKernelEx(&config, mhc_pre_h_aggregate_kernel, residual, pre,
                            output, hidden_size);
}

inline cudaError_t launch_mhc_pre_h_aggregate_backward(
    __nv_bfloat16 const* grad_output, __nv_bfloat16 const* residual,
    float const* pre, float* grad_residual, float* grad_pre,
    int64_t num_tokens, int64_t hidden_size, cudaStream_t stream,
    bool enable_pdl) {
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(static_cast<unsigned int>(num_tokens));
  config.blockDim = dim3(kMhcPreHAggregateBackwardThreads);
  config.stream = stream;

  cudaLaunchAttribute attribute{};
  if (enable_pdl) {
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = 1;
    config.attrs = &attribute;
    config.numAttrs = 1;
  }

  return cudaLaunchKernelEx(&config, mhc_pre_h_aggregate_backward_kernel,
                            grad_output, residual, pre, grad_residual, grad_pre,
                            hidden_size);
}

}
