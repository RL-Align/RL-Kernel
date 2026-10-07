#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace rl_kernel::mhc {

constexpr int64_t kMhcPreHcMult = 4;
constexpr int64_t kMhcPreHiddenSize = 4096;
constexpr int kMhcPreHAggregateDecodeThreads = 1024;
constexpr int kMhcPreHAggregateBatchThreads = 512;
constexpr int kMhcPreHAggregateBackwardThreads = 256;

static_assert(kMhcPreHAggregateBackwardThreads >= kMhcPreHcMult,
              "one backward thread per stream is required for the dPRE fold");
static_assert(kMhcPreHAggregateBackwardThreads <= 1024,
              "MHC H Aggregate backward exceeds the CUDA block limit");

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

__global__ void mhc_pre_h_aggregate_backward_kernel(
    __nv_bfloat16 const* grad_output, __nv_bfloat16 const* residual,
    float const* pre, float* grad_residual, float* grad_pre,
    int64_t hidden_size) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaGridDependencySynchronize();
#endif

  int64_t const token = static_cast<int64_t>(blockIdx.x);
  int64_t const output_offset = token * hidden_size;
  int64_t const residual_offset = token * 4 * hidden_size;
  float const weight_0 = pre[token * 4];
  float const weight_1 = pre[token * 4 + 1];
  float const weight_2 = pre[token * 4 + 2];
  float const weight_3 = pre[token * 4 + 3];

  // dR[i, d] = dH[d] * PRE[i]: every element has one writer, no reduction.
  for (int64_t hidden = threadIdx.x; hidden < hidden_size;
       hidden += blockDim.x) {
    float const dy = __bfloat162float(grad_output[output_offset + hidden]);
    grad_residual[residual_offset + hidden] = __fmul_rn(dy, weight_0);
    grad_residual[residual_offset + hidden_size + hidden] =
        __fmul_rn(dy, weight_1);
    grad_residual[residual_offset + 2 * hidden_size + hidden] =
        __fmul_rn(dy, weight_2);
    grad_residual[residual_offset + 3 * hidden_size + hidden] =
        __fmul_rn(dy, weight_3);
  }

  // dPRE[i] = sum_d dH[d] * R[i, d] with the P1 `fixed_sum` contract: one
  // FP32 accumulator per stream, ascending d, mul then add, no tree. Thread i
  // owns stream i so the four folds run in parallel without changing bytes.
  if (threadIdx.x < kMhcPreHcMult) {
    int64_t const stream_offset = residual_offset + threadIdx.x * hidden_size;
    float acc = 0.0f;
    for (int64_t hidden = 0; hidden < hidden_size; ++hidden) {
      float const dy = __bfloat162float(grad_output[output_offset + hidden]);
      float const r = __bfloat162float(residual[stream_offset + hidden]);
      acc = __fadd_rn(acc, __fmul_rn(dy, r));
    }
    grad_pre[token * 4 + threadIdx.x] = acc;
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
