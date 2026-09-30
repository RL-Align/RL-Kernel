// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors

// Forward and backward for issue #386 using fixed 256-key tiles. Each logical
// row stays on one CTA: tiles use a fixed reduction tree, then merge strictly
// from left to right with the online-softmax recurrence. The exponential uses
// the same fixed FP32 sequence as the CPU reference instead of platform libm.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cmath>
#include <vector>

namespace {

constexpr int kTileK = 256;

__device__ __forceinline__ float portable_exp_nonpositive(float value) {
  if (isnan(value)) {
    return value;
  }
  if (value < -104.0f) {
    return 0.0f;
  }

  const float scaled =
      __fadd_rn(__fmul_rn(value, __int_as_float(0x3fb8aa3b)), 0.5f);
  const int exponent = __float2int_rd(scaled);
  const float exponent_fp32 = __int2float_rn(exponent);
  float remainder = __fsub_rn(
      value, __fmul_rn(exponent_fp32, __int_as_float(0x3f317200)));
  remainder = __fsub_rn(
      remainder, __fmul_rn(exponent_fp32, __int_as_float(0x35bfbe8e)));

  float polynomial = __int_as_float(0x39500d01);
  polynomial = __fadd_rn(
      __fmul_rn(polynomial, remainder), __int_as_float(0x3ab60b61));
  polynomial = __fadd_rn(
      __fmul_rn(polynomial, remainder), __int_as_float(0x3c088889));
  polynomial = __fadd_rn(
      __fmul_rn(polynomial, remainder), __int_as_float(0x3d2aaaab));
  polynomial = __fadd_rn(
      __fmul_rn(polynomial, remainder), __int_as_float(0x3e2aaaab));
  polynomial = __fadd_rn(
      __fmul_rn(polynomial, remainder), __int_as_float(0x3f000000));
  polynomial = __fadd_rn(
      __fmul_rn(polynomial, remainder), __int_as_float(0x3f800000));
  polynomial = __fadd_rn(
      __fmul_rn(polynomial, remainder), __int_as_float(0x3f800000));

  if (exponent >= -126) {
    const float scale = __int_as_float((exponent + 127) << 23);
    return __fmul_rn(polynomial, scale);
  }

  const float scale = __int_as_float((exponent + 64 + 127) << 23);
  return __fmul_rn(
      __fmul_rn(polynomial, scale), __int_as_float(0x1f800000));
}

template <typename input_t, typename output_t>
__global__ void joint_attn_softmax_forward_kernel(
    const input_t* __restrict__ scores,
    output_t* __restrict__ probabilities,
    float* __restrict__ saved_probabilities,
    int64_t key_length) {
  const int64_t row_index = blockIdx.x;
  const int tid = threadIdx.x;
  const int64_t row_offset = row_index * key_length;

  __shared__ float reduction[kTileK];
  __shared__ float online_max;
  __shared__ float online_sum;

  if (tid == 0) {
    online_max = -INFINITY;
    online_sum = 0.0f;
  }

  for (int64_t tile_start = 0; tile_start < key_length;
       tile_start += kTileK) {
    const int64_t column = tile_start + tid;
    const float score = column < key_length
        ? static_cast<float>(scores[row_offset + column])
        : -INFINITY;

    reduction[tid] = score;
    __syncthreads();

    for (int stride = kTileK / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
      }
      __syncthreads();
    }
    const float tile_max = reduction[0];
    // Finish every warp's read before reusing reduction for the sum tree.
    __syncthreads();

    const float exp_value = column < key_length
        ? (score == -INFINITY
               ? 0.0f
               : portable_exp_nonpositive(__fsub_rn(score, tile_max)))
        : 0.0f;
    reduction[tid] = exp_value;
    __syncthreads();

    for (int stride = kTileK / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        reduction[tid] = __fadd_rn(reduction[tid], reduction[tid + stride]);
      }
      __syncthreads();
    }
    const float tile_sum = reduction[0];

    if (tid == 0 && tile_max != -INFINITY) {
      if (online_max == -INFINITY) {
        online_max = tile_max;
        online_sum = tile_sum;
      } else {
        const float new_max = fmaxf(online_max, tile_max);
        const float old_scale =
            portable_exp_nonpositive(__fsub_rn(online_max, new_max));
        const float tile_scale =
            portable_exp_nonpositive(__fsub_rn(tile_max, new_max));
        const float scaled_old = __fmul_rn(online_sum, old_scale);
        const float scaled_tile = __fmul_rn(tile_sum, tile_scale);
        online_max = new_max;
        online_sum = __fadd_rn(scaled_old, scaled_tile);
      }
    }
    __syncthreads();
  }

  const float row_max = online_max;
  const float row_sum = online_sum;
  for (int64_t column = tid; column < key_length;
       column += kTileK) {
    const float exp_value =
        portable_exp_nonpositive(
            __fsub_rn(
                static_cast<float>(scores[row_offset + column]), row_max));
    const float probability = __fdiv_rn(exp_value, row_sum);
    probabilities[row_offset + column] = static_cast<output_t>(probability);
    if (saved_probabilities != nullptr) {
      saved_probabilities[row_offset + column] = probability;
    }
  }
}

template <typename grad_t, typename output_t>
__global__ void joint_attn_softmax_backward_kernel(
    const float* __restrict__ probabilities,
    const grad_t* __restrict__ grad_probabilities,
    output_t* __restrict__ grad_scores,
    int64_t key_length) {
  const int64_t row_index = blockIdx.x;
  const int tid = threadIdx.x;
  const int64_t row_offset = row_index * key_length;

  __shared__ float reduction[kTileK];
  __shared__ float row_delta;

  for (int64_t tile_start = 0; tile_start < key_length;
       tile_start += kTileK) {
    const int64_t column = tile_start + tid;
    const float product = column < key_length
        ? __fmul_rn(
              probabilities[row_offset + column],
              static_cast<float>(grad_probabilities[row_offset + column]))
        : 0.0f;
    reduction[tid] = product;
    __syncthreads();

    for (int stride = kTileK / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        reduction[tid] = __fadd_rn(reduction[tid], reduction[tid + stride]);
      }
      __syncthreads();
    }

    if (tid == 0) {
      row_delta = tile_start == 0
          ? reduction[0]
          : __fadd_rn(row_delta, reduction[0]);
    }
    __syncthreads();
  }

  const float delta = row_delta;
  for (int64_t column = tid; column < key_length;
       column += kTileK) {
    const float centered =
        __fsub_rn(
            static_cast<float>(grad_probabilities[row_offset + column]), delta);
    grad_scores[row_offset + column] =
        static_cast<output_t>(
            __fmul_rn(probabilities[row_offset + column], centered));
  }
}

int64_t validate_joint_attn_softmax_scores(const torch::Tensor& scores) {
  TORCH_CHECK(scores.is_cuda(), "joint_attn_softmax: scores must be a CUDA tensor");
  TORCH_CHECK(scores.is_contiguous(), "joint_attn_softmax: scores must be contiguous");
  TORCH_CHECK(
      scores.scalar_type() == at::kFloat ||
          scores.scalar_type() == at::kBFloat16,
      "joint_attn_softmax: scores must use BF16 or FP32");
  TORCH_CHECK(scores.dim() >= 1, "joint_attn_softmax: scores must have shape [..., K]");

  const int64_t key_length = scores.size(-1);
  TORCH_CHECK(key_length > 0, "joint_attn_softmax: K must be non-empty");
  return key_length;
}

}  // namespace

torch::Tensor joint_attn_softmax_forward_fp32(torch::Tensor scores) {
  const int64_t key_length = validate_joint_attn_softmax_scores(scores);

  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(scores));
  auto probabilities = torch::empty(
      scores.sizes(), scores.options().dtype(at::kFloat));
  const int64_t row_count = scores.numel() / key_length;
  if (row_count == 0) {
    return probabilities;
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  if (scores.scalar_type() == at::kFloat) {
    joint_attn_softmax_forward_kernel<float, float>
        <<<row_count, kTileK, 0, stream>>>(
            scores.data_ptr<float>(),
            probabilities.data_ptr<float>(),
            nullptr,
            key_length);
  } else {
    joint_attn_softmax_forward_kernel<at::BFloat16, float>
        <<<row_count, kTileK, 0, stream>>>(
            scores.data_ptr<at::BFloat16>(),
            probabilities.data_ptr<float>(),
            nullptr,
            key_length);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return probabilities;
}

torch::Tensor joint_attn_softmax_forward(torch::Tensor scores) {
  if (scores.scalar_type() == at::kFloat) {
    return joint_attn_softmax_forward_fp32(scores);
  }
  const int64_t key_length = validate_joint_attn_softmax_scores(scores);

  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(scores));
  auto probabilities = torch::empty_like(scores);
  const int64_t row_count = scores.numel() / key_length;
  if (row_count == 0) {
    return probabilities;
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  joint_attn_softmax_forward_kernel<at::BFloat16, at::BFloat16>
      <<<row_count, kTileK, 0, stream>>>(
          scores.data_ptr<at::BFloat16>(),
          probabilities.data_ptr<at::BFloat16>(),
          nullptr,
          key_length);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return probabilities;
}

std::vector<torch::Tensor> joint_attn_softmax_forward_with_state(
    torch::Tensor scores) {
  if (scores.scalar_type() == at::kFloat) {
    auto probabilities = joint_attn_softmax_forward_fp32(scores);
    return {probabilities, probabilities};
  }
  const int64_t key_length = validate_joint_attn_softmax_scores(scores);

  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(scores));
  auto probabilities = torch::empty_like(scores);
  auto saved_probabilities = torch::empty(
      scores.sizes(), scores.options().dtype(at::kFloat));
  const int64_t row_count = scores.numel() / key_length;
  if (row_count == 0) {
    return {probabilities, saved_probabilities};
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  joint_attn_softmax_forward_kernel<at::BFloat16, at::BFloat16>
      <<<row_count, kTileK, 0, stream>>>(
          scores.data_ptr<at::BFloat16>(),
          probabilities.data_ptr<at::BFloat16>(),
          saved_probabilities.data_ptr<float>(),
          key_length);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {probabilities, saved_probabilities};
}

torch::Tensor joint_attn_softmax_backward(
    torch::Tensor probabilities,
    torch::Tensor grad_probabilities,
    bool output_bf16) {
  TORCH_CHECK(
      probabilities.is_cuda() && grad_probabilities.is_cuda(),
      "joint_attn_softmax backward: tensors must be on CUDA");
  TORCH_CHECK(
      probabilities.device() == grad_probabilities.device(),
      "joint_attn_softmax backward: tensors must share a device");
  TORCH_CHECK(
      probabilities.is_contiguous() && grad_probabilities.is_contiguous(),
      "joint_attn_softmax backward: tensors must be contiguous");
  TORCH_CHECK(
      probabilities.scalar_type() == at::kFloat,
      "joint_attn_softmax backward: saved probabilities must use FP32");
  TORCH_CHECK(
      grad_probabilities.scalar_type() == at::kFloat ||
          grad_probabilities.scalar_type() == at::kBFloat16,
      "joint_attn_softmax backward: output gradients must use BF16 or FP32");
  TORCH_CHECK(
      probabilities.sizes() == grad_probabilities.sizes(),
      "joint_attn_softmax backward: tensor shapes must match");
  TORCH_CHECK(
      probabilities.dim() >= 1,
      "joint_attn_softmax backward: tensors must have shape [..., K]");

  const int64_t key_length = probabilities.size(-1);
  TORCH_CHECK(key_length > 0, "joint_attn_softmax backward: K must be non-empty");

  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(probabilities));
  auto grad_scores = torch::empty(
      probabilities.sizes(),
      probabilities.options().dtype(output_bf16 ? at::kBFloat16 : at::kFloat));
  const int64_t row_count = probabilities.numel() / key_length;
  if (row_count == 0) {
    return grad_scores;
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  if (grad_probabilities.scalar_type() == at::kFloat && !output_bf16) {
    joint_attn_softmax_backward_kernel<float, float>
        <<<row_count, kTileK, 0, stream>>>(
            probabilities.data_ptr<float>(),
            grad_probabilities.data_ptr<float>(),
            grad_scores.data_ptr<float>(),
            key_length);
  } else if (grad_probabilities.scalar_type() == at::kFloat) {
    joint_attn_softmax_backward_kernel<float, at::BFloat16>
        <<<row_count, kTileK, 0, stream>>>(
            probabilities.data_ptr<float>(),
            grad_probabilities.data_ptr<float>(),
            grad_scores.data_ptr<at::BFloat16>(),
            key_length);
  } else if (output_bf16) {
    joint_attn_softmax_backward_kernel<at::BFloat16, at::BFloat16>
        <<<row_count, kTileK, 0, stream>>>(
            probabilities.data_ptr<float>(),
            grad_probabilities.data_ptr<at::BFloat16>(),
            grad_scores.data_ptr<at::BFloat16>(),
            key_length);
  } else {
    joint_attn_softmax_backward_kernel<at::BFloat16, float>
        <<<row_count, kTileK, 0, stream>>>(
            probabilities.data_ptr<float>(),
            grad_probabilities.data_ptr<at::BFloat16>(),
            grad_scores.data_ptr<float>(),
            key_length);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_scores;
}
