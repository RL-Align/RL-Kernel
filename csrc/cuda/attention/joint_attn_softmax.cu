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
#include <limits>
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

template <typename input_t, typename output_t, bool has_key_layout>
__global__ void joint_attn_softmax_forward_kernel(
    const input_t* __restrict__ scores,
    output_t* __restrict__ probabilities,
    float* __restrict__ saved_probabilities,
    const int32_t* __restrict__ logical_to_physical,
    const int32_t* __restrict__ valid_key_counts,
    int64_t rows_per_batch,
    int64_t key_length) {
  const int64_t row_index = blockIdx.x;
  const int tid = threadIdx.x;
  const int64_t row_offset = row_index * key_length;
  int64_t key_map_offset = 0;
  int64_t valid_key_count = key_length;
  if constexpr (has_key_layout) {
    const int64_t batch_index = row_index / rows_per_batch;
    key_map_offset = batch_index * key_length;
    valid_key_count = valid_key_counts[batch_index];
    for (int64_t physical_column = tid; physical_column < key_length;
         physical_column += kTileK) {
      probabilities[row_offset + physical_column] = static_cast<output_t>(0.0f);
      if (saved_probabilities != nullptr) {
        saved_probabilities[row_offset + physical_column] = 0.0f;
      }
    }
    __syncthreads();
  }

  __shared__ float reduction[kTileK];
  __shared__ float online_max;
  __shared__ float online_sum;

  if (tid == 0) {
    online_max = -INFINITY;
    online_sum = 0.0f;
  }

  for (int64_t tile_start = 0; tile_start < key_length;
       tile_start += kTileK) {
    const int64_t logical_column = tile_start + tid;
    const bool valid = logical_column < valid_key_count;
    const int64_t physical_column = has_key_layout && valid
        ? logical_to_physical[key_map_offset + logical_column]
        : logical_column;
    const float score = valid
        ? static_cast<float>(scores[row_offset + physical_column])
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

    const float exp_value = valid
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
    // A NaN sum marks unsupported NaN/+inf input without changing the finite
    // max tree.
    const float tile_state_max = isnan(tile_sum) ? NAN : tile_max;

    if (tid == 0 && tile_state_max != -INFINITY) {
      if (online_max == -INFINITY) {
        online_max = tile_state_max;
        online_sum = tile_sum;
      } else {
        const float new_max = fmaxf(online_max, tile_state_max);
        const float old_scale =
            portable_exp_nonpositive(__fsub_rn(online_max, new_max));
        const float tile_scale =
            portable_exp_nonpositive(__fsub_rn(tile_state_max, new_max));
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
  for (int64_t logical_column = tid; logical_column < valid_key_count;
       logical_column += kTileK) {
    const int64_t physical_column = has_key_layout
        ? logical_to_physical[key_map_offset + logical_column]
        : logical_column;
    float probability = 0.0f;
    if (row_sum != 0.0f) {
      const float exp_value =
          portable_exp_nonpositive(
              __fsub_rn(
                  static_cast<float>(scores[row_offset + physical_column]), row_max));
      probability = __fdiv_rn(exp_value, row_sum);
    }
    probabilities[row_offset + physical_column] = static_cast<output_t>(probability);
    if (saved_probabilities != nullptr) {
      saved_probabilities[row_offset + physical_column] = probability;
    }
  }
}

template <typename grad_t, typename output_t, bool has_key_layout>
__global__ void joint_attn_softmax_backward_kernel(
    const float* __restrict__ probabilities,
    const grad_t* __restrict__ grad_probabilities,
    output_t* __restrict__ grad_scores,
    const int32_t* __restrict__ logical_to_physical,
    const int32_t* __restrict__ valid_key_counts,
    int64_t rows_per_batch,
    int64_t key_length) {
  const int64_t row_index = blockIdx.x;
  const int tid = threadIdx.x;
  const int64_t row_offset = row_index * key_length;
  int64_t key_map_offset = 0;
  int64_t valid_key_count = key_length;
  if constexpr (has_key_layout) {
    const int64_t batch_index = row_index / rows_per_batch;
    key_map_offset = batch_index * key_length;
    valid_key_count = valid_key_counts[batch_index];
    for (int64_t physical_column = tid; physical_column < key_length;
         physical_column += kTileK) {
      grad_scores[row_offset + physical_column] = static_cast<output_t>(0.0f);
    }
    __syncthreads();
  }

  __shared__ float reduction[kTileK];
  __shared__ float row_delta;

  for (int64_t tile_start = 0; tile_start < key_length;
       tile_start += kTileK) {
    const int64_t logical_column = tile_start + tid;
    const bool valid = logical_column < valid_key_count;
    const int64_t physical_column = has_key_layout && valid
        ? logical_to_physical[key_map_offset + logical_column]
        : logical_column;
    const float product = valid
        ? __fmul_rn(
              probabilities[row_offset + physical_column],
              static_cast<float>(grad_probabilities[row_offset + physical_column]))
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
  for (int64_t logical_column = tid; logical_column < valid_key_count;
       logical_column += kTileK) {
    const int64_t physical_column = has_key_layout
        ? logical_to_physical[key_map_offset + logical_column]
        : logical_column;
    const float probability = probabilities[row_offset + physical_column];
    const float centered =
        __fsub_rn(
            static_cast<float>(grad_probabilities[row_offset + physical_column]), delta);
    const float raw_grad_score = __fmul_rn(probability, centered);
    // Padding can change the sign of an exact zero; always store +0.
    const float grad_score = raw_grad_score == 0.0f ? 0.0f : raw_grad_score;
    grad_scores[row_offset + physical_column] = static_cast<output_t>(grad_score);
  }
}

__global__ void joint_attn_softmax_key_mapping_kernel(
    const bool* __restrict__ key_padding_mask,
    int32_t* __restrict__ logical_to_physical,
    int32_t* __restrict__ valid_key_counts,
    int64_t key_length) {
  const int64_t batch_index = blockIdx.x;
  const int tid = threadIdx.x;
  const int64_t batch_offset = batch_index * key_length;

  __shared__ int32_t scan[kTileK];
  __shared__ int32_t next_logical_key;
  if (tid == 0) {
    next_logical_key = 0;
  }
  __syncthreads();

  for (int64_t tile_start = 0; tile_start < key_length;
       tile_start += kTileK) {
    const int64_t physical_key = tile_start + tid;
    const int32_t is_valid = physical_key < key_length &&
            key_padding_mask[batch_offset + physical_key]
        ? 1
        : 0;
    scan[tid] = is_valid;
    __syncthreads();

    for (int offset = 1; offset < kTileK; offset <<= 1) {
      const int32_t addend = tid >= offset ? scan[tid - offset] : 0;
      __syncthreads();
      if (tid >= offset) {
        scan[tid] += addend;
      }
      __syncthreads();
    }

    if (is_valid != 0) {
      logical_to_physical[
          batch_offset + next_logical_key + scan[tid] - 1] =
          static_cast<int32_t>(physical_key);
    }
    __syncthreads();
    if (tid == 0) {
      next_logical_key += scan[kTileK - 1];
    }
    __syncthreads();
  }

  if (tid == 0) {
    valid_key_counts[batch_index] = next_logical_key;
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

struct KeyLayoutView {
  bool enabled;
  const int32_t* logical_to_physical;
  const int32_t* valid_key_counts;
  int64_t rows_per_batch;
};

KeyLayoutView validate_key_layout(
    const torch::Tensor& values,
    const torch::optional<torch::Tensor>& logical_to_physical,
    const torch::optional<torch::Tensor>& valid_key_counts,
    int64_t rows_per_batch) {
  const bool has_indices = logical_to_physical.has_value();
  const bool has_counts = valid_key_counts.has_value();
  TORCH_CHECK(
      has_indices == has_counts,
      "joint_attn_softmax: key indices and counts must be provided together");
  if (!has_indices) {
    return {false, nullptr, nullptr, 1};
  }

  const auto& indices = logical_to_physical.value();
  const auto& counts = valid_key_counts.value();
  const int64_t key_length = values.size(-1);
  TORCH_CHECK(
      values.dim() >= 2,
      "joint_attn_softmax: key layout requires scores with shape [B, ..., K]");
  TORCH_CHECK(
      indices.is_cuda() && counts.is_cuda(),
      "joint_attn_softmax: key layout tensors must be on CUDA");
  TORCH_CHECK(
      indices.device() == values.device() && counts.device() == values.device(),
      "joint_attn_softmax: key layout tensors must share the scores device");
  TORCH_CHECK(
      indices.is_contiguous() && counts.is_contiguous(),
      "joint_attn_softmax: key layout tensors must be contiguous");
  TORCH_CHECK(
      indices.scalar_type() == at::kInt && counts.scalar_type() == at::kInt,
      "joint_attn_softmax: key layout tensors must use int32");
  TORCH_CHECK(
      indices.dim() == 2 &&
          indices.size(0) == values.size(0) &&
          indices.size(1) == key_length,
      "joint_attn_softmax: key indices must have shape [B, K]");
  TORCH_CHECK(
      counts.dim() == 1 && counts.size(0) == values.size(0),
      "joint_attn_softmax: key counts must have shape [B]");

  int64_t expected_rows_per_batch = 1;
  for (int64_t dimension = 1; dimension + 1 < values.dim(); ++dimension) {
    expected_rows_per_batch *= values.size(dimension);
  }
  TORCH_CHECK(
      rows_per_batch == expected_rows_per_batch,
      "joint_attn_softmax: rows_per_batch does not match scores shape");
  return {
      true,
      indices.data_ptr<int32_t>(),
      counts.data_ptr<int32_t>(),
      rows_per_batch,
  };
}

template <typename input_t, typename output_t>
void launch_joint_attn_softmax_forward(
    const input_t* scores,
    output_t* probabilities,
    float* saved_probabilities,
    int64_t row_count,
    int64_t key_length,
    const KeyLayoutView& key_layout,
    cudaStream_t stream) {
  if (key_layout.enabled) {
    joint_attn_softmax_forward_kernel<input_t, output_t, true>
        <<<row_count, kTileK, 0, stream>>>(
            scores,
            probabilities,
            saved_probabilities,
            key_layout.logical_to_physical,
            key_layout.valid_key_counts,
            key_layout.rows_per_batch,
            key_length);
  } else {
    joint_attn_softmax_forward_kernel<input_t, output_t, false>
        <<<row_count, kTileK, 0, stream>>>(
            scores,
            probabilities,
            saved_probabilities,
            nullptr,
            nullptr,
            1,
            key_length);
  }
}

template <typename grad_t, typename output_t>
void launch_joint_attn_softmax_backward(
    const float* probabilities,
    const grad_t* grad_probabilities,
    output_t* grad_scores,
    int64_t row_count,
    int64_t key_length,
    const KeyLayoutView& key_layout,
    cudaStream_t stream) {
  if (key_layout.enabled) {
    joint_attn_softmax_backward_kernel<grad_t, output_t, true>
        <<<row_count, kTileK, 0, stream>>>(
            probabilities,
            grad_probabilities,
            grad_scores,
            key_layout.logical_to_physical,
            key_layout.valid_key_counts,
            key_layout.rows_per_batch,
            key_length);
  } else {
    joint_attn_softmax_backward_kernel<grad_t, output_t, false>
        <<<row_count, kTileK, 0, stream>>>(
            probabilities,
            grad_probabilities,
            grad_scores,
            nullptr,
            nullptr,
            1,
            key_length);
  }
}

}  // namespace

std::vector<torch::Tensor> joint_attn_softmax_build_key_mapping(
    torch::Tensor key_padding_mask) {
  TORCH_CHECK(
      key_padding_mask.is_cuda(),
      "joint_attn_softmax: key_padding_mask must be on CUDA");
  TORCH_CHECK(
      key_padding_mask.is_contiguous(),
      "joint_attn_softmax: key_padding_mask must be contiguous");
  TORCH_CHECK(
      key_padding_mask.scalar_type() == at::kBool,
      "joint_attn_softmax: key_padding_mask must use bool dtype");
  TORCH_CHECK(
      key_padding_mask.dim() == 2,
      "joint_attn_softmax: key_padding_mask must have shape [B, K]");

  const int64_t batch_size = key_padding_mask.size(0);
  const int64_t key_length = key_padding_mask.size(1);
  TORCH_CHECK(
      key_length <= std::numeric_limits<int32_t>::max(),
      "joint_attn_softmax: K exceeds the int32 key-map range");
  const at::cuda::OptionalCUDAGuard device_guard(
      at::device_of(key_padding_mask));
  auto int_options = key_padding_mask.options().dtype(at::kInt);
  auto logical_to_physical = torch::empty(
      {batch_size, key_length}, int_options);
  auto valid_key_counts = torch::empty({batch_size}, int_options);
  if (batch_size == 0) {
    return {logical_to_physical, valid_key_counts};
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  joint_attn_softmax_key_mapping_kernel
      <<<batch_size, kTileK, 0, stream>>>(
          key_padding_mask.data_ptr<bool>(),
          logical_to_physical.data_ptr<int32_t>(),
          valid_key_counts.data_ptr<int32_t>(),
          key_length);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {logical_to_physical, valid_key_counts};
}

torch::Tensor joint_attn_softmax_forward_fp32(
    torch::Tensor scores,
    torch::optional<torch::Tensor> logical_to_physical,
    torch::optional<torch::Tensor> valid_key_counts,
    int64_t rows_per_batch) {
  const int64_t key_length = validate_joint_attn_softmax_scores(scores);
  const auto key_layout = validate_key_layout(
      scores, logical_to_physical, valid_key_counts, rows_per_batch);

  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(scores));
  auto probabilities = torch::empty(
      scores.sizes(), scores.options().dtype(at::kFloat));
  const int64_t row_count = scores.numel() / key_length;
  if (row_count == 0) {
    return probabilities;
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  if (scores.scalar_type() == at::kFloat) {
    launch_joint_attn_softmax_forward<float, float>(
        scores.data_ptr<float>(),
        probabilities.data_ptr<float>(),
        nullptr,
        row_count,
        key_length,
        key_layout,
        stream);
  } else {
    launch_joint_attn_softmax_forward<at::BFloat16, float>(
        scores.data_ptr<at::BFloat16>(),
        probabilities.data_ptr<float>(),
        nullptr,
        row_count,
        key_length,
        key_layout,
        stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return probabilities;
}

torch::Tensor joint_attn_softmax_forward(
    torch::Tensor scores,
    torch::optional<torch::Tensor> logical_to_physical,
    torch::optional<torch::Tensor> valid_key_counts,
    int64_t rows_per_batch) {
  if (scores.scalar_type() == at::kFloat) {
    return joint_attn_softmax_forward_fp32(
        scores, logical_to_physical, valid_key_counts, rows_per_batch);
  }
  const int64_t key_length = validate_joint_attn_softmax_scores(scores);
  const auto key_layout = validate_key_layout(
      scores, logical_to_physical, valid_key_counts, rows_per_batch);

  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(scores));
  auto probabilities = torch::empty_like(scores);
  const int64_t row_count = scores.numel() / key_length;
  if (row_count == 0) {
    return probabilities;
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  launch_joint_attn_softmax_forward<at::BFloat16, at::BFloat16>(
      scores.data_ptr<at::BFloat16>(),
      probabilities.data_ptr<at::BFloat16>(),
      nullptr,
      row_count,
      key_length,
      key_layout,
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return probabilities;
}

std::vector<torch::Tensor> joint_attn_softmax_forward_with_state(
    torch::Tensor scores,
    torch::optional<torch::Tensor> logical_to_physical,
    torch::optional<torch::Tensor> valid_key_counts,
    int64_t rows_per_batch) {
  if (scores.scalar_type() == at::kFloat) {
    auto probabilities = joint_attn_softmax_forward_fp32(
        scores, logical_to_physical, valid_key_counts, rows_per_batch);
    return {probabilities, probabilities};
  }
  const int64_t key_length = validate_joint_attn_softmax_scores(scores);
  const auto key_layout = validate_key_layout(
      scores, logical_to_physical, valid_key_counts, rows_per_batch);

  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(scores));
  auto probabilities = torch::empty_like(scores);
  auto saved_probabilities = torch::empty(
      scores.sizes(), scores.options().dtype(at::kFloat));
  const int64_t row_count = scores.numel() / key_length;
  if (row_count == 0) {
    return {probabilities, saved_probabilities};
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  launch_joint_attn_softmax_forward<at::BFloat16, at::BFloat16>(
      scores.data_ptr<at::BFloat16>(),
      probabilities.data_ptr<at::BFloat16>(),
      saved_probabilities.data_ptr<float>(),
      row_count,
      key_length,
      key_layout,
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {probabilities, saved_probabilities};
}

torch::Tensor joint_attn_softmax_backward(
    torch::Tensor probabilities,
    torch::Tensor grad_probabilities,
    bool output_bf16,
    torch::optional<torch::Tensor> logical_to_physical,
    torch::optional<torch::Tensor> valid_key_counts,
    int64_t rows_per_batch) {
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
  const auto key_layout = validate_key_layout(
      probabilities, logical_to_physical, valid_key_counts, rows_per_batch);

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
    launch_joint_attn_softmax_backward<float, float>(
        probabilities.data_ptr<float>(),
        grad_probabilities.data_ptr<float>(),
        grad_scores.data_ptr<float>(),
        row_count,
        key_length,
        key_layout,
        stream);
  } else if (grad_probabilities.scalar_type() == at::kFloat) {
    launch_joint_attn_softmax_backward<float, at::BFloat16>(
        probabilities.data_ptr<float>(),
        grad_probabilities.data_ptr<float>(),
        grad_scores.data_ptr<at::BFloat16>(),
        row_count,
        key_length,
        key_layout,
        stream);
  } else if (output_bf16) {
    launch_joint_attn_softmax_backward<at::BFloat16, at::BFloat16>(
        probabilities.data_ptr<float>(),
        grad_probabilities.data_ptr<at::BFloat16>(),
        grad_scores.data_ptr<at::BFloat16>(),
        row_count,
        key_length,
        key_layout,
        stream);
  } else {
    launch_joint_attn_softmax_backward<at::BFloat16, float>(
        probabilities.data_ptr<float>(),
        grad_probabilities.data_ptr<at::BFloat16>(),
        grad_scores.data_ptr<float>(),
        row_count,
        key_length,
        key_layout,
        stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_scores;
}
