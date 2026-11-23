// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
// Inkling 2 Mini global GQA attention (INK-15).

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <vector>

namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ bool key_is_valid(
    int64_t b,
    int64_t q,
    int64_t key,
    int64_t query_len,
    int64_t key_len,
    bool causal,
    const int64_t* query_documents,
    const int64_t* key_documents) {
  if (causal && key > key_len - query_len + q) return false;
  if (query_documents != nullptr && query_documents[b * query_len + q] != key_documents[b * key_len + key]) {
    return false;
  }
  return true;
}

__device__ __forceinline__ float sigmoid_f32(float value) {
  return 1.0f / (1.0f + expf(-value));
}

template <typename scalar_t>
__global__ void full_attention_forward_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const float* __restrict__ relative_bias,
    const float* __restrict__ sink,
    const float* __restrict__ output_gate,
    const int64_t* __restrict__ query_documents,
    const int64_t* __restrict__ key_documents,
    scalar_t* __restrict__ output,
    float* __restrict__ probabilities,
    float* __restrict__ pre_gate,
    float* __restrict__ sink_probability,
    int64_t batch,
    int64_t query_len,
    int64_t key_len,
    int64_t query_heads,
    int64_t key_heads,
    int64_t head_dim,
    bool causal,
    float scale) {
  const int64_t row = blockIdx.x;
  const int64_t total_rows = batch * query_heads * query_len;
  if (row >= total_rows || threadIdx.x != 0) return;
  const int64_t q = row % query_len;
  const int64_t qh = (row / query_len) % query_heads;
  const int64_t b = row / (query_heads * query_len);
  const int64_t kvh = qh / (query_heads / key_heads);
  const int64_t q_offset = ((b * query_len + q) * query_heads + qh) * head_dim;
  const int64_t row_offset = row * key_len;
  const int64_t bias_offset = row_offset;

  float max_score = sink == nullptr ? -3.402823466e+38F : sink[qh];
  for (int64_t k = 0; k < key_len; ++k) {
    if (!key_is_valid(b, q, k, query_len, key_len, causal, query_documents, key_documents)) continue;
    float score = 0.0f;
    for (int64_t d = 0; d < head_dim; ++d) {
      score += static_cast<float>(query[q_offset + d]) *
          static_cast<float>(key[((b * key_len + k) * key_heads + kvh) * head_dim + d]);
    }
    score *= scale;
    if (relative_bias != nullptr) score += relative_bias[bias_offset + k];
    max_score = fmaxf(max_score, score);
  }
  float sum = sink == nullptr ? 0.0f : expf(sink[qh] - max_score);
  for (int64_t k = 0; k < key_len; ++k) {
    if (!key_is_valid(b, q, k, query_len, key_len, causal, query_documents, key_documents)) {
      probabilities[row_offset + k] = 0.0f;
      continue;
    }
    float score = 0.0f;
    for (int64_t d = 0; d < head_dim; ++d) {
      score += static_cast<float>(query[q_offset + d]) *
          static_cast<float>(key[((b * key_len + k) * key_heads + kvh) * head_dim + d]);
    }
    score *= scale;
    if (relative_bias != nullptr) score += relative_bias[bias_offset + k];
    const float probability = expf(score - max_score);
    probabilities[row_offset + k] = probability;
    sum += probability;
  }
  const float inverse_sum = 1.0f / sum;
  const float sink_p = sink == nullptr ? 0.0f : expf(sink[qh] - max_score) * inverse_sum;
  sink_probability[row] = sink_p;
  for (int64_t k = 0; k < key_len; ++k) probabilities[row_offset + k] *= inverse_sum;
  const float gate = output_gate == nullptr ? 1.0f : sigmoid_f32(output_gate[qh]);
  for (int64_t d = 0; d < head_dim; ++d) {
    float reduced = 0.0f;
    for (int64_t k = 0; k < key_len; ++k) {
      reduced += probabilities[row_offset + k] *
          static_cast<float>(value[((b * key_len + k) * key_heads + kvh) * head_dim + d]);
    }
    pre_gate[q_offset + d] = reduced;
    output[q_offset + d] = static_cast<scalar_t>(reduced * gate);
  }
}

template <typename scalar_t>
__global__ void full_attention_dscore_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ value,
    const float* __restrict__ probabilities,
    const float* __restrict__ pre_gate,
    const float* __restrict__ output_gate,
    float* __restrict__ dscore,
    int64_t batch,
    int64_t query_len,
    int64_t key_len,
    int64_t query_heads,
    int64_t key_heads,
    int64_t head_dim,
    bool causal,
    const int64_t* query_documents,
    const int64_t* key_documents) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = batch * query_heads * query_len * key_len;
  if (index >= total) return;
  const int64_t k = index % key_len;
  const int64_t row = index / key_len;
  const int64_t q = row % query_len;
  const int64_t qh = (row / query_len) % query_heads;
  const int64_t b = row / (query_heads * query_len);
  const int64_t kvh = qh / (query_heads / key_heads);
  if (!key_is_valid(b, q, k, query_len, key_len, causal, query_documents, key_documents)) {
    dscore[index] = 0.0f;
    return;
  }
  const float gate = output_gate == nullptr ? 1.0f : sigmoid_f32(output_gate[qh]);
  float d_probability = 0.0f;
  const int64_t q_offset = ((b * query_len + q) * query_heads + qh) * head_dim;
  const int64_t k_offset = ((b * key_len + k) * key_heads + kvh) * head_dim;
  for (int64_t d = 0; d < head_dim; ++d) {
    d_probability += static_cast<float>(grad_output[q_offset + d]) * gate *
        static_cast<float>(value[k_offset + d]);
  }
  float sum_probability_gradient = 0.0f;
  const int64_t row_offset = row * key_len;
  for (int64_t j = 0; j < key_len; ++j) {
    if (!key_is_valid(b, q, j, query_len, key_len, causal, query_documents, key_documents)) continue;
    const int64_t j_offset = ((b * key_len + j) * key_heads + kvh) * head_dim;
    float dp = 0.0f;
    for (int64_t d = 0; d < head_dim; ++d) {
      dp += static_cast<float>(grad_output[q_offset + d]) * gate *
          static_cast<float>(value[j_offset + d]);
    }
    sum_probability_gradient += probabilities[row_offset + j] * dp;
  }
  dscore[index] = probabilities[index] * (d_probability - sum_probability_gradient);
}

template <typename scalar_t>
__global__ void full_attention_dq_kernel(
    const float* __restrict__ dscore,
    const scalar_t* __restrict__ key,
    scalar_t* __restrict__ dquery,
    int64_t batch,
    int64_t query_len,
    int64_t key_len,
    int64_t query_heads,
    int64_t key_heads,
    int64_t head_dim,
    float scale,
    bool causal,
    const int64_t* query_documents,
    const int64_t* key_documents) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = batch * query_heads * query_len * head_dim;
  if (index >= total) return;
  const int64_t d = index % head_dim;
  const int64_t qh = (index / head_dim) % query_heads;
  const int64_t q = (index / (head_dim * query_heads)) % query_len;
  const int64_t b = index / (query_heads * query_len * head_dim);
  const int64_t kvh = qh / (query_heads / key_heads);
  float value = 0.0f;
  for (int64_t k = 0; k < key_len; ++k) {
    if (key_is_valid(b, q, k, query_len, key_len, causal, query_documents, key_documents)) {
      value += dscore[((b * query_heads + qh) * query_len + q) * key_len + k] *
          static_cast<float>(key[((b * key_len + k) * key_heads + kvh) * head_dim + d]);
    }
  }
  dquery[index] = static_cast<scalar_t>(value * scale);
}

template <typename scalar_t>
__global__ void full_attention_dkv_kernel(
    const float* __restrict__ dscore,
    const float* __restrict__ probabilities,
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ grad_output,
    const float* __restrict__ output_gate,
    scalar_t* __restrict__ dkey,
    scalar_t* __restrict__ dvalue,
    int64_t batch,
    int64_t query_len,
    int64_t key_len,
    int64_t query_heads,
    int64_t key_heads,
    int64_t head_dim,
    bool causal,
    const int64_t* query_documents,
    const int64_t* key_documents) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = batch * key_heads * key_len * head_dim;
  if (index >= total) return;
  const int64_t d = index % head_dim;
  const int64_t kvh = (index / head_dim) % key_heads;
  const int64_t k = (index / (head_dim * key_heads)) % key_len;
  const int64_t b = index / (key_heads * key_len * head_dim);
  const int64_t repeats = query_heads / key_heads;
  float key_gradient = 0.0f;
  float value_gradient = 0.0f;
  for (int64_t qh = kvh * repeats; qh < (kvh + 1) * repeats; ++qh) {
    for (int64_t q = 0; q < query_len; ++q) {
      if (!key_is_valid(b, q, k, query_len, key_len, causal, query_documents, key_documents)) continue;
      const int64_t row = (b * query_heads + qh) * query_len + q;
      const int64_t q_offset = ((b * query_len + q) * query_heads + qh) * head_dim;
      key_gradient += dscore[row * key_len + k] * static_cast<float>(query[q_offset + d]);
      const float gate = output_gate == nullptr ? 1.0f : sigmoid_f32(output_gate[qh]);
      value_gradient += probabilities[row * key_len + k] * static_cast<float>(grad_output[q_offset + d]) * gate;
    }
  }
  dkey[index] = static_cast<scalar_t>(key_gradient);
  dvalue[index] = static_cast<scalar_t>(value_gradient);
}

template <typename scalar_t>
__global__ void full_attention_dsink_kernel(
    const float* __restrict__ sink_probability,
    const float* __restrict__ probabilities,
    const scalar_t* __restrict__ value_tensor,
    const scalar_t* __restrict__ grad_output,
    float* __restrict__ d_sink,
    int64_t batch,
    int64_t query_len,
    int64_t key_len,
    int64_t query_heads,
    int64_t key_heads,
    int64_t head_dim,
    const float* __restrict__ output_gate,
    bool causal,
    const int64_t* query_documents,
    const int64_t* key_documents) {
  const int64_t qh = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (qh >= query_heads) return;
  const int64_t repeats = query_heads / key_heads;
  const int64_t kvh = qh / repeats;
  float value = 0.0f;
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t q = 0; q < query_len; ++q) {
      const int64_t row = (b * query_heads + qh) * query_len + q;
      const int64_t q_offset = ((b * query_len + q) * query_heads + qh) * head_dim;
      const float gate = output_gate == nullptr ? 1.0f : sigmoid_f32(output_gate[qh]);
      float sum_probability_gradient = 0.0f;
      for (int64_t k = 0; k < key_len; ++k) {
        if (!key_is_valid(b, q, k, query_len, key_len, causal, query_documents, key_documents)) continue;
        float dp = 0.0f;
        for (int64_t d = 0; d < head_dim; ++d) {
          dp += static_cast<float>(grad_output[q_offset + d]) * gate *
              static_cast<float>(value_tensor[((b * key_len + k) * key_heads + kvh) * head_dim + d]);
        }
        sum_probability_gradient += probabilities[row * key_len + k] * dp;
      }
      value += sink_probability[row] * (0.0f - sum_probability_gradient);
    }
  }
  d_sink[qh] = value;
}

template <typename scalar_t>
__global__ void full_attention_dgate_kernel(
    const float* __restrict__ pre_gate,
    const scalar_t* __restrict__ grad_output,
    const float* __restrict__ output_gate,
    float* __restrict__ d_gate,
    int64_t batch,
    int64_t query_len,
    int64_t query_heads,
    int64_t head_dim) {
  const int64_t qh = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (qh >= query_heads) return;
  const float gate = sigmoid_f32(output_gate[qh]);
  float value = 0.0f;
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t q = 0; q < query_len; ++q) {
      const int64_t row = (b * query_heads + qh) * query_len + q;
      for (int64_t d = 0; d < head_dim; ++d) {
        const int64_t q_offset = ((b * query_len + q) * query_heads + qh) * head_dim;
        value += static_cast<float>(grad_output[q_offset + d]) * pre_gate[q_offset + d];
      }
    }
  }
  d_gate[qh] = value * gate * (1.0f - gate);
}

static void check_qkv(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous(), name, " must be contiguous CUDA");
  TORCH_CHECK(tensor.scalar_type() == at::kFloat || tensor.scalar_type() == at::kHalf ||
                  tensor.scalar_type() == at::kBFloat16,
              name, " must be float32, float16, or bfloat16");
}

static void check_same(const torch::Tensor& lhs, const torch::Tensor& rhs, const char* lhs_name,
                       const char* rhs_name) {
  TORCH_CHECK(lhs.device() == rhs.device(), lhs_name, " and ", rhs_name, " must share device");
  TORCH_CHECK(lhs.scalar_type() == rhs.scalar_type(), lhs_name, " and ", rhs_name, " must share dtype");
}

}  // namespace

std::vector<torch::Tensor> inkling2_full_attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor relative_bias,
    torch::Tensor sink,
    torch::Tensor output_gate,
    torch::Tensor query_documents,
    torch::Tensor key_documents,
    bool causal,
    double scale) {
  check_qkv(query, "query");
  check_qkv(key, "key");
  check_qkv(value, "value");
  check_same(query, key, "query", "key");
  check_same(key, value, "key", "value");
  TORCH_CHECK(query.dim() == 4 && key.dim() == 4 && value.sizes() == key.sizes(),
              "query/key/value must be [batch, sequence, heads, dim]");
  TORCH_CHECK(query.size(0) == key.size(0) && query.size(3) == key.size(3), "invalid attention geometry");
  TORCH_CHECK(query.size(2) % key.size(2) == 0, "query heads must be divisible by key heads");
  TORCH_CHECK(relative_bias.numel() == 0 || (relative_bias.is_cuda() && relative_bias.is_contiguous() &&
                  relative_bias.scalar_type() == at::kFloat), "relative_bias must be contiguous CUDA float32");
  TORCH_CHECK(sink.numel() == 0 || (sink.is_cuda() && sink.is_contiguous() && sink.scalar_type() == at::kFloat),
              "sink must be contiguous CUDA float32");
  TORCH_CHECK(output_gate.numel() == 0 || (output_gate.is_cuda() && output_gate.is_contiguous() &&
                  output_gate.scalar_type() == at::kFloat), "output_gate must be contiguous CUDA float32");
  TORCH_CHECK(query_documents.numel() == 0 || (query_documents.is_cuda() && query_documents.is_contiguous() &&
                  query_documents.scalar_type() == at::kLong), "query_documents must be contiguous CUDA int64");
  TORCH_CHECK(key_documents.numel() == 0 || (key_documents.is_cuda() && key_documents.is_contiguous() &&
                  key_documents.scalar_type() == at::kLong), "key_documents must be contiguous CUDA int64");
  const int64_t batch = query.size(0), query_len = query.size(1), query_heads = query.size(2);
  const int64_t key_len = key.size(1), key_heads = key.size(2), head_dim = query.size(3);
  TORCH_CHECK(relative_bias.numel() == 0 || relative_bias.sizes() == torch::IntArrayRef({batch, query_heads, query_len, key_len}),
              "relative_bias must be [batch, heads, query, key]");
  TORCH_CHECK(sink.numel() == 0 || sink.sizes() == torch::IntArrayRef({query_heads}), "sink must be [heads]");
  TORCH_CHECK(output_gate.numel() == 0 || output_gate.sizes() == torch::IntArrayRef({query_heads}),
              "output_gate must be [heads]");
  TORCH_CHECK(query_documents.numel() == 0 || query_documents.sizes() == torch::IntArrayRef({batch, query_len}),
              "query_documents shape mismatch");
  TORCH_CHECK(key_documents.numel() == 0 || key_documents.sizes() == torch::IntArrayRef({batch, key_len}),
              "key_documents shape mismatch");
  TORCH_CHECK(query_documents.numel() == 0 || key_documents.numel() != 0,
              "query and key documents must be provided together");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  auto output = torch::empty_like(query);
  auto probabilities = torch::empty({batch, query_heads, query_len, key_len}, query.options().dtype(torch::kFloat32));
  auto pre_gate = torch::empty_like(query, query.options().dtype(torch::kFloat32));
  auto sink_probability = torch::empty({batch, query_heads, query_len}, query.options().dtype(torch::kFloat32));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t rows = batch * query_heads * query_len;
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, query.scalar_type(),
                                  "inkling2_full_attention_forward", [&] {
    full_attention_forward_kernel<scalar_t><<<rows, 1, 0, stream>>>(
        query.data_ptr<scalar_t>(), key.data_ptr<scalar_t>(), value.data_ptr<scalar_t>(),
        relative_bias.numel() == 0 ? nullptr : relative_bias.data_ptr<float>(),
        sink.numel() == 0 ? nullptr : sink.data_ptr<float>(),
        output_gate.numel() == 0 ? nullptr : output_gate.data_ptr<float>(),
        query_documents.numel() == 0 ? nullptr : query_documents.data_ptr<int64_t>(),
        key_documents.numel() == 0 ? nullptr : key_documents.data_ptr<int64_t>(),
        output.data_ptr<scalar_t>(), probabilities.data_ptr<float>(), pre_gate.data_ptr<float>(),
        sink_probability.data_ptr<float>(), batch, query_len, key_len, query_heads, key_heads,
        head_dim, causal, static_cast<float>(scale));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, probabilities, pre_gate, sink_probability};
}

std::vector<torch::Tensor> inkling2_full_attention_backward(
    torch::Tensor grad_output,
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor probabilities,
    torch::Tensor pre_gate,
    torch::Tensor sink_probability,
    torch::Tensor relative_bias,
    torch::Tensor sink,
    torch::Tensor output_gate,
    torch::Tensor query_documents,
    torch::Tensor key_documents,
    bool causal,
    double scale) {
  check_qkv(grad_output, "grad_output");
  check_qkv(query, "query");
  check_qkv(key, "key");
  check_qkv(value, "value");
  check_same(grad_output, query, "grad_output", "query");
  const int64_t batch = query.size(0), query_len = query.size(1), query_heads = query.size(2);
  const int64_t key_len = key.size(1), key_heads = key.size(2), head_dim = query.size(3);
  auto d_query = torch::empty_like(query);
  auto d_key = torch::empty_like(key);
  auto d_value = torch::empty_like(value);
  auto d_relative_bias = relative_bias.numel() == 0 ? torch::Tensor() : torch::empty_like(relative_bias);
  auto d_sink = sink.numel() == 0 ? torch::Tensor() : torch::empty_like(sink);
  auto d_output_gate = output_gate.numel() == 0 ? torch::Tensor() : torch::empty_like(output_gate);
  auto dscore = torch::empty_like(probabilities);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t score_total = probabilities.numel();
  const int64_t q_total = query.numel();
  const int64_t kv_total = key.numel();
  const int score_blocks = static_cast<int>((score_total + kThreads - 1) / kThreads);
  const int q_blocks = static_cast<int>((q_total + kThreads - 1) / kThreads);
  const int kv_blocks = static_cast<int>((kv_total + kThreads - 1) / kThreads);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, query.scalar_type(),
                                  "inkling2_full_attention_backward", [&] {
    full_attention_dscore_kernel<scalar_t><<<score_blocks, kThreads, 0, stream>>>(
        grad_output.data_ptr<scalar_t>(), value.data_ptr<scalar_t>(), probabilities.data_ptr<float>(),
        pre_gate.data_ptr<float>(), output_gate.numel() == 0 ? nullptr : output_gate.data_ptr<float>(),
        dscore.data_ptr<float>(), batch, query_len, key_len, query_heads, key_heads, head_dim,
        causal, query_documents.numel() == 0 ? nullptr : query_documents.data_ptr<int64_t>(),
        key_documents.numel() == 0 ? nullptr : key_documents.data_ptr<int64_t>());
    full_attention_dq_kernel<scalar_t><<<q_blocks, kThreads, 0, stream>>>(
        dscore.data_ptr<float>(), key.data_ptr<scalar_t>(), d_query.data_ptr<scalar_t>(), batch,
        query_len, key_len, query_heads, key_heads, head_dim, static_cast<float>(scale), causal,
        query_documents.numel() == 0 ? nullptr : query_documents.data_ptr<int64_t>(),
        key_documents.numel() == 0 ? nullptr : key_documents.data_ptr<int64_t>());
    full_attention_dkv_kernel<scalar_t><<<kv_blocks, kThreads, 0, stream>>>(
        dscore.data_ptr<float>(), probabilities.data_ptr<float>(), query.data_ptr<scalar_t>(),
        grad_output.data_ptr<scalar_t>(), output_gate.numel() == 0 ? nullptr : output_gate.data_ptr<float>(),
        d_key.data_ptr<scalar_t>(), d_value.data_ptr<scalar_t>(), batch, query_len, key_len, query_heads, key_heads, head_dim, causal,
        query_documents.numel() == 0 ? nullptr : query_documents.data_ptr<int64_t>(),
        key_documents.numel() == 0 ? nullptr : key_documents.data_ptr<int64_t>());
    if (relative_bias.numel() != 0) d_relative_bias.copy_(dscore);
    if (sink.numel() != 0) {
      full_attention_dsink_kernel<scalar_t><<<(query_heads + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
          sink_probability.data_ptr<float>(), probabilities.data_ptr<float>(), value.data_ptr<scalar_t>(),
          grad_output.data_ptr<scalar_t>(), d_sink.data_ptr<float>(), batch, query_len, key_len,
          query_heads, key_heads, head_dim,
          output_gate.numel() == 0 ? nullptr : output_gate.data_ptr<float>(), causal,
          query_documents.numel() == 0 ? nullptr : query_documents.data_ptr<int64_t>(),
          key_documents.numel() == 0 ? nullptr : key_documents.data_ptr<int64_t>());
    }
    if (output_gate.numel() != 0) {
      full_attention_dgate_kernel<scalar_t><<<(query_heads + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
          pre_gate.data_ptr<float>(), grad_output.data_ptr<scalar_t>(), output_gate.data_ptr<float>(),
          d_output_gate.data_ptr<float>(), batch, query_len, query_heads, head_dim);
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {d_query, d_key, d_value, d_relative_bias, d_sink, d_output_gate};
}
