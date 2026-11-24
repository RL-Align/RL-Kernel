// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// Inkling 2 Mini global GQA attention (INK-15), native ROCm implementation.
// This file is compiled only for ROCm builds (csrc/cuda/inkling2/full_attention.cu
// is the CUDA twin and exports the same symbols); it is tuned for MI300X (gfx942,
// wave64) and is NOT hipified.
//
// Determinism contract (no float atomics anywhere, results independent of grid
// shape, batch size and other rows' contents, bitwise run-to-run identical):
//  * QK score: one lane per key, sequential FP32 sum over head_dim (same order
//    as the CUDA kernel); the query row is staged in LDS.
//  * PV and dK/dV/dQ: one lane per output element, sequential FP32 sum over keys
//    (or over (head, query) for dK/dV) in the CUDA kernel's order.
//  * Softmax max is order independent. The softmax denominator, the softmax
//    backward row sum, and the dGate reduction use a FIXED reduction tree
//    (lane-strided partial sums followed by a wave64 __shfl_xor butterfly /
//    256-thread LDS tree) that depends only on compile-time constants. Compared
//    with the single-thread CUDA kernel these three reductions differ by float
//    rounding only; every other value follows the CUDA accumulation order.
//  * dSink keeps the CUDA sequential (batch, query) order, fed by the per-row
//    backward sum computed once instead of being recomputed O(K) times.

#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <c10/core/DeviceGuard.h>
#include <c10/hip/HIPException.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#include <cstdint>
#include <vector>

namespace {

constexpr int kWave = 64;
constexpr int kThreads = 256;

__device__ __forceinline__ bool key_is_valid(
    int64_t b, int64_t q, int64_t key, int64_t query_len, int64_t key_len, bool causal,
    const int64_t* query_documents, const int64_t* key_documents) {
  if (causal && key > key_len - query_len + q) return false;
  if (query_documents != nullptr &&
      query_documents[b * query_len + q] != key_documents[b * key_len + key]) {
    return false;
  }
  return true;
}

__device__ __forceinline__ float sigmoid_f32(float value) { return 1.0f / (1.0f + expf(-value)); }

__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
  for (int off = kWave / 2; off > 0; off >>= 1) v += __shfl_xor(v, off, kWave);
  return v;
}

__device__ __forceinline__ float wave_max(float v) {
#pragma unroll
  for (int off = kWave / 2; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor(v, off, kWave));
  return v;
}

// One wave (64 threads) per (batch, head, query) row.
template <typename scalar_t>
__global__ void __launch_bounds__(kWave) full_attention_forward_kernel(
    const scalar_t* __restrict__ query, const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value, const float* __restrict__ relative_bias,
    const float* __restrict__ sink, const float* __restrict__ output_gate,
    const int64_t* __restrict__ query_documents, const int64_t* __restrict__ key_documents,
    scalar_t* __restrict__ output, float* __restrict__ probabilities, float* __restrict__ pre_gate,
    float* __restrict__ sink_probability, int64_t batch, int64_t query_len, int64_t key_len,
    int64_t query_heads, int64_t key_heads, int64_t head_dim, bool causal, float scale) {
  extern __shared__ float q_lds[];
  const int64_t row = blockIdx.x;
  const int lane = threadIdx.x;
  const int64_t q = row % query_len;
  const int64_t qh = (row / query_len) % query_heads;
  const int64_t b = row / (query_heads * query_len);
  const int64_t kvh = qh / (query_heads / key_heads);
  const int64_t q_offset = ((b * query_len + q) * query_heads + qh) * head_dim;
  const int64_t row_offset = row * key_len;
  const int64_t kv_stride = key_heads * head_dim;
  const int64_t kv_base = b * key_len * kv_stride + kvh * head_dim;

  for (int64_t d = lane; d < head_dim; d += kWave) {
    q_lds[d] = static_cast<float>(query[q_offset + d]);
  }
  __syncthreads();

  // Pass 1: raw scores (stored in the probabilities buffer) and the row max.
  float max_score = sink == nullptr ? -3.402823466e+38F : sink[qh];
  for (int64_t k = lane; k < key_len; k += kWave) {
    if (!key_is_valid(b, q, k, query_len, key_len, causal, query_documents, key_documents)) continue;
    const scalar_t* key_row = key + kv_base + k * kv_stride;
    float score = 0.0f;
    for (int64_t d = 0; d < head_dim; ++d) score += q_lds[d] * static_cast<float>(key_row[d]);
    score *= scale;
    if (relative_bias != nullptr) score += relative_bias[row_offset + k];
    probabilities[row_offset + k] = score;
    max_score = fmaxf(max_score, score);
  }
  max_score = wave_max(max_score);

  // Pass 2: exponentials (each lane re-reads only its own scores) + fixed-tree sum.
  float partial = 0.0f;
  for (int64_t k = lane; k < key_len; k += kWave) {
    if (!key_is_valid(b, q, k, query_len, key_len, causal, query_documents, key_documents)) {
      probabilities[row_offset + k] = 0.0f;
      continue;
    }
    const float probability = expf(probabilities[row_offset + k] - max_score);
    probabilities[row_offset + k] = probability;
    partial += probability;
  }
  const float sink_exp = sink == nullptr ? 0.0f : expf(sink[qh] - max_score);
  const float sum = sink_exp + wave_sum(partial);
  const float inverse_sum = 1.0f / sum;
  if (lane == 0) sink_probability[row] = sink == nullptr ? 0.0f : sink_exp * inverse_sum;
  for (int64_t k = lane; k < key_len; k += kWave) probabilities[row_offset + k] *= inverse_sum;
  __syncthreads();  // make the normalized row visible to every lane of the wave

  // PV: lane owns output dims d, d+64, ...; sequential over keys.
  const float gate = output_gate == nullptr ? 1.0f : sigmoid_f32(output_gate[qh]);
  for (int64_t d = lane; d < head_dim; d += kWave) {
    float reduced = 0.0f;
    const scalar_t* value_col = value + kv_base + d;
    for (int64_t k = 0; k < key_len; ++k) {
      reduced += probabilities[row_offset + k] * static_cast<float>(value_col[k * kv_stride]);
    }
    pre_gate[q_offset + d] = reduced;
    output[q_offset + d] = static_cast<scalar_t>(reduced * gate);
  }
}

// One wave per row: dp[k] for all valid keys, fixed-tree row sum, then dscore.
template <typename scalar_t>
__global__ void __launch_bounds__(kWave) full_attention_dscore_kernel(
    const scalar_t* __restrict__ grad_output, const scalar_t* __restrict__ value,
    const float* __restrict__ probabilities, const float* __restrict__ output_gate,
    float* __restrict__ dscore, float* __restrict__ row_sum, int64_t batch, int64_t query_len,
    int64_t key_len, int64_t query_heads, int64_t key_heads, int64_t head_dim, bool causal,
    const int64_t* query_documents, const int64_t* key_documents) {
  extern __shared__ float go_lds[];  // grad_output * gate for this row
  const int64_t row = blockIdx.x;
  const int lane = threadIdx.x;
  const int64_t q = row % query_len;
  const int64_t qh = (row / query_len) % query_heads;
  const int64_t b = row / (query_heads * query_len);
  const int64_t kvh = qh / (query_heads / key_heads);
  const int64_t q_offset = ((b * query_len + q) * query_heads + qh) * head_dim;
  const int64_t row_offset = row * key_len;
  const float gate = output_gate == nullptr ? 1.0f : sigmoid_f32(output_gate[qh]);
  for (int64_t d = lane; d < head_dim; d += kWave) {
    go_lds[d] = static_cast<float>(grad_output[q_offset + d]) * gate;
  }
  __syncthreads();
  float partial = 0.0f;
  for (int64_t k = lane; k < key_len; k += kWave) {
    if (!key_is_valid(b, q, k, query_len, key_len, causal, query_documents, key_documents)) {
      dscore[row_offset + k] = 0.0f;
      continue;
    }
    const scalar_t* value_row = value + ((b * key_len + k) * key_heads + kvh) * head_dim;
    float dp = 0.0f;
    for (int64_t d = 0; d < head_dim; ++d) dp += go_lds[d] * static_cast<float>(value_row[d]);
    dscore[row_offset + k] = dp;  // temporarily dp; each lane only rewrites its own keys
    partial += probabilities[row_offset + k] * dp;
  }
  const float sum = wave_sum(partial);
  if (lane == 0) row_sum[row] = sum;
  for (int64_t k = lane; k < key_len; k += kWave) {
    if (!key_is_valid(b, q, k, query_len, key_len, causal, query_documents, key_documents)) continue;
    dscore[row_offset + k] = probabilities[row_offset + k] * (dscore[row_offset + k] - sum);
  }
}

// One wave per (b, q, qh): lane owns d, d+64, ...; sequential over keys.
template <typename scalar_t>
__global__ void __launch_bounds__(kWave) full_attention_dq_kernel(
    const float* __restrict__ dscore, const scalar_t* __restrict__ key,
    scalar_t* __restrict__ dquery, int64_t batch, int64_t query_len, int64_t key_len,
    int64_t query_heads, int64_t key_heads, int64_t head_dim, float scale, bool causal,
    const int64_t* query_documents, const int64_t* key_documents) {
  const int64_t qrow = blockIdx.x;  // (b * query_len + q) * query_heads + qh
  const int64_t qh = qrow % query_heads;
  const int64_t q = (qrow / query_heads) % query_len;
  const int64_t b = qrow / (query_heads * query_len);
  const int64_t kvh = qh / (query_heads / key_heads);
  const float* ds_row = dscore + ((b * query_heads + qh) * query_len + q) * key_len;
  for (int64_t d = threadIdx.x; d < head_dim; d += kWave) {
    float value = 0.0f;
    for (int64_t k = 0; k < key_len; ++k) {
      if (key_is_valid(b, q, k, query_len, key_len, causal, query_documents, key_documents)) {
        value += ds_row[k] *
            static_cast<float>(key[((b * key_len + k) * key_heads + kvh) * head_dim + d]);
      }
    }
    dquery[qrow * head_dim + d] = static_cast<scalar_t>(value * scale);
  }
}

template <typename scalar_t>
__global__ void full_attention_dkv_kernel(
    const float* __restrict__ dscore, const float* __restrict__ probabilities,
    const scalar_t* __restrict__ query, const scalar_t* __restrict__ grad_output,
    const float* __restrict__ output_gate, scalar_t* __restrict__ dkey,
    scalar_t* __restrict__ dvalue, int64_t batch, int64_t query_len, int64_t key_len,
    int64_t query_heads, int64_t key_heads, int64_t head_dim, bool causal,
    const int64_t* query_documents, const int64_t* key_documents) {
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
    const float gate = output_gate == nullptr ? 1.0f : sigmoid_f32(output_gate[qh]);
    for (int64_t q = 0; q < query_len; ++q) {
      if (!key_is_valid(b, q, k, query_len, key_len, causal, query_documents, key_documents)) continue;
      const int64_t row = (b * query_heads + qh) * query_len + q;
      const int64_t q_offset = ((b * query_len + q) * query_heads + qh) * head_dim;
      key_gradient += dscore[row * key_len + k] * static_cast<float>(query[q_offset + d]);
      value_gradient +=
          probabilities[row * key_len + k] * static_cast<float>(grad_output[q_offset + d]) * gate;
    }
  }
  dkey[index] = static_cast<scalar_t>(key_gradient);
  dvalue[index] = static_cast<scalar_t>(value_gradient);
}

// Sequential (batch, query) order as in the CUDA kernel; row sums precomputed.
__global__ void full_attention_dsink_kernel(
    const float* __restrict__ sink_probability, const float* __restrict__ row_sum,
    float* __restrict__ d_sink, int64_t batch, int64_t query_len, int64_t query_heads) {
  const int64_t qh = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (qh >= query_heads) return;
  float value = 0.0f;
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t q = 0; q < query_len; ++q) {
      const int64_t row = (b * query_heads + qh) * query_len + q;
      value += sink_probability[row] * (0.0f - row_sum[row]);
    }
  }
  d_sink[qh] = value;
}

// One 256-thread block per head: thread t owns rows t, t+256, ... (each summed
// sequentially over d), then a fixed LDS tree.
template <typename scalar_t>
__global__ void __launch_bounds__(kThreads) full_attention_dgate_kernel(
    const float* __restrict__ pre_gate, const scalar_t* __restrict__ grad_output,
    const float* __restrict__ output_gate, float* __restrict__ d_gate, int64_t batch,
    int64_t query_len, int64_t query_heads, int64_t head_dim) {
  __shared__ float lds[kThreads];
  const int64_t qh = blockIdx.x;
  const int64_t rows = batch * query_len;
  float partial = 0.0f;
  for (int64_t r = threadIdx.x; r < rows; r += kThreads) {
    const int64_t q_offset = (r * query_heads + qh) * head_dim;  // r = b * query_len + q
    float acc = 0.0f;
    for (int64_t d = 0; d < head_dim; ++d) {
      acc += static_cast<float>(grad_output[q_offset + d]) * pre_gate[q_offset + d];
    }
    partial += acc;
  }
  lds[threadIdx.x] = partial;
  __syncthreads();
  for (int s = kThreads / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) lds[threadIdx.x] += lds[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const float gate = sigmoid_f32(output_gate[qh]);
    d_gate[qh] = lds[0] * gate * (1.0f - gate);
  }
}

static void check_qkv(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous(), name, " must be contiguous GPU");
  TORCH_CHECK(tensor.scalar_type() == at::kFloat || tensor.scalar_type() == at::kHalf ||
                  tensor.scalar_type() == at::kBFloat16,
              name, " must be float32, float16, or bfloat16");
}

static void check_same(const torch::Tensor& lhs, const torch::Tensor& rhs, const char* lhs_name,
                       const char* rhs_name) {
  TORCH_CHECK(lhs.device() == rhs.device(), lhs_name, " and ", rhs_name, " must share device");
  TORCH_CHECK(lhs.scalar_type() == rhs.scalar_type(), lhs_name, " and ", rhs_name,
              " must share dtype");
}

template <typename T>
const T* optional_ptr(const torch::Tensor& tensor) {
  return tensor.numel() == 0 ? nullptr : tensor.data_ptr<T>();
}

}  // namespace

std::vector<torch::Tensor> inkling2_full_attention_forward(
    torch::Tensor query, torch::Tensor key, torch::Tensor value, torch::Tensor relative_bias,
    torch::Tensor sink, torch::Tensor output_gate, torch::Tensor query_documents,
    torch::Tensor key_documents, bool causal, double scale) {
  check_qkv(query, "query");
  check_qkv(key, "key");
  check_qkv(value, "value");
  check_same(query, key, "query", "key");
  check_same(key, value, "key", "value");
  TORCH_CHECK(query.dim() == 4 && key.dim() == 4 && value.sizes() == key.sizes(),
              "query/key/value must be [batch, sequence, heads, dim]");
  TORCH_CHECK(query.size(0) == key.size(0) && query.size(3) == key.size(3),
              "invalid attention geometry");
  TORCH_CHECK(key.size(2) > 0 && query.size(2) % key.size(2) == 0,
              "query heads must be divisible by key heads");
  TORCH_CHECK(relative_bias.numel() == 0 ||
                  (relative_bias.is_cuda() && relative_bias.is_contiguous() &&
                   relative_bias.scalar_type() == at::kFloat),
              "relative_bias must be contiguous GPU float32");
  TORCH_CHECK(sink.numel() == 0 ||
                  (sink.is_cuda() && sink.is_contiguous() && sink.scalar_type() == at::kFloat),
              "sink must be contiguous GPU float32");
  TORCH_CHECK(output_gate.numel() == 0 ||
                  (output_gate.is_cuda() && output_gate.is_contiguous() &&
                   output_gate.scalar_type() == at::kFloat),
              "output_gate must be contiguous GPU float32");
  TORCH_CHECK(query_documents.numel() == 0 ||
                  (query_documents.is_cuda() && query_documents.is_contiguous() &&
                   query_documents.scalar_type() == at::kLong),
              "query_documents must be contiguous GPU int64");
  TORCH_CHECK(key_documents.numel() == 0 ||
                  (key_documents.is_cuda() && key_documents.is_contiguous() &&
                   key_documents.scalar_type() == at::kLong),
              "key_documents must be contiguous GPU int64");
  const int64_t batch = query.size(0), query_len = query.size(1), query_heads = query.size(2);
  const int64_t key_len = key.size(1), key_heads = key.size(2), head_dim = query.size(3);
  TORCH_CHECK(relative_bias.numel() == 0 ||
                  relative_bias.sizes() == torch::IntArrayRef({batch, query_heads, query_len, key_len}),
              "relative_bias must be [batch, heads, query, key]");
  TORCH_CHECK(sink.numel() == 0 || sink.sizes() == torch::IntArrayRef({query_heads}),
              "sink must be [heads]");
  TORCH_CHECK(output_gate.numel() == 0 || output_gate.sizes() == torch::IntArrayRef({query_heads}),
              "output_gate must be [heads]");
  TORCH_CHECK(query_documents.numel() == 0 ||
                  query_documents.sizes() == torch::IntArrayRef({batch, query_len}),
              "query_documents shape mismatch");
  TORCH_CHECK(key_documents.numel() == 0 ||
                  key_documents.sizes() == torch::IntArrayRef({batch, key_len}),
              "key_documents shape mismatch");
  TORCH_CHECK(query_documents.numel() == 0 || key_documents.numel() != 0,
              "query and key documents must be provided together");
  const c10::OptionalDeviceGuard device_guard(device_of(query));
  auto output = torch::empty_like(query);
  auto probabilities =
      torch::empty({batch, query_heads, query_len, key_len}, query.options().dtype(torch::kFloat32));
  auto pre_gate = torch::empty_like(query, query.options().dtype(torch::kFloat32));
  auto sink_probability =
      torch::empty({batch, query_heads, query_len}, query.options().dtype(torch::kFloat32));
  const int64_t rows = batch * query_heads * query_len;
  if (rows == 0 || head_dim == 0) {
    pre_gate.zero_();
    output.zero_();
    sink_probability.zero_();
    probabilities.zero_();
    return {output, probabilities, pre_gate, sink_probability};
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  const size_t lds_bytes = static_cast<size_t>(head_dim) * sizeof(float);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, query.scalar_type(), "inkling2_full_attention_forward", [&] {
        full_attention_forward_kernel<scalar_t><<<rows, kWave, lds_bytes, stream>>>(
            query.data_ptr<scalar_t>(), key.data_ptr<scalar_t>(), value.data_ptr<scalar_t>(),
            optional_ptr<float>(relative_bias), optional_ptr<float>(sink),
            optional_ptr<float>(output_gate), optional_ptr<int64_t>(query_documents),
            optional_ptr<int64_t>(key_documents), output.data_ptr<scalar_t>(),
            probabilities.data_ptr<float>(), pre_gate.data_ptr<float>(),
            sink_probability.data_ptr<float>(), batch, query_len, key_len, query_heads, key_heads,
            head_dim, causal, static_cast<float>(scale));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, probabilities, pre_gate, sink_probability};
}

std::vector<torch::Tensor> inkling2_full_attention_backward(
    torch::Tensor grad_output, torch::Tensor query, torch::Tensor key, torch::Tensor value,
    torch::Tensor probabilities, torch::Tensor pre_gate, torch::Tensor sink_probability,
    torch::Tensor relative_bias, torch::Tensor sink, torch::Tensor output_gate,
    torch::Tensor query_documents, torch::Tensor key_documents, bool causal, double scale) {
  check_qkv(grad_output, "grad_output");
  check_qkv(query, "query");
  check_qkv(key, "key");
  check_qkv(value, "value");
  check_same(grad_output, query, "grad_output", "query");
  const int64_t batch = query.size(0), query_len = query.size(1), query_heads = query.size(2);
  const int64_t key_len = key.size(1), key_heads = key.size(2), head_dim = query.size(3);
  const c10::OptionalDeviceGuard device_guard(device_of(query));
  auto d_query = torch::empty_like(query);
  auto d_key = torch::empty_like(key);
  auto d_value = torch::empty_like(value);
  auto d_relative_bias = relative_bias.numel() == 0 ? torch::Tensor() : torch::empty_like(relative_bias);
  auto d_sink = sink.numel() == 0 ? torch::Tensor() : torch::empty_like(sink);
  auto d_output_gate = output_gate.numel() == 0 ? torch::Tensor() : torch::empty_like(output_gate);
  auto dscore = torch::empty_like(probabilities);
  const int64_t rows = batch * query_heads * query_len;
  auto row_sum = torch::empty({rows}, probabilities.options());
  if (rows == 0 || key_len == 0 || head_dim == 0) {
    d_query.zero_();
    d_key.zero_();
    d_value.zero_();
    if (d_relative_bias.defined()) d_relative_bias.zero_();
    if (d_sink.defined()) d_sink.zero_();
    if (d_output_gate.defined()) d_output_gate.zero_();
    return {d_query, d_key, d_value, d_relative_bias, d_sink, d_output_gate};
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t kv_total = key.numel();
  const int kv_blocks = static_cast<int>((kv_total + kThreads - 1) / kThreads);
  const size_t lds_bytes = static_cast<size_t>(head_dim) * sizeof(float);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, query.scalar_type(), "inkling2_full_attention_backward", [&] {
        full_attention_dscore_kernel<scalar_t><<<rows, kWave, lds_bytes, stream>>>(
            grad_output.data_ptr<scalar_t>(), value.data_ptr<scalar_t>(),
            probabilities.data_ptr<float>(), optional_ptr<float>(output_gate),
            dscore.data_ptr<float>(), row_sum.data_ptr<float>(), batch, query_len, key_len,
            query_heads, key_heads, head_dim, causal, optional_ptr<int64_t>(query_documents),
            optional_ptr<int64_t>(key_documents));
        full_attention_dq_kernel<scalar_t><<<batch * query_len * query_heads, kWave, 0, stream>>>(
            dscore.data_ptr<float>(), key.data_ptr<scalar_t>(), d_query.data_ptr<scalar_t>(), batch,
            query_len, key_len, query_heads, key_heads, head_dim, static_cast<float>(scale), causal,
            optional_ptr<int64_t>(query_documents), optional_ptr<int64_t>(key_documents));
        full_attention_dkv_kernel<scalar_t><<<kv_blocks, kThreads, 0, stream>>>(
            dscore.data_ptr<float>(), probabilities.data_ptr<float>(), query.data_ptr<scalar_t>(),
            grad_output.data_ptr<scalar_t>(), optional_ptr<float>(output_gate),
            d_key.data_ptr<scalar_t>(), d_value.data_ptr<scalar_t>(), batch, query_len, key_len,
            query_heads, key_heads, head_dim, causal, optional_ptr<int64_t>(query_documents),
            optional_ptr<int64_t>(key_documents));
        if (relative_bias.numel() != 0) d_relative_bias.copy_(dscore);
        if (sink.numel() != 0) {
          full_attention_dsink_kernel<<<(query_heads + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
              sink_probability.data_ptr<float>(), row_sum.data_ptr<float>(), d_sink.data_ptr<float>(),
              batch, query_len, query_heads);
        }
        if (output_gate.numel() != 0) {
          full_attention_dgate_kernel<scalar_t><<<query_heads, kThreads, 0, stream>>>(
              pre_gate.data_ptr<float>(), grad_output.data_ptr<scalar_t>(),
              output_gate.data_ptr<float>(), d_output_gate.data_ptr<float>(), batch, query_len,
              query_heads, head_dim);
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {d_query, d_key, d_value, d_relative_bias, d_sink, d_output_gate};
}
