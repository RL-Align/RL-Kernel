// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// Single-card SM90 batch-invariant embedding and LM-head reference kernels.
//
// These kernels intentionally avoid Split-K. For LM-head, each output element
// owns the full hidden-dimension reduction inside one CTA, so the K traversal
// and reduction tree depend only on hidden_size, not on batch/sequence layout.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <array>
#include <limits>
#include <mutex>
#include <torch/extension.h>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kMaxCachedDevices = 64;

// Reference path: one CTA per logit is intentionally throughput-heavy for large
// vocab prefill. It exists to preserve a fixed K reduction order for WS1.

template <typename T>
__device__ __forceinline__ float to_float(T value) {
    return static_cast<float>(value);
}

template <typename T>
__device__ __forceinline__ T from_float(float value) {
    return static_cast<T>(value);
}

template <int BlockSize>
__device__ __forceinline__ float block_sum(float value) {
    static_assert(BlockSize % 32 == 0, "BlockSize must be a warp multiple");
    __shared__ float shared[BlockSize / 32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }

    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();

    value = threadIdx.x < (BlockSize / 32) ? shared[lane] : 0.0f;
    if (warp == 0) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffff, value, offset);
        }
    }
    return value;
}

template <typename input_t, typename output_t>
__global__ void embedding_sm90_forward_kernel(const int64_t *__restrict__ token_ids,
                                              const input_t *__restrict__ weight,
                                              output_t *__restrict__ output,
                                              int64_t num_tokens, int64_t hidden_size,
                                              int64_t /*vocab_size*/) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = num_tokens * hidden_size;
    if (idx >= total) {
        return;
    }

    const int64_t token_row = idx / hidden_size;
    const int64_t hidden_col = idx - token_row * hidden_size;
    const int64_t token_id = token_ids[token_row];
    output[idx] = static_cast<output_t>(weight[token_id * hidden_size + hidden_col]);
}

template <typename scalar_t, typename output_t>
__global__ void lm_head_sm90_forward_kernel(const scalar_t *__restrict__ hidden,
                                            const scalar_t *__restrict__ weight,
                                            const float *__restrict__ bias,
                                            output_t *__restrict__ output,
                                            int64_t num_tokens, int64_t hidden_size,
                                            int64_t vocab_size) {
    const int64_t out_idx = static_cast<int64_t>(blockIdx.x);
    const int64_t total = num_tokens * vocab_size;
    if (out_idx >= total) {
        return;
    }

    const int64_t token_row = out_idx / vocab_size;
    const int64_t vocab_col = out_idx - token_row * vocab_size;
    const scalar_t *hidden_row = hidden + token_row * hidden_size;
    const scalar_t *weight_row = weight + vocab_col * hidden_size;

    float acc = 0.0f;
    for (int64_t k = threadIdx.x; k < hidden_size; k += blockDim.x) {
        acc += to_float(hidden_row[k]) * to_float(weight_row[k]);
    }
    acc = block_sum<kThreads>(acc);

    if (threadIdx.x == 0) {
        if (bias != nullptr) {
            acc += bias[vocab_col];
        }
        output[out_idx] = from_float<output_t>(acc);
    }
}

std::vector<int64_t> embedding_output_sizes(torch::Tensor token_ids, int64_t hidden_size) {
    std::vector<int64_t> sizes;
    sizes.reserve(static_cast<size_t>(token_ids.dim()) + 1);
    for (int64_t i = 0; i < token_ids.dim(); ++i) {
        sizes.push_back(token_ids.size(i));
    }
    sizes.push_back(hidden_size);
    return sizes;
}

std::vector<int64_t> lm_head_output_sizes(torch::Tensor hidden, int64_t vocab_size) {
    std::vector<int64_t> sizes;
    sizes.reserve(static_cast<size_t>(hidden.dim()));
    for (int64_t i = 0; i < hidden.dim() - 1; ++i) {
        sizes.push_back(hidden.size(i));
    }
    sizes.push_back(vocab_size);
    return sizes;
}

bool is_supported_float_dtype(at::ScalarType dtype) {
    return dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16;
}

void check_sm90_device() {
    int device = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    static std::array<int, kMaxCachedDevices> cached_major{};
    static std::array<int, kMaxCachedDevices> cached_minor{};
    static std::array<std::once_flag, kMaxCachedDevices> cached_once{};

    if (device >= 0 && device < kMaxCachedDevices) {
        std::call_once(cached_once[device], [device]() {
            C10_CUDA_CHECK(cudaDeviceGetAttribute(&cached_major[device],
                                                  cudaDevAttrComputeCapabilityMajor, device));
            C10_CUDA_CHECK(cudaDeviceGetAttribute(&cached_minor[device],
                                                  cudaDevAttrComputeCapabilityMinor, device));
        });
        TORCH_CHECK(cached_major[device] == 9,
                    "SM90 embedding/lm_head kernels require Hopper, got sm_",
                    cached_major[device], cached_minor[device]);
        return;
    }

    int major = 0;
    int minor = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
    TORCH_CHECK(major == 9, "SM90 embedding/lm_head kernels require Hopper, got sm_",
                major, minor);
}

void check_cuda_same_device(torch::Tensor a, torch::Tensor b, const char *a_name,
                            const char *b_name) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), a_name, " and ", b_name,
                " must be CUDA tensors");
    TORCH_CHECK(a.device() == b.device(), a_name, " and ", b_name,
                " must be on the same CUDA device");
}

torch::Tensor embedding_sm90_forward_impl(torch::Tensor token_ids, torch::Tensor weight,
                                          bool output_fp32) {
    TORCH_CHECK(token_ids.is_cuda() && weight.is_cuda(),
                "token_ids and weight must be CUDA tensors");
    TORCH_CHECK(token_ids.device() == weight.device(),
                "token_ids and weight must be on the same CUDA device");
    TORCH_CHECK(weight.dim() == 2, "embedding weight must be [vocab, hidden]");
    TORCH_CHECK(weight.is_contiguous(), "embedding weight must be contiguous");
    TORCH_CHECK(is_supported_float_dtype(weight.scalar_type()),
                "embedding_sm90 supports fp32, fp16, and bf16 weights");

    c10::cuda::CUDAGuard device_guard(weight.device());
    check_sm90_device();
    const int64_t vocab_size = weight.size(0);
    const int64_t hidden_size = weight.size(1);
    const int64_t num_tokens = token_ids.numel();
    auto ids = token_ids.reshape({num_tokens}).to(at::kLong).contiguous();
    if (num_tokens > 0) {
        const int64_t min_id = ids.min().item<int64_t>();
        const int64_t max_id = ids.max().item<int64_t>();
        TORCH_CHECK(min_id >= 0 && max_id < vocab_size,
                    "embedding_sm90 token ids must be in [0, ", vocab_size - 1,
                    "], got [", min_id, ", ", max_id, "]");
    }
    auto out_options = weight.options().dtype(output_fp32 ? at::kFloat : weight.scalar_type());
    auto output = torch::empty(embedding_output_sizes(token_ids, hidden_size), out_options);
    if (num_tokens == 0 || hidden_size == 0) {
        return output;
    }

    const int64_t total = num_tokens * hidden_size;
    TORCH_CHECK(total <= std::numeric_limits<int>::max(),
                "embedding_sm90 launch size exceeds CUDA grid limit");
    const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
    auto output_2d = output.reshape({num_tokens, hidden_size});

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, weight.scalar_type(), "embedding_sm90_forward", [&] {
            if (output_fp32) {
                embedding_sm90_forward_kernel<scalar_t, float>
                    <<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
                        ids.data_ptr<int64_t>(), weight.data_ptr<scalar_t>(),
                        output_2d.data_ptr<float>(), num_tokens, hidden_size, vocab_size);
            } else {
                embedding_sm90_forward_kernel<scalar_t, scalar_t>
                    <<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
                        ids.data_ptr<int64_t>(), weight.data_ptr<scalar_t>(),
                        output_2d.data_ptr<scalar_t>(), num_tokens, hidden_size, vocab_size);
            }
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor lm_head_sm90_forward_impl(torch::Tensor hidden, torch::Tensor weight,
                                        torch::optional<torch::Tensor> bias,
                                        bool output_fp32) {
    check_cuda_same_device(hidden, weight, "hidden", "weight");
    TORCH_CHECK(hidden.dim() >= 2, "hidden must have shape [..., hidden]");
    TORCH_CHECK(weight.dim() == 2, "lm_head weight must be [vocab, hidden]");
    TORCH_CHECK(hidden.size(-1) == weight.size(1), "hidden/weight hidden-dim mismatch");
    TORCH_CHECK(is_supported_float_dtype(hidden.scalar_type()),
                "lm_head_sm90 supports fp32, fp16, and bf16 hidden states");
    TORCH_CHECK(is_supported_float_dtype(weight.scalar_type()),
                "lm_head_sm90 supports fp32, fp16, and bf16 weights");

    c10::cuda::CUDAGuard device_guard(hidden.device());
    check_sm90_device();
    const int64_t hidden_size = hidden.size(-1);
    TORCH_CHECK(hidden_size > 0, "lm_head hidden dimension must be non-zero");
    const int64_t vocab_size = weight.size(0);
    const int64_t num_tokens = hidden.numel() / hidden_size;
    const at::ScalarType compute_dtype = output_fp32 ? at::kFloat : hidden.scalar_type();

    auto hidden_2d = hidden.reshape({num_tokens, hidden_size}).to(compute_dtype).contiguous();
    auto weight_2d = weight.to(compute_dtype).contiguous();
    torch::Tensor bias_f;
    const float *bias_ptr = nullptr;
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda(), "lm_head bias must be a CUDA tensor");
        TORCH_CHECK(bias->device() == hidden.device(),
                    "lm_head bias must be on the same CUDA device as hidden");
        TORCH_CHECK(bias->dim() == 1, "lm_head bias must be 1-D [vocab]");
        TORCH_CHECK(bias->numel() == vocab_size, "lm_head bias must have vocab elements");
        TORCH_CHECK(is_supported_float_dtype(bias->scalar_type()),
                    "lm_head_sm90 supports fp32, fp16, and bf16 bias");
        bias_f = bias->reshape({vocab_size}).to(at::kFloat).contiguous();
        bias_ptr = bias_f.data_ptr<float>();
    }

    auto out_options = hidden.options().dtype(output_fp32 ? at::kFloat : hidden.scalar_type());
    auto output = torch::empty(lm_head_output_sizes(hidden, vocab_size), out_options);
    if (num_tokens == 0 || vocab_size == 0) {
        return output;
    }

    const int64_t total = num_tokens * vocab_size;
    TORCH_CHECK(total <= std::numeric_limits<int>::max(),
                "lm_head_sm90 launch size exceeds CUDA grid limit");
    auto output_2d = output.reshape({num_tokens, vocab_size});

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, compute_dtype, "lm_head_sm90_forward", [&] {
            if (output_fp32) {
                lm_head_sm90_forward_kernel<scalar_t, float>
                    <<<static_cast<int>(total), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
                        hidden_2d.data_ptr<scalar_t>(), weight_2d.data_ptr<scalar_t>(),
                        bias_ptr, output_2d.data_ptr<float>(), num_tokens, hidden_size,
                        vocab_size);
            } else {
                lm_head_sm90_forward_kernel<scalar_t, scalar_t>
                    <<<static_cast<int>(total), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
                        hidden_2d.data_ptr<scalar_t>(), weight_2d.data_ptr<scalar_t>(),
                        bias_ptr, output_2d.data_ptr<scalar_t>(), num_tokens, hidden_size,
                        vocab_size);
            }
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

} // namespace

torch::Tensor embedding_sm90_forward(torch::Tensor token_ids, torch::Tensor weight) {
    return embedding_sm90_forward_impl(token_ids, weight, false);
}

torch::Tensor embedding_sm90_forward_fp32(torch::Tensor token_ids, torch::Tensor weight) {
    return embedding_sm90_forward_impl(token_ids, weight, true);
}

torch::Tensor lm_head_sm90_forward(torch::Tensor hidden, torch::Tensor weight,
                                   torch::optional<torch::Tensor> bias) {
    return lm_head_sm90_forward_impl(hidden, weight, bias, false);
}

torch::Tensor lm_head_sm90_forward_fp32(torch::Tensor hidden, torch::Tensor weight,
                                        torch::optional<torch::Tensor> bias) {
    return lm_head_sm90_forward_impl(hidden, weight, bias, true);
}

torch::Tensor det_gemm_rowwise_fwd_fp32(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2,
                "det_gemm_rowwise_fwd_fp32 expects [M,K] @ [K,N]");
    TORCH_CHECK(a.size(1) == b.size(0),
                "det_gemm_rowwise_fwd_fp32: K mismatch");
    // lm_head_sm90_forward_impl computes one output element per CTA using a
    // fixed 256-thread block reduction. Passing B^T as [N,K] exposes that
    // deterministic rowwise reduction as a general GEMM configuration.
    return lm_head_sm90_forward_impl(
        a, b.transpose(0, 1).contiguous(), torch::optional<torch::Tensor>{}, true);
}
