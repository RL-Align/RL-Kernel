#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <limits>
#include <torch/extension.h>

constexpr int kFusedLogpLogicalWarpSize = 32;

template <typename T>
__device__ __forceinline__ T fused_logp_shfl_down_32(T value, unsigned int delta) {
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
    return __shfl_down(value, delta, kFusedLogpLogicalWarpSize);
#else
    return __shfl_down_sync(0xffffffffu, value, delta, kFusedLogpLogicalWarpSize);
#endif
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t blockReduceMax(scalar_t val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    float f_val = static_cast<float>(val);

    for (int offset = 16; offset > 0; offset /= 2)
        f_val = max(f_val, fused_logp_shfl_down_32(f_val, offset));

    if (lane == 0) shared[wid] = f_val;
    __syncthreads();

    f_val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : -1e20f;
    if (wid == 0) {
        for (int offset = 16; offset > 0; offset /= 2)
            f_val = max(f_val, fused_logp_shfl_down_32(f_val, offset));
    }
    return static_cast<scalar_t>(f_val);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t blockReduceSum(scalar_t val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    float f_val = static_cast<float>(val);

    for (int offset = 16; offset > 0; offset /= 2)
        f_val += fused_logp_shfl_down_32(f_val, offset);

    if (lane == 0) shared[wid] = f_val;
    __syncthreads();

    f_val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0) {
        for (int offset = 16; offset > 0; offset /= 2)
            f_val += fused_logp_shfl_down_32(f_val, offset);
    }
    return static_cast<scalar_t>(f_val);
}

struct LogSumExpState {
    float max_val;
    float sum_exp;
};

__device__ __forceinline__ LogSumExpState merge_logsumexp_state(
    LogSumExpState a,
    LogSumExpState b) {
    if (a.sum_exp == 0.0f) {
        return b;
    }
    if (b.sum_exp == 0.0f) {
        return a;
    }

    if (a.max_val >= b.max_val) {
        return {a.max_val, a.sum_exp + b.sum_exp * expf(b.max_val - a.max_val)};
    }
    return {b.max_val, b.sum_exp + a.sum_exp * expf(a.max_val - b.max_val)};
}

template <int BlockSize>
__device__ __forceinline__ LogSumExpState blockReduceLogSumExp(LogSumExpState state) {
    static_assert(BlockSize > 0 && BlockSize <= 1024 && BlockSize % 32 == 0,
                  "online fused_logp block size must be a positive warp multiple up to 1024");
    static __shared__ float shared_max[32];
    static __shared__ float shared_sum[32];
    constexpr int kWarpCount = BlockSize / 32;
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    for (int offset = 16; offset > 0; offset /= 2) {
        LogSumExpState other{
            fused_logp_shfl_down_32(state.max_val, offset),
            fused_logp_shfl_down_32(state.sum_exp, offset)};
        state = merge_logsumexp_state(state, other);
    }

    if (lane == 0) {
        shared_max[wid] = state.max_val;
        shared_sum[wid] = state.sum_exp;
    }
    __syncthreads();

    state = { -1e20f, 0.0f };
    if (threadIdx.x < kWarpCount) {
        state = {shared_max[lane], shared_sum[lane]};
    }
    if (wid == 0) {
        for (int offset = 16; offset > 0; offset /= 2) {
            LogSumExpState other{
                fused_logp_shfl_down_32(state.max_val, offset),
                fused_logp_shfl_down_32(state.sum_exp, offset)};
            state = merge_logsumexp_state(state, other);
        }
    }
    return state;
}

template <typename scalar_t, typename output_t>
__global__ void fused_logp_forward_kernel(
    const scalar_t* __restrict__ logits,      // [TotalTokens, VocabSize]
    const int64_t* __restrict__ token_ids,   // [TotalTokens]
    output_t* __restrict__ output,           // [TotalTokens]
    const int64_t* __restrict__ row_indices, // Optional [ValidTokens]
    int64_t total_tokens,
    int vocab_size) {

    int64_t row = row_indices == nullptr ? blockIdx.x : row_indices[blockIdx.x];
    if (row < 0 || row >= total_tokens) {
        return;
    }

    const scalar_t* row_logits = logits + row * vocab_size;

    float local_max = -1e20f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        local_max = max(local_max, static_cast<float>(row_logits[i]));
    }
    float max_val = blockReduceMax<float>(local_max);

    __shared__ float res_max;
    if (threadIdx.x == 0) res_max = max_val;
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        local_sum += expf(static_cast<float>(row_logits[i]) - res_max);
    }
    float sum_val = blockReduceSum<float>(local_sum);

    __shared__ float res_sum;
    if (threadIdx.x == 0) res_sum = sum_val;
    __syncthreads();

    if (threadIdx.x == 0) {
        int64_t target_id = token_ids[row];
        if (target_id >= 0 && target_id < vocab_size) {
            float target_logit = static_cast<float>(row_logits[target_id]);
            output[row] = static_cast<output_t>(target_logit - res_max - logf(res_sum));
        } else {
            output[row] = static_cast<output_t>(0.0f);
        }
    }
}

// dlogits = grad * (one_hot(target) - softmax(logits)) for one row per block.
// The max / sum-exp passes visit each thread's elements in the same order as
// fused_logp_forward_kernel (same block size and reductions), so the softmax is
// bitwise consistent with the forward logp; loads are only batched kUnroll at a
// time for memory-level parallelism.  No [N, V] FP32 workspace is materialized.
// Rows whose target is outside [0, V) have a constant forward output and get a
// zero gradient.
template <typename scalar_t, int kUnroll>
__global__ void fused_logp_backward_kernel(
    const scalar_t* __restrict__ logits,      // [TotalTokens, VocabSize]
    const int64_t* __restrict__ token_ids,   // [TotalTokens]
    const float* __restrict__ grad_output,   // [TotalTokens]
    scalar_t* __restrict__ grad_logits,      // [TotalTokens, VocabSize]
    int vocab_size) {

    int64_t row = blockIdx.x;
    const scalar_t* row_logits = logits + row * vocab_size;
    scalar_t* row_grad = grad_logits + row * vocab_size;
    const int stride = blockDim.x;
    const int tile = stride * kUnroll;

    int64_t target_id = token_ids[row];
    if (target_id < 0 || target_id >= vocab_size) {
        for (int i = threadIdx.x; i < vocab_size; i += stride) {
            row_grad[i] = static_cast<scalar_t>(0.0f);
        }
        return;
    }

    float local_max = -1e20f;
    for (int base = threadIdx.x; base < vocab_size; base += tile) {
        float values[kUnroll];
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            int i = base + u * stride;
            values[u] = i < vocab_size ? static_cast<float>(row_logits[i]) : -1e20f;
        }
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            local_max = max(local_max, values[u]);
        }
    }
    float max_val = blockReduceMax<float>(local_max);

    __shared__ float res_max;
    if (threadIdx.x == 0) res_max = max_val;
    __syncthreads();

    float local_sum = 0.0f;
    for (int base = threadIdx.x; base < vocab_size; base += tile) {
        float values[kUnroll];
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            int i = base + u * stride;
            values[u] = i < vocab_size ? static_cast<float>(row_logits[i]) : 0.0f;
        }
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            if (base + u * stride < vocab_size) {
                local_sum += expf(values[u] - res_max);
            }
        }
    }
    float sum_val = blockReduceSum<float>(local_sum);

    __shared__ float res_sum;
    if (threadIdx.x == 0) res_sum = sum_val;
    __syncthreads();

    float g = grad_output[row];
    for (int base = threadIdx.x; base < vocab_size; base += tile) {
        float values[kUnroll];
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            int i = base + u * stride;
            values[u] = i < vocab_size ? static_cast<float>(row_logits[i]) : 0.0f;
        }
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            int i = base + u * stride;
            if (i < vocab_size) {
                float prob = expf(values[u] - res_max) / res_sum;
                float one_hot = i == target_id ? 1.0f : 0.0f;
                row_grad[i] = static_cast<scalar_t>(g * (one_hot - prob));
            }
        }
    }
}

template <typename scalar_t, typename output_t, int BlockSize>
__global__ void __launch_bounds__(BlockSize) fused_logp_forward_online_kernel(
    const scalar_t* __restrict__ logits,      // [TotalTokens, VocabSize]
    const int64_t* __restrict__ token_ids,   // [TotalTokens]
    output_t* __restrict__ output,           // [TotalTokens]
    const int64_t* __restrict__ row_indices, // Optional [ValidTokens]
    int64_t total_tokens,
    int vocab_size) {

    int64_t row = row_indices == nullptr ? blockIdx.x : row_indices[blockIdx.x];
    if (row < 0 || row >= total_tokens) {
        return;
    }

    const scalar_t* row_logits = logits + row * vocab_size;

    LogSumExpState local_state{-1e20f, 0.0f};
    for (int i = threadIdx.x; i < vocab_size; i += BlockSize) {
        float value = static_cast<float>(row_logits[i]);
        if (local_state.sum_exp == 0.0f) {
            local_state = {value, 1.0f};
        } else if (value <= local_state.max_val) {
            local_state.sum_exp += expf(value - local_state.max_val);
        } else {
            local_state.sum_exp = local_state.sum_exp * expf(local_state.max_val - value) + 1.0f;
            local_state.max_val = value;
        }
    }

    LogSumExpState global_state = blockReduceLogSumExp<BlockSize>(local_state);

    __shared__ float res_max;
    __shared__ float res_sum;
    if (threadIdx.x == 0) {
        res_max = global_state.max_val;
        res_sum = global_state.sum_exp;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int64_t target_id = token_ids[row];
        if (target_id >= 0 && target_id < vocab_size) {
            float target_logit = static_cast<float>(row_logits[target_id]);
            output[row] = static_cast<output_t>(target_logit - res_max - logf(res_sum));
        } else {
            output[row] = static_cast<output_t>(0.0f);
        }
    }
}

namespace {

#ifndef FUSED_LOGP_TWOPASS_BLOCK_SIZE
#define FUSED_LOGP_TWOPASS_BLOCK_SIZE 256
#endif

#ifndef FUSED_LOGP_ONLINE_BLOCK_SIZE
#define FUSED_LOGP_ONLINE_BLOCK_SIZE 128
#endif

#ifndef FUSED_LOGP_ONLINE_SPARSE_LARGE_VOCAB_BLOCK_SIZE
#define FUSED_LOGP_ONLINE_SPARSE_LARGE_VOCAB_BLOCK_SIZE 256
#endif

#ifndef FUSED_LOGP_ONLINE_LARGE_ROW_BYTES_THRESHOLD
#define FUSED_LOGP_ONLINE_LARGE_ROW_BYTES_THRESHOLD 65536
#endif

#ifndef FUSED_LOGP_ONLINE_SPARSE_DENSITY_NUMERATOR
#define FUSED_LOGP_ONLINE_SPARSE_DENSITY_NUMERATOR 1
#endif

#ifndef FUSED_LOGP_ONLINE_SPARSE_DENSITY_DENOMINATOR
#define FUSED_LOGP_ONLINE_SPARSE_DENSITY_DENOMINATOR 2
#endif

#ifndef FUSED_LOGP_ONLINE_MIN_BLOCKS_PER_SM
#define FUSED_LOGP_ONLINE_MIN_BLOCKS_PER_SM 1
#endif

constexpr int kFusedLogpTwoPassBlockSize = FUSED_LOGP_TWOPASS_BLOCK_SIZE;
constexpr int kFusedLogpBackwardUnroll = 8;
constexpr int kFusedLogpOnlineBlockSize = FUSED_LOGP_ONLINE_BLOCK_SIZE;
constexpr int kFusedLogpOnlineSparseLargeVocabBlockSize =
    FUSED_LOGP_ONLINE_SPARSE_LARGE_VOCAB_BLOCK_SIZE;
constexpr int64_t kFusedLogpOnlineLargeRowBytesThreshold =
    FUSED_LOGP_ONLINE_LARGE_ROW_BYTES_THRESHOLD;
constexpr int64_t kFusedLogpOnlineSparseDensityNumerator =
    FUSED_LOGP_ONLINE_SPARSE_DENSITY_NUMERATOR;
constexpr int64_t kFusedLogpOnlineSparseDensityDenominator =
    FUSED_LOGP_ONLINE_SPARSE_DENSITY_DENOMINATOR;
constexpr int64_t kFusedLogpOnlineMinBlocksPerSm =
    FUSED_LOGP_ONLINE_MIN_BLOCKS_PER_SM;

enum class FusedLogpOnlineLaunchVariant {
    kDefault,
    kSparseLargeRow,
};

FusedLogpOnlineLaunchVariant select_fused_logp_online_launch_variant(
    const int64_t* row_indices_ptr,
    int64_t launch_rows,
    int64_t total_tokens,
    int64_t row_bytes,
    int sm_count) {
    bool indexed_sparse =
        row_indices_ptr != nullptr &&
        launch_rows * kFusedLogpOnlineSparseDensityDenominator <=
            total_tokens * kFusedLogpOnlineSparseDensityNumerator;
    bool large_streaming_row = row_bytes >= kFusedLogpOnlineLargeRowBytesThreshold;
    bool enough_blocks = launch_rows >= static_cast<int64_t>(sm_count) * kFusedLogpOnlineMinBlocksPerSm;
    if (indexed_sparse && large_streaming_row && enough_blocks) {
        return FusedLogpOnlineLaunchVariant::kSparseLargeRow;
    }
    return FusedLogpOnlineLaunchVariant::kDefault;
}

template <int BlockSize, typename input_t, typename output_t>
void launch_fused_logp_online_variant(
    const torch::Tensor& logits,
    const torch::Tensor& token_ids,
    const torch::Tensor& output,
    const int64_t* row_indices_ptr,
    int64_t launch_rows,
    int64_t total_tokens,
    int64_t vocab_size) {
    static_assert(BlockSize > 0 && BlockSize <= 1024 && BlockSize % 32 == 0,
                  "online fused_logp block size must be a positive warp multiple up to 1024");
    fused_logp_forward_online_kernel<input_t, output_t, BlockSize><<<
        static_cast<int>(launch_rows),
        BlockSize,
        0,
        at::cuda::getCurrentCUDAStream()>>>(
        logits.data_ptr<input_t>(),
        token_ids.data_ptr<int64_t>(),
        output.data_ptr<output_t>(),
        row_indices_ptr,
        total_tokens,
        static_cast<int>(vocab_size));
}

void check_fused_logp_inputs(
    const torch::Tensor& logits,
    const torch::Tensor& token_ids,
    const torch::Tensor& output) {
    TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor");
    TORCH_CHECK(token_ids.is_cuda(), "token_ids must be a CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
    TORCH_CHECK(logits.device() == token_ids.device(), "logits and token_ids must be on the same CUDA device");
    TORCH_CHECK(logits.device() == output.device(), "logits and output must be on the same CUDA device");
    TORCH_CHECK(logits.dim() == 2, "logits must be a 2D tensor");
    TORCH_CHECK(token_ids.dim() == 1, "token_ids must be a 1D tensor");
    TORCH_CHECK(output.dim() == 1, "output must be a 1D tensor");
    TORCH_CHECK(token_ids.scalar_type() == at::ScalarType::Long, "token_ids must be int64");
    TORCH_CHECK(
        token_ids.numel() == logits.size(0),
        "token_ids length must match logits rows");
    TORCH_CHECK(output.numel() == logits.size(0), "output length must match logits rows");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(logits.size(1) > 0, "logits vocab dimension must be non-empty");
    TORCH_CHECK(
        logits.size(0) <= std::numeric_limits<int>::max(),
        "logits row count exceeds CUDA grid-x limit");
    TORCH_CHECK(
        logits.size(1) <= std::numeric_limits<int>::max(),
        "logits vocab dimension exceeds int32 kernel limit");
    TORCH_CHECK(
        output.scalar_type() == at::ScalarType::Float ||
            output.scalar_type() == at::ScalarType::Double ||
            output.scalar_type() == at::ScalarType::Half ||
            output.scalar_type() == at::ScalarType::BFloat16,
        "output dtype must be float64, float32, float16, or bfloat16");
}

void check_fused_logp_indices(
    const torch::Tensor& logits,
    const torch::Tensor& row_indices) {
    TORCH_CHECK(row_indices.is_cuda(), "row_indices must be a CUDA tensor");
    TORCH_CHECK(logits.device() == row_indices.device(), "logits and row_indices must be on the same CUDA device");
    TORCH_CHECK(row_indices.dim() == 1, "row_indices must be a 1D tensor");
    TORCH_CHECK(row_indices.scalar_type() == at::ScalarType::Long, "row_indices must be int64");
    TORCH_CHECK(
        row_indices.numel() <= std::numeric_limits<int>::max(),
        "row_indices length exceeds CUDA grid-x limit");
}

void launch_fused_logp_kernel(
    const torch::Tensor& logits,
    const torch::Tensor& token_ids,
    const torch::Tensor& output,
    const int64_t* row_indices_ptr,
    int64_t launch_rows,
    int64_t total_tokens,
    int64_t vocab_size) {
    if (launch_rows == 0) {
        return;
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        logits.scalar_type(),
        "fused_logp_kernel",
        ([&] {
            using input_t = scalar_t;
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::ScalarType::Half,
                at::ScalarType::BFloat16,
                output.scalar_type(),
                "fused_logp_output_kernel",
                ([&] {
                    using output_t = scalar_t;
                    fused_logp_forward_kernel<input_t, output_t><<<
                        static_cast<int>(launch_rows),
                        kFusedLogpTwoPassBlockSize,
                        0,
                        at::cuda::getCurrentCUDAStream()>>>(
                        logits.data_ptr<input_t>(),
                        token_ids.data_ptr<int64_t>(),
                        output.data_ptr<output_t>(),
                        row_indices_ptr,
                        total_tokens,
                        static_cast<int>(vocab_size));
                }));
        }));

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_fused_logp_online_kernel(
    const torch::Tensor& logits,
    const torch::Tensor& token_ids,
    const torch::Tensor& output,
    const int64_t* row_indices_ptr,
    int64_t launch_rows,
    int64_t total_tokens,
    int64_t vocab_size) {
    if (launch_rows == 0) {
        return;
    }
    const auto* device_props = at::cuda::getCurrentDeviceProperties();
    FusedLogpOnlineLaunchVariant launch_variant = select_fused_logp_online_launch_variant(
        row_indices_ptr,
        launch_rows,
        total_tokens,
        vocab_size * logits.element_size(),
        device_props->multiProcessorCount);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        logits.scalar_type(),
        "fused_logp_online_kernel",
        ([&] {
            using input_t = scalar_t;
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::ScalarType::Half,
                at::ScalarType::BFloat16,
                output.scalar_type(),
                "fused_logp_online_output_kernel",
                ([&] {
                    using output_t = scalar_t;
                    if (launch_variant == FusedLogpOnlineLaunchVariant::kSparseLargeRow) {
                        launch_fused_logp_online_variant<
                            kFusedLogpOnlineSparseLargeVocabBlockSize,
                            input_t,
                            output_t>(
                            logits,
                            token_ids,
                            output,
                            row_indices_ptr,
                            launch_rows,
                            total_tokens,
                            vocab_size);
                    } else {
                        launch_fused_logp_online_variant<
                            kFusedLogpOnlineBlockSize,
                            input_t,
                            output_t>(
                            logits,
                            token_ids,
                            output,
                            row_indices_ptr,
                            launch_rows,
                            total_tokens,
                            vocab_size);
                    }
                }));
        }));

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

torch::Tensor fused_logp_forward_out(
    torch::Tensor logits,
    torch::Tensor token_ids,
    torch::Tensor output) {
    check_fused_logp_inputs(logits, token_ids, output);

    auto logits_contig = logits.contiguous();
    auto token_ids_contig = token_ids.contiguous();

    int64_t total_tokens = logits_contig.size(0);
    int64_t vocab_size = logits_contig.size(1);
    launch_fused_logp_kernel(
        logits_contig,
        token_ids_contig,
        output,
        nullptr,
        total_tokens,
        total_tokens,
        vocab_size);

    return output;
}

torch::Tensor fused_logp_forward_indexed_out(
    torch::Tensor logits,
    torch::Tensor token_ids,
    torch::Tensor row_indices,
    torch::Tensor output) {
    check_fused_logp_inputs(logits, token_ids, output);
    check_fused_logp_indices(logits, row_indices);

    auto logits_contig = logits.contiguous();
    auto token_ids_contig = token_ids.contiguous();
    auto row_indices_contig = row_indices.contiguous();

    int64_t total_tokens = logits_contig.size(0);
    int64_t vocab_size = logits_contig.size(1);
    int64_t valid_tokens = row_indices_contig.numel();

    launch_fused_logp_kernel(
        logits_contig,
        token_ids_contig,
        output,
        row_indices_contig.data_ptr<int64_t>(),
        valid_tokens,
        total_tokens,
        vocab_size);

    return output;
}

torch::Tensor fused_logp_forward_online_out(
    torch::Tensor logits,
    torch::Tensor token_ids,
    torch::Tensor output) {
    check_fused_logp_inputs(logits, token_ids, output);

    auto logits_contig = logits.contiguous();
    auto token_ids_contig = token_ids.contiguous();

    int64_t total_tokens = logits_contig.size(0);
    int64_t vocab_size = logits_contig.size(1);
    launch_fused_logp_online_kernel(
        logits_contig,
        token_ids_contig,
        output,
        nullptr,
        total_tokens,
        total_tokens,
        vocab_size);

    return output;
}

torch::Tensor fused_logp_forward_online_indexed_out(
    torch::Tensor logits,
    torch::Tensor token_ids,
    torch::Tensor row_indices,
    torch::Tensor output) {
    check_fused_logp_inputs(logits, token_ids, output);
    check_fused_logp_indices(logits, row_indices);

    auto logits_contig = logits.contiguous();
    auto token_ids_contig = token_ids.contiguous();
    auto row_indices_contig = row_indices.contiguous();

    int64_t total_tokens = logits_contig.size(0);
    int64_t vocab_size = logits_contig.size(1);
    int64_t valid_tokens = row_indices_contig.numel();

    launch_fused_logp_online_kernel(
        logits_contig,
        token_ids_contig,
        output,
        row_indices_contig.data_ptr<int64_t>(),
        valid_tokens,
        total_tokens,
        vocab_size);

    return output;
}

torch::Tensor fused_logp_forward(torch::Tensor logits, torch::Tensor token_ids) {
    TORCH_CHECK(logits.dim() == 2, "logits must be a 2D tensor");
    auto output = torch::empty({logits.size(0)}, logits.options());
    return fused_logp_forward_out(logits, token_ids, output);
}

torch::Tensor fused_logp_forward_fp32(torch::Tensor logits, torch::Tensor token_ids) {
    TORCH_CHECK(logits.dim() == 2, "logits must be a 2D tensor");
    auto output = torch::empty({logits.size(0)}, logits.options().dtype(at::ScalarType::Float));
    return fused_logp_forward_out(logits, token_ids, output);
}

torch::Tensor fused_logp_forward_indexed_fp32(
    torch::Tensor logits,
    torch::Tensor token_ids,
    torch::Tensor row_indices) {
    TORCH_CHECK(logits.dim() == 2, "logits must be a 2D tensor");
    auto output = torch::zeros({logits.size(0)}, logits.options().dtype(at::ScalarType::Float));
    return fused_logp_forward_indexed_out(logits, token_ids, row_indices, output);
}

torch::Tensor fused_logp_forward_online_fp32(torch::Tensor logits, torch::Tensor token_ids) {
    TORCH_CHECK(logits.dim() == 2, "logits must be a 2D tensor");
    auto output = torch::empty({logits.size(0)}, logits.options().dtype(at::ScalarType::Float));
    return fused_logp_forward_online_out(logits, token_ids, output);
}

torch::Tensor fused_logp_forward_online_indexed_fp32(
    torch::Tensor logits,
    torch::Tensor token_ids,
    torch::Tensor row_indices) {
    TORCH_CHECK(logits.dim() == 2, "logits must be a 2D tensor");
    auto output = torch::zeros({logits.size(0)}, logits.options().dtype(at::ScalarType::Float));
    return fused_logp_forward_online_indexed_out(logits, token_ids, row_indices, output);
}

torch::Tensor fused_logp_backward(
    torch::Tensor logits,
    torch::Tensor token_ids,
    torch::Tensor grad_output) {
    TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor");
    TORCH_CHECK(token_ids.is_cuda(), "token_ids must be a CUDA tensor");
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");
    TORCH_CHECK(logits.device() == token_ids.device(), "logits and token_ids must be on the same CUDA device");
    TORCH_CHECK(logits.device() == grad_output.device(), "logits and grad_output must be on the same CUDA device");
    TORCH_CHECK(logits.dim() == 2, "logits must be a 2D tensor");
    TORCH_CHECK(token_ids.dim() == 1, "token_ids must be a 1D tensor");
    TORCH_CHECK(grad_output.dim() == 1, "grad_output must be a 1D tensor");
    TORCH_CHECK(token_ids.scalar_type() == at::ScalarType::Long, "token_ids must be int64");
    TORCH_CHECK(grad_output.scalar_type() == at::ScalarType::Float, "grad_output must be float32");
    TORCH_CHECK(token_ids.numel() == logits.size(0), "token_ids length must match logits rows");
    TORCH_CHECK(grad_output.numel() == logits.size(0), "grad_output length must match logits rows");
    TORCH_CHECK(logits.size(1) > 0, "logits vocab dimension must be non-empty");
    TORCH_CHECK(
        logits.size(1) <= std::numeric_limits<int>::max(),
        "logits vocab dimension exceeds int32 kernel indexing");
    TORCH_CHECK(
        logits.size(0) <= std::numeric_limits<int>::max(),
        "logits rows exceed the CUDA grid limit");

    auto logits_contig = logits.contiguous();
    auto token_ids_contig = token_ids.contiguous();
    auto grad_output_contig = grad_output.contiguous();
    auto grad_logits = torch::empty_like(logits_contig);
    int64_t total_tokens = logits_contig.size(0);
    if (total_tokens == 0) {
        return grad_logits;
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        logits_contig.scalar_type(),
        "fused_logp_backward_kernel",
        ([&] {
            fused_logp_backward_kernel<scalar_t, kFusedLogpBackwardUnroll><<<
                static_cast<int>(total_tokens),
                kFusedLogpTwoPassBlockSize,
                0,
                at::cuda::getCurrentCUDAStream()>>>(
                logits_contig.data_ptr<scalar_t>(),
                token_ids_contig.data_ptr<int64_t>(),
                grad_output_contig.data_ptr<float>(),
                grad_logits.data_ptr<scalar_t>(),
                static_cast<int>(logits_contig.size(1)));
        }));

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_logits;
}
