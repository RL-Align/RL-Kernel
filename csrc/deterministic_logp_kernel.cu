#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <limits>
#include <torch/extension.h>

namespace {

constexpr int kDeterministicLogpSmallBlockSize = 128;
constexpr int kDeterministicLogpMediumBlockSize = 256;
constexpr int kDeterministicLogpLargeBlockSize = 512;
constexpr int kDeterministicLogpSmallVocabLimit = 128;
constexpr int kDeterministicLogpMediumVocabLimit = 4096;
constexpr int kDeterministicLogpWarpSize = 32;
constexpr float kDeterministicLogpNegInf = -3.4028234663852886e38F;

template <int BlockSize>
struct DeterministicLogpBlockTraits {
    static_assert(
        BlockSize == kDeterministicLogpSmallBlockSize ||
            BlockSize == kDeterministicLogpMediumBlockSize ||
            BlockSize == kDeterministicLogpLargeBlockSize,
        "deterministic logp reduction topology requires a supported fixed block size");
    static_assert(BlockSize % kDeterministicLogpWarpSize == 0, "block size must be warp-aligned");
    static constexpr int WarpCount = BlockSize / kDeterministicLogpWarpSize;
};

template <int BlockSize>
__device__ __forceinline__ float deterministicBlockReduceMax(float val) {
    constexpr int WarpCount = DeterministicLogpBlockTraits<BlockSize>::WarpCount;
    __shared__ float shared[WarpCount];

    int lane = threadIdx.x & (kDeterministicLogpWarpSize - 1);
    int wid = threadIdx.x / kDeterministicLogpWarpSize;

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    val = threadIdx.x < WarpCount ? shared[lane] : kDeterministicLogpNegInf;
    if (wid == 0) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        }
    }
    return val;
}

template <int BlockSize>
__device__ __forceinline__ float deterministicBlockReduceSum(float val) {
    constexpr int WarpCount = DeterministicLogpBlockTraits<BlockSize>::WarpCount;
    __shared__ float shared[WarpCount];

    int lane = threadIdx.x & (kDeterministicLogpWarpSize - 1);
    int wid = threadIdx.x / kDeterministicLogpWarpSize;

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    val = threadIdx.x < WarpCount ? shared[lane] : 0.0f;
    if (wid == 0) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
    }
    return val;
}

template <typename input_t, typename output_t, int BlockSize>
__global__ void __launch_bounds__(BlockSize) deterministic_logp_forward_kernel(
    const input_t* __restrict__ logits,
    const int64_t* __restrict__ token_ids,
    output_t* __restrict__ output,
    const int64_t* __restrict__ row_indices,
    int64_t total_rows,
    int vocab_size) {
    int64_t row = row_indices == nullptr ? blockIdx.x : row_indices[blockIdx.x];
    if (row < 0 || row >= total_rows) {
        return;
    }

    const input_t* row_logits = logits + row * vocab_size;

    float local_max = kDeterministicLogpNegInf;
    for (int col = threadIdx.x; col < vocab_size; col += BlockSize) {
        local_max = fmaxf(local_max, static_cast<float>(row_logits[col]));
    }

    float max_val = deterministicBlockReduceMax<BlockSize>(local_max);

    __shared__ float row_max;
    if (threadIdx.x == 0) {
        row_max = max_val;
    }
    __syncthreads();

    float local_sum = 0.0f;
    for (int col = threadIdx.x; col < vocab_size; col += BlockSize) {
        local_sum += expf(static_cast<float>(row_logits[col]) - row_max);
    }

    float sum_val = deterministicBlockReduceSum<BlockSize>(local_sum);

    __shared__ float row_sum;
    if (threadIdx.x == 0) {
        row_sum = sum_val;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int64_t target_id = token_ids[row];
        if (target_id >= 0 && target_id < vocab_size) {
            float target_logit = static_cast<float>(row_logits[target_id]);
            output[row] = static_cast<output_t>(target_logit - row_max - logf(row_sum));
        } else {
            output[row] = static_cast<output_t>(0.0f);
        }
    }
}

void check_deterministic_logp_inputs(
    const torch::Tensor& logits,
    const torch::Tensor& token_ids,
    const torch::Tensor& output) {
    TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor");
    TORCH_CHECK(token_ids.is_cuda(), "token_ids must be a CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
    TORCH_CHECK(
        logits.device() == token_ids.device(),
        "logits and token_ids must be on the same CUDA device");
    TORCH_CHECK(
        logits.device() == output.device(),
        "logits and output must be on the same CUDA device");
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

void check_deterministic_logp_indices(
    const torch::Tensor& logits,
    const torch::Tensor& row_indices) {
    TORCH_CHECK(row_indices.is_cuda(), "row_indices must be a CUDA tensor");
    TORCH_CHECK(
        logits.device() == row_indices.device(),
        "logits and row_indices must be on the same CUDA device");
    TORCH_CHECK(row_indices.dim() == 1, "row_indices must be a 1D tensor");
    TORCH_CHECK(row_indices.scalar_type() == at::ScalarType::Long, "row_indices must be int64");
    TORCH_CHECK(
        row_indices.numel() <= std::numeric_limits<int>::max(),
        "row_indices length exceeds CUDA grid-x limit");
}

void launch_deterministic_logp_kernel(
    const torch::Tensor& logits,
    const torch::Tensor& token_ids,
    const torch::Tensor& output,
    const int64_t* row_indices_ptr,
    int64_t launch_rows,
    int64_t total_rows,
    int64_t vocab_size) {
    if (launch_rows == 0) {
        return;
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        logits.scalar_type(),
        "deterministic_logp_kernel",
        ([&] {
            using input_t = scalar_t;
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::ScalarType::Half,
                at::ScalarType::BFloat16,
                output.scalar_type(),
                "deterministic_logp_output_kernel",
                ([&] {
                    using output_t = scalar_t;
                    const int vocab_size_i32 = static_cast<int>(vocab_size);
                    const int launch_rows_i32 = static_cast<int>(launch_rows);
                    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

                    if (vocab_size <= kDeterministicLogpSmallVocabLimit) {
                        deterministic_logp_forward_kernel<
                            input_t,
                            output_t,
                            kDeterministicLogpSmallBlockSize><<<
                            launch_rows_i32,
                            kDeterministicLogpSmallBlockSize,
                            0,
                            stream>>>(
                            logits.data_ptr<input_t>(),
                            token_ids.data_ptr<int64_t>(),
                            output.data_ptr<output_t>(),
                            row_indices_ptr,
                            total_rows,
                            vocab_size_i32);
                    } else if (vocab_size <= kDeterministicLogpMediumVocabLimit) {
                        deterministic_logp_forward_kernel<
                            input_t,
                            output_t,
                            kDeterministicLogpMediumBlockSize><<<
                            launch_rows_i32,
                            kDeterministicLogpMediumBlockSize,
                            0,
                            stream>>>(
                            logits.data_ptr<input_t>(),
                            token_ids.data_ptr<int64_t>(),
                            output.data_ptr<output_t>(),
                            row_indices_ptr,
                            total_rows,
                            vocab_size_i32);
                    } else {
                        deterministic_logp_forward_kernel<
                            input_t,
                            output_t,
                            kDeterministicLogpLargeBlockSize><<<
                            launch_rows_i32,
                            kDeterministicLogpLargeBlockSize,
                            0,
                            stream>>>(
                            logits.data_ptr<input_t>(),
                            token_ids.data_ptr<int64_t>(),
                            output.data_ptr<output_t>(),
                            row_indices_ptr,
                            total_rows,
                            vocab_size_i32);
                    }
                }));
        }));

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

torch::Tensor deterministic_logp_forward_out(
    torch::Tensor logits,
    torch::Tensor token_ids,
    torch::Tensor output) {
    check_deterministic_logp_inputs(logits, token_ids, output);

    auto logits_contig = logits.contiguous();
    auto token_ids_contig = token_ids.contiguous();

    int64_t total_rows = logits_contig.size(0);
    int64_t vocab_size = logits_contig.size(1);
    launch_deterministic_logp_kernel(
        logits_contig,
        token_ids_contig,
        output,
        nullptr,
        total_rows,
        total_rows,
        vocab_size);

    return output;
}

torch::Tensor deterministic_logp_forward_indexed_out(
    torch::Tensor logits,
    torch::Tensor token_ids,
    torch::Tensor row_indices,
    torch::Tensor output) {
    check_deterministic_logp_inputs(logits, token_ids, output);
    check_deterministic_logp_indices(logits, row_indices);

    auto logits_contig = logits.contiguous();
    auto token_ids_contig = token_ids.contiguous();
    auto row_indices_contig = row_indices.contiguous();

    int64_t total_rows = logits_contig.size(0);
    int64_t vocab_size = logits_contig.size(1);
    int64_t valid_rows = row_indices_contig.numel();

    launch_deterministic_logp_kernel(
        logits_contig,
        token_ids_contig,
        output,
        row_indices_contig.data_ptr<int64_t>(),
        valid_rows,
        total_rows,
        vocab_size);

    return output;
}

torch::Tensor deterministic_logp_forward(torch::Tensor logits, torch::Tensor token_ids) {
    TORCH_CHECK(logits.dim() == 2, "logits must be a 2D tensor");
    auto output = torch::empty({logits.size(0)}, logits.options());
    return deterministic_logp_forward_out(logits, token_ids, output);
}

torch::Tensor deterministic_logp_forward_fp32(torch::Tensor logits, torch::Tensor token_ids) {
    TORCH_CHECK(logits.dim() == 2, "logits must be a 2D tensor");
    auto output = torch::empty({logits.size(0)}, logits.options().dtype(at::ScalarType::Float));
    return deterministic_logp_forward_out(logits, token_ids, output);
}

torch::Tensor deterministic_logp_forward_indexed_fp32(
    torch::Tensor logits,
    torch::Tensor token_ids,
    torch::Tensor row_indices) {
    TORCH_CHECK(logits.dim() == 2, "logits must be a 2D tensor");
    auto output = torch::zeros({logits.size(0)}, logits.options().dtype(at::ScalarType::Float));
    return deterministic_logp_forward_indexed_out(logits, token_ids, row_indices, output);
}
