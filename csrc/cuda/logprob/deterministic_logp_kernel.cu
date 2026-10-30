#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <limits>
#include <torch/extension.h>

namespace {

constexpr int kDeterministicLogpTinyBlockSize = 64;
constexpr int kDeterministicLogpSmallBlockSize = 128;
constexpr int kDeterministicLogpMediumBlockSize = 256;
constexpr int kDeterministicLogpLargeBlockSize = 512;
constexpr int kDeterministicLogpSmallVocabLimit = 128;
constexpr int kDeterministicLogpMediumVocabLimit = 4096;
constexpr int kDeterministicLogpWarpSize = 32;
constexpr float kDeterministicLogpNegInf = -3.4028234663852886e38F;

template <typename T>
__device__ __forceinline__ T deterministic_logp_shfl_down_32(T value, unsigned int delta) {
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
    return __shfl_down(value, delta, kDeterministicLogpWarpSize);
#else
    return __shfl_down_sync(0xffffffffu, value, delta, kDeterministicLogpWarpSize);
#endif
}

template <int BlockSize>
struct DeterministicLogpBlockTraits {
    static_assert(
        BlockSize == kDeterministicLogpTinyBlockSize ||
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
        val = fmaxf(val, deterministic_logp_shfl_down_32(val, offset));
    }

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    const bool has_warp_value = threadIdx.x < WarpCount;
    const int shared_idx = has_warp_value ? threadIdx.x : 0;
    val = has_warp_value ? shared[shared_idx] : kDeterministicLogpNegInf;
    if (wid == 0) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val = fmaxf(val, deterministic_logp_shfl_down_32(val, offset));
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
        val += deterministic_logp_shfl_down_32(val, offset);
    }

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    const bool has_warp_value = threadIdx.x < WarpCount;
    const int shared_idx = has_warp_value ? threadIdx.x : 0;
    val = has_warp_value ? shared[shared_idx] : 0.0f;
    if (wid == 0) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += deterministic_logp_shfl_down_32(val, offset);
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

    // Indexed mode may launch duplicate row ids. The writes are idempotent:
    // every duplicate writer computes and stores the same deterministic value.
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
    TORCH_CHECK(
        reinterpret_cast<uintptr_t>(logits.data_ptr()) % 16 == 0,
        "logits must be 16-byte aligned");
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

namespace {

// Tuned on sm_90 (H100) for the Qwen3 shape (V=151936 split into 64 tiles of
// 2374 columns). Two things dominate: 16-byte vector loads, and a block small
// enough that a tile splits into several balanced chunks. The previous fixed
// block of 256 left only ~1.2 chunks per tile, so most of each pass ran with
// idle lanes and the kernel stalled well short of memory bandwidth.
#ifndef DETERMINISTIC_LOGP_TILE_BLOCK_SIZE_NARROW
#define DETERMINISTIC_LOGP_TILE_BLOCK_SIZE_NARROW 64  // 1- and 2-byte inputs
#endif
#ifndef DETERMINISTIC_LOGP_TILE_BLOCK_SIZE_WIDE
#define DETERMINISTIC_LOGP_TILE_BLOCK_SIZE_WIDE 128  // 4-byte inputs
#endif
#ifndef DETERMINISTIC_LOGP_TILE_VECTOR_BYTES
#define DETERMINISTIC_LOGP_TILE_VECTOR_BYTES 16
#endif

template <int Bytes>
struct DeterministicLogpPacked;
template <>
struct DeterministicLogpPacked<16> {
    using type = int4;
};
template <>
struct DeterministicLogpPacked<8> {
    using type = int2;
};
template <>
struct DeterministicLogpPacked<4> {
    using type = int;
};
template <>
struct DeterministicLogpPacked<2> {
    using type = short;
};

template <typename scalar_t, int Vec>
__device__ __forceinline__ void deterministicLogpLoadVector(
    const scalar_t* __restrict__ pointer,
    scalar_t (&out)[Vec]) {
    constexpr int Bytes = Vec * static_cast<int>(sizeof(scalar_t));
    constexpr int PieceBytes = Bytes < 16 ? Bytes : 16;
    constexpr int Pieces = Bytes / PieceBytes;
    using packed_t = typename DeterministicLogpPacked<PieceBytes>::type;
    if ((reinterpret_cast<uintptr_t>(pointer) % alignof(packed_t)) == 0) {
        packed_t packed[Pieces];
#pragma unroll
        for (int piece = 0; piece < Pieces; ++piece) {
            packed[piece] = reinterpret_cast<const packed_t*>(pointer)[piece];
        }
        __builtin_memcpy(out, packed, Bytes);
    } else {
#pragma unroll
        for (int i = 0; i < Vec; ++i) {
            out[i] = pointer[i];
        }
    }
}

template <typename scalar_t, int Vec>
__device__ __forceinline__ void deterministicLogpStoreVector(
    scalar_t* __restrict__ pointer,
    const scalar_t (&in)[Vec]) {
    constexpr int Bytes = Vec * static_cast<int>(sizeof(scalar_t));
    constexpr int PieceBytes = Bytes < 16 ? Bytes : 16;
    constexpr int Pieces = Bytes / PieceBytes;
    using packed_t = typename DeterministicLogpPacked<PieceBytes>::type;
    if ((reinterpret_cast<uintptr_t>(pointer) % alignof(packed_t)) == 0) {
        packed_t packed[Pieces];
        __builtin_memcpy(packed, in, Bytes);
#pragma unroll
        for (int piece = 0; piece < Pieces; ++piece) {
            reinterpret_cast<packed_t*>(pointer)[piece] = packed[piece];
        }
    } else {
#pragma unroll
        for (int i = 0; i < Vec; ++i) {
            pointer[i] = in[i];
        }
    }
}

// Per-row, per-tile FP32 (max, sumexp) partials over the real-vocabulary part of
// the tile. The reduction tree is fixed by (BlockSize, Vec) and the position
// inside the tile, never by the row count or the tile's place in the global
// order, so the partials stay batch-invariant and TP-replicated.
template <typename scalar_t, int BlockSize, int Vec>
__global__ void __launch_bounds__(BlockSize) deterministic_logp_tile_stats_kernel(
    const scalar_t* __restrict__ logits,
    float* __restrict__ tile_max,
    float* __restrict__ tile_sum,
    int64_t rows,
    int64_t local_vocab,
    int64_t vocab_start,
    int64_t real_vocab,
    int64_t tile_size,
    int64_t local_tiles) {
    constexpr int Chunk = BlockSize * Vec;
    const int64_t tile_index = static_cast<int64_t>(blockIdx.y);
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= rows || tile_index >= local_tiles) {
        return;
    }
    const int64_t col_begin = tile_index * tile_size;
    const int64_t col_end = min(col_begin + tile_size, local_vocab);
    // Columns at or beyond the real vocabulary are padding; hoisting the bound
    // keeps the per-element predicate out of the inner loop.
    const int64_t real_end =
        min(col_end, max(real_vocab - vocab_start, static_cast<int64_t>(0)));
    const scalar_t* __restrict__ row_pointer = logits + row * local_vocab;

    float local_max = -std::numeric_limits<float>::infinity();
    for (int64_t base = col_begin + static_cast<int64_t>(threadIdx.x) * Vec; base < real_end;
         base += Chunk) {
        scalar_t values[Vec];
        if (base + Vec <= real_end) {
            deterministicLogpLoadVector<scalar_t, Vec>(row_pointer + base, values);
#pragma unroll
            for (int i = 0; i < Vec; ++i) {
                local_max = fmaxf(local_max, static_cast<float>(values[i]));
            }
        } else {
#pragma unroll
            for (int i = 0; i < Vec; ++i) {
                if (base + i < real_end) {
                    local_max = fmaxf(local_max, static_cast<float>(row_pointer[base + i]));
                }
            }
        }
    }
    const float max_value = deterministicBlockReduceMax<BlockSize>(local_max);
    __shared__ float row_max;
    if (threadIdx.x == 0) row_max = max_value;
    __syncthreads();
    const float tile_max_value = row_max;

    float sum_value = 0.0f;
    if (isfinite(tile_max_value)) {
        float local_sum = 0.0f;
        for (int64_t base = col_begin + static_cast<int64_t>(threadIdx.x) * Vec; base < real_end;
             base += Chunk) {
            scalar_t values[Vec];
            if (base + Vec <= real_end) {
                deterministicLogpLoadVector<scalar_t, Vec>(row_pointer + base, values);
#pragma unroll
                for (int i = 0; i < Vec; ++i) {
                    local_sum += expf(static_cast<float>(values[i]) - tile_max_value);
                }
            } else {
#pragma unroll
                for (int i = 0; i < Vec; ++i) {
                    if (base + i < real_end) {
                        local_sum +=
                            expf(static_cast<float>(row_pointer[base + i]) - tile_max_value);
                    }
                }
            }
        }
        sum_value = deterministicBlockReduceSum<BlockSize>(local_sum);
    }
    if (threadIdx.x == 0) {
        const int64_t output_index = row * local_tiles + tile_index;
        tile_max[output_index] = tile_max_value;
        tile_sum[output_index] = sum_value;
    }
}


#ifndef DETERMINISTIC_LOGP_BACKWARD_BLOCK_SIZE
#define DETERMINISTIC_LOGP_BACKWARD_BLOCK_SIZE 256
#endif

// grad = coef_logp * (onehot - p) + coef_lse * p, with p = exp(z - lse) on finite
// rows, 0 on non-finite rows and on padding columns. Purely elementwise, so the
// result does not depend on the launch geometry or on the batch.
template <typename scalar_t, int BlockSize, int Vec>
__global__ void __launch_bounds__(BlockSize) deterministic_logp_backward_kernel(
    const scalar_t* __restrict__ logits,
    const float* __restrict__ lse,
    const float* __restrict__ coef_logp,
    const float* __restrict__ coef_lse,
    const int64_t* __restrict__ target_local,
    scalar_t* __restrict__ grad,
    int64_t rows,
    int64_t local_vocab,
    int64_t vocab_start,
    int64_t real_vocab,
    bool has_lse_grad) {
    constexpr int Chunk = BlockSize * Vec;
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= rows) {
        return;
    }
    const float row_lse = lse[row];
    const bool finite_row = isfinite(row_lse);
    const float lse_safe = finite_row ? row_lse : 0.0f;
    const float g_logp = coef_logp[row];
    const float g_lse = has_lse_grad ? coef_lse[row] : 0.0f;
    const int64_t hit = target_local[row];
    const int64_t real_end =
        min(local_vocab, max(real_vocab - vocab_start, static_cast<int64_t>(0)));
    const scalar_t* __restrict__ row_in = logits + row * local_vocab;
    scalar_t* __restrict__ row_out = grad + row * local_vocab;
    const int64_t stride = static_cast<int64_t>(gridDim.y) * Chunk;

    for (int64_t base = static_cast<int64_t>(blockIdx.y) * Chunk +
                        static_cast<int64_t>(threadIdx.x) * Vec;
         base < local_vocab;
         base += stride) {
        scalar_t values[Vec];
        scalar_t outputs[Vec];
        const bool full = base + Vec <= local_vocab;
        if (full) {
            deterministicLogpLoadVector<scalar_t, Vec>(row_in + base, values);
        } else {
#pragma unroll
            for (int i = 0; i < Vec; ++i) {
                values[i] =
                    (base + i < local_vocab) ? row_in[base + i] : static_cast<scalar_t>(0.0f);
            }
        }
#pragma unroll
        for (int i = 0; i < Vec; ++i) {
            const int64_t col = base + i;
            float value = 0.0f;
            if (col < real_end) {
                const float p =
                    finite_row ? expf(static_cast<float>(values[i]) - lse_safe) : 0.0f;
                const float onehot = (col == hit) ? 1.0f : 0.0f;
                value = g_logp * (onehot - p);
                if (has_lse_grad) {
                    value = value + g_lse * p;
                }
            }
            outputs[i] = static_cast<scalar_t>(value);
        }
        if (full) {
            deterministicLogpStoreVector<scalar_t, Vec>(row_out + base, outputs);
        } else {
#pragma unroll
            for (int i = 0; i < Vec; ++i) {
                if (base + i < local_vocab) {
                    row_out[base + i] = outputs[i];
                }
            }
        }
    }
}

} // namespace

std::vector<torch::Tensor> deterministic_logp_tile_stats(
    torch::Tensor logits,
    int64_t vocab_start,
    int64_t real_vocab,
    int64_t num_tiles) {
    TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA/ROCm tensor");
    TORCH_CHECK(logits.dim() == 2, "logits must be 2D [tokens, local_vocab]");
    TORCH_CHECK(logits.scalar_type() == at::ScalarType::Half ||
                    logits.scalar_type() == at::ScalarType::BFloat16 ||
                    logits.scalar_type() == at::ScalarType::Float,
                "logits must be float16, bfloat16, or float32");
    TORCH_CHECK(vocab_start >= 0 && real_vocab > 0 && num_tiles > 0,
                "invalid vocabulary metadata");
    auto input = logits.contiguous();
    const int64_t rows = input.size(0);
    const int64_t local_vocab = input.size(1);
    TORCH_CHECK(local_vocab > 0 && local_vocab % num_tiles == 0,
                "local_vocab must be divisible by num_tiles");
    const int64_t tile_size = local_vocab / num_tiles;
    auto options = input.options().dtype(at::ScalarType::Float);
    auto tile_max = torch::empty({rows, num_tiles}, options);
    auto tile_sum = torch::empty({rows, num_tiles}, options);
    const dim3 grid(static_cast<unsigned int>(rows), static_cast<unsigned int>(num_tiles), 1);
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
        "deterministic_logp_tile_stats", ([&] {
            // 16-byte loads per thread; narrow inputs also want the smaller block
            // so a tile splits into enough chunks to keep every lane busy.
            constexpr int Vec = DETERMINISTIC_LOGP_TILE_VECTOR_BYTES / sizeof(scalar_t);
            constexpr int BlockSize = sizeof(scalar_t) >= 4
                                          ? DETERMINISTIC_LOGP_TILE_BLOCK_SIZE_WIDE
                                          : DETERMINISTIC_LOGP_TILE_BLOCK_SIZE_NARROW;
            deterministic_logp_tile_stats_kernel<scalar_t, BlockSize, Vec>
                <<<grid, BlockSize, 0, stream>>>(
                    input.data_ptr<scalar_t>(), tile_max.data_ptr<float>(),
                    tile_sum.data_ptr<float>(), rows, local_vocab, vocab_start,
                    real_vocab, tile_size, num_tiles);
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {tile_max, tile_sum};
}

torch::Tensor deterministic_logp_backward(
    torch::Tensor logits,
    torch::Tensor lse,
    torch::Tensor coef_logp,
    torch::Tensor coef_lse,
    torch::Tensor target_local,
    int64_t vocab_start,
    int64_t real_vocab,
    bool has_lse_grad) {
    TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA/ROCm tensor");
    TORCH_CHECK(logits.dim() == 2, "logits must be 2D [tokens, local_vocab]");
    TORCH_CHECK(logits.scalar_type() == at::ScalarType::Half ||
                    logits.scalar_type() == at::ScalarType::BFloat16 ||
                    logits.scalar_type() == at::ScalarType::Float,
                "logits must be float16, bfloat16, or float32");
    TORCH_CHECK(vocab_start >= 0 && real_vocab > 0, "invalid vocabulary metadata");
    auto input = logits.contiguous();
    const int64_t rows = input.size(0);
    const int64_t local_vocab = input.size(1);
    auto check_row_vector =
        [&](const torch::Tensor& tensor, at::ScalarType dtype, const char* name) {
            TORCH_CHECK(tensor.is_cuda() && tensor.device() == input.device(), name,
                        " must live on the logits device");
            TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
            TORCH_CHECK(tensor.dim() == 1 && tensor.size(0) == rows, name,
                        " must have one entry per token");
            TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
        };
    check_row_vector(lse, at::ScalarType::Float, "lse");
    check_row_vector(coef_logp, at::ScalarType::Float, "coef_logp");
    check_row_vector(coef_lse, at::ScalarType::Float, "coef_lse");
    check_row_vector(target_local, at::ScalarType::Long, "target_local");
    auto grad = torch::empty_like(input);
    if (rows == 0 || local_vocab == 0) {
        return grad;
    }
    TORCH_CHECK(rows <= std::numeric_limits<int>::max(), "row count exceeds CUDA grid-x limit");
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
        "deterministic_logp_backward", ([&] {
            constexpr int Vec = DETERMINISTIC_LOGP_TILE_VECTOR_BYTES / sizeof(scalar_t);
            constexpr int BlockSize = DETERMINISTIC_LOGP_BACKWARD_BLOCK_SIZE;
            constexpr int Chunk = BlockSize * Vec;
            const int64_t chunks = (local_vocab + Chunk - 1) / Chunk;
            const dim3 grid(static_cast<unsigned int>(rows),
                            static_cast<unsigned int>(std::min<int64_t>(chunks, 65535)), 1);
            deterministic_logp_backward_kernel<scalar_t, BlockSize, Vec>
                <<<grid, BlockSize, 0, stream>>>(
                    input.data_ptr<scalar_t>(), lse.data_ptr<float>(),
                    coef_logp.data_ptr<float>(), coef_lse.data_ptr<float>(),
                    target_local.data_ptr<int64_t>(), grad.data_ptr<scalar_t>(), rows,
                    local_vocab, vocab_start, real_vocab, has_lse_grad);
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad;
}
