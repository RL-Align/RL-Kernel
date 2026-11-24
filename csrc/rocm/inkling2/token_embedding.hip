// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// INK-06 (Inkling 2 Mini) padded token embedding, ROCm-only.
// Compiled directly by hipcc for ROCm builds; the CUDA path lives in the
// SM90-only csrc/cuda/embedding/embedding_lm_head_sm90.cu and is not built here.
//
// Determinism contract
//   forward : a pure gather. Each output row is a byte-exact copy of one weight
//             row (no arithmetic, no cast), so the result is bitwise equal to
//             torch.nn.functional.embedding for every dtype and independent of
//             the batch, of other rows and of the launch configuration.
//             Out-of-range ids never read outside the table: the row is
//             zero-filled. The Python entry point validates ids first and
//             raises ValueError exactly like the reference, so that path is
//             only a memory-safety backstop.
//   backward: no float atomics. Token positions are stable-sorted by id, then one
//             block owns each distinct id and accumulates its gradient rows in
//             ascending token order in FP32 and casts once at the end. This is
//             the same accumulation order and cast point as the CUDA kernel's
//             token loop, independent of batch composition and launch config.
//
// Tuned for MI300X (gfx942, wave64): 256-thread blocks (4 waves), one block per
// token row, widest of 16/8/4/2/1-byte vector copies that the row size and the
// pointer alignment allow (hidden 2304 in bf16 = 4608 B -> 16-byte copies).

#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPException.h>
#include <c10/hip/HIPGuard.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>
#include <torch/extension.h>

namespace {

constexpr int kThreads = 256;  // multiple of the 64-lane wavefront

bool is_supported_float_dtype(at::ScalarType dtype) {
    return dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16;
}

template <typename V>
struct VecTraits;
template <> struct VecTraits<uint4> { static constexpr int kBytes = 16; };
template <> struct VecTraits<uint2> { static constexpr int kBytes = 8; };
template <> struct VecTraits<uint32_t> { static constexpr int kBytes = 4; };
template <> struct VecTraits<uint16_t> { static constexpr int kBytes = 2; };
template <> struct VecTraits<uint8_t> { static constexpr int kBytes = 1; };

template <typename id_t, typename V>
__global__ __launch_bounds__(kThreads) void token_embedding_forward_kernel(
    const id_t *__restrict__ token_ids,
    const V *__restrict__ weight,
    V *__restrict__ output,
    int64_t vocab_size,
    int64_t row_vecs) {
    const int64_t token = blockIdx.x;
    const int64_t id = static_cast<int64_t>(token_ids[token]);
    const bool valid = id >= 0 && id < vocab_size;
    const V *src = weight + (valid ? id : 0) * row_vecs;
    V *dst = output + token * row_vecs;
    V zero;
    __builtin_memset(&zero, 0, sizeof(V));
    for (int64_t i = threadIdx.x; i < row_vecs; i += kThreads) {
        dst[i] = valid ? src[i] : zero;
    }
}

// sorted_ids/sorted_pos come from a stable sort, so equal ids appear in
// ascending token order. Only the first block of each equal-id run does work.
template <typename scalar_t>
__global__ __launch_bounds__(kThreads) void token_embedding_backward_kernel(
    const int64_t *__restrict__ sorted_ids,
    const int64_t *__restrict__ sorted_pos,
    const scalar_t *__restrict__ grad_output,
    scalar_t *__restrict__ grad_weight,
    int64_t num_tokens,
    int64_t hidden_size,
    int64_t vocab_size) {
    const int64_t first = blockIdx.x;
    const int64_t id = sorted_ids[first];
    if (first > 0 && sorted_ids[first - 1] == id) return;
    if (id < 0 || id >= vocab_size) return;
    for (int64_t col = threadIdx.x; col < hidden_size; col += kThreads) {
        float acc = 0.0f;
        for (int64_t k = first; k < num_tokens && sorted_ids[k] == id; ++k) {
            acc += static_cast<float>(grad_output[sorted_pos[k] * hidden_size + col]);
        }
        grad_weight[id * hidden_size + col] = static_cast<scalar_t>(acc);
    }
}

template <typename id_t, typename V>
void launch_forward(const at::Tensor &ids, const at::Tensor &weight, at::Tensor &output,
                    int64_t num_tokens, int64_t row_bytes, hipStream_t stream) {
    constexpr int64_t vb = VecTraits<V>::kBytes;
    token_embedding_forward_kernel<id_t, V>
        <<<static_cast<unsigned int>(num_tokens), kThreads, 0, stream>>>(
            ids.data_ptr<id_t>(), reinterpret_cast<const V *>(weight.data_ptr()),
            reinterpret_cast<V *>(output.data_ptr()), weight.size(0), row_bytes / vb);
}

template <typename id_t>
void dispatch_forward(const at::Tensor &ids, const at::Tensor &weight, at::Tensor &output,
                      int64_t num_tokens, hipStream_t stream) {
    const int64_t row_bytes = weight.size(1) * weight.element_size();
    const auto wp = reinterpret_cast<uintptr_t>(weight.data_ptr());
    const auto op = reinterpret_cast<uintptr_t>(output.data_ptr());
    auto aligned = [&](int64_t n) {
        return row_bytes % n == 0 && wp % n == 0 && op % n == 0;
    };
    if (aligned(16)) launch_forward<id_t, uint4>(ids, weight, output, num_tokens, row_bytes, stream);
    else if (aligned(8)) launch_forward<id_t, uint2>(ids, weight, output, num_tokens, row_bytes, stream);
    else if (aligned(4)) launch_forward<id_t, uint32_t>(ids, weight, output, num_tokens, row_bytes, stream);
    else if (aligned(2)) launch_forward<id_t, uint16_t>(ids, weight, output, num_tokens, row_bytes, stream);
    else launch_forward<id_t, uint8_t>(ids, weight, output, num_tokens, row_bytes, stream);
}

}  // namespace

torch::Tensor inkling2_token_embedding_rocm(torch::Tensor token_ids, torch::Tensor weight) {
    TORCH_CHECK(token_ids.is_cuda() && weight.is_cuda(),
                "token_ids and weight must be HIP tensors");
    TORCH_CHECK(token_ids.device() == weight.device(),
                "token_ids and weight must share a device");
    TORCH_CHECK(token_ids.scalar_type() == at::kLong || token_ids.scalar_type() == at::kInt,
                "token_ids must be int32 or int64");
    TORCH_CHECK(weight.dim() == 2, "embedding weight must be [vocab, hidden]");
    TORCH_CHECK(is_supported_float_dtype(weight.scalar_type()),
                "inkling2 token embedding supports fp32, fp16, and bf16 weights");
    TORCH_CHECK(weight.is_contiguous(), "embedding weight must be contiguous");
    const c10::cuda::CUDAGuard device_guard(weight.device());
    auto ids = token_ids.contiguous();
    std::vector<int64_t> out_shape(ids.sizes().begin(), ids.sizes().end());
    out_shape.push_back(weight.size(1));
    auto output = torch::empty(out_shape, weight.options());
    const int64_t num_tokens = ids.numel();
    if (num_tokens == 0 || weight.size(1) == 0) return output;
    TORCH_CHECK(num_tokens <= std::numeric_limits<int>::max(),
                "too many tokens for one launch");
    auto stream = at::cuda::getCurrentCUDAStream();
    if (ids.scalar_type() == at::kLong) {
        dispatch_forward<int64_t>(ids, weight, output, num_tokens, stream);
    } else {
        dispatch_forward<int32_t>(ids, weight, output, num_tokens, stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor inkling2_token_embedding_backward_rocm(torch::Tensor token_ids,
                                                     torch::Tensor grad_output,
                                                     int64_t vocab_size) {
    TORCH_CHECK(token_ids.is_cuda() && grad_output.is_cuda(),
                "token_ids and grad_output must be HIP tensors");
    TORCH_CHECK(token_ids.device() == grad_output.device(),
                "token_ids and grad_output must share a device");
    TORCH_CHECK(token_ids.scalar_type() == at::kLong || token_ids.scalar_type() == at::kInt,
                "token_ids must be int32 or int64");
    TORCH_CHECK(grad_output.dim() == token_ids.dim() + 1,
                "grad_output must have token_ids dimensions plus hidden dimension");
    TORCH_CHECK(is_supported_float_dtype(grad_output.scalar_type()),
                "embedding gradient must be fp32, fp16, or bf16");
    TORCH_CHECK(vocab_size > 0, "vocab_size must be positive");
    const c10::cuda::CUDAGuard device_guard(grad_output.device());
    const int64_t num_tokens = token_ids.numel();
    const int64_t hidden_size = grad_output.size(-1);
    TORCH_CHECK(grad_output.numel() == num_tokens * hidden_size,
                "grad_output shape does not match token_ids");
    auto grad_2d = grad_output.contiguous().reshape({num_tokens, hidden_size});
    auto grad_weight = torch::zeros({vocab_size, hidden_size}, grad_output.options());
    if (num_tokens == 0 || hidden_size == 0) return grad_weight;
    TORCH_CHECK(num_tokens <= std::numeric_limits<int>::max(),
                "too many tokens for one launch");
    auto ids = token_ids.reshape({num_tokens}).to(at::kLong).contiguous();
    auto sorted = at::sort(ids, /*stable=*/true, /*dim=*/0, /*descending=*/false);
    auto sorted_ids = std::get<0>(sorted).contiguous();
    auto sorted_pos = std::get<1>(sorted).contiguous();
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, grad_2d.scalar_type(), "inkling2_token_embedding_backward_rocm",
        [&] {
            token_embedding_backward_kernel<scalar_t>
                <<<static_cast<unsigned int>(num_tokens), kThreads, 0, stream>>>(
                    sorted_ids.data_ptr<int64_t>(), sorted_pos.data_ptr<int64_t>(),
                    grad_2d.data_ptr<scalar_t>(), grad_weight.data_ptr<scalar_t>(), num_tokens,
                    hidden_size, vocab_size);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_weight;
}
