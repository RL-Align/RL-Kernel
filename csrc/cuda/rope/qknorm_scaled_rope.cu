// SPDX-License-Identifier: Apache-2.0
// F2 forward: one 128-thread block per token/head, FP32 half-split tree v1.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <limits>

namespace {
using bf16 = __nv_bfloat16;

// One block scans positions, then publishes one integer flag for one host read.
// This integer predicate reduction is separate from the fixed FP32 norm tree.
__global__ void f2_check_positions(const int64_t* positions, int64_t count,
                                   int64_t coverage, int* invalid) {
    bool bad = false;
    for (int64_t i = threadIdx.x; i < count; i += blockDim.x) {
        const int64_t position = positions[i];
        bad |= position < 0 || position >= coverage;
    }
    const int any_bad = __syncthreads_or(bad);
    if (threadIdx.x == 0) *invalid = any_bad;
}

__device__ __forceinline__ float round_bf16(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}

// Preserve the ordered 64/32/16/8/4/2/1 FP32 tree.
// Only SM80 uses the validated shared-memory/shuffle mapping.
__device__ __forceinline__ void fixed_tree_reduce(float* acc, int d) {
#if __CUDA_ARCH__ == 800
    __syncthreads();
    if (d < 64) acc[d] = __fadd_rn(acc[d], acc[d + 64]);
    __syncthreads();
    if (d < 32) acc[d] = __fadd_rn(acc[d], acc[d + 32]);
    __syncthreads();
    if (d < 32) {
        float v = acc[d];
        #pragma unroll
        for (int stride = 16; stride; stride >>= 1) {
            const float peer = __shfl_down_sync(0xffffffffu, v, stride, 32);
            if (d < stride) v = __fadd_rn(v, peer);
        }
        if (d == 0) acc[0] = v;
    }
    __syncthreads();
#else
    __syncthreads();
    #pragma unroll
    for (int stride = 64; stride; stride >>= 1) {
        if (d < stride) acc[d] = __fadd_rn(acc[d], acc[d + stride]);
        __syncthreads();
    }
#endif
}

template<bool Local>
__global__ void f2_forward(
    const bf16* q, const bf16* k, const int64_t* positions,
    const bf16* cosine, const bf16* sine, bf16* oq, bf16* ok,
    float* rq, float* rk, int64_t hq, int64_t hk
) {
    const int d = threadIdx.x;
    const int64_t row = blockIdx.x;
    const int64_t token = row / (hq + hk);
    const int64_t combined = row % (hq + hk);
    const bool is_q = combined < hq;
    const int64_t head = is_q ? combined : combined - hq;
    const int64_t head_row = token * (is_q ? hq : hk) + head;
    const bf16* src = is_q ? q : k;
    bf16* dst = is_q ? oq : ok;
    float* rdst = is_q ? rq : rk;
    __shared__ float acc[128];
    __shared__ float normalized[128];
    const float x = __bfloat162float(src[head_row * 128 + d]);
    acc[d] = __fmul_rn(x, x);
    fixed_tree_reduce(acc, d);
    if (d == 0) {
        const float variance = __fadd_rn(__fmul_rn(acc[0], 1.0f / 128), 1.0e-5f);
        acc[0] = rsqrtf(variance);
        rdst[head_row] = acc[0];
    }
    __syncthreads();
    float value = round_bf16(__fmul_rn(x, acc[0]));
    if (is_q) value = round_bf16(__fmul_rn(value, 3.87f));
    if constexpr (Local) {
        normalized[d] = value;
        __syncthreads();
        const int64_t idx = positions[token] * 64 + d % 64;
        const float c = __bfloat162float(cosine[idx]);
        const float s = __bfloat162float(sine[idx]);
        const float other = normalized[(d + 64) % 128];
        const float rotated = d < 64 ? -other : other;
        const float a = round_bf16(__fmul_rn(value, c));
        const float b = round_bf16(__fmul_rn(rotated, s));
        value = __fadd_rn(a, b);
    }
    dst[head_row * 128 + d] = __float2bfloat16_rn(value);
}
template<bool Local>
__global__ void f2_backward(
    const bf16* q, const bf16* k, const bf16* gq, const bf16* gk,
    const float* rq, const float* rk, const int64_t* positions,
    const bf16* cosine, const bf16* sine, bf16* dq, bf16* dk,
    int64_t hq, int64_t hk
) {
    const int d = threadIdx.x;
    const int64_t row = blockIdx.x;
    const int64_t token = row / (hq + hk);
    const int64_t combined = row % (hq + hk);
    const bool is_q = combined < hq;
    const int64_t head = is_q ? combined : combined - hq;
    const int64_t head_row = token * (is_q ? hq : hk) + head;
    const bf16* src = is_q ? q : k;
    const bf16* grad = is_q ? gq : gk;
    const float* saved_r = is_q ? rq : rk;
    bf16* dst = is_q ? dq : dk;
    __shared__ float acc[128];
    __shared__ float incoming[128];
    const float x = __bfloat162float(src[head_row * 128 + d]);
    float g = __bfloat162float(grad[head_row * 128 + d]);
    if constexpr (Local) {
        incoming[d] = g;
        __syncthreads();
        const int64_t idx = positions[token] * 64 + d % 64;
        const float c = __bfloat162float(cosine[idx]);
        const float s = __bfloat162float(sine[idx]);
        const float partner = incoming[(d + 64) % 128];
        const float a = round_bf16(__fmul_rn(g, c));
        const float b = round_bf16(__fmul_rn(partner, s));
        g = round_bf16(__fadd_rn(a, d < 64 ? b : -b));
    }
    if (is_q) g = round_bf16(__fmul_rn(g, 3.87f));
    // VJP v2: normalize in FP32 before the dot, avoiding r^3 underflow.
    const float r = saved_r[head_row];
    const float u = __fmul_rn(x, r);
    acc[d] = __fmul_rn(g, u);
    fixed_tree_reduce(acc, d);
    const float mean_dot = __fmul_rn(acc[0], 1.0f / 128);
    const float correction = __fmul_rn(u, mean_dot);
    const float dx = __fmul_rn(r, __fsub_rn(g, correction));
    dst[head_row * 128 + d] = __float2bfloat16_rn(dx);
}
}  // namespace

struct F2Inputs {
    int64_t rows;
    const int64_t* positions;
    const bf16* cosine;
    const bf16* sine;
};

static F2Inputs check_f2_inputs(
    const torch::Tensor& q, const torch::Tensor& k,
    const c10::optional<torch::Tensor>& positions,
    const c10::optional<torch::Tensor>& cosine,
    const c10::optional<torch::Tensor>& sine, bool local
) {
#ifdef RL_KERNEL_F2_FAST_MATH_BUILD
    TORCH_CHECK(false, "F2 strict forward is unavailable in a fast-math build");
#endif
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && q.device() == k.device(),
                "F2 Q/K must share a CUDA device");
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && q.size(2) == 128 && k.size(2) == 128,
                "F2 requires [T, H, 128]");
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(1) > 0 && k.size(1) > 0,
                "F2 invalid token/head counts");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16 &&
                q.is_contiguous() && k.is_contiguous(), "F2 requires contiguous BF16 Q/K");
    const c10::cuda::CUDAGuard guard(q.device());
    TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major >= 8, "F2 requires SM80+");
    const int64_t heads = q.size(1) + k.size(1);
    TORCH_CHECK(q.size(0) <= std::numeric_limits<int>::max() / heads,
                "F2 row count exceeds supported grid");
    const int64_t rows = q.size(0) * heads;
    const int64_t* p = nullptr;
    const bf16* c = nullptr;
    const bf16* s = nullptr;
    if (local) {
        TORCH_CHECK(positions.has_value() && cosine.has_value() && sine.has_value(),
                    "F2 local layers require positions and tables");
        const auto& pos = *positions;
        const auto& cos = *cosine;
        const auto& sin = *sine;
        TORCH_CHECK(pos.device() == q.device() && pos.scalar_type() == at::kLong &&
                    pos.is_contiguous() && pos.dim() == 1 && pos.size(0) == q.size(0),
                    "F2 positions must be contiguous int64 [T] on the Q/K device");
        TORCH_CHECK(cos.dim() == 2 && cos.size(1) == 64 && cos.sizes() == sin.sizes() &&
                    cos.size(0) <= 131072, "F2 tables require matching [P, 64], P <= 131072");
        TORCH_CHECK(cos.device() == q.device() && sin.device() == q.device() &&
                    cos.scalar_type() == at::kBFloat16 && sin.scalar_type() == at::kBFloat16 &&
                    cos.is_contiguous() && sin.is_contiguous(),
                    "F2 tables must be contiguous BF16 on the Q/K device");
        if (pos.numel()) {
            auto invalid = torch::empty({}, pos.options().dtype(at::kInt));
            f2_check_positions<<<1, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                pos.data_ptr<int64_t>(), pos.numel(), cos.size(0), invalid.data_ptr<int>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            TORCH_CHECK(invalid.item<int>() == 0, "F2 positions outside shared table");
        }
        p = pos.data_ptr<int64_t>();
        c = reinterpret_cast<const bf16*>(cos.data_ptr<at::BFloat16>());
        s = reinterpret_cast<const bf16*>(sin.data_ptr<at::BFloat16>());
    }
    return {rows, p, c, s};
}

std::vector<torch::Tensor> qknorm_scaled_rope_forward_cuda(
    torch::Tensor q, torch::Tensor k,
    c10::optional<torch::Tensor> positions,
    c10::optional<torch::Tensor> cosine,
    c10::optional<torch::Tensor> sine, bool local
) {
    const auto inputs = check_f2_inputs(q, k, positions, cosine, sine, local);
    TORCH_CHECK(!(at::GradMode::is_enabled() && (q.requires_grad() || k.requires_grad())),
                "F2 raw forward has no autograd; use qknorm_scaled_rope for backward");
    const c10::cuda::CUDAGuard guard(q.device());
    const auto rows = inputs.rows;
    const auto* p = inputs.positions;
    const auto* c = inputs.cosine;
    const auto* s = inputs.sine;
    auto oq = torch::empty_like(q);
    auto ok = torch::empty_like(k);
    auto rq = torch::empty({q.size(0), q.size(1)}, q.options().dtype(at::kFloat));
    auto rk = torch::empty({k.size(0), k.size(1)}, k.options().dtype(at::kFloat));
    if (!rows) return {oq, ok, rq, rk};
    const auto* qp = reinterpret_cast<const bf16*>(q.data_ptr<at::BFloat16>());
    const auto* kp = reinterpret_cast<const bf16*>(k.data_ptr<at::BFloat16>());
    auto* qop = reinterpret_cast<bf16*>(oq.data_ptr<at::BFloat16>());
    auto* kop = reinterpret_cast<bf16*>(ok.data_ptr<at::BFloat16>());
    auto stream = at::cuda::getCurrentCUDAStream();
    if (local) {
        f2_forward<true><<<rows, 128, 0, stream>>>(qp, kp, p, c, s, qop, kop,
            rq.data_ptr<float>(), rk.data_ptr<float>(), q.size(1), k.size(1));
    } else {
        f2_forward<false><<<rows, 128, 0, stream>>>(qp, kp, p, c, s, qop, kop,
            rq.data_ptr<float>(), rk.data_ptr<float>(), q.size(1), k.size(1));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {oq, ok, rq, rk};
}

std::vector<torch::Tensor> qknorm_scaled_rope_backward_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor gq, torch::Tensor gk,
    torch::Tensor rq, torch::Tensor rk,
    c10::optional<torch::Tensor> positions,
    c10::optional<torch::Tensor> cosine,
    c10::optional<torch::Tensor> sine, bool local
) {
    const auto inputs = check_f2_inputs(q, k, positions, cosine, sine, local);
    const c10::cuda::CUDAGuard guard(q.device());
    TORCH_CHECK(gq.device() == q.device() && gk.device() == q.device() &&
                gq.scalar_type() == at::kBFloat16 && gk.scalar_type() == at::kBFloat16 &&
                gq.sizes() == q.sizes() && gk.sizes() == k.sizes() &&
                gq.is_contiguous() && gk.is_contiguous(),
                "F2 backward gradients must match contiguous BF16 Q/K");
    TORCH_CHECK(rq.device() == q.device() && rk.device() == q.device() &&
                rq.scalar_type() == at::kFloat && rk.scalar_type() == at::kFloat &&
                rq.dim() == 2 && rk.dim() == 2 && rq.size(0) == q.size(0) &&
                rq.size(1) == q.size(1) && rk.size(0) == k.size(0) &&
                rk.size(1) == k.size(1) && rq.is_contiguous() && rk.is_contiguous(),
                "F2 backward requires contiguous FP32 rstd matching [T, H]");
    auto dq = torch::empty_like(q);
    auto dk = torch::empty_like(k);
    if (!inputs.rows) return {dq, dk};
    const auto* qp = reinterpret_cast<const bf16*>(q.data_ptr<at::BFloat16>());
    const auto* kp = reinterpret_cast<const bf16*>(k.data_ptr<at::BFloat16>());
    const auto* gqp = reinterpret_cast<const bf16*>(gq.data_ptr<at::BFloat16>());
    const auto* gkp = reinterpret_cast<const bf16*>(gk.data_ptr<at::BFloat16>());
    auto* dqp = reinterpret_cast<bf16*>(dq.data_ptr<at::BFloat16>());
    auto* dkp = reinterpret_cast<bf16*>(dk.data_ptr<at::BFloat16>());
    auto stream = at::cuda::getCurrentCUDAStream();
    if (local) {
        f2_backward<true><<<inputs.rows, 128, 0, stream>>>(qp, kp, gqp, gkp,
            rq.data_ptr<float>(), rk.data_ptr<float>(), inputs.positions,
            inputs.cosine, inputs.sine, dqp, dkp, q.size(1), k.size(1));
    } else {
        f2_backward<false><<<inputs.rows, 128, 0, stream>>>(qp, kp, gqp, gkp,
            rq.data_ptr<float>(), rk.data_ptr<float>(), nullptr,
            nullptr, nullptr, dqp, dkp, q.size(1), k.size(1));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dq, dk};
}
