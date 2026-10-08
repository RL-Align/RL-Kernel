// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors

// v1: FP32 throughout, precise non-FMA arithmetic, no atomics. Unfused stages:
// coefficients -> step/mean -> density -> tile tree -> ascending tile fold.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace {
constexpr int kTile = 256;
constexpr int64_t kMaxElements = 1LL << 24;

__global__ void coefficients(const float* p, float* c, float* std_out, int batch) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch) return;
    float s = p[4 * row], sn = p[4 * row + 1];
    float sm = p[4 * row + 2], level = p[4 * row + 3];
    float dt = __fsub_rn(sn, s);
    float std = __fmul_rn(__fsqrt_rn(__fdiv_rn(s, __fsub_rn(1.f, s == 1.f ? sm : s))), level);
    float v = __fmul_rn(std, std);
    float denominator = __fmul_rn(2.f, s);
    c[4 * row] = __fadd_rn(1.f, __fmul_rn(__fdiv_rn(v, denominator), dt));
    c[4 * row + 1] = __fmul_rn(
        __fadd_rn(1.f, __fdiv_rn(__fmul_rn(v, __fsub_rn(1.f, s)), denominator)), dt);
    c[4 * row + 2] = __fmul_rn(std, __fsqrt_rn(-dt));
    c[4 * row + 3] = std;
    std_out[row] = std;
}

__global__ void step(const float* x, const float* v, const float* aux, const float* c,
                     float* mean, float* target, int64_t width, bool replay) {
    int64_t col = int64_t(blockIdx.x) * kTile + threadIdx.x;
    int row = blockIdx.y;
    if (col >= width) return;
    int64_t i = int64_t(row) * width + col;
    float m = __fadd_rn(__fmul_rn(x[i], c[4 * row]), __fmul_rn(v[i], c[4 * row + 1]));
    mean[i] = m;
    target[i] = replay ? aux[i] : __fadd_rn(m, __fmul_rn(c[4 * row + 2], aux[i]));
}

__global__ void density(const float* target, const float* mean, const float* c,
                        float* values, int64_t width) {
    int64_t col = int64_t(blockIdx.x) * kTile + threadIdx.x;
    int row = blockIdx.y;
    if (col >= width) return;
    int64_t i = int64_t(row) * width + col;
    float r = __fsub_rn(target[i], mean[i]);
    float tau = c[4 * row + 2];
    float denom = __fmul_rn(2.f, __fmul_rn(tau, tau));
    float value = __fdiv_rn(-__fmul_rn(r, r), denom);
    value = __fsub_rn(value, logf(tau));
    values[i] = __fsub_rn(value, logf(__fsqrt_rn(6.283185307179586f)));
}

__global__ void tile_sum(const float* values, float* partial, int64_t width, int tiles) {
    __shared__ float tree[kTile];
    int lane = threadIdx.x;
    int64_t col = int64_t(blockIdx.x) * kTile + lane;
    int row = blockIdx.y;
    tree[lane] = col < width ? values[int64_t(row) * width + col] : 0.f;
    __syncthreads();
    for (int stride = 1; stride < kTile; stride *= 2) {
        if (lane % (2 * stride) == 0) tree[lane] = __fadd_rn(tree[lane], tree[lane + stride]);
        __syncthreads();
    }
    if (lane == 0) partial[int64_t(row) * tiles + blockIdx.x] = tree[0];
}

__global__ void final_mean(const float* partial, float* logp, int64_t width, int tiles,
                          int batch) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch) return;
    float total = 0.f;
    for (int tile = 0; tile < tiles; ++tile)
        total = __fadd_rn(total, partial[int64_t(row) * tiles + tile]);
    logp[row] = __fdiv_rn(total, float(width));
}

__global__ void vjp(const float* target, const float* mean, const float* c,
                    const float* gt, const float* gl, const float* gm,
                    float* dx, float* dv, int64_t width, bool replay) {
    int64_t col = int64_t(blockIdx.x) * kTile + threadIdx.x;
    int row = blockIdx.y;
    if (col >= width) return;
    int64_t i = int64_t(row) * width + col;
    float tau = c[4 * row + 2];
    float dmean = __fmul_rn(
        __fdiv_rn(gl[row], float(width)),
        __fdiv_rn(__fsub_rn(target[i], mean[i]), __fmul_rn(tau, tau)));
    dmean = __fadd_rn(dmean, gm[i]);
    if (!replay) dmean = __fadd_rn(dmean, gt[i]);
    dx[i] = __fmul_rn(dmean, c[4 * row]);
    dv[i] = __fmul_rn(dmean, c[4 * row + 1]);
}

void check_tensor(const torch::Tensor& t, const torch::Tensor& anchor, const char* name) {
    TORCH_CHECK(t.is_cuda() && t.device() == anchor.device(), name, ": same CUDA device required");
    TORCH_CHECK(t.scalar_type() == torch::kFloat32 && t.is_contiguous(), name,
                ": contiguous FP32 required");
}

void check_geometry(const torch::Tensor& x) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32 && x.is_contiguous(),
                "sample: contiguous CUDA FP32 required");
    TORCH_CHECK(x.dim() >= 2 && x.size(0) > 0 && x.numel() > 0, "nonempty [B,...] required");
    TORCH_CHECK(x.size(0) <= 65535, "CUDA grid supports at most 65535 samples");
    TORCH_CHECK(x.numel() / x.size(0) <= kMaxElements, "latent width exceeds v1 limit");
}
}  // namespace

bool flow_sde_strict_math() {
#if defined(RLK_FLOW_SDE_FAST_MATH) || defined(__FAST_MATH__) || \
    (defined(__CUDA_FTZ) && __CUDA_FTZ) || \
    (defined(__CUDA_PREC_DIV) && !__CUDA_PREC_DIV) || \
    (defined(__CUDA_PREC_SQRT) && !__CUDA_PREC_SQRT)
    return false;
#else
    return true;
#endif
}

std::vector<torch::Tensor> flow_sde_step_logp_forward(
    torch::Tensor sample, torch::Tensor velocity, torch::Tensor params,
    torch::Tensor auxiliary, bool replay) {
    TORCH_CHECK(flow_sde_strict_math(), "Flow SDE rejects fast-math builds");
    check_geometry(sample);
    check_tensor(velocity, sample, "velocity");
    check_tensor(auxiliary, sample, "auxiliary");
    check_tensor(params, sample, "params");
    TORCH_CHECK(velocity.sizes() == sample.sizes() && auxiliary.sizes() == sample.sizes(),
                "latent shapes must match");
    int batch = sample.size(0);
    TORCH_CHECK(params.dim() == 2 && params.size(0) == batch && params.size(1) == 4,
                "params must be [B,4]; use Python wrapper for value validation");
    c10::cuda::CUDAGuard guard(sample.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    int64_t width = sample.numel() / batch;
    int tiles = (width + kTile - 1) / kTile;
    auto mean = torch::empty_like(sample), target = torch::empty_like(sample);
    auto values = torch::empty_like(sample);
    auto coeff = torch::empty({batch, 4}, sample.options());
    auto std = torch::empty({batch}, sample.options()), logp = torch::empty({batch}, sample.options());
    auto partial = torch::empty({batch, tiles}, sample.options());
    coefficients<<<(batch + kTile - 1) / kTile, kTile, 0, stream>>>(
        params.data_ptr<float>(), coeff.data_ptr<float>(), std.data_ptr<float>(), batch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    dim3 grid(tiles, batch);
    step<<<grid, kTile, 0, stream>>>(sample.data_ptr<float>(), velocity.data_ptr<float>(),
        auxiliary.data_ptr<float>(), coeff.data_ptr<float>(), mean.data_ptr<float>(),
        target.data_ptr<float>(), width, replay);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    density<<<grid, kTile, 0, stream>>>(target.data_ptr<float>(), mean.data_ptr<float>(),
        coeff.data_ptr<float>(), values.data_ptr<float>(), width);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    tile_sum<<<grid, kTile, 0, stream>>>(values.data_ptr<float>(), partial.data_ptr<float>(), width, tiles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    final_mean<<<(batch + kTile - 1) / kTile, kTile, 0, stream>>>(
        partial.data_ptr<float>(), logp.data_ptr<float>(), width, tiles, batch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {target, logp, mean, std, coeff};
}

std::vector<torch::Tensor> flow_sde_step_logp_backward(
    torch::Tensor target, torch::Tensor mean, torch::Tensor coeff,
    torch::Tensor grad_target, torch::Tensor grad_logp, torch::Tensor grad_mean, bool replay) {
    TORCH_CHECK(flow_sde_strict_math(), "Flow SDE rejects fast-math builds");
    check_geometry(mean);
    for (auto item : {target, coeff, grad_target, grad_logp, grad_mean})
        check_tensor(item, mean, "backward input");
    TORCH_CHECK(target.sizes() == mean.sizes() && grad_target.sizes() == mean.sizes()
                && grad_mean.sizes() == mean.sizes(), "backward latent shapes must match");
    int batch = mean.size(0);
    TORCH_CHECK(coeff.dim() == 2 && coeff.size(0) == batch && coeff.size(1) == 4,
                "coeff must be [B,4]");
    TORCH_CHECK(grad_logp.dim() == 1 && grad_logp.size(0) == batch, "grad_logp must be [B]");
    c10::cuda::CUDAGuard guard(mean.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    int64_t width = mean.numel() / batch;
    dim3 grid((width + kTile - 1) / kTile, batch);
    auto dx = torch::empty_like(mean), dv = torch::empty_like(mean);
    vjp<<<grid, kTile, 0, stream>>>(target.data_ptr<float>(), mean.data_ptr<float>(),
        coeff.data_ptr<float>(), grad_target.data_ptr<float>(), grad_logp.data_ptr<float>(),
        grad_mean.data_ptr<float>(), dx.data_ptr<float>(), dv.data_ptr<float>(), width, replay);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dx, dv};
}
