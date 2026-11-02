#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#if !defined(USE_ROCM)
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#endif
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <assert.h>

template <typename scalar_t>
__device__ __forceinline__ float load_as_float(const scalar_t* ptr) {
    return static_cast<float>(*ptr);
}

template <>
__device__ __forceinline__ float load_as_float<at::Half>(const at::Half* ptr) {
    const __half* p = reinterpret_cast<const __half*>(ptr);
    return __half2float(*p);
}

template <>
__device__ __forceinline__ float load_as_float<at::BFloat16>(const at::BFloat16* ptr) {
    const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(ptr);
    return __bfloat162float(*p);
}


template <typename scalar_t>
__device__ __forceinline__ void store_from_float(scalar_t* ptr, float v) {
    *ptr = static_cast<scalar_t>(v);
}

template <>
__device__ __forceinline__ void store_from_float<at::Half>(at::Half* ptr, float v) {
    __half* p = reinterpret_cast<__half*>(ptr);
    *p = __float2half(v);
}

template <>
__device__ __forceinline__ void store_from_float<at::BFloat16>(at::BFloat16* ptr, float v) {
    __nv_bfloat16* p = reinterpret_cast<__nv_bfloat16*>(ptr);
    *p = __float2bfloat16(v);
}


__device__ __forceinline__ float block_reduce_sum(float v) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    assert((blockDim.x & (blockDim.x - 1)) == 0);

    smem[tid] = v;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    return smem[0];
}


static int choose_threads(int H) {
    if (H <= 64) return 64;
    if (H <= 128) return 128;
    if (H <= 256) return 256;
    return 512;
}

constexpr int RMSNORM_DW_ROWS_PER_CHUNK = 256;
constexpr int RMSNORM_DW_H_TILE = 128;

#if !defined(USE_ROCM)
constexpr int FP32_LEFT_FOLD_THREADS = 256;

__global__ void reduce_rows_fp32_left_fold_kernel(
    const float* __restrict__ rows,
    float* __restrict__ output,
    int64_t row_count,
    int64_t columns
) {
    int64_t column = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (column >= columns) {
        return;
    }

    float acc = 0.0f;
#pragma unroll 1
    for (int64_t row = 0; row < row_count; ++row) {
        acc = __fadd_rn(acc, rows[row * columns + column]);
    }
    output[column] = acc;
}
#endif

int64_t rmsnorm_backward_dw_chunks_cuda(int64_t rows) {
    return (rows + RMSNORM_DW_ROWS_PER_CHUNK - 1) / RMSNORM_DW_ROWS_PER_CHUNK;
}

template <typename scalar_t, typename weight_t>
__global__ void rmsnorm_fwd_kernel(
    const scalar_t* __restrict__ x,
    const weight_t* __restrict__ weight,
    scalar_t* __restrict__ y,
    float* __restrict__ rstd,
    int T,
    int H,
    float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    const scalar_t* x_row = x + row * H;
    scalar_t* y_row = y + row * H;

    float local_sum = 0.0f;

    // Compute sum(x^2), with each thread handling a subset of columns.
    for (int col = tid; col < H; col += blockDim.x) {
        float xv = load_as_float<scalar_t>(x_row + col);
        local_sum += xv * xv;
    }

    // Reduce within the block in a fixed order.
    float sum = block_reduce_sum(local_sum);

    float row_rstd = rsqrtf(sum / static_cast<float>(H) + eps);

    if (tid == 0) {
        rstd[row] = row_rstd;
    }

    __syncthreads();

    // Write y = x * rstd * weight.
    for (int col = tid; col < H; col += blockDim.x) {
        float xv = load_as_float<scalar_t>(x_row + col);
        float wv = load_as_float<weight_t>(weight + col);
        float out = xv * row_rstd * wv;
        store_from_float<scalar_t>(y_row + col, out);
    }
}


template <typename scalar_t, typename weight_t>
__global__ void rmsnorm_bwd_dx_kernel(
    const scalar_t* __restrict__ dy,
    const scalar_t* __restrict__ x,
    const weight_t* __restrict__ weight,
    const float* __restrict__ rstd,
    scalar_t* __restrict__ dx,
    int T,
    int H
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    const scalar_t* dy_row = dy + row * H;
    const scalar_t* x_row = x + row * H;
    scalar_t* dx_row = dx + row * H;

    float local_dot = 0.0f;

    for (int col = tid; col < H; col += blockDim.x) {
        float dyv = load_as_float<scalar_t>(dy_row + col);
        float xv = load_as_float<scalar_t>(x_row + col);
        float wv = load_as_float<weight_t>(weight + col);
        local_dot += dyv * wv * xv;
    }

    float dot = block_reduce_sum(local_dot);

    float r = rstd[row];
    float coeff = dot * r * r * r / static_cast<float>(H);

    for (int col = tid; col < H; col += blockDim.x) {
        float dyv = load_as_float<scalar_t>(dy_row + col);
        float xv = load_as_float<scalar_t>(x_row + col);
        float wv = load_as_float<weight_t>(weight + col);

        float out = r * dyv * wv - xv * coeff;
        store_from_float<scalar_t>(dx_row + col, out);
    }
}


template <typename scalar_t>
__global__ void rmsnorm_partial_dw_kernel(
    const scalar_t* __restrict__ dy,
    const scalar_t* __restrict__ x,
    const float* __restrict__ rstd,
    const bool* __restrict__ mask,
    float* __restrict__ partial_dw,
    int T,
    int H
) {
    int chunk = blockIdx.x;
    int h = blockIdx.y * RMSNORM_DW_H_TILE + threadIdx.x;
    if (h >= H) {
        return;
    }

    int t0 = chunk * RMSNORM_DW_ROWS_PER_CHUNK;
    float acc = 0.0f;

#pragma unroll
    for (int i = 0; i < RMSNORM_DW_ROWS_PER_CHUNK; ++i) {
        int row = t0 + i;
        float contrib = 0.0f;
        if (row < T && mask[row]) {
            int idx = row * H + h;
            float dyv = load_as_float<scalar_t>(dy + idx);
            float xv = load_as_float<scalar_t>(x + idx);
            contrib = dyv * xv * rstd[row];
        }
        acc += contrib;
    }

    partial_dw[chunk * H + h] = acc;
}


__global__ void rmsnorm_reduce_dw_kernel(
    const float* __restrict__ partial_dw,
    float* __restrict__ dw,
    int chunks,
    int H
) {
    int h = blockIdx.x * RMSNORM_DW_H_TILE + threadIdx.x;
    if (h >= H) {
        return;
    }

    float acc = 0.0f;
    for (int chunk = 0; chunk < chunks; ++chunk) {
        acc += partial_dw[chunk * H + h];
    }
    dw[h] = acc;
}


void rmsnorm_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor y,
    torch::Tensor rstd,
    double eps
) {
    int T = x.size(0);
    int H = x.size(1);
    int threads = choose_threads(H);
    size_t smem = threads * sizeof(float);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "rmsnorm_forward_cuda", [&] {
        using x_t = scalar_t;
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, weight.scalar_type(), "rmsnorm_forward_weight_cuda", [&] {
            using w_t = scalar_t;
            rmsnorm_fwd_kernel<x_t, w_t><<<T, threads, smem, stream>>>(
                x.data_ptr<x_t>(),
                weight.data_ptr<w_t>(),
                y.data_ptr<x_t>(),
                rstd.data_ptr<float>(),
                T,
                H,
                static_cast<float>(eps)
            );
        });
    });
}


void rmsnorm_backward_dx_cuda(
    torch::Tensor dy,
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor rstd,
    torch::Tensor dx
) {
    int T = x.size(0);
    int H = x.size(1);
    int threads = choose_threads(H);
    size_t smem = threads * sizeof(float);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "rmsnorm_backward_dx_cuda", [&] {
        using x_t = scalar_t;
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, weight.scalar_type(), "rmsnorm_backward_dx_weight_cuda", [&] {
            using w_t = scalar_t;
            rmsnorm_bwd_dx_kernel<x_t, w_t><<<T, threads, smem, stream>>>(
                dy.data_ptr<x_t>(),
                x.data_ptr<x_t>(),
                weight.data_ptr<w_t>(),
                rstd.data_ptr<float>(),
                dx.data_ptr<x_t>(),
                T,
                H
            );
        });
    });
}


void rmsnorm_backward_partial_dw_cuda(
    torch::Tensor dy,
    torch::Tensor x,
    torch::Tensor rstd,
    torch::Tensor mask,
    torch::Tensor partial_dw
) {
    int T = x.size(0);
    int H = x.size(1);

    int chunks = (T + RMSNORM_DW_ROWS_PER_CHUNK - 1) / RMSNORM_DW_ROWS_PER_CHUNK;
    dim3 blocks(chunks, (H + RMSNORM_DW_H_TILE - 1) / RMSNORM_DW_H_TILE);
    dim3 threads(RMSNORM_DW_H_TILE);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "rmsnorm_partial_dw_cuda", [&] {
        rmsnorm_partial_dw_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            dy.data_ptr<scalar_t>(),
            x.data_ptr<scalar_t>(),
            rstd.data_ptr<float>(),
            mask.data_ptr<bool>(),
            partial_dw.data_ptr<float>(),
            T,
            H
        );
    });
}


void rmsnorm_backward_reduce_dw_cuda(
    torch::Tensor partial_dw,
    torch::Tensor dw
) {
    int chunks = partial_dw.size(0);
    int H = partial_dw.size(1);

    dim3 blocks((H + RMSNORM_DW_H_TILE - 1) / RMSNORM_DW_H_TILE);
    dim3 threads(RMSNORM_DW_H_TILE);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    rmsnorm_reduce_dw_kernel<<<blocks, threads, 0, stream>>>(
        partial_dw.data_ptr<float>(),
        dw.data_ptr<float>(),
        chunks,
        H
    );
}

#if !defined(USE_ROCM)
void reduce_rows_fp32_left_fold_cuda(
    torch::Tensor rows,
    torch::Tensor output
) {
    const c10::cuda::CUDAGuard device_guard(rows.device());
    int64_t columns = rows.size(1);
    int64_t blocks = (columns + FP32_LEFT_FOLD_THREADS - 1) / FP32_LEFT_FOLD_THREADS;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    reduce_rows_fp32_left_fold_kernel<<<blocks, FP32_LEFT_FOLD_THREADS, 0, stream>>>(
        rows.data_ptr<float>(),
        output.data_ptr<float>(),
        rows.size(0),
        columns
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
#endif
