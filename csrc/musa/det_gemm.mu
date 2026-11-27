// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors

#include <musa_runtime.h>
#include <c10/core/DeviceGuard.h>
#include <torch/extension.h>
#include <torch_musa/csrc/aten/musa/Exceptions.h>
#include <torch_musa/csrc/aten/musa/MUSAContext.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace {

constexpr int kBlockSize = 256;

template <
    typename a_t,
    typename b_t,
    typename output_t,
    bool TransposeA,
    bool TransposeB,
    bool TransposeOutput>
__global__ void det_gemm_kernel(
    const a_t* a,
    const b_t* b,
    output_t* c,
    int64_t m,
    int64_t n,
    int64_t k) {
    const int64_t index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= m * n) {
        return;
    }
    const int64_t row = index / n;
    const int64_t column = index % n;
    float accumulator = 0.0f;
    for (int64_t inner = 0; inner < k; ++inner) {
        const int64_t a_index = TransposeA ? inner * m + row : row * k + inner;
        const int64_t b_index =
            TransposeB ? column * k + inner : inner * n + column;
        accumulator +=
            static_cast<float>(a[a_index]) * static_cast<float>(b[b_index]);
    }
    const int64_t output_index =
        TransposeOutput ? column * m + row : index;
    c[output_index] = static_cast<output_t>(accumulator);
}

int64_t checked_product(
    int64_t lhs, int64_t rhs, const char* description) {
    TORCH_CHECK(
        lhs >= 0 && rhs >= 0,
        "MUSA det_gemm: dimensions must be nonnegative");
    TORCH_CHECK(
        lhs == 0 || rhs <= std::numeric_limits<int64_t>::max() / lhs,
        "MUSA det_gemm: ",
        description,
        " exceeds int64 range");
    return lhs * rhs;
}

int64_t validate_launch_shape(
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t max_grid_x) {
    TORCH_CHECK(
        max_grid_x > 0,
        "MUSA det_gemm: launch-grid limit must be positive");
    const int64_t output_elements =
        checked_product(m, n, "output element count");
    checked_product(m, k, "A input element count");
    checked_product(k, n, "B input element count");
    const int64_t blocks =
        output_elements / kBlockSize +
        static_cast<int64_t>(output_elements % kBlockSize != 0);
    TORCH_CHECK(
        blocks <= max_grid_x,
        "MUSA det_gemm: required block count ",
        blocks,
        " exceeds supported launch-grid limit ",
        max_grid_x);
    TORCH_CHECK(
        blocks <= std::numeric_limits<unsigned int>::max(),
        "MUSA det_gemm: required block count ",
        blocks,
        " exceeds launch-grid type limit");
    return blocks;
}

template <typename a_t, typename b_t, typename output_t>
void launch(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor output,
    int64_t m,
    int64_t n,
    int64_t k,
    unsigned int blocks,
    bool transpose_a,
    bool transpose_b,
    bool transpose_output) {
    auto stream = at::musa::getCurrentMUSAStream();
    if (transpose_a) {
        if (transpose_output) {
            det_gemm_kernel<a_t, b_t, output_t, true, false, true>
                <<<blocks, kBlockSize, 0, stream>>>(
                    a.data_ptr<a_t>(),
                    b.data_ptr<b_t>(),
                    output.data_ptr<output_t>(),
                    m,
                    n,
                    k);
        } else {
            det_gemm_kernel<a_t, b_t, output_t, true, false, false>
                <<<blocks, kBlockSize, 0, stream>>>(
                    a.data_ptr<a_t>(),
                    b.data_ptr<b_t>(),
                    output.data_ptr<output_t>(),
                    m,
                    n,
                    k);
        }
    } else if (transpose_b) {
        det_gemm_kernel<a_t, b_t, output_t, false, true, false>
            <<<blocks, kBlockSize, 0, stream>>>(
                a.data_ptr<a_t>(),
                b.data_ptr<b_t>(),
                output.data_ptr<output_t>(),
                m,
                n,
                k);
    } else {
        det_gemm_kernel<a_t, b_t, output_t, false, false, false>
            <<<blocks, kBlockSize, 0, stream>>>(
                a.data_ptr<a_t>(),
                b.data_ptr<b_t>(),
                output.data_ptr<output_t>(),
                m,
                n,
                k);
    }
}

void check_inputs(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(
        a.device().type() == c10::kPrivateUse1 &&
            b.device().type() == c10::kPrivateUse1,
        "det_gemm requires MUSA tensors");
    TORCH_CHECK(a.device() == b.device(), "det_gemm tensors must share a device");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "det_gemm expects 2-D tensors");
    TORCH_CHECK(a.scalar_type() == b.scalar_type(), "det_gemm dtypes must match");
    TORCH_CHECK(
        a.scalar_type() == at::ScalarType::BFloat16,
        "MUSA det_gemm currently supports bfloat16 inputs");
}

void check_accum_inputs(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(
        a.device().type() == c10::kPrivateUse1 &&
            b.device().type() == c10::kPrivateUse1,
        "det_gemm requires MUSA tensors");
    TORCH_CHECK(a.device() == b.device(), "det_gemm tensors must share a device");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "det_gemm expects 2-D tensors");
    const auto supported = [](at::ScalarType dtype) {
        return dtype == at::ScalarType::BFloat16 ||
            dtype == at::ScalarType::Float;
    };
    TORCH_CHECK(
        supported(a.scalar_type()) && supported(b.scalar_type()),
        "MUSA FP32-accumulation det_gemm supports bfloat16 or float32 inputs");
}

torch::Tensor dispatch(
    torch::Tensor a,
    torch::Tensor b,
    int64_t m,
    int64_t n,
    int64_t k,
    bool transpose_a,
    bool transpose_b,
    bool transpose_output,
    bool output_fp32) {
    const c10::DeviceGuard device_guard(a.device());
    musaDeviceProp properties{};
    C10_MUSA_CHECK(
        musaGetDeviceProperties(&properties, a.device().index()));
    const int64_t grid_limit = std::min<int64_t>(
        properties.maxGridSize[0],
        std::numeric_limits<unsigned int>::max());
    const auto blocks = static_cast<unsigned int>(
        validate_launch_shape(m, n, k, grid_limit));
    a = a.contiguous();
    b = b.contiguous();
    auto options = output_fp32 ? a.options().dtype(torch::kFloat) : a.options();
    auto output = torch::empty(
        transpose_output ? std::vector<int64_t>{n, m}
                         : std::vector<int64_t>{m, n},
        options);
    if (m == 0 || n == 0) {
        return output;
    }
    if (output_fp32) {
        if (a.scalar_type() == at::ScalarType::Float &&
            b.scalar_type() == at::ScalarType::Float) {
            launch<float, float, float>(
                a, b, output, m, n, k, blocks,
                transpose_a, transpose_b, transpose_output);
        } else if (a.scalar_type() == at::ScalarType::Float) {
            launch<float, c10::BFloat16, float>(
                a, b, output, m, n, k, blocks,
                transpose_a, transpose_b, transpose_output);
        } else if (b.scalar_type() == at::ScalarType::Float) {
            launch<c10::BFloat16, float, float>(
                a, b, output, m, n, k, blocks,
                transpose_a, transpose_b, transpose_output);
        } else {
            launch<c10::BFloat16, c10::BFloat16, float>(
                a, b, output, m, n, k, blocks,
                transpose_a, transpose_b, transpose_output);
        }
    } else {
        launch<c10::BFloat16, c10::BFloat16, c10::BFloat16>(
            a, b, output, m, n, k, blocks,
            transpose_a, transpose_b, transpose_output);
    }
    C10_MUSA_KERNEL_LAUNCH_CHECK();
    return output;
}

}  // namespace

torch::Tensor det_gemm_fwd(torch::Tensor a, torch::Tensor b) {
    check_inputs(a, b);
    TORCH_CHECK(b.size(0) == a.size(1), "det_gemm_fwd: K mismatch");
    return dispatch(a, b, a.size(0), b.size(1), a.size(1), false, false, false, false);
}

torch::Tensor det_gemm_fwd_fp32(torch::Tensor a, torch::Tensor b) {
    check_inputs(a, b);
    TORCH_CHECK(b.size(0) == a.size(1), "det_gemm_fwd_fp32: K mismatch");
    return dispatch(a, b, a.size(0), b.size(1), a.size(1), false, false, false, true);
}

torch::Tensor det_gemm_accum_fwd_fp32(torch::Tensor a, torch::Tensor b) {
    check_accum_inputs(a, b);
    TORCH_CHECK(
        b.size(0) == a.size(1),
        "det_gemm_accum_fwd_fp32: K mismatch");
    return dispatch(
        a, b, a.size(0), b.size(1), a.size(1),
        false, false, false, true);
}

int64_t det_gemm_validate_launch(
    int64_t m, int64_t n, int64_t k, int64_t max_grid_x) {
    return validate_launch_shape(m, n, k, max_grid_x);
}

torch::Tensor det_gemm_fwd_rhs_transposed(torch::Tensor a, torch::Tensor bt) {
    check_inputs(a, bt);
    TORCH_CHECK(bt.size(1) == a.size(1), "det_gemm_fwd_rhs_transposed: K mismatch");
    return dispatch(a, bt, a.size(0), bt.size(0), a.size(1), false, true, false, false);
}

torch::Tensor det_gemm_da(torch::Tensor dc, torch::Tensor b) {
    check_inputs(dc, b);
    TORCH_CHECK(b.size(1) == dc.size(1), "det_gemm_da: N mismatch");
    return dispatch(dc, b, dc.size(0), b.size(0), dc.size(1), false, true, false, false);
}

torch::Tensor det_gemm_db(torch::Tensor a, torch::Tensor dc) {
    check_inputs(a, dc);
    TORCH_CHECK(a.size(0) == dc.size(0), "det_gemm_db: M mismatch");
    return dispatch(a, dc, a.size(1), dc.size(1), a.size(0), true, false, false, false);
}

torch::Tensor det_gemm_db_transposed(torch::Tensor a, torch::Tensor dc) {
    check_inputs(a, dc);
    TORCH_CHECK(a.size(0) == dc.size(0), "det_gemm_db_transposed: M mismatch");
    return dispatch(a, dc, a.size(1), dc.size(1), a.size(0), true, false, true, false);
}
