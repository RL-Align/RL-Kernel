// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors

#include <torch/extension.h>

torch::Tensor det_gemm_fwd(torch::Tensor a, torch::Tensor b);
torch::Tensor det_gemm_fwd_fp32(torch::Tensor a, torch::Tensor b);
torch::Tensor det_gemm_fwd_rhs_transposed(torch::Tensor a, torch::Tensor bt);
torch::Tensor det_gemm_da(torch::Tensor dc, torch::Tensor b);
torch::Tensor det_gemm_db(torch::Tensor a, torch::Tensor dc);
torch::Tensor det_gemm_db_transposed(torch::Tensor a, torch::Tensor dc);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("det_gemm_fwd", &det_gemm_fwd);
    m.def("det_gemm_fwd_fp32", &det_gemm_fwd_fp32);
    m.def("det_gemm_fwd_rhs_transposed", &det_gemm_fwd_rhs_transposed);
    m.def("det_gemm_da", &det_gemm_da);
    m.def("det_gemm_db", &det_gemm_db);
    m.def("det_gemm_db_transposed", &det_gemm_db_transposed);
}
