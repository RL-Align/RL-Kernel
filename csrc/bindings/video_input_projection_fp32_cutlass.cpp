// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors

#include <torch/extension.h>

torch::Tensor cutlass_gemm_fp32_tn(torch::Tensor a, torch::Tensor b_transposed);
int cutlass_cuda_version();
void cutlass_gemm_fp32_tree_leaves_tn(
    torch::Tensor a, torch::Tensor bt, torch::Tensor workspace,
    torch::Tensor starts, torch::Tensor lengths, torch::Tensor nodes);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm_fp32_tn", &cutlass_gemm_fp32_tn,
        "Fixed FP32 SIMT GEMM: [M,K] times transposed [N,K]");
  m.def("gemm_fp32_tree_leaves_tn", &cutlass_gemm_fp32_tree_leaves_tn,
        "Batch all FP32 leaves into a validated canonical tree workspace");
  m.def("build_info", []() {
    pybind11::dict info;
    info["cutlass_revision"] = "56ca28adb32ffadd097566545b8f7d83b652e998";
    info["headers_sha256"] = "7b5321860c269f8e44eab02e59c32c2d4d0e2fa7a3ef72e0a161de9f58b361fa";
    info["cuda_version"] = cutlass_cuda_version();
    info["architecture"] = "sm_90";
    info["tile"] = pybind11::make_tuple(64, 64, 8);
    info["operator_class"] = "simt";
    info["split_k"] = 1;
    info["fmad"] = false;
    info["ftz"] = false;
    info["tree_leaf_size"] = 32;
    info["tree_leaf_launcher"] = "cutlass_device_gemm_array";
    return info;
  });
}
