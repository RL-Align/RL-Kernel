// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors

#include <torch/extension.h>

torch::Tensor fused_logp_forward(torch::Tensor logits, torch::Tensor token_ids);

#if defined(__CUDACC__) || defined(KERNEL_ALIGN_WITH_SM90)
torch::Tensor fused_logp_sm90_forward(torch::Tensor logits, torch::Tensor labels);
#endif

#if defined(__CUDACC__) || defined(KERNEL_ALIGN_WITH_CUDA)
torch::Tensor fused_logp_forward_out(
    torch::Tensor logits,
    torch::Tensor token_ids,
    torch::Tensor output);
torch::Tensor fused_logp_forward_fp32(torch::Tensor logits, torch::Tensor token_ids);
torch::Tensor fused_logp_forward_indexed_out(
    torch::Tensor logits,
    torch::Tensor token_ids,
    torch::Tensor row_indices,
    torch::Tensor output);
torch::Tensor fused_logp_forward_indexed_fp32(
    torch::Tensor logits,
    torch::Tensor token_ids,
    torch::Tensor row_indices);
torch::Tensor fused_logp_forward_online_out(
    torch::Tensor logits,
    torch::Tensor token_ids,
    torch::Tensor output);
torch::Tensor fused_logp_forward_online_fp32(torch::Tensor logits, torch::Tensor token_ids);
torch::Tensor fused_logp_forward_online_indexed_out(
    torch::Tensor logits,
    torch::Tensor token_ids,
    torch::Tensor row_indices,
    torch::Tensor output);
torch::Tensor fused_logp_forward_online_indexed_fp32(
    torch::Tensor logits,
    torch::Tensor token_ids,
    torch::Tensor row_indices);
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "RL-Kernel High-Performance Operator Extension Library";

    m.def("fused_logp", &fused_logp_forward, "Fused logp forward fallback");

#if defined(__CUDACC__) || defined(KERNEL_ALIGN_WITH_SM90)
    m.def("fused_logp_sm90", &fused_logp_sm90_forward, "TMA-accelerated Online Softmax Fused LogP");
#endif

#if defined(__CUDACC__) || defined(KERNEL_ALIGN_WITH_CUDA)
    m.def("fused_logp_forward_out", &fused_logp_forward_out, "Fused logp out");
    m.def("fused_logp_forward_fp32", &fused_logp_forward_fp32, "Fused logp fp32");
    m.def(
        "fused_logp_forward_indexed_out",
        &fused_logp_forward_indexed_out,
        "Fused logp indexed out");
    m.def(
        "fused_logp_forward_indexed_fp32",
        &fused_logp_forward_indexed_fp32,
        "Fused logp indexed fp32");
    m.def(
        "fused_logp_forward_online_out",
        &fused_logp_forward_online_out,
        "Fused logp online out");
    m.def(
        "fused_logp_forward_online_fp32",
        &fused_logp_forward_online_fp32,
        "Fused logp online fp32");
    m.def(
        "fused_logp_forward_online_indexed_out",
        &fused_logp_forward_online_indexed_out,
        "Fused logp online indexed out");
    m.def(
        "fused_logp_forward_online_indexed_fp32",
        &fused_logp_forward_online_indexed_fp32,
        "Fused logp online indexed fp32");
#endif
}
