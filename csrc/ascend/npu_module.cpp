// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors

#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> batch_invariant_logp_ascend_forward(torch::Tensor logits,
                                                               torch::Tensor target,
                                                               int64_t ignore_index);

torch::Tensor rope_apply_ascend_forward(torch::Tensor x,
                                        torch::Tensor cos,
                                        torch::Tensor sin,
                                        double sin_sign);

std::vector<torch::Tensor> deterministic_attention_ascend_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool causal,
    double scale,
    c10::optional<torch::Tensor> key_padding_mask);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("batch_invariant_logp_ascend",
          &batch_invariant_logp_ascend_forward,
          "Batch-invariant selected-token log-probability (Ascend C forward)");
    m.def("rope_apply_ascend",
          &rope_apply_ascend_forward,
          "GPT-NeoX/HF rotate-half RoPE apply (Ascend C forward/backward primitive)");
    m.def("deterministic_attention_ascend",
          &deterministic_attention_ascend_forward,
          "Deterministic batch-invariant standard-softmax attention (Ascend C forward)");
}
