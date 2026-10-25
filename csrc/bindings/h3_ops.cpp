// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
#include <torch/extension.h>

#include <map>
#include <string>
#include <vector>

std::map<std::string, std::string> h3_attention_build_info();
std::vector<torch::Tensor> h3_attention_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v);
torch::Tensor h3_attention_probabilities(
    torch::Tensor q, torch::Tensor k, torch::Tensor v);
std::vector<torch::Tensor> h3_attention_backward_accumulate(
    torch::Tensor grad_output, torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor probabilities,
    torch::Tensor dk, torch::Tensor dv);

// Expose the original core in the same strict build for numerical comparisons.
std::vector<torch::Tensor> deterministic_attention_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal, double scale,
    torch::optional<torch::Tensor> key_padding_mask);
std::vector<torch::Tensor> deterministic_attention_backward(
    torch::Tensor grad_output, torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor probabilities, bool causal, double scale,
    torch::optional<torch::Tensor> key_padding_mask);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("h3_attention_build_info", &h3_attention_build_info);
  m.def("h3_attention_forward", &h3_attention_forward);
  m.def("h3_attention_probabilities", &h3_attention_probabilities);
  m.def("h3_attention_backward_accumulate", &h3_attention_backward_accumulate);
  m.def("deterministic_attention_forward", &deterministic_attention_forward);
  m.def("deterministic_attention_backward", &deterministic_attention_backward);
}
