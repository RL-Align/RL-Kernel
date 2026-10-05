// SPDX-License-Identifier: Apache-2.0
// Standalone pybind for T06 local GPU verification when rl_engine._C is absent.

#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> mqa_joint_attention_sink_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor sink,
    torch::Tensor valid,
    double scale,
    bool output_fp32);

void mqa_joint_attention_sink_forward_into(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor sink,
    torch::Tensor valid,
    double scale,
    bool output_fp32,
    torch::Tensor out,
    torch::Tensor scores,
    torch::Tensor p_sink,
    torch::Tensor m,
    torch::Tensor z);

std::vector<torch::Tensor> mqa_joint_attention_sink_backward(
    torch::Tensor dO,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor sink,
    torch::Tensor valid,
    torch::Tensor P,
    torch::Tensor p_sink,
    double scale,
    bool sink_was_shared);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mqa_joint_attention_sink_forward", &mqa_joint_attention_sink_forward);
  m.def("mqa_joint_attention_sink_forward_into", &mqa_joint_attention_sink_forward_into);
  m.def("mqa_joint_attention_sink_backward", &mqa_joint_attention_sink_backward);
  m.attr("mqa_joint_attention_sink_workspace_validation_version") = 2;
}
