// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors

#include <torch/extension.h>

#include <limits>

torch::Tensor fused_logp_forward_musa(torch::Tensor logits, torch::Tensor token_ids);
torch::Tensor fused_logp_backward_musa(
    torch::Tensor logits, torch::Tensor token_ids, torch::Tensor grad_output);

torch::Tensor fused_logp_forward(torch::Tensor logits, torch::Tensor token_ids) {
  TORCH_CHECK(logits.device().type() == c10::kPrivateUse1,
              "logits must be a MUSA tensor, got ", logits.device());
  TORCH_CHECK(token_ids.device().type() == c10::kPrivateUse1,
              "token_ids must be a MUSA tensor, got ", token_ids.device());
  TORCH_CHECK(logits.device() == token_ids.device(),
              "logits and token_ids must share a device");
  TORCH_CHECK(logits.dim() == 2, "logits must be a 2D tensor");
  TORCH_CHECK(token_ids.dim() == 1, "token_ids must be a 1D tensor");
  TORCH_CHECK(token_ids.scalar_type() == at::ScalarType::Long,
              "token_ids must be int64");
  TORCH_CHECK(token_ids.numel() == logits.size(0),
              "token_ids length must match logits rows");
  TORCH_CHECK(logits.size(0) <= std::numeric_limits<int>::max(),
              "too many logits rows");
  TORCH_CHECK(logits.size(1) > 0, "logits vocabulary dimension must be non-empty");
  if (token_ids.numel() > 0) {
    TORCH_CHECK(token_ids.min().item<int64_t>() >= 0 &&
                    token_ids.max().item<int64_t>() < logits.size(1),
                "token_ids must be within the logits vocabulary dimension");
  }
  TORCH_CHECK(logits.scalar_type() == at::ScalarType::Float ||
                  logits.scalar_type() == at::ScalarType::Half ||
                  logits.scalar_type() == at::ScalarType::BFloat16,
              "MUSA fused_logp supports float32, float16, and bfloat16 logits");

  return fused_logp_forward_musa(logits.contiguous(), token_ids.contiguous());
}

torch::Tensor fused_logp_backward(
    torch::Tensor logits,
    torch::Tensor token_ids,
    torch::Tensor grad_output) {
  TORCH_CHECK(logits.device().type() == c10::kPrivateUse1,
              "logits must be a MUSA tensor, got ", logits.device());
  TORCH_CHECK(token_ids.device() == logits.device() &&
                  grad_output.device() == logits.device(),
              "all tensors must share the same MUSA device");
  TORCH_CHECK(logits.dim() == 2 && token_ids.dim() == 1 &&
                  grad_output.dim() == 1,
              "expected logits [rows, vocab], token_ids [rows], and grad_output [rows]");
  TORCH_CHECK(token_ids.scalar_type() == at::ScalarType::Long,
              "token_ids must be int64");
  TORCH_CHECK(grad_output.scalar_type() == logits.scalar_type(),
              "grad_output dtype must match logits dtype");
  TORCH_CHECK(token_ids.numel() == logits.size(0) &&
                  grad_output.numel() == logits.size(0),
              "token_ids and grad_output length must match logits rows");
  TORCH_CHECK(logits.size(1) > 0, "logits vocabulary dimension must be non-empty");
  if (token_ids.numel() > 0) {
    TORCH_CHECK(token_ids.min().item<int64_t>() >= 0 &&
                    token_ids.max().item<int64_t>() < logits.size(1),
                "token_ids must be within the logits vocabulary dimension");
  }
  return fused_logp_backward_musa(
      logits.contiguous(), token_ids.contiguous(), grad_output.contiguous());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_logp", &fused_logp_forward,
        "MUSA fused selected-token log-probability");
  m.def("fused_logp_backward", &fused_logp_backward,
        "MUSA fused selected-token log-probability backward");
}
