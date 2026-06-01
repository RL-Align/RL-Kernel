# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import torch


class NativeLogpOp:
    """Pure PyTorch native fallback for Fused LogP."""

    def __init__(self):
        pass

    def __call__(self, logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        return self.apply(logits, token_ids)

    def apply(self, logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """Baseline cross-entropy log prob extraction using torch.gather."""
        orig_shape = logits.shape[:-1]
        logits_2d = logits.view(-1, logits.size(-1))
        token_ids_1d = token_ids.view(-1).unsqueeze(1)
        log_probs = torch.nn.functional.log_softmax(logits_2d.float(), dim=-1).to(logits.dtype)
        selected_log_probs = torch.gather(log_probs, dim=-1, index=token_ids_1d.long()).squeeze(-1)
        return selected_log_probs.view(orig_shape)

    def apply_fp32(self, logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """Same as apply but forces float32 output for numerical stability."""
        logits_fp32 = logits.float()
        return self.apply(logits_fp32, token_ids)

    def indexed_fp32(
        self, logits: torch.Tensor, token_ids: torch.Tensor, row_indices: torch.Tensor
    ) -> torch.Tensor:
        orig_shape = logits.shape[:-1]
        logits_2d = logits.view(-1, logits.size(-1))
        token_ids_1d = token_ids.view(-1)
        valid_logits = logits_2d[row_indices]
        valid_token_ids = token_ids_1d[row_indices]
        valid_log_probs = self.apply_fp32(valid_logits.unsqueeze(0), valid_token_ids.unsqueeze(0))
        output = torch.zeros(orig_shape, device=logits.device, dtype=torch.float32).view(-1)
        output[row_indices] = valid_log_probs.view(-1)
        return output.view(orig_shape)

    def online_fp32(self, logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        return self.apply_fp32(logits, token_ids)

    def online_indexed_fp32(
        self, logits: torch.Tensor, token_ids: torch.Tensor, row_indices: torch.Tensor
    ) -> torch.Tensor:
        return self.indexed_fp32(logits, token_ids, row_indices)

    def out(
        self, logits: torch.Tensor, token_ids: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        result = self.apply(logits, token_ids)
        output.copy_(result)
        return output
