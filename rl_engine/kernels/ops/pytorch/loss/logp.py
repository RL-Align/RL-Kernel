# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import torch


class NativeLogpOp:
    """Pure PyTorch native fallback for Fused LogP."""

    def __init__(self):
        pass

    def __call__(self, logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        return self.apply(logits, token_ids)

    def _selected_logps(
        self,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
        *,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        if logits.shape[:-1] != token_ids.shape:
            raise ValueError(
                f"logits leading shape {tuple(logits.shape[:-1])} must match "
                f"token_ids shape {tuple(token_ids.shape)}"
            )
        logits_2d = logits.reshape(-1, logits.size(-1))
        token_ids_1d = token_ids.reshape(-1).to(device=logits.device, dtype=torch.long)
        log_probs = torch.nn.functional.log_softmax(logits_2d.float(), dim=-1)
        selected_log_probs = torch.gather(log_probs, dim=-1, index=token_ids_1d.unsqueeze(1))
        return selected_log_probs.squeeze(-1).to(output_dtype).reshape(logits.shape[:-1])

    def _flat_row_indices(self, row_indices: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        total_rows = int(logits.numel() // logits.size(-1))
        indices = row_indices.to(device=logits.device, dtype=torch.long).reshape(-1)
        if indices.numel() and (indices.min().item() < 0 or indices.max().item() >= total_rows):
            raise ValueError("row_indices contains an out-of-range row")
        return indices

    def _validate_output_shape(self, output: torch.Tensor, logits: torch.Tensor) -> None:
        if output.shape != logits.shape[:-1]:
            raise ValueError(
                f"output shape {tuple(output.shape)} must match logits leading shape "
                f"{tuple(logits.shape[:-1])}"
            )

    def apply(self, logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """Baseline selected-token log probability extraction using torch.gather."""
        return self._selected_logps(logits, token_ids, output_dtype=logits.dtype)

    def apply_fp32(self, logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """Same as apply but forces float32 output for numerical stability."""
        return self._selected_logps(logits, token_ids, output_dtype=torch.float32)

    def indexed_out(
        self,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
        row_indices: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_output_shape(output, logits)
        indices = self._flat_row_indices(row_indices, logits)
        values = self._selected_logps(logits, token_ids, output_dtype=output.dtype)
        output.reshape(-1)[indices] = values.reshape(-1)[indices]
        return output

    def indexed_fp32(
        self, logits: torch.Tensor, token_ids: torch.Tensor, row_indices: torch.Tensor
    ) -> torch.Tensor:
        output = torch.zeros(logits.shape[:-1], device=logits.device, dtype=torch.float32)
        return self.indexed_out(logits, token_ids, row_indices, output)

    def online_out(
        self, logits: torch.Tensor, token_ids: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        return self.out(logits, token_ids, output)

    def online_fp32(self, logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        return self.apply_fp32(logits, token_ids)

    def online_indexed_out(
        self,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
        row_indices: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        return self.indexed_out(logits, token_ids, row_indices, output)

    def online_indexed_fp32(
        self, logits: torch.Tensor, token_ids: torch.Tensor, row_indices: torch.Tensor
    ) -> torch.Tensor:
        return self.indexed_fp32(logits, token_ids, row_indices)

    def out(
        self, logits: torch.Tensor, token_ids: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        self._validate_output_shape(output, logits)
        result = self._selected_logps(logits, token_ids, output_dtype=output.dtype)
        output.copy_(result)
        return output
