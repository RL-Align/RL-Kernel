# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Shared selected-logprob mechanics; extension linking and dispatch stay in backends."""

from typing import Any, Callable

import torch

# FP32 elements per backward row chunk (512 MiB each; upcast + softmax keep two alive).
# Smaller chunks cost launch overhead: 2^24 is ~30% slower than unchunked at 32k x 152k.
FUSED_LOGP_BWD_CHUNK_ELEMS = 1 << 27


def fused_logp_backward_rows_per_chunk(
    vocab: int, chunk_elems: int = FUSED_LOGP_BWD_CHUNK_ELEMS
) -> int:
    """Rows of FP32 softmax workspace the chunked backward materializes at once."""
    return max(1, chunk_elems // max(vocab, 1))


def fused_logp_backward_chunked(
    logits: torch.Tensor,
    labels: torch.Tensor,
    grad_output: torch.Tensor,
    out_dtype: torch.dtype,
    chunk_elems: int = FUSED_LOGP_BWD_CHUNK_ELEMS,
) -> torch.Tensor:
    """Row-chunked ``dlogits = -grad * (softmax - one_hot(target))``.

    The FP32 workspace is bounded by ``chunk_elems`` instead of growing with
    ``N * V``, so 8k-32k token rows at 150k+ vocab do not allocate tens of GB
    of transient softmax.  Each row is computed independently, so the result is
    bitwise identical for any chunk size.  Rows whose target is outside
    ``[0, V)`` have a constant forward output of zero and get a zero gradient.
    """
    n_rows, vocab = logits.shape
    # The fused binding rejects these; without the checks a length-1 grad would broadcast.
    if grad_output.numel() != n_rows:
        raise ValueError(f"grad_output length {grad_output.numel()} != {n_rows} logits rows")
    if labels.numel() != n_rows:
        raise ValueError(f"labels length {labels.numel()} != {n_rows} logits rows")
    grad_logits = torch.empty((n_rows, vocab), device=logits.device, dtype=out_dtype)
    neg_grad = -grad_output.reshape(-1).float()
    valid = (labels >= 0) & (labels < vocab)
    safe_labels = torch.where(valid, labels, torch.zeros_like(labels))
    one_hot_scale = valid.float()
    rows_per_chunk = fused_logp_backward_rows_per_chunk(vocab, chunk_elems)
    for r0 in range(0, n_rows, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, n_rows)
        probs = torch.softmax(logits[r0:r1].float(), dim=-1)
        rows = torch.arange(r1 - r0, device=logits.device)
        probs[rows, safe_labels[r0:r1]] -= one_hot_scale[r0:r1]
        probs.mul_((neg_grad[r0:r1] * one_hot_scale[r0:r1]).unsqueeze(1))
        # Multiplication by zero does not clear NaNs from fully masked rows.
        probs.masked_fill_(~valid[r0:r1].unsqueeze(1), 0)
        grad_logits[r0:r1].copy_(probs)
        del probs  # keep at most two FP32 chunks (upcast + softmax) alive
    return grad_logits


class FusedLogpAutogradForward(torch.autograd.Function):
    """Existing forward normalization and saved state for backend-owned bridges."""

    @staticmethod
    def forward(ctx, logits: torch.Tensor, token_ids: torch.Tensor, backend):
        logits_2d = logits.reshape(-1, logits.size(-1)).contiguous()
        labels = token_ids.reshape(-1).to(device=logits.device, dtype=torch.long).contiguous()
        output = backend.fused_logp(logits_2d, labels)
        ctx.save_for_backward(logits_2d, labels)
        ctx.backend = backend
        ctx.input_shape = tuple(logits.shape)
        ctx.input_dtype = logits.dtype
        return output.reshape(logits.shape[:-1])


class FusedLogpOp:
    """Shared input handling and native variant delegation for selected logprob."""

    is_fused_logp = True
    _backend: Any
    apply: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

    def __call__(self, logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        return self.apply(logits, token_ids)

    def _prepare_inputs(
        self,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Size]:
        orig_shape = logits.shape[:-1]
        logits_2d = logits.reshape(-1, logits.size(-1))
        token_ids_1d = token_ids.reshape(-1).to(device=logits.device, dtype=torch.long).contiguous()
        return logits_2d, token_ids_1d, orig_shape

    def _prepare_output(self, output: torch.Tensor, orig_shape: torch.Size) -> torch.Tensor:
        if output.shape != orig_shape:
            raise ValueError(
                f"output shape {tuple(output.shape)} must match logits leading shape "
                f"{tuple(orig_shape)}"
            )
        return output.view(-1)

    def _prepare_indices(self, row_indices: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        return row_indices.reshape(-1).to(device=logits.device, dtype=torch.long).contiguous()

    def apply_fp32(self, logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        logits_2d, token_ids_1d, orig_shape = self._prepare_inputs(logits, token_ids)
        results = self._backend.fused_logp_forward_fp32(logits_2d, token_ids_1d)
        return results.view(orig_shape)

    def out(
        self, logits: torch.Tensor, token_ids: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        logits_2d, token_ids_1d, orig_shape = self._prepare_inputs(logits, token_ids)
        output_1d = self._prepare_output(output, orig_shape)
        results = self._backend.fused_logp_forward_out(logits_2d, token_ids_1d, output_1d)
        return results.view(orig_shape)

    def indexed_out(
        self,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
        row_indices: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        logits_2d, token_ids_1d, orig_shape = self._prepare_inputs(logits, token_ids)
        row_indices_1d = self._prepare_indices(row_indices, logits)
        output_1d = self._prepare_output(output, orig_shape)
        results = self._backend.fused_logp_forward_indexed_out(
            logits_2d, token_ids_1d, row_indices_1d, output_1d
        )
        return results.view(orig_shape)

    def indexed_fp32(
        self, logits: torch.Tensor, token_ids: torch.Tensor, row_indices: torch.Tensor
    ) -> torch.Tensor:
        logits_2d, token_ids_1d, orig_shape = self._prepare_inputs(logits, token_ids)
        row_indices_1d = self._prepare_indices(row_indices, logits)
        results = self._backend.fused_logp_forward_indexed_fp32(
            logits_2d, token_ids_1d, row_indices_1d
        )
        return results.view(orig_shape)

    def online_out(
        self, logits: torch.Tensor, token_ids: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        logits_2d, token_ids_1d, orig_shape = self._prepare_inputs(logits, token_ids)
        output_1d = self._prepare_output(output, orig_shape)
        results = self._backend.fused_logp_forward_online_out(logits_2d, token_ids_1d, output_1d)
        return results.view(orig_shape)

    def online_fp32(self, logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        logits_2d, token_ids_1d, orig_shape = self._prepare_inputs(logits, token_ids)
        results = self._backend.fused_logp_forward_online_fp32(logits_2d, token_ids_1d)
        return results.view(orig_shape)

    def online_indexed_out(
        self,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
        row_indices: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        logits_2d, token_ids_1d, orig_shape = self._prepare_inputs(logits, token_ids)
        row_indices_1d = self._prepare_indices(row_indices, logits)
        output_1d = self._prepare_output(output, orig_shape)
        results = self._backend.fused_logp_forward_online_indexed_out(
            logits_2d, token_ids_1d, row_indices_1d, output_1d
        )
        return results.view(orig_shape)

    def online_indexed_fp32(
        self, logits: torch.Tensor, token_ids: torch.Tensor, row_indices: torch.Tensor
    ) -> torch.Tensor:
        logits_2d, token_ids_1d, orig_shape = self._prepare_inputs(logits, token_ids)
        row_indices_1d = self._prepare_indices(row_indices, logits)
        results = self._backend.fused_logp_forward_online_indexed_fp32(
            logits_2d, token_ids_1d, row_indices_1d
        )
        return results.view(orig_shape)
