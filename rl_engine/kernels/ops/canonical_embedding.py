# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Logical-row canonical embedding parameter gradient."""

from __future__ import annotations

from collections.abc import Callable

import torch

from rl_engine.kernels.ops.backward_runtime import record_backward
from rl_engine.kernels.ops.canonical_backward import active_session


def _canonical_embedding_family(family: str) -> str:
    requested = str(family)
    normalized = "cuda" if requested.startswith("cuda") else requested
    if normalized == "ascend":
        # The Ascend embedding's deterministic grad-weight reuses the CUDA
        # construction bit-for-bit (see ascend/linear/embedding.py), so the
        # canonical family normalizes to the same implementation.
        return "cuda"
    if normalized not in {"cuda", "pytorch", "triton"}:
        raise RuntimeError(f"unsupported canonical embedding backend family: {requested!r}")
    return normalized


def _reduce_embedding_grad_weight(
    ids: torch.Tensor,
    grad_rows: torch.Tensor,
    *,
    weight_shape: tuple[int, int],
    weight_dtype: torch.dtype,
    family: str,
) -> torch.Tensor:
    if family == "triton":
        from rl_engine.kernels.ops.triton.linear.embedding import _embedding_grad_weight

        return _embedding_grad_weight(
            ids.reshape(-1).to(dtype=torch.long),
            grad_rows,
            weight_shape=weight_shape,
            weight_dtype=weight_dtype,
        )

    if family not in {"cuda", "pytorch"}:
        raise RuntimeError(f"unsupported canonical embedding backend family: {family!r}")

    from rl_engine.kernels.ops.cuda.linear.embedding import _deterministic_embedding_grad_weight

    return _deterministic_embedding_grad_weight(
        ids.reshape(-1).to(dtype=torch.long),
        grad_rows.float(),
        weight_shape=weight_shape,
        weight_dtype=weight_dtype,
    )


class _CanonicalEmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        token_ids: torch.Tensor,
        weight: torch.Tensor,
        logical_keys: torch.Tensor,
        parameter_id: str,
        forward_op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        family: str,
    ) -> torch.Tensor:
        session = active_session()
        if session is None:
            raise RuntimeError("canonical embedding requires an active backward session")
        if weight.ndim != 2:
            raise ValueError("canonical embedding weight must be [vocab, hidden]")
        if logical_keys.ndim < 2 or logical_keys.shape[-1] not in (2, 3):
            raise ValueError("canonical embedding logical keys must end in width 2 or 3")
        if logical_keys.device != token_ids.device:
            raise ValueError("canonical embedding token ids and logical keys must share a device")

        ids = token_ids.reshape(-1).to(dtype=torch.long).contiguous()
        keys = logical_keys.reshape(-1, logical_keys.shape[-1])
        if keys.shape[0] != ids.numel():
            raise ValueError("logical key count does not match embedding token count")
        normalized_family = _canonical_embedding_family(family)

        ctx.save_for_backward(ids)
        ctx.session = session
        ctx.parameter_id = str(parameter_id)
        ctx.slot = session.register(ctx.parameter_id, keys) if weight.requires_grad else None
        ctx.weight_shape = (int(weight.shape[0]), int(weight.shape[1]))
        ctx.weight_dtype = weight.dtype
        ctx.family = normalized_family
        with torch.no_grad():
            return forward_op(token_ids, weight)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (ids,) = ctx.saved_tensors
        grad_weight = None
        if ctx.needs_input_grad[1]:
            if ctx.slot is None:
                raise RuntimeError("canonical embedding gradient was not registered")
            hidden = ctx.weight_shape[1]
            grad_rows = grad_output.reshape(-1, hidden).contiguous()

            def reducer(ordered_ids: torch.Tensor, ordered_grads: torch.Tensor) -> torch.Tensor:
                return _reduce_embedding_grad_weight(
                    ordered_ids,
                    ordered_grads,
                    weight_shape=ctx.weight_shape,
                    weight_dtype=ctx.weight_dtype,
                    family=ctx.family,
                )

            grad_weight = ctx.session.submit_linear(
                ctx.parameter_id,
                ctx.slot,
                ids.unsqueeze(1),
                grad_rows,
                reducer,
            )
            record_backward(
                "embedding",
                kernel_id=(
                    "rl_engine.kernels.ops.triton.linear.embedding._embedding_bwd"
                    if ctx.family == "triton"
                    else (
                        "rl_engine.kernels.ops.cuda.linear.embedding."
                        "_deterministic_embedding_grad_weight"
                    )
                ),
                impl=f"{ctx.family}_embedding_canonical_rowfold",
                family=ctx.family,
            )
        return None, grad_weight, None, None, None, None


def canonical_embedding(
    token_ids: torch.Tensor,
    weight: torch.Tensor,
    logical_keys: torch.Tensor,
    *,
    forward_op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    family: str,
    parameter_id: str = "embed_tokens.weight",
) -> torch.Tensor:
    return _CanonicalEmbeddingFunction.apply(
        token_ids,
        weight,
        logical_keys,
        parameter_id,
        forward_op,
        family,
    )


__all__ = ["canonical_embedding"]
