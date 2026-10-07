# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""H3 gated residual (RFC #420 row ``adaln_gate_residual``).

After each sublayer of a block, diffusers adds the sublayer output back to
the residual stream through the per-row AdaLN gate::

    hidden = residual + gate.index_select(0, adaln_indices) * sublayer_output

``gate`` is ``gate_msa`` (after attention) or ``gate_mlp`` (after the FFN),
an ``(R, H)`` row view of the AdaLN table; ``residual`` and the sublayer
output are ``(..., S, H)``. The order is fixed by RFC #420 §4:
``residual + gate[row] * sublayer_output``.
"""

from __future__ import annotations

import torch

_DTYPES = (torch.float32, torch.bfloat16, torch.float16)


def validate_h3_gate_residual(
    residual: torch.Tensor,
    y: torch.Tensor,
    gate: torch.Tensor,
    index: torch.Tensor,
    *,
    check_range: bool = True,
    same_dtype: bool = True,
) -> None:
    if residual.dtype not in _DTYPES:
        raise TypeError(f"residual must be fp32, bf16 or fp16, got {residual.dtype}")
    if y.shape != residual.shape or y.dtype != residual.dtype or y.device != residual.device:
        raise ValueError("y must match the residual's shape, dtype and device")
    if residual.dim() < 2 or residual.numel() == 0:
        raise ValueError(
            f"residual must be a non-empty (..., S, H) tensor, got {tuple(residual.shape)}"
        )
    hidden = residual.shape[-1]
    if gate.dim() != 2 or gate.shape[1] != hidden or gate.device != residual.device:
        raise ValueError(f"gate must be (R, {hidden}) on the residual's device")
    if same_dtype and gate.dtype != residual.dtype:
        raise TypeError(f"gate dtype {gate.dtype} must match the residual dtype {residual.dtype}")
    if index.dtype != torch.int64 or index.dim() != 1 or index.device != residual.device:
        raise TypeError("index must be a 1-D int64 tensor on the residual's device")
    if residual.shape[-2] != index.shape[0]:
        raise ValueError(f"residual must be (..., S, H) with S = len(index) = {index.shape[0]}")
    if check_range and bool(((index < 0) | (index >= gate.shape[0])).any()):
        raise IndexError(f"index must lie in [0, {gate.shape[0]})")


class _DeterministicGateResidual(torch.autograd.Function):
    """Eager forward; the gate gradient as FP32 per-row sums instead of
    ``index_select``'s BF16 atomic scatter-add (``d_residual``/``d_y`` are the
    eager VJPs)."""

    @staticmethod
    def forward(ctx, residual, y, gate, index):
        gathered = gate.index_select(0, index)
        ctx.save_for_backward(y, gathered, index)
        ctx.gate_rows = gate.shape[0]
        return residual + gathered * y

    @staticmethod
    def backward(ctx, grad):
        y, gathered, index = ctx.saved_tensors
        contrib = (grad.float() * y.float()).reshape(-1, index.shape[0], y.shape[-1])
        per_position = contrib.sum(dim=0)  # (S, H): fixed reduction over the batch
        d_gate = per_position.new_zeros((ctx.gate_rows, y.shape[-1]))
        for row in range(ctx.gate_rows):
            positions = torch.nonzero(index == row).flatten()
            if positions.numel():
                d_gate[row] = per_position.index_select(0, positions).sum(dim=0)
        return grad, grad * gathered, d_gate.to(gathered.dtype), None


class NativeH3GateResidualOp:
    """PyTorch reference: ``forward`` is the eager provider expression (with a
    deterministic gate gradient), ``forward_fp32`` the FP64 golden returned in
    FP32. The raw diffusers path, atomic backward included, is
    ``rl_engine.testing.h3_provider.provider_gate_residual``."""

    op_class = "reduction"  # forward is elementwise; the gate VJP sums positions

    def __call__(self, residual, y, gate, index):
        return self.forward(residual, y, gate, index)

    def forward(self, residual, y, gate, index, *, check_range: bool = True):
        validate_h3_gate_residual(residual, y, gate, index, check_range=check_range)
        return _DeterministicGateResidual.apply(residual, y, gate, index)

    def forward_fp32(self, residual, y, gate, index):
        validate_h3_gate_residual(residual, y, gate, index, same_dtype=False)
        out = residual.double() + gate.double().index_select(0, index) * y.double()
        return out.float()
