# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CPU byte reference for the indexed FP32 tree/ascending-token profile.

This is separate from both framework select01 fallback and the accuracy gold.
No backend normalization or reduction helper is used.
"""

import torch

from rl_engine.reference.norm.adaln_modulation import _eps_supported_fp32, _validate_indexed_inputs


def _hidden_tree(values):
    width = 1 << (values.shape[-1] - 1).bit_length()
    values = torch.nn.functional.pad(values, (0, width - values.shape[-1]))
    while width > 1:
        width //= 2
        values = values[..., :width] + values[..., width:]
    return values[..., 0]


class _IndexedFixed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, modulation, index, eps):
        batch, seq, hidden = x.shape
        x32 = x.float()
        mean = _hidden_tree(x32) / hidden
        centered = x32 - mean[..., None]
        variance = _hidden_tree(centered * centered) / hidden
        root = (variance + eps).double().sqrt().float()
        rstd = 1.0 / root
        norm = centered * rstd[..., None]
        rows = torch.arange(batch)[:, None] + index.long() * batch
        selected = modulation[rows]
        shift, scale, gate = selected.chunk(3, dim=-1)
        shift, scale = shift.float(), scale.float()
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(norm, rstd, scale, index)
        ctx.mod_shape = modulation.shape
        ctx.dtype = x.dtype
        return (norm * (1.0 + scale) + shift).to(x.dtype), gate

    @staticmethod
    def backward(ctx, dy, dg):
        norm, rstd, scale, index = ctx.saved_tensors
        batch, seq, hidden = norm.shape
        if dg is not None:
            dg = dg.float()
        dx = None
        dm = torch.zeros(ctx.mod_shape, dtype=torch.float32)
        if dy is not None:
            dy = dy.float()
            dnorm = dy * (1.0 + scale)
            dx = (
                dnorm
                - (_hidden_tree(dnorm) / hidden)[..., None]
                - norm * (_hidden_tree(dnorm * norm) / hidden)[..., None]
            ) * rstd[..., None]
        for b in range(batch):
            for s in range(seq):
                bank = int(index[b if index.shape[0] > 1 else 0, s])
                row = b + bank * batch
                if dy is not None:
                    dm[row, :hidden] = dm[row, :hidden] + dy[b, s]
                    contribution = dy[b, s] * norm[b, s]
                    dm[row, hidden : 2 * hidden] = dm[row, hidden : 2 * hidden] + contribution
                if dg is not None:
                    dm[row, 2 * hidden :] = dm[row, 2 * hidden :] + dg[b, s]
        return None if dx is None else dx.to(ctx.dtype), dm.to(ctx.dtype), None, None


class IndexedAdaLNFixedReference:
    """FP32/BF16 CPU callable for fixed-order byte tests, not a dispatch fallback."""

    def __call__(self, x, modulation, *, modulate_index, eps=1e-6):
        if x.device.type != "cpu" or modulation.device.type != "cpu":
            raise ValueError("indexed byte reference requires CPU tensors")
        if x.dtype not in (torch.float32, torch.bfloat16) or modulation.dtype != x.dtype:
            raise TypeError("indexed byte reference requires matching FP32 or BF16 inputs")
        modulate_index = modulate_index.cpu()
        eps = float(eps)
        _validate_indexed_inputs(x, modulation, modulate_index, eps)
        if not _eps_supported_fp32(eps):
            raise ValueError("eps must remain finite FP32 without positive underflow")
        return _IndexedFixed.apply(x, modulation, modulate_index, eps)
