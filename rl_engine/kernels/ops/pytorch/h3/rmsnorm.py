# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""H3 RMSNorm and its fused AdaLN modulation (RFC #420 row ``h3_rmsnorm``).

Every H3 RMSNorm is ``nn.RMSNorm(5376, eps=1e-5)`` with an affine weight in the
block dtype: the block ``norm1``/``norm2``, the token-refiner norms and the
final ``norm_out.norm``. Inside a transformer block (and in ``norm_out``) the
normalised rows are immediately modulated by per-row AdaLN parameters::

    n   = rms_norm(x, weight, eps)
    out = n * (1.0 + scale[index]) + shift[index]      # eager, rounded after every op

``index`` is ``adaln_indices`` in a block and ``timestep_indices`` in
``norm_out``; ``shift``/``scale`` are ``(R, H)`` rows (views of the AdaLN
table). ``x`` is ``(..., S, H)`` and ``index`` has one entry per position ``S``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

H3_NORM_EPS = 1e-5
_DTYPES = (torch.float32, torch.bfloat16, torch.float16)


def validate_h3_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> int:
    for name, tensor in (("x", x), ("weight", weight)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dtype not in _DTYPES:
            raise TypeError(f"{name} must be fp32, bf16 or fp16, got {tensor.dtype}")
    if weight.dtype != x.dtype:
        raise TypeError(f"weight dtype {weight.dtype} must match x dtype {x.dtype}")
    if weight.device != x.device:
        raise ValueError(f"weight is on {weight.device}, x is on {x.device}")
    if x.dim() < 2 or x.numel() == 0:
        raise ValueError(f"x must be a non-empty (..., N) tensor, got {tuple(x.shape)}")
    if weight.shape != (x.shape[-1],):
        raise ValueError(f"weight must be ({x.shape[-1]},), got {tuple(weight.shape)}")
    if not eps > 0:
        raise ValueError(f"eps must be positive, got {eps}")
    return x.shape[-1]


def validate_h3_modulation(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    index: torch.Tensor,
    *,
    check_range: bool = True,
    same_dtype: bool = True,
) -> None:
    hidden = x.shape[-1]
    for name, tensor in (("shift", shift), ("scale", scale)):
        if tensor.device != x.device or (same_dtype and tensor.dtype != x.dtype):
            raise TypeError(f"{name} must match x's dtype and device")
        if tensor.dim() != 2 or tensor.shape[1] != hidden:
            raise ValueError(f"{name} must be (R, {hidden}), got {tuple(tensor.shape)}")
    if shift.shape != scale.shape:
        raise ValueError("shift and scale must have the same shape")
    if index.dtype != torch.int64 or index.dim() != 1 or index.device != x.device:
        raise TypeError("index must be a 1-D int64 tensor on x's device")
    if x.dim() < 2 or x.shape[-2] != index.shape[0]:
        raise ValueError(
            f"x must be (..., S, {hidden}) with S = len(index) = {index.shape[0]}, "
            f"got {tuple(x.shape)}"
        )
    if check_range and bool(((index < 0) | (index >= shift.shape[0])).any()):
        raise IndexError(f"index must lie in [0, {shift.shape[0]})")


class _DeterministicModulation(torch.autograd.Function):
    """Eager ``n * (1.0 + scale[i]) + shift[i]``; the ``shift``/``scale``
    gradients as FP32 per-row sums instead of ``index_select``'s atomic
    scatter-add (``d_n`` is the eager VJP)."""

    @staticmethod
    def forward(ctx, n, shift, scale, index):
        t1 = 1.0 + scale.index_select(0, index)
        ctx.save_for_backward(n, t1, index)
        ctx.rows = shift.shape[0]
        return n * t1 + shift.index_select(0, index)

    @staticmethod
    def backward(ctx, grad):
        n, t1, index = ctx.saved_tensors
        hidden = n.shape[-1]
        g = grad.float().reshape(-1, index.shape[0], hidden)
        per_shift = g.sum(dim=0)
        per_scale = (g * n.float().reshape(-1, index.shape[0], hidden)).sum(dim=0)
        d_shift = per_shift.new_zeros((ctx.rows, hidden))
        d_scale = per_scale.new_zeros((ctx.rows, hidden))
        for row in range(ctx.rows):
            positions = torch.nonzero(index == row).flatten()
            if positions.numel():
                d_shift[row] = per_shift.index_select(0, positions).sum(dim=0)
                d_scale[row] = per_scale.index_select(0, positions).sum(dim=0)
        return grad * t1, d_shift.to(t1.dtype), d_scale.to(t1.dtype), None


class NativeH3RMSNormOp:
    """PyTorch reference.

    ``forward`` / ``forward_modulated`` are the provider path (``F.rms_norm`` and
    the eager modulation expression, with a deterministic shift/scale gradient;
    the raw diffusers path is ``h3_provider.provider_norm_modulate``). ``forward_fp32`` /
    ``forward_modulated_fp32`` are the golden: the same math in FP64 with no
    intermediate rounding, returned in FP32.
    """

    op_class = "reduction"

    def __call__(self, x, weight, eps: float = H3_NORM_EPS):
        return self.forward(x, weight, eps)

    def forward(self, x, weight, eps: float = H3_NORM_EPS) -> torch.Tensor:
        hidden = validate_h3_rmsnorm(x, weight, eps)
        return F.rms_norm(x, (hidden,), weight, eps)

    def forward_fp32(self, x, weight, eps: float = H3_NORM_EPS) -> torch.Tensor:
        validate_h3_rmsnorm(x, weight, eps)
        x64 = x.double()
        rstd = torch.rsqrt(x64.square().mean(dim=-1, keepdim=True) + eps)
        return (x64 * rstd * weight.double()).float()

    def forward_modulated(self, x, weight, shift, scale, index, eps: float = H3_NORM_EPS):
        validate_h3_modulation(x, shift, scale, index)
        n = self.forward(x, weight, eps)
        return _DeterministicModulation.apply(n, shift, scale, index)

    def forward_modulated_fp32(self, x, weight, shift, scale, index, eps: float = H3_NORM_EPS):
        # The golden may be fed FP32 golden modulation rows for a BF16 x.
        validate_h3_modulation(x, shift, scale, index, same_dtype=False)
        validate_h3_rmsnorm(x, weight, eps)
        x64 = x.double()
        n = x64 * torch.rsqrt(x64.square().mean(dim=-1, keepdim=True) + eps) * weight.double()
        out = n * (1.0 + scale.double().index_select(0, index)) + shift.double().index_select(
            0, index
        )
        return out.float()
