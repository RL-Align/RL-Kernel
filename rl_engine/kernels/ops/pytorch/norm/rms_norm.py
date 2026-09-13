# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import torch


@torch.library.custom_op("rl_kernel::strict_rms_norm", mutates_args=())
def _strict_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Preserve PyTorch eager RMSNorm arithmetic across graph compilation."""

    return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight, eps)


@_strict_rms_norm.register_fake
def _strict_rms_norm_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    del weight, eps
    return torch.empty_like(x)


@torch.library.custom_op("rl_kernel::strict_add_rms_norm", mutates_args=())
def _strict_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preserve vLLM's eager residual-add and RMSNorm contract."""

    updated_residual = x + residual
    normalized = torch.nn.functional.rms_norm(
        updated_residual,
        (updated_residual.shape[-1],),
        weight,
        eps,
    )
    return normalized, updated_residual


@_strict_add_rms_norm.register_fake
def _strict_add_rms_norm_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del residual, weight, eps
    return torch.empty_like(x), torch.empty_like(x)


def strict_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    return _strict_rms_norm(x, weight, eps)


def strict_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _strict_add_rms_norm(x, residual, weight, eps)




def shape_invariant_rstd(x_f: torch.Tensor, eps: float) -> torch.Tensor:
    """Shape-invariant per-row rstd (the shared RMSNorm statistic).

    torch mean/sum select shape-dependent reduction kernels on NPU and flip
    single-ULP results between batch layouts (e.g. [1,7,H] vs [1,20,H]),
    which breaks the chunked-vs-full model invariance. This reduction sums
    in FIXED 32-wide chunks first, so the intermediate shapes -- and hence
    the reduction kernels -- never depend on the batch layout, and the
    result is bitwise identical for every layout on every device.
    """
    hidden = x_f.shape[-1]
    if hidden % 32 != 0:
        var = x_f.pow(2).mean(dim=-1)
        return torch.rsqrt(var + float(eps))
    sq = x_f.pow(2).reshape(*x_f.shape[:-1], -1, 32)
    partial = sq.sum(dim=-1)      # [*, C] — fixed 32-wide chunks
    sumsq = partial.sum(dim=-1)   # [*lead]
    var = sumsq / float(hidden)
    return torch.rsqrt(var + float(eps))

class NativeRMSNormOp:
    """
    Pure Pytorch native RMSNorm reference
    out = x * rsqrt(mean(x^2, dim=-1) + eps) * weight
    """

    def __init__(self) -> None:
        pass

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        *,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        return self.forward(x, weight, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        *,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Canonical entry: accumulate in fp32, cast the result back to x.dtype.
        This is the dtype-behavior path used as the Axis-B accuracy candidate.
        """
        return self._rms_norm(x, weight, eps=eps, output_dtype=x.dtype)

    def forward_fp32(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        *,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Ground-truth: accumulate in fp32 and force fp32 output."""
        return self._rms_norm(x, weight, eps=eps, output_dtype=torch.float32)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _rms_norm(
        x: torch.Tensor,
        weight: torch.Tensor,
        *,
        eps: float,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        if weight.dim() != 1 or weight.shape[0] != x.shape[-1]:
            raise ValueError(
                f"weight must be 1-D of size x.shape[-1]={x.shape[-1]}, "
                f"got tuple(weight.shape)={tuple(weight.shape)}"
            )
        x_f = x.float()
        rstd = shape_invariant_rstd(x_f, float(eps)).unsqueeze(-1)
        normed = x_f * rstd
        out = normed * weight.float()
        return out.to(output_dtype)
