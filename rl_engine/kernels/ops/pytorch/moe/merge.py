# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import torch
from torch import Tensor


@torch.no_grad()
def _merge(routed: Tensor, shared: Tensor, residual: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    # Compilation may reassociate additions; this is the eager FP32 reference.
    if torch.compiler.is_compiling():
        raise RuntimeError("shared_residual_merge_fwd requires PyTorch eager execution.")
    for name, value in (("routed", routed), ("shared", shared), ("residual", residual)):
        if type(value) is not Tensor:
            raise TypeError(f"{name} must be a torch.Tensor.")
        if value.layout != torch.strided or value.ndim != 2 or not value.is_contiguous():
            raise ValueError(f"{name} must be a contiguous [T, H] tensor.")
        if value.shape[0] == 0 or value.shape[1] == 0:
            raise ValueError(f"{name} must have non-empty token and hidden dimensions.")
        if value.shape != routed.shape:
            raise ValueError("routed, shared and residual must share shape.")
        if value.device != routed.device:
            raise RuntimeError("routed, shared and residual must be on the same device.")
        if value.device.type not in {"cpu", "cuda"}:
            raise RuntimeError("shared_residual_merge_fwd supports CPU and CUDA/ROCm tensors.")
        dtypes = (torch.float32,) if name == "routed" else (torch.float32, torch.bfloat16)
        if value.dtype not in dtypes:
            raise TypeError(f"{name} must have dtype {dtypes}, got {value.dtype}.")
    if routed.is_cuda:
        with torch.cuda.device(routed.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("shared_residual_merge_fwd does not support CUDA Graph capture.")

    with torch.autocast(device_type=routed.device.type, enabled=False):
        after_shared = routed + shared.float()
        after_residual = after_shared + residual.float()
        output = after_residual.to(torch.bfloat16)
    return after_shared, after_residual, output


def shared_residual_merge_fwd(routed: Tensor, shared: Tensor, residual: Tensor) -> Tensor:
    """Compute ``(routed + shared) + residual`` in FP32, then cast once to BF16.

    Inputs are contiguous [T, H] rows on the same device. Routed rows are already
    weighted and combined in FP32; shared and residual may be FP32 or BF16.
    This is a forward-only reference. Producer identity and ownership checks
    belong to the caller's validation path.
    """
    return _merge(routed, shared, residual)[-1]


class MoeMergeOp(torch.nn.Module):
    """Pure PyTorch reference for the MoE shared/residual merge."""

    def forward(self, routed: Tensor, shared: Tensor, residual: Tensor) -> Tensor:
        return shared_residual_merge_fwd(routed, shared, residual)

    def forward_with_intermediates(
        self, routed: Tensor, shared: Tensor, residual: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return after-shared FP32, after-residual FP32 and final BF16 tensors."""
        return _merge(routed, shared, residual)
