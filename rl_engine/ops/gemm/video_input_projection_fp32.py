# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""FP32 MiniMax-H3 video input projection with logical-row parameter VJPs."""

from __future__ import annotations

from collections.abc import Callable

import torch

from rl_engine.ops.autograd.canonical_backward import active_session
from rl_engine.ops.gemm.canonical_linear import _gemm_fp32, canonical_linear_fp32

VIDEO_PATCH_WIDTH = 96
VIDEO_HIDDEN_WIDTH = 5376


def validate_projection_inputs(x, weight, bias):
    """Validate the public FP32 tensor boundary before backend dispatch."""
    tensors = {"x": x, "weight": weight}
    if bias is not None:
        tensors["bias"] = bias
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dtype != torch.float32:
            raise TypeError(f"{name} must have dtype torch.float32")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if tensor.device != x.device:
            raise ValueError("projection tensors must be on the same device")
    if x.ndim not in (2, 3) or x.shape[-1] != VIDEO_PATCH_WIDTH or x.numel() == 0:
        raise ValueError("x must have non-empty shape [M,96] or [B,S,96]")
    if tuple(weight.shape) != (VIDEO_HIDDEN_WIDTH, VIDEO_PATCH_WIDTH):
        raise ValueError("weight must have shape [5376,96]")
    if bias is not None and tuple(bias.shape) != (VIDEO_HIDDEN_WIDTH,):
        raise ValueError("bias must have shape [5376]")


def _validate_keys(keys, rows, device, parameter_id, session, weight, bias):
    if not isinstance(keys, torch.Tensor):
        raise TypeError("parameter gradients require logical_keys")
    if keys.dtype != torch.int64 or keys.device != device or not keys.is_contiguous():
        raise ValueError("logical_keys must be contiguous int64 on the projection device")
    if keys.ndim != 2 or keys.shape[0] != rows or keys.shape[1] not in (2, 3):
        raise ValueError("logical_keys must have shape [M,2] or [M,3]")
    if not isinstance(parameter_id, str) or not parameter_id.strip():
        raise ValueError("parameter gradients require a non-empty parameter_id")
    # A duplicate active identity leaves tie order dependent on chunk arrival.
    checked = set()
    for name, needed in (
        ("weight", weight.requires_grad),
        ("bias", bias is not None and bias.requires_grad),
    ):
        if not needed:
            continue
        previous = session.uses.get(f"{parameter_id}.{name}", [])
        signature = tuple(use.contribution for use in previous)
        if all(token is not None for token in signature):
            if signature in checked:
                continue
            checked.add(signature)
        if any(use.keys.shape[1] != keys.shape[1] for use in previous):
            raise ValueError("logical key columns must agree across parameter uses")
        merged = torch.cat([use.keys for use in previous] + [keys], dim=0) if previous else keys
        active = merged[merged[:, 0] >= 0]
        if torch.unique(active, dim=0).shape[0] != active.shape[0]:
            raise ValueError("active logical keys must be unique across parameter uses")


class VideoInputProjectionFP32Op:
    """Shared semantic adapter; subclasses select one qualified GEMM backend."""

    family: str
    backend_id: str
    matmul_op: Callable[..., torch.Tensor] | None = None
    reuse_canonical_order: bool = False

    def _check_backend(self, x):
        raise NotImplementedError

    def __call__(self, x, weight, bias=None, *, logical_keys=None, parameter_id=None):
        validate_projection_inputs(x, weight, bias)
        self._check_backend(x)
        rows = x.reshape(-1, VIDEO_PATCH_WIDTH)
        needs_grad = torch.is_grad_enabled() and any(
            tensor is not None and tensor.requires_grad for tensor in (x, weight, bias)
        )
        if needs_grad:
            if x.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
                raise RuntimeError("video projection backward requires eager execution")
            if weight.requires_grad or (bias is not None and bias.requires_grad):
                session = active_session()
                if session is None:
                    raise RuntimeError("parameter gradients require canonical_backward_session")
                _validate_keys(
                    logical_keys, rows.shape[0], x.device, parameter_id, session, weight, bias
                )
                logical_keys = logical_keys.detach().clone()
            output = canonical_linear_fp32(
                rows,
                weight,
                logical_keys,
                parameter_id=f"{parameter_id}.weight",
                family=self.family,
                bias=bias,
                bias_parameter_id=f"{parameter_id}.bias",
                matmul_op=self.matmul_op,
                reuse_canonical_order=self.reuse_canonical_order,
            )
        else:
            output = _gemm_fp32(
                rows,
                weight.t().contiguous(),
                self.family,
                bias=bias,
                matmul_op=self.matmul_op,
            )
        return output.reshape(*x.shape[:-1], VIDEO_HIDDEN_WIDTH)

    forward = __call__


__all__ = ["VideoInputProjectionFP32Op"]
