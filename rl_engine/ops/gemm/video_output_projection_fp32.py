# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""FP32 MiniMax-H3 video output projection with logical-row parameter gradients."""

from __future__ import annotations

from collections.abc import Callable

import torch

from rl_engine.ops.autograd.canonical_backward import active_session
from rl_engine.ops.gemm.canonical_linear import _gemm_fp32, canonical_linear_fp32

VIDEO_PATCH_WIDTH = 96
VIDEO_HIDDEN_WIDTH = 5376


def validate_projection_inputs(x, weight, bias):
    """Check contiguous FP32 projection tensors and fixed H3 dimensions."""

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
    if x.ndim not in (2, 3) or x.shape[-1] != VIDEO_HIDDEN_WIDTH or x.numel() == 0:
        raise ValueError("x must have non-empty shape [M,5376] or [B,S,5376]")
    if tuple(weight.shape) != (VIDEO_PATCH_WIDTH, VIDEO_HIDDEN_WIDTH):
        raise ValueError("weight must have shape [96,5376]")
    if bias is not None and tuple(bias.shape) != (VIDEO_PATCH_WIDTH,):
        raise ValueError("bias must have shape [96]")


def _validate_keys(keys, rows, device, parameter_id, session, weight, bias):
    """Require unique active logical keys across each parameter's registered uses.

    Each input row has a two- or three-column key. A negative first component
    marks padding, which is excluded from the uniqueness check.
    """

    if not isinstance(keys, torch.Tensor):
        raise TypeError("parameter gradients require logical_keys")
    if keys.dtype != torch.int64 or keys.device != device or not keys.is_contiguous():
        raise ValueError("logical_keys must be contiguous int64 on the projection device")
    if keys.ndim != 2 or keys.shape[0] != rows or keys.shape[1] not in (2, 3):
        raise ValueError("logical_keys must have shape [M,2] or [M,3]")
    if not isinstance(parameter_id, str) or not parameter_id.strip():
        raise ValueError("parameter gradients require a non-empty parameter_id")
    # A duplicate active identity leaves tie order dependent on chunk arrival.
    for name, needed in (
        ("weight", weight.requires_grad),
        ("bias", bias is not None and bias.requires_grad),
    ):
        if not needed:
            continue
        previous = session.uses.get(f"{parameter_id}.{name}", [])
        if any(use.keys.shape[1] != keys.shape[1] for use in previous):
            raise ValueError("logical key columns must agree across parameter uses")
        merged = torch.cat([use.keys for use in previous] + [keys], dim=0) if previous else keys
        active = merged[merged[:, 0] >= 0]
        if torch.unique(active, dim=0).shape[0] != active.shape[0]:
            raise ValueError("active logical keys must be unique across parameter uses")


class VideoOutputProjectionFP32Op:
    """Shared semantic adapter; subclasses select one qualified GEMM backend."""

    family: str
    backend_id: str
    matmul_op: Callable[..., torch.Tensor] | None = None

    def _check_backend(self, x):
        """Require subclasses to check device and architecture eligibility."""

        raise NotImplementedError

    def __call__(self, x, weight, bias=None, *, logical_keys=None, parameter_id=None):
        """Project [M, 5376] or [B, S, 5376] to FP32 video patch width 96.

        When gradients are enabled, trainable weight or bias requires an active
        canonical backward session, logical_keys, and parameter_id. Calls that
        need only input gradients do not require parameter-gradient registration.
        """

        validate_projection_inputs(x, weight, bias)
        self._check_backend(x)
        # Flatten batch and sequence axes without changing logical row order.
        rows = x.reshape(-1, VIDEO_HIDDEN_WIDTH)
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
                # Preserve registered row identities if the caller later edits its keys.
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
            )
        else:
            # Calls without gradients bypass the backward session.
            output = _gemm_fp32(
                rows,
                weight.t().contiguous(),
                self.family,
                bias=bias,
                matmul_op=self.matmul_op,
            )
        return output.reshape(*x.shape[:-1], VIDEO_PATCH_WIDTH)

    forward = __call__


__all__ = ["VideoOutputProjectionFP32Op"]
