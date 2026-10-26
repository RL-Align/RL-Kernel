# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Python surface of the H3 deterministic row-wise linear (contract h3-det-linear-v1).

See ``csrc/cuda/h3/det_linear.cu`` for the reduction order. Every helper
raises instead of falling back to cuBLAS when the extension is missing.
"""

from __future__ import annotations

import torch

from rl_engine.backends.extension import _C, _EXT_AVAILABLE

ACT_NONE = 0
ACT_SILU = 1
CONTRACT = "h3-det-linear-v1"
_SYMBOLS = (
    "h3_det_linear_forward",
    "h3_det_linear_backward_input",
    "h3_det_linear_backward_weight",
)


def det_linear_available() -> bool:
    """Return whether the extension exposes all three H3 deterministic linear APIs."""

    return bool(_EXT_AVAILABLE and all(hasattr(_C, name) for name in _SYMBOLS))


def det_linear_device_supported(device: torch.device | int | str | None = None) -> bool:
    """Return whether ``device`` can run the BF16 forward (``mma.sync``, SM80+).

    ``device`` defaults to the current CUDA device. Without CUDA, or for a non-CUDA device, the
    answer is False: constructors use this as an early check so the registry falls back to the
    PyTorch reference, and ``det_linear_forward`` repeats it for the device of the actual tensors.
    """

    if not torch.cuda.is_available():
        return False
    if device is not None and torch.device(device).type != "cuda":
        return False
    return torch.cuda.get_device_capability(device)[0] >= 8


def _require() -> None:
    """Raise when a deterministic linear entry point is unavailable."""

    if not det_linear_available():
        raise RuntimeError(
            "rl_engine._C lacks the h3_det_linear_* symbols; rebuild the CUDA extension "
            "with csrc/cuda/h3/det_linear.cu (no cuBLAS fallback is allowed)"
        )


def det_linear_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    activation: int = ACT_NONE,
    save_pre_activation: bool = False,
) -> list[torch.Tensor]:
    """Project CUDA ``x`` of shape ``(T, K)`` with ``(N, K)`` weights and optional bias.

    Inputs share an FP32 or BF16 dtype/device, ``T > 0``, and ``K`` supports
    16-byte vector loads; BF16 forward needs SM80 or newer. Return ``[out]``
    in the input dtype, plus an FP32 pre-activation when requested. Activation
    is identity or SiLU; contiguous copies preserve the reduction contract.
    """

    _require()
    if x.dtype == torch.bfloat16 and not det_linear_device_supported(x.device):
        raise RuntimeError(
            f"the BF16 h3 det_linear forward needs SM80 or newer (tensors are on {x.device})"
        )
    return _C.h3_det_linear_forward(
        x.contiguous(),
        weight.contiguous(),
        None if bias is None else bias.contiguous(),
        int(activation),
        bool(save_pre_activation),
    )


def det_linear_backward_input(
    grad: torch.Tensor, weight: torch.Tensor, out_dtype: torch.dtype
) -> torch.Tensor:
    """Compute a deterministic ``(T, K)`` input VJP on the inputs' CUDA device.

    Convert nonempty ``(T, N)`` gradients to FP32 and combine them with FP32
    or BF16 ``(N, K)`` weights; return FP32 or BF16 as specified by ``out_dtype``.
    """

    _require()
    return _C.h3_det_linear_backward_input(
        grad.float().contiguous(), weight.contiguous(), out_dtype
    )


def det_linear_backward_weight(
    grad: torch.Tensor, x: torch.Tensor, w_dtype: torch.dtype, *, with_bias: bool = True
) -> list[torch.Tensor]:
    """Fold nonempty CUDA ``(T, N)`` gradients and ``(T, K)`` inputs in row order.

    Convert gradients to FP32; ``x`` is FP32 or BF16 on the same device.
    Return ``[dweight]`` of shape ``(N, K)``, optionally followed by ``dbias``
    of shape ``(N,)``, both in the requested FP32 or BF16 ``w_dtype``.
    """

    _require()
    return _C.h3_det_linear_backward_weight(
        grad.float().contiguous(), x.contiguous(), w_dtype, bool(with_bias)
    )


def silu_backward_fp32(grad: torch.Tensor, pre_activation: torch.Tensor) -> torch.Tensor:
    """Elementwise SiLU VJP in FP32: g * s * (1 + z * (1 - s)), s = sigmoid(z)."""

    z = pre_activation.float()
    s = torch.sigmoid(z)
    return grad.float() * (s * (1.0 + z * (1.0 - s)))
