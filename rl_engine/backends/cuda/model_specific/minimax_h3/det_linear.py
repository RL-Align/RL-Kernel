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
    return bool(_EXT_AVAILABLE and all(hasattr(_C, name) for name in _SYMBOLS))


def _require() -> None:
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
    _require()
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
    _require()
    return _C.h3_det_linear_backward_input(
        grad.float().contiguous(), weight.contiguous(), out_dtype
    )


def det_linear_backward_weight(
    grad: torch.Tensor, x: torch.Tensor, w_dtype: torch.dtype, *, with_bias: bool = True
) -> list[torch.Tensor]:
    _require()
    return _C.h3_det_linear_backward_weight(
        grad.float().contiguous(), x.contiguous(), w_dtype, bool(with_bias)
    )


def silu_backward_fp32(grad: torch.Tensor, pre_activation: torch.Tensor) -> torch.Tensor:
    """Elementwise SiLU VJP in FP32: g * s * (1 + z * (1 - s)), s = sigmoid(z)."""

    z = pre_activation.float()
    s = torch.sigmoid(z)
    return grad.float() * (s * (1.0 + z * (1.0 - s)))
