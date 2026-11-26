# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""MiniMax-H3-Image velocity-form Euler step (RFC #738 ``image_velocity_step``, reference).

    x_next = xt + (sigma - sigma_next) * v

FP32 throughout, one cast at the end, in exactly this order: ``d = sigma - sigma_next`` is
one FP32 subtraction, then one multiply, then one add (no fused multiply-add). This is
``MiniMaxH3Scheduler.step`` with ``euler_update_form="velocity"`` (Diffusers 875373ac), the
form the MiniMax-H3-Image scheduler config selects.

It is not the MiniMax-H3 step (RFC #420 ``h3_ode_step``, :mod:`.ode_step`), which writes
the same Euler update as the blend ``r * xt + (1 - r) * (xt + sigma * v)`` with
``r = sigma_next / sigma``: equal in real arithmetic, a different FP32 program (the H13
probe of #420). The input checks are the H3 ones, reused from :mod:`.ode_step`, so a caller
sees the same rejections from both steps.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor

from rl_engine.reference.flow.ode_step import (
    _as_sigma,
    _check_sigma_layout,
    _check_sigma_values,
    _check_xt_v,
)


def _velocity_step_fp32(xt: Tensor, v: Tensor, sigma: Tensor, sigma_next: Tensor) -> Tensor:
    """The declared FP32 program; this is the gold expression order."""
    return xt + (sigma - sigma_next) * v


class NativeH3ImageVelocityStepOp:
    """PyTorch reference: FP32 velocity-form Euler update, one cast on return."""

    op_class = "elementwise"

    def __call__(self, xt: Tensor, v: Tensor, sigma: Any, sigma_next: Any) -> Tensor:
        """Alias for :meth:`forward` so the op can be invoked like a function."""
        return self.forward(xt, v, sigma, sigma_next)

    def forward(self, xt: Tensor, v: Tensor, sigma: Any, sigma_next: Any) -> Tensor:
        """Return ``x_next`` in ``xt``'s dtype, computed in FP32."""
        s, sn = self._prepare(xt, v, sigma, sigma_next)
        return _velocity_step_fp32(xt.float(), v.float(), s, sn).to(xt.dtype)

    def forward_fp32(self, xt: Tensor, v: Tensor, sigma: Any, sigma_next: Any) -> Tensor:
        """Gold path: FP32 in, FP32 out."""
        s, sn = self._prepare(xt, v, sigma, sigma_next)
        return _velocity_step_fp32(xt.float(), v.float(), s, sn)

    @staticmethod
    def _prepare(xt: Tensor, v: Tensor, sigma: Any, sigma_next: Any) -> tuple[Tensor, Tensor]:
        """Validate the inputs and normalise both sigma arguments (the H3 checks)."""
        _check_xt_v(xt, v)
        s = _as_sigma(sigma, xt, "sigma")
        sn = _as_sigma(sigma_next, xt, "sigma_next")
        _check_sigma_layout(s, sn)
        _check_sigma_values(s, sn)
        return s, sn
