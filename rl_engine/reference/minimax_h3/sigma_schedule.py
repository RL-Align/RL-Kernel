# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""FP32 reference for the MiniMax-H3 sigma grid (RFC #420 ``h3_sigma_schedule``).

Upstream: ``MiniMaxH3Scheduler.set_timesteps`` in diffusers
``schedulers/scheduling_minimax_h3.py``, pinned to git blob sha
``ca69ae6a2f6aae5bb68498216316a71eb07c1f7b``. The file was added once, in
commit f53d552036a0d1bd5570782a39cd40cfabf112bc ("Add MiniMax-H3 (#14355)",
2026-08-05), and has not changed since:
https://github.com/huggingface/diffusers/blob/f53d552/src/diffusers/schedulers/scheduling_minimax_h3.py
The construction order is byte-frozen against that source:

    linspace(1, 0, N, float32)             # CPU
    -> shift * s / (1 + (shift - 1) * s)   # python-float scalar operands
    -> unique_consecutive                  # collapse fp32 collisions near s=1
    -> timesteps = 1 - sigmas[:-1]         # t = 1 - sigma, t = 1 is clean

Released checkpoints run one grid per modality — shift=12.0 for video,
shift=3.0 for audio — so the grid is a per-call argument, not scheduler state.
Every arithmetic step runs on CPU in float32; tensors move to ``device`` only
after the grid is fully built. Consumers take sigma as a per-row input
(PR #471 ``h3_ode_step``), so this operator carries no step index.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class H3SigmaSchedule:
    """One MiniMax-H3 sigma grid.

    Attributes:
        sigmas: float32, shape ``[M]`` with ``M <= num_inference_steps``.
            Strictly decreasing; ``sigmas[0] == 1.0`` exactly for the released
            shifts (3.0, 12.0) and ``sigmas[-1] == 0.0`` exactly (the shift
            maps 0 to 0).
        timesteps: float32, shape ``[M - 1]``; ``timesteps = 1 - sigmas[:-1]``,
            strictly increasing from ``0.0`` towards ``1.0``.
        num_model_evals: ``M - 1``, the number of transformer evaluations the
            grid drives (dedup included).
    """

    sigmas: Tensor
    timesteps: Tensor
    num_model_evals: int


class NativeH3SigmaScheduleOp:
    """Pure PyTorch reference for the H3 sigma grid — forward only."""

    def __call__(
        self,
        num_inference_steps: int,
        *,
        shift: float,
        device: str | torch.device | None = None,
    ) -> H3SigmaSchedule:
        return self.forward(num_inference_steps, shift=shift, device=device)

    def forward(
        self,
        num_inference_steps: int,
        *,
        shift: float,
        device: str | torch.device | None = None,
    ) -> H3SigmaSchedule:
        """Build the grid in the frozen order; see the module docstring."""
        # Both checks mirror upstream's own failures and add no extra policy:
        # num_inference_steps < 2 is set_timesteps' check (pinned blob,
        # lines 149-153) and shift <= 0 the constructor's (lines 75-76).
        if num_inference_steps < 2:
            raise ValueError(
                "num_inference_steps must be >= 2 (terminal zero included), got "
                f"{num_inference_steps}."
            )
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}.")

        # Scalar operands must stay python floats in exactly this expression:
        # pre-casting shift to a tensor or reassociating the division changes
        # fp32 rounding and breaks byte equality with upstream (lines 156-159).
        # device="cpu" is explicit because factory functions without a device
        # follow the ambient default device (meta/CUDA context) instead.
        base = torch.linspace(1.0, 0.0, int(num_inference_steps), dtype=torch.float32, device="cpu")
        shift = float(shift)
        sigmas = shift * base / (1 + (shift - 1) * base)
        # The shift compresses spacing near sigma=1; drop the fp32 collisions
        # it creates before deriving timesteps and the evaluation count.
        sigmas = torch.unique_consecutive(sigmas)
        timesteps = 1.0 - sigmas[:-1]

        return H3SigmaSchedule(
            sigmas=sigmas.to(device=device),
            timesteps=timesteps.to(device=device),
            num_model_evals=int(sigmas.numel()) - 1,
        )
