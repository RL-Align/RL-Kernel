# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""MiniMax-H3-Image denoising update (RFC #738 ``image_velocity_step``).

At step ``i`` the released loop updates the generated rows of the image selection, and only
those, with the FP32 velocity-form Euler step on the sealed grid:

    rows[R:] = cast(fp32(rows[R:]) + (sigma_i - sigma_{i+1}) * fp32(velocity[R:]))

where ``R`` is the reference-row count, ``sigma`` the round-tripped PDD grid
(``image_sigma_pdd_schedule``) and the step is the registered
``h3_image_velocity_step`` operator (PyTorch reference, or the Triton kernel on CUDA/ROCm).
The reference rows are context: they are never written, and :class:`ImageVelocitySampler`
checks they are byte-identical after every update.

:class:`ImageVelocitySampler` replaces the scheduler of the released loop: it owns the
schedule, takes the six steps in order (head ``i`` at step ``i``; a seventh call, a skipped
or repeated step, or a head other than the step fails), and records what it did.

Upstream: ``MiniMaxH3ImageLoopSchedulerStep`` (``denoise.py``) and
``MiniMaxH3Scheduler.step`` (``euler_update_form="velocity"``) of Diffusers at commit
875373ac7f5856071e345a3e394c74efccc6b9e6.
"""

from __future__ import annotations

from typing import Any

import torch

from rl_engine.models.minimax_h3_image import patch_pack as pp
from rl_engine.models.minimax_h3_image import pdd_schedule as ps
from rl_engine.reference.flow.velocity_step import NativeH3ImageVelocityStepOp
from rl_engine.reference.minimax_h3.sigma_schedule import H3SigmaSchedule

CONTRACT_ID = "h3-image-velocity-step-v1"


class GeneratedRowsOnly(RuntimeError):
    """An update changed a reference row."""


class ImageVelocitySampler:
    """The six released Euler updates of one request, generated rows only."""

    def __init__(
        self,
        schedule: H3SigmaSchedule | None = None,
        *,
        op: Any | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.schedule = schedule if schedule is not None else ps.released_schedule(device)
        self.op = op if op is not None else NativeH3ImageVelocityStepOp()
        self.step_index = 0
        self.records: list[dict[str, Any]] = []

    @property
    def backend_id(self) -> str:
        return str(getattr(self.op, "backend_id", type(self.op).__name__))

    @property
    def timesteps(self) -> torch.Tensor:
        return self.schedule.timesteps

    def sigma_pair(self, step: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """``(sigma_i, sigma_{i+1})`` as FP32 scalars on ``device``, read from the grid."""

        sigmas = self.schedule.sigmas
        return (
            sigmas[step].to(device=device, dtype=torch.float32),
            sigmas[step + 1].to(device=device, dtype=torch.float32),
        )

    def step(
        self,
        step: int,
        velocity: torch.Tensor,
        image_rows: torch.Tensor,
        num_reference_rows: int,
        *,
        pdd_head: int | None = None,
    ) -> torch.Tensor:
        """Update ``image_rows[R:]`` in place and return ``image_rows``.

        ``velocity`` is the head output for every image row, ``(1, N, 96)`` or ``(N, 96)``;
        ``image_rows`` is ``(N, 96)`` (references first).
        """

        if step != self.step_index:
            raise ps.OffGridTimestep(f"expected step {self.step_index}, got {step}")
        ps.head_for_step(step)
        if pdd_head is not None and pdd_head != step:
            raise ps.OffGridTimestep(f"step {step} must use PDD head {step}, got {pdd_head}")
        if step >= self.schedule.num_model_evals:
            raise ps.OffGridTimestep("the schedule has no further model step")
        v = velocity[0] if velocity.ndim == 3 and velocity.shape[0] == 1 else velocity
        if v.ndim != 2 or image_rows.ndim != 2 or v.shape != image_rows.shape:
            raise ValueError(
                f"velocity {list(velocity.shape)} and image rows {list(image_rows.shape)} must "
                f"both be (N, {pp.WIDTH})"
            )
        if int(image_rows.shape[1]) != pp.WIDTH:
            raise ValueError(f"image rows must be {pp.WIDTH} wide")
        r = int(num_reference_rows)
        if isinstance(num_reference_rows, bool) or not 0 <= r < int(image_rows.shape[0]):
            raise ValueError("num_reference_rows must leave at least one generated row")
        before = image_rows[:r].clone()
        sigma, sigma_next = self.sigma_pair(step, image_rows.device)
        image_rows[r:] = self.op(image_rows[r:], v[r:].float(), sigma, sigma_next)
        if r and not _same_bits(image_rows[:r], before):
            raise GeneratedRowsOnly(f"step {step} changed a reference row")
        self.records.append(
            {
                "step": step,
                "pdd_head": step,
                "sigma": float(sigma).hex(),
                "sigma_next": float(sigma_next).hex(),
                "generated_rows": int(image_rows.shape[0]) - r,
                "reference_rows": r,
            }
        )
        self.step_index += 1
        return image_rows

    def provenance(self) -> dict[str, Any]:
        """Contract, operator backend and schedule digest of this sampler."""

        return {
            "contract_id": CONTRACT_ID,
            "semantic_op": "h3_image_velocity_step",
            "backend_id": self.backend_id,
            "schedule": ps.artifact(self.schedule)["sha256"],
            "update": "fp32(x) + (sigma_i - sigma_{i+1}) * fp32(v), generated rows only",
        }


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bit equality (``-0.0`` and NaN payloads included)."""

    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    views = {4: torch.int32, 2: torch.int16, 8: torch.int64}
    view = views.get(a.element_size())
    if view is None:
        return bool(torch.equal(a, b))
    return bool(torch.equal(a.contiguous().view(view), b.contiguous().view(view)))
