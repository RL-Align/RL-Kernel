# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""MiniMax-H3-Image PDD sampling schedule (RFC #738 ``image_sigma_pdd_schedule``).

MiniMax-H3-Image samples on a fixed seven-point sigma grid its six per-step output heads
(PDD heads) were distilled on. Nothing about it is computed from a step count or a shift:

* **Grid.** ``pdd_config.json`` ``sigmas`` (equal to ``modular_model_index.json``
  ``sample_sigmas``), seven FP32-representable values from 1.0 to 0.0. The scheduler config
  has ``shift_provided_sigmas = false``, so the grid is used verbatim, not shifted.
* **FP32 round trip.** The released sampler rounds every sigma through its FP32 timestep and
  back: ``t_i = fp32(1 - sigma_i)`` for the six model sigmas, then ``sigma_i' = fp32(1 - t_i)``
  and the terminal 0. It steps on ``sigma'``. Below 0.5 that round trip can move a sigma by
  an ulp; for the released grid it moves none, and :func:`round_trip_moved` says so.
* **Six model calls.** One transformer call per timestep, none for the terminal sigma.
* **Head i at step i.** PDD head ``i`` is the head of step ``i`` and of nothing else.
  :func:`head_for_timestep` accepts only a bit-exact grid timestep: choosing a head by
  nearest sigma, interpolating or averaging heads is a different model (RFC #738 section 6)
  and fails closed.
* **Per-row timestep table.** At step ``i`` every packed row gets ``t_i`` except the
  reference rows, which get ``fp32(0.999)``. The distinct values are sorted and each row
  indexes them (``torch.unique(sorted=True, return_inverse=True)``); the transformer reads
  its AdaLN table at ``timestep_index * 3 + token_tag``.

The output type is the MiniMax-H3 one (RFC #420 ``h3_sigma_schedule``,
:class:`~rl_engine.reference.minimax_h3.sigma_schedule.H3SigmaSchedule`), so consumers of
the H3 grid take this one unchanged. The H3 operator builds a shifted ``linspace`` grid from a
step count; H3-Image has no step count to build from, so this module reads the sealed grid
instead, and the H3 construction is a probe arm here.

Upstream: ``MiniMaxH3ImageSetTimestepsStep`` (``before_denoise.py``) and
``MiniMaxH3Scheduler.set_timesteps`` of Diffusers at commit
875373ac7f5856071e345a3e394c74efccc6b9e6.
"""

from __future__ import annotations

import hashlib
import json
import struct
from typing import Any, Sequence

import torch

from rl_engine.models.minimax_h3_image import fingerprint as fp
from rl_engine.models.minimax_h3_image import packed_layout as pl
from rl_engine.reference.minimax_h3.sigma_schedule import H3SigmaSchedule

CONTRACT_ID = "h3-image-pdd-schedule-v1"
REFERENCE_TIMESTEP = fp.REFERENCE_TIMESTEP
MODALITY_NUM = len(fp.MODALITY_TAGS)  # AdaLN rows per distinct timestep


class OffGridTimestep(ValueError):
    """A timestep or step index that is not one of the six sealed PDD steps."""


def config_sigmas() -> list[float]:
    """The sealed grid from the pinned ``pdd_config.json`` (cross-checked by the fingerprint)."""

    return [float(s) for s in fp.sigma_grid()["config_sigmas"]]


def build_schedule(
    sigmas: Sequence[float], device: torch.device | str | None = None
) -> H3SigmaSchedule:
    """Grid -> FP32 timesteps -> round-tripped step sigmas, as the released sampler builds it.

    ``timesteps = fp32(1 - s)`` with ``1 - s`` evaluated in Python (FP64) and rounded once;
    ``sigmas = cat(1 - timesteps, [0])`` with ``1 - t`` an FP32 tensor operation.
    """

    grid = [float(s) for s in sigmas]
    if len(grid) < 2 or grid[-1] != 0.0 or any(b >= a for a, b in zip(grid, grid[1:])):
        raise ValueError("sigmas must hold at least two strictly decreasing values ending at 0.0")
    timesteps = torch.tensor([1.0 - s for s in grid[:-1]], dtype=torch.float32)
    step_sigmas = torch.cat([1.0 - timesteps, torch.zeros(1)])
    return H3SigmaSchedule(
        sigmas=step_sigmas.to(device=device),
        timesteps=timesteps.to(device=device),
        num_model_evals=int(timesteps.numel()),
    )


def released_schedule(device: torch.device | str | None = None) -> H3SigmaSchedule:
    """The six-step PDD schedule of the release; refuses a grid that does not fit the heads."""

    schedule = build_schedule(config_sigmas(), device)
    if schedule.num_model_evals != fp.NUM_PDD_HEADS:
        raise ValueError(f"{schedule.num_model_evals} model calls for {fp.NUM_PDD_HEADS} PDD heads")
    return schedule


def round_trip_moved(sigmas: Sequence[float]) -> list[int]:
    """Indices whose step sigma differs from the FP32 grid value after the round trip."""

    schedule = build_schedule(sigmas)
    grid = torch.tensor([float(s) for s in sigmas], dtype=torch.float32)
    return [i for i in range(grid.numel()) if _bits(schedule.sigmas[i]) != _bits(grid[i])]


def head_for_step(step: int) -> int:
    """PDD head of denoising step ``step``: the step itself, for steps 0..5 only."""

    if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step < fp.NUM_PDD_HEADS:
        raise OffGridTimestep(f"step must be an int in [0, {fp.NUM_PDD_HEADS}), got {step!r}")
    return step


def head_for_timestep(schedule: H3SigmaSchedule, timestep: float | torch.Tensor) -> int:
    """PDD head of a timestep: bit-exact grid match only.

    A timestep one ulp off the grid, a sigma used in place of a timestep or a value between
    two steps is refused: selecting the nearest head is a different model.
    """

    value = torch.as_tensor(timestep, dtype=torch.float64).reshape(-1)
    if value.numel() != 1:
        raise OffGridTimestep("one timestep at a time")
    as_f32 = value.to(torch.float32)
    if float(as_f32) != float(value):
        raise OffGridTimestep(f"{float(value)!r} is not an FP32 value")
    table = schedule.timesteps.detach().cpu()
    hits = (table.view(torch.int32) == as_f32.view(torch.int32)).nonzero().flatten()
    if hits.numel() != 1:
        raise OffGridTimestep(
            f"timestep {float(value)!r} is not on the sealed PDD grid "
            f"{[float(t) for t in table]}; nearest-head selection is a different model"
        )
    return head_for_step(int(hits[0]))


def row_timestep_table(
    layout: pl.PackedLayout, timestep: float | torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(sorted distinct FP32 timesteps, int64 index of every packed row)`` for one step."""

    image_indices = layout.image_indices.cpu()
    rows = torch.full((layout.sequence_length,), float(timestep), dtype=torch.float32)
    rows[image_indices[: layout.num_reference_rows]] = REFERENCE_TIMESTEP
    unique, inverse = torch.unique(rows, sorted=True, return_inverse=True)
    return unique, inverse


def row_timestep_plan(
    layout: pl.PackedLayout, schedule: H3SigmaSchedule | None = None
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """The table of every step, in step order."""

    schedule = schedule if schedule is not None else released_schedule()
    return [row_timestep_table(layout, t) for t in schedule.timesteps.cpu()]


def adaln_indices(timestep_indices: torch.Tensor, token_tags: torch.Tensor) -> torch.Tensor:
    """Row of the AdaLN modulation table each packed row reads: ``index * 3 + tag``."""

    if timestep_indices.shape != token_tags.shape:
        raise ValueError("timestep_indices and token_tags must have one entry per packed row")
    return timestep_indices.long() * MODALITY_NUM + token_tags.long()


def artifact(schedule: H3SigmaSchedule | None = None) -> dict[str, Any]:
    """Sealed description: grid, step sigmas and timesteps as FP32 hex, head map, call count."""

    schedule = schedule if schedule is not None else released_schedule()
    body = {
        "contract_id": CONTRACT_ID,
        "config_sigmas": [_hex(s) for s in config_sigmas()],
        "step_sigmas": [_hex(float(s)) for s in schedule.sigmas.cpu()],
        "timesteps": [_hex(float(t)) for t in schedule.timesteps.cpu()],
        "reference_timestep": _hex(REFERENCE_TIMESTEP),
        "num_model_evals": schedule.num_model_evals,
        "head_for_step": [head_for_step(i) for i in range(schedule.num_model_evals)],
        "round_trip_moved": round_trip_moved(config_sigmas()),
        "shift_provided_sigmas": False,
        "adaln_index": "timestep_index * 3 + token_tag",
    }
    body["sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return body


def _hex(value: float) -> str:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0].hex()


def _bits(value: torch.Tensor) -> int:
    return int(value.detach().cpu().to(torch.float32).reshape(1).view(torch.int32)[0])
