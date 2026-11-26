# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Full six-step MiniMax-H3-Image trajectories on the strict transformer.

RFC #738 ``ws1_full_h3_image_t2i`` and ``ws1_full_h3_image_edit``: all 50 layers, the six PDD
heads and the six generated-row updates, for a T2I request and for an edit request whose
references have different geometries.

The trajectory itself is ``full_chain.run_chain`` with :class:`h3_image_step.StrictImageDenoiser`
as the denoiser. That is the ``phase_b_image_full_chain`` runner: text presentation, reference
encode and conditioning, packed layout, the ``image_sigma_pdd_schedule`` grid, the
``image_velocity_step`` updates, unpack and decode, sealed as an ``h3_image_evidence_schema``
artifact. Nothing in the chain changes; only the transformer is RL-Kernel's.

This module adds what the closeout rows check on top of that artifact:

* :func:`run_loop` runs the same six steps from the chain's conditioned start rows, for a batch of
  requests that share one layout, optionally in a physically permuted layout. It records each
  request's steps in the artifact's own step-record form, so :func:`first_record_drift` compares
  a loop row with a chain artifact field by field: repeat (I-C2), request alone versus batch
  position (I-C1, I-C5) and a fully consistent row permutation (I-C3), at every step.
* :func:`fixed_reference_rows` reads back that every edit reference row hashes to its conditioned
  value after every update (I-C6).
* :func:`schedule_records` is the head, sigma pair and per-row timestep table of each step; two
  arms on the same request must agree on it exactly.
* :func:`trajectory_distance` is the per-step distance between two arms that are not bitwise
  equal (strict against the Diffusers transformer), split into reference and target rows, and
  :func:`psnr` the distance of the decoded images.
* :class:`CountingDenoiser` counts the model calls (six, one per head, in order).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

from rl_engine.models.minimax_h3_image import fingerprint as fp
from rl_engine.models.minimax_h3_image import packed_layout as pl
from rl_engine.models.minimax_h3_image import pdd_schedule as ps
from rl_engine.models.minimax_h3_image import trajectory as tj
from rl_engine.models.minimax_h3_image import velocity_step as vs
from rl_engine.validation.models import h3_image_step as hs

CONTRACT_ID = "h3-image-ws1-full-v1"
SCHEDULE_KEYS = ("pdd_head", "sigma", "sigma_next", "timestep_table")
# Comparison order of one step: what the model is given, what it returns, what the update leaves.
RECORD_KEYS = (
    *SCHEDULE_KEYS,
    "velocity_sha256",
    "generated_velocity_sha256",
    "reference_rows_sha256",
    "latent_sha256",
)


class TrajectoryError(ValueError):
    """The start rows or the batch do not describe requests of one layout."""


@dataclass(frozen=True, eq=False)
class TrajectoryStart:
    """What enters step 0: image rows (conditioned references, then target noise) and text."""

    image_rows: torch.Tensor  # FP32 (B, N_image, 96)
    prompt_embeds: torch.Tensor  # (B, N_text, D)
    layout: pl.PackedLayout

    def __post_init__(self) -> None:
        rows, embeds = self.image_rows, self.prompt_embeds
        if rows.ndim != 3 or rows.dtype != torch.float32 or rows.shape[-1] != 96:
            raise TrajectoryError("image_rows must be FP32 (B, N, 96)")
        if embeds.ndim != 3 or len(embeds) != len(rows):
            raise TrajectoryError("prompt_embeds must be (B, T, D) with the image rows' batch")
        if rows.shape[1] != self.layout.image_indices.numel():
            raise TrajectoryError("one image row per image index is required")
        if embeds.shape[1] != self.layout.text_indices.numel():
            raise TrajectoryError("one prompt row per text index is required")

    @property
    def batch(self) -> int:
        return int(self.image_rows.shape[0])


def chain_layout(artifact: Mapping[str, Any], text_token_tags: torch.Tensor) -> pl.PackedLayout:
    """The packed layout a ``run_chain`` artifact was run on, rebuilt and checked against it."""

    inputs = artifact["trajectory"]["inputs"]
    grids = [(int(ref["latent"][0]), int(ref["latent"][1])) for ref in inputs["references"]]
    height, width = (int(v) for v in inputs["canvas"]["latent"])
    target = (height, width)
    layout = pl.build_layout(text_token_tags.cpu().long(), grids, target)
    built = {
        "sequence_length": layout.sequence_length,
        "image_indices_sha256": tj.tensor_digest(layout.image_indices.cpu()),
        "text_indices_sha256": tj.tensor_digest(layout.text_indices.cpu()),
        "num_reference_rows": layout.num_reference_rows,
        "position_sha256_fp64": tj.tensor_digest(layout.position_ids.to(torch.float64).cpu()),
    }
    wrong = sorted(k for k, v in built.items() if inputs["layout"][k] != v)
    tags = tj.tensor_digest(text_token_tags.cpu().long())
    if inputs["presentation"]["token_tags_sha256"] != tags:
        wrong.append("token_tags_sha256")
    if wrong:
        raise TrajectoryError(f"the rebuilt layout is not the chain's: {wrong}")
    return layout


def chain_start(run: Any, text: Any) -> TrajectoryStart:
    """What entered step 0 of a ``run_chain`` run kept with tensors (``text``: its conditions)."""

    rows = run.tensors.get("reference_conditioning.image_rows")
    if rows is None:
        raise TrajectoryError("run the chain with keep_tensors=True")
    layout = chain_layout(run.artifact, text.text_token_tags)
    return TrajectoryStart(rows[None].float(), text.prompt_embeds.cpu(), layout)


def stack(starts: Iterable[TrajectoryStart]) -> TrajectoryStart:
    """Batch requests that share one packed layout (each keeps its rows and text)."""

    items = list(starts)
    digest = pl.layout_digest(items[0].layout)
    if any(pl.layout_digest(s.layout) != digest for s in items[1:]):
        raise TrajectoryError("batched requests must share one packed layout")
    return TrajectoryStart(
        torch.cat([s.image_rows for s in items]),
        torch.cat([s.prompt_embeds.to(items[0].prompt_embeds) for s in items]),
        items[0].layout,
    )


# --- the loop ----------------------------------------------------------------------------------


@dataclass
class LoopRun:
    """Per request (batch row), the six step records; optionally the rows entering each step."""

    records: list[list[dict[str, Any]]]
    final_rows: torch.Tensor  # FP32 (B, N_image, 96), on the CPU
    calls: list[int] = field(default_factory=list)  # PDD step of each model call
    step_inputs: list[torch.Tensor] = field(default_factory=list)  # (B, N_image, 96) per step
    updates: list[list[dict[str, Any]]] = field(default_factory=list)  # sampler records


def _step_record(
    index: int,
    sigma: tuple[torch.Tensor, torch.Tensor],
    unique: torch.Tensor,
    indices: torch.Tensor,
    velocity: torch.Tensor,
    rows: torch.Tensor,
    r: int,
) -> dict[str, Any]:
    """The fields of ``trajectory.ImageStepRecord`` (no chain hash), for one request."""

    return {
        "index": index,
        "pdd_head": index,
        "sigma": tj.f32_hex(float(sigma[0])),
        "sigma_next": tj.f32_hex(float(sigma[1])),
        "timestep_table": {
            "values": [tj.f32_hex(float(v)) for v in unique.reshape(-1).tolist()],
            "indices_sha256": tj.tensor_digest(indices),
        },
        "velocity_sha256": tj.tensor_digest(velocity),
        "generated_velocity_sha256": tj.tensor_digest(velocity[:, r:]),
        "reference_rows_sha256": tj.tensor_digest(rows[:, :r]) if r else "",
        "latent_sha256": tj.tensor_digest(rows[:, r:]),
    }


def run_loop(
    transformer: hs.StrictImageTransformer,
    start: TrajectoryStart,
    *,
    sampler: Callable[[], vs.ImageVelocitySampler],
    order: torch.Tensor | None = None,
    keep_inputs: bool = False,
) -> LoopRun:
    """The six strict steps of every request in ``start``, batched in one transformer call.

    ``sampler()`` returns a fresh ``ImageVelocitySampler``; each request gets its own. With
    ``order`` the transformer runs on ``packed_layout.permute(layout, order)`` with the timestep
    indices moved alongside; image rows and velocity stay in selection order, and the records
    keep the canonical timestep table, so they compare with an unpermuted run.
    """

    layout = start.layout
    plan = ps.row_timestep_plan(layout)
    used = pl.permute(layout, order) if order is not None else layout
    device = transformer.device
    rows = start.image_rows.to(device).clone()
    embeds = start.prompt_embeds.to(device)
    r = layout.num_reference_rows
    conditioned = rows[:, :r].clone()
    samplers = [sampler() for _ in range(start.batch)]
    num_steps = samplers[0].schedule.num_model_evals
    out = LoopRun(records=[[] for _ in range(start.batch)], final_rows=rows)
    for i in range(num_steps):
        unique, indices = plan[i]
        physical = indices.cpu()[order] if order is not None else indices
        if keep_inputs:
            out.step_inputs.append(rows.cpu().clone())
        inputs = hs.StepInputs(rows, embeds, used, unique, physical, i)
        with torch.no_grad():
            velocity = transformer(inputs)
        out.calls.append(i)
        for b, s in enumerate(samplers):
            s.step(i, velocity[b], rows[b], r, pdd_head=i)
        if r and not vs._same_bits(rows[:, :r], conditioned):
            raise vs.GeneratedRowsOnly(f"step {i} changed a reference row")
        for b, s in enumerate(samplers):
            out.records[b].append(
                _step_record(
                    i,
                    s.sigma_pair(i, device),
                    unique,
                    indices,
                    velocity[b : b + 1],
                    rows[b : b + 1],
                    r,
                )
            )
    out.final_rows = rows.cpu()
    out.updates = [s.records for s in samplers]
    return out


# --- reading artifacts -------------------------------------------------------------------------


def chain_records(artifact: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The step records of a ``run_chain`` artifact, without the chain hash."""

    steps = artifact["trajectory"]["step_records"]
    return [{"index": s["index"], **{k: s[k] for k in RECORD_KEYS}} for s in steps]


def first_record_drift(
    lhs: Sequence[Mapping[str, Any]],
    rhs: Sequence[Mapping[str, Any]],
    keys: Sequence[str] = RECORD_KEYS,
) -> dict[str, Any] | None:
    """First step and field where two step-record lists differ, or ``None``."""

    if len(lhs) != len(rhs):
        return {"step": min(len(lhs), len(rhs)), "field": "steps", "lengths": [len(lhs), len(rhs)]}
    for a, b in zip(lhs, rhs):
        for key in keys:
            if a[key] != b[key]:
                rows = {"reference_rows_sha256": "reference", "velocity_sha256": "all_image"}
                return {
                    "step": int(a["index"]),
                    "pdd_head": a["pdd_head"],
                    "field": key,
                    "rows": "schedule" if key in SCHEDULE_KEYS else rows.get(key, "generated"),
                }
    return None


def schedule_records(artifact: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Head, sigma pair and per-row timestep table of each step."""

    return [{k: s[k] for k in SCHEDULE_KEYS} for s in artifact["trajectory"]["step_records"]]


def fixed_reference_rows(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Whether every reference row hashes to its conditioned value after every update (I-C6)."""

    traj = artifact["trajectory"]
    layout = traj["inputs"]["layout"]
    if not layout["num_reference_rows"]:
        return {"num_reference_rows": 0, "fixed": None}
    conditioned = layout["conditioned_reference_rows_sha256"]
    per_step = [s["reference_rows_sha256"] for s in traj["step_records"]]
    return {
        "num_reference_rows": layout["num_reference_rows"],
        "conditioned_sha256": conditioned,
        "steps_equal_to_conditioned": [d == conditioned for d in per_step],
        "fixed": all(d == conditioned for d in per_step) and len(per_step) == fp.NUM_PDD_HEADS,
    }


# --- distance between arms ---------------------------------------------------------------------


def _relative(error: torch.Tensor, base: torch.Tensor) -> float:
    norm = float(base.norm())
    return float(error.norm()) / norm if norm else float(error.norm())


def trajectory_distance(
    lhs: Mapping[str, torch.Tensor],
    rhs: Mapping[str, torch.Tensor],
    num_reference_rows: int,
    steps: int = fp.NUM_PDD_HEADS,
) -> list[dict[str, Any]]:
    """Per step: velocity distance on reference and target rows, and the latent distance.

    ``lhs`` / ``rhs`` are ``ChainRun.tensors`` of two runs kept with tensors (keys
    ``trajectory.<step>.velocity`` and ``trajectory.<step>.latent``); ``rhs`` is the base.
    """

    r = int(num_reference_rows)
    out = []
    for i in range(steps):
        va = lhs[f"trajectory.{i}.velocity"].double()
        vb = rhs[f"trajectory.{i}.velocity"].double()
        la = lhs[f"trajectory.{i}.latent"].double()
        lb = rhs[f"trajectory.{i}.latent"].double()
        row: dict[str, Any] = {
            "step": i,
            "velocity_target": _relative(va[:, r:] - vb[:, r:], vb[:, r:]),
            "latent": _relative(la - lb, lb),
            "latent_max_abs": float((la - lb).abs().max()),
            "velocity_bitwise": bool(torch.equal(va, vb)),
            "latent_bitwise": bool(torch.equal(la, lb)),
        }
        if r:
            row["velocity_reference"] = _relative(va[:, :r] - vb[:, :r], vb[:, :r])
        out.append(row)
    return out


def psnr(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    """PSNR in dB of two uint8 images of one shape (``inf`` when equal)."""

    if lhs.dtype != torch.uint8 or rhs.dtype != torch.uint8 or lhs.shape != rhs.shape:
        raise ValueError("psnr compares two uint8 images of one shape")
    mse = float((lhs.double() - rhs.double()).pow(2).mean())
    return math.inf if mse == 0 else 10 * math.log10(255.0**2 / mse)


# --- denoiser readback -------------------------------------------------------------------------


class CountingDenoiser:
    """Any ``full_chain.ImageDenoiser``, with the PDD step of every call kept."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.calls: list[int] = []

    @property
    def backend_id(self) -> str:
        return str(getattr(self.inner, "backend_id", type(self.inner).__name__))

    def __call__(self, image_rows, prompt_embeds, layout, unique_timesteps, timestep_indices, step):
        self.calls.append(int(step))
        return self.inner(
            image_rows, prompt_embeds, layout, unique_timesteps, timestep_indices, step
        )

    def readback(self) -> dict[str, Any]:
        expected = list(range(fp.NUM_PDD_HEADS))
        return {"calls": list(self.calls), "six_in_order": self.calls == expected}


__all__ = [
    "CONTRACT_ID",
    "RECORD_KEYS",
    "SCHEDULE_KEYS",
    "CountingDenoiser",
    "LoopRun",
    "TrajectoryError",
    "TrajectoryStart",
    "chain_layout",
    "chain_records",
    "chain_start",
    "first_record_drift",
    "fixed_reference_rows",
    "psnr",
    "run_loop",
    "schedule_records",
    "stack",
    "trajectory_distance",
]
