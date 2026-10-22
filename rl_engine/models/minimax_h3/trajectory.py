# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Sealed trajectory artifact for MiniMax-H3 (RFC #420 ``h3_trajectory_artifact``).

A sampling (or SDE rollout) trajectory is the unit that end-to-end claims are made about:
"native and strict differ from step 3", "strict is bitwise identical across runs". This
module defines the record that carries such a claim and the comparison that locates it.

An artifact holds, per modality (video, audio):

* the initial noise digest and the exact FP32 sigma grid;

and per step: the sigmas used, and the digest of each modality's velocity and of the
latent after the step. Steps are hash-chained (``chain_sha256``) and the whole record is
sealed (``artifact_sha256``), so an edited, truncated or reordered artifact fails to load.
It is bound to the architecture fingerprint (``h3_arch_fingerprint``) and optionally to an
operator trace (``h3_operator_trace``), so "what ran" and "what it produced" travel together.

Digests cover dtype, shape and raw bytes: one flipped bit anywhere is a different digest,
and a reshape or dtype change is too. Tensors can be kept next to the artifact
(``TrajectoryRecorder(keep_tensors=True)`` + ``save_tensors``) so ``first_drift`` can also
report how far apart two diverging tensors are (mismatch count, max abs difference).

The core is stdlib; torch is imported only to hash or load tensors.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from rl_engine.models.minimax_h3 import fingerprint as fp

SCHEMA_VERSION = "rlkernel.h3.trajectory.v1"
MODALITIES = ("video", "audio")
# Order in which a step's tensors are compared: a velocity diverges before the latent it feeds.
KINDS = ("velocity", "latent")


# --- hashing ------------------------------------------------------------------------------


def tensor_digest(tensor: Any) -> str:
    """sha256 over dtype, shape and raw bytes of a tensor (any device, any stride)."""

    import torch

    cpu = tensor.detach().contiguous().cpu()
    header = f"{cpu.dtype}|{tuple(cpu.shape)}|".encode()
    raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes() if cpu.numel() else b""
    return hashlib.sha256(header + raw).hexdigest()


def f32_hex(value: float) -> str:
    """Exact text form of a FP32 value (round to FP32, then ``float.hex``)."""

    return struct.unpack("<f", struct.pack("<f", float(value)))[0].hex()


def f32_from_hex(text: str) -> float:
    return float.fromhex(text)


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


# --- recording ----------------------------------------------------------------------------


@dataclass(frozen=True)
class StepRecord:
    index: int
    sigma: Mapping[str, str]  # modality -> FP32 hex, the sigma this step starts from
    sigma_next: Mapping[str, str]
    velocity_sha256: Mapping[str, str]
    latent_sha256: Mapping[str, str]  # after the step
    chain_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "sigma": dict(self.sigma),
            "sigma_next": dict(self.sigma_next),
            "velocity_sha256": dict(self.velocity_sha256),
            "latent_sha256": dict(self.latent_sha256),
            "chain_sha256": self.chain_sha256,
        }


def _chain(previous: str, body: Mapping[str, Any]) -> str:
    return _sha({"previous": previous, "step": body})


GENESIS = "0" * 64


class TrajectoryRecorder:
    """Builds an artifact step by step; ``seal()`` returns the immutable dict."""

    def __init__(
        self,
        *,
        variant: str,
        label: str,
        seed: int,
        sigma_grids: Mapping[str, Sequence[float]],
        initial_noise: Mapping[str, Any],
        shifts: Mapping[str, float] | None = None,
        operator_trace_sha256: str = "",
        metadata: Mapping[str, Any] | None = None,
        keep_tensors: bool = False,
    ) -> None:
        if variant not in fp.VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if set(sigma_grids) != set(initial_noise) or not set(sigma_grids) <= set(MODALITIES):
            raise ValueError(
                f"modalities must be a subset of {MODALITIES}, the same for grids and noise"
            )
        steps = {len(grid) - 1 for grid in sigma_grids.values()}
        if len(steps) != 1 or steps.pop() < 1:
            raise ValueError("every sigma grid needs the same length of at least 2 (steps + 1)")
        self.variant, self.label, self.seed = variant, label, int(seed)
        self.modalities = tuple(m for m in MODALITIES if m in sigma_grids)
        self.grids = {m: [f32_hex(s) for s in sigma_grids[m]] for m in self.modalities}
        self.steps = len(self.grids[self.modalities[0]]) - 1
        self.shifts = {m: float((shifts or {}).get(m, 0.0)) for m in self.modalities}
        self.operator_trace_sha256 = operator_trace_sha256
        self.metadata = dict(metadata or {})
        self.initial = {
            m: {
                "sha256": tensor_digest(initial_noise[m]),
                "shape": list(initial_noise[m].shape),
                "dtype": str(initial_noise[m].dtype),
            }
            for m in self.modalities
        }
        self.records: list[StepRecord] = []
        self.keep_tensors = keep_tensors
        self.tensors: dict[str, Any] = {}
        if keep_tensors:
            for m in self.modalities:
                self.tensors[tensor_key(-1, "latent", m)] = initial_noise[m].detach().cpu().clone()

    def record_step(
        self,
        index: int,
        *,
        velocity: Mapping[str, Any],
        latent: Mapping[str, Any],
    ) -> StepRecord:
        """Record step ``index`` (consecutive from 0): its velocities and resulting latents."""

        if index != len(self.records):
            raise ValueError(f"expected step {len(self.records)}, got {index}")
        if index >= self.steps:
            raise ValueError(f"trajectory has only {self.steps} steps")
        for name, group in (("velocity", velocity), ("latent", latent)):
            if set(group) != set(self.modalities):
                raise ValueError(f"{name} must cover exactly {self.modalities}")
        body = {
            "index": index,
            "sigma": {m: self.grids[m][index] for m in self.modalities},
            "sigma_next": {m: self.grids[m][index + 1] for m in self.modalities},
            "velocity_sha256": {m: tensor_digest(velocity[m]) for m in self.modalities},
            "latent_sha256": {m: tensor_digest(latent[m]) for m in self.modalities},
        }
        previous = self.records[-1].chain_sha256 if self.records else GENESIS
        record = StepRecord(chain_sha256=_chain(previous, body), **body)
        self.records.append(record)
        if self.keep_tensors:
            for kind, group in (("velocity", velocity), ("latent", latent)):
                for m in self.modalities:
                    self.tensors[tensor_key(index, kind, m)] = group[m].detach().cpu().clone()
        return record

    def seal(self) -> dict[str, Any]:
        if len(self.records) != self.steps:
            raise ValueError(f"recorded {len(self.records)} of {self.steps} steps")
        artifact = {
            "schema_version": SCHEMA_VERSION,
            "architecture_fingerprint_sha256": fp.build_fingerprint()["fingerprint_sha256"],
            "variant": self.variant,
            "label": self.label,
            "seed": self.seed,
            "steps": self.steps,
            "modalities": list(self.modalities),
            "shifts": self.shifts,
            "sigma_grids": self.grids,
            "initial_noise": self.initial,
            "operator_trace_sha256": self.operator_trace_sha256,
            "metadata": self.metadata,
            "step_records": [record.to_dict() for record in self.records],
            "final_latent_sha256": dict(self.records[-1].latent_sha256),
        }
        artifact["artifact_sha256"] = _sha(artifact)
        return artifact

    def save_tensors(self, path: Path) -> None:
        """Write kept tensors as safetensors (keys ``step.kind.modality``; -1 = initial noise)."""

        if not self.keep_tensors:
            raise ValueError("recorder was created with keep_tensors=False")
        from safetensors.torch import save_file

        save_file({key: value.contiguous() for key, value in self.tensors.items()}, str(path))


def tensor_key(step: int, kind: str, modality: str) -> str:
    return f"{step}.{kind}.{modality}"


# --- validation ---------------------------------------------------------------------------


def verify(artifact: Mapping[str, Any], *, check_fingerprint: bool = True) -> None:
    """Fail closed unless the artifact is internally consistent and sealed."""

    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported trajectory schema {artifact.get('schema_version')!r}")
    body = {k: v for k, v in artifact.items() if k != "artifact_sha256"}
    if artifact.get("artifact_sha256") != _sha(body):
        raise ValueError("artifact_sha256 does not match the content")
    modalities = tuple(artifact["modalities"])
    if not modalities or not set(modalities) <= set(MODALITIES):
        raise ValueError("invalid modalities")
    steps = artifact["steps"]
    if len(artifact["step_records"]) != steps:
        raise ValueError("step_records length differs from steps")
    for modality in modalities:
        grid = [f32_from_hex(s) for s in artifact["sigma_grids"][modality]]
        if len(grid) != steps + 1:
            raise ValueError(f"{modality}: sigma grid needs steps + 1 entries")
        if any(b > a for a, b in zip(grid, grid[1:])):
            raise ValueError(f"{modality}: sigma grid is not non-increasing")
        if grid[0] <= 0.0 or grid[-1] < 0.0 or grid[0] > 1.0:
            raise ValueError(f"{modality}: sigma grid must start in (0, 1] and end >= 0")
    previous = GENESIS
    for index, record in enumerate(artifact["step_records"]):
        body = {k: v for k, v in record.items() if k != "chain_sha256"}
        if record["index"] != index:
            raise ValueError(f"step {index}: out-of-order record {record['index']}")
        for modality in modalities:
            if record["sigma"][modality] != artifact["sigma_grids"][modality][index]:
                raise ValueError(f"step {index}: sigma disagrees with the grid")
            if record["sigma_next"][modality] != artifact["sigma_grids"][modality][index + 1]:
                raise ValueError(f"step {index}: sigma_next disagrees with the grid")
        if record["chain_sha256"] != _chain(previous, body):
            raise ValueError(f"step {index}: hash chain broken")
        previous = record["chain_sha256"]
    if artifact["final_latent_sha256"] != artifact["step_records"][-1]["latent_sha256"]:
        raise ValueError("final_latent_sha256 differs from the last step")
    if check_fingerprint and (
        artifact["architecture_fingerprint_sha256"] != fp.build_fingerprint()["fingerprint_sha256"]
    ):
        raise ValueError("artifact was recorded against a different H3 architecture fingerprint")


def dumps(artifact: Mapping[str, Any]) -> str:
    verify(artifact)
    return json.dumps(artifact, indent=1, sort_keys=True) + "\n"


def loads(text: str, *, check_fingerprint: bool = True) -> dict[str, Any]:
    artifact = json.loads(text)
    verify(artifact, check_fingerprint=check_fingerprint)
    return artifact


# --- comparison ---------------------------------------------------------------------------


def input_mismatches(lhs: Mapping[str, Any], rhs: Mapping[str, Any]) -> list[str]:
    """Differences in what the two runs were *given*: a drift comparison needs none."""

    problems: list[str] = []
    for key in ("variant", "seed", "steps", "modalities", "shifts", "sigma_grids"):
        if lhs[key] != rhs[key]:
            problems.append(f"{key} differs")
    for modality in lhs["modalities"]:
        if lhs["initial_noise"].get(modality) != rhs["initial_noise"].get(modality):
            problems.append(f"initial noise differs for {modality}")
    return problems


def first_drift(
    lhs: Mapping[str, Any],
    rhs: Mapping[str, Any],
    *,
    tensors_lhs: Path | None = None,
    tensors_rhs: Path | None = None,
    require_same_inputs: bool = True,
) -> dict[str, Any] | None:
    """First (step, kind, modality) at which two trajectories differ, or ``None`` if bitwise equal.

    With both tensor files the result also carries ``mismatch_count``, ``numel`` and
    ``max_abs_diff``. Differing inputs (seed, noise, grids) make a drift meaningless, so
    they raise unless ``require_same_inputs=False``.
    """

    verify(lhs)
    verify(rhs)
    if require_same_inputs:
        problems = input_mismatches(lhs, rhs)
        if problems:
            raise ValueError("trajectories are not comparable: " + "; ".join(problems))
    for a, b in zip(lhs["step_records"], rhs["step_records"]):
        for kind in KINDS:
            for modality in lhs["modalities"]:
                if a[f"{kind}_sha256"][modality] != b[f"{kind}_sha256"][modality]:
                    drift = {"step": a["index"], "kind": kind, "modality": modality}
                    if tensors_lhs is not None and tensors_rhs is not None:
                        drift.update(
                            _tensor_difference(
                                tensors_lhs, tensors_rhs, tensor_key(a["index"], kind, modality)
                            )
                        )
                    return drift
    return None


def _tensor_difference(path_a: Path, path_b: Path, key: str) -> dict[str, Any]:
    import torch
    from safetensors import safe_open

    with safe_open(str(path_a), framework="pt") as fa, safe_open(str(path_b), framework="pt") as fb:
        a, b = fa.get_tensor(key), fb.get_tensor(key)
    if a.shape != b.shape or a.dtype != b.dtype:
        return {"shape_or_dtype_differs": True}
    mismatch = a != b
    nan_mismatch = torch.isnan(a) != torch.isnan(b)
    differing = mismatch & ~(torch.isnan(a) & torch.isnan(b)) | nan_mismatch
    diff = (a.double() - b.double()).abs()
    diff = torch.where(torch.isfinite(diff), diff, torch.zeros_like(diff))
    return {
        "mismatch_count": int(differing.sum()),
        "numel": int(a.numel()),
        "max_abs_diff": float(diff.max()) if diff.numel() else 0.0,
    }


def drift_report(
    lhs: Mapping[str, Any],
    rhs: Mapping[str, Any],
    *,
    tensors_lhs: Path | None = None,
    tensors_rhs: Path | None = None,
) -> dict[str, Any]:
    """Every (step, kind, modality) that differs, in order, plus the headline numbers.

    ``bitwise_equal`` is the strict claim; ``differing`` is how many of the compared
    tensors differ; with tensor files each entry carries mismatch count and max abs diff.
    """

    verify(lhs)
    verify(rhs)
    problems = input_mismatches(lhs, rhs)
    if problems:
        raise ValueError("trajectories are not comparable: " + "; ".join(problems))
    entries: list[dict[str, Any]] = []
    total = 0
    for a, b in zip(lhs["step_records"], rhs["step_records"]):
        for kind in KINDS:
            for modality in lhs["modalities"]:
                total += 1
                if a[f"{kind}_sha256"][modality] != b[f"{kind}_sha256"][modality]:
                    entry = {"step": a["index"], "kind": kind, "modality": modality}
                    if tensors_lhs is not None and tensors_rhs is not None:
                        entry.update(
                            _tensor_difference(
                                tensors_lhs, tensors_rhs, tensor_key(a["index"], kind, modality)
                            )
                        )
                    entries.append(entry)
    return {
        "labels": [lhs["label"], rhs["label"]],
        "steps": lhs["steps"],
        "compared": total,
        "differing": len(entries),
        "bitwise_equal": not entries,
        "first_drift": entries[0] if entries else None,
        "final_latent_equal": lhs["final_latent_sha256"] == rhs["final_latent_sha256"],
        "entries": entries,
    }


def sigmas_of(artifact: Mapping[str, Any], modality: str) -> list[float]:
    return [f32_from_hex(text) for text in artifact["sigma_grids"][modality]]


def modalities_of(artifact: Mapping[str, Any]) -> Iterable[str]:
    return tuple(artifact["modalities"])
