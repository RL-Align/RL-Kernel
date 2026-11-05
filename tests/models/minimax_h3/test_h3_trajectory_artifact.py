# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""``h3_trajectory_artifact`` (RFC #420): sealed trajectories and where two of them drift.

CPU only. Synthetic Euler-like trajectories stand in for a model: what is under test
is the record (digests, hash chain, seal), its fail-closed validation and the drift locator.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from rl_engine.models.minimax_h3 import trajectory as tj

STEPS = 6
SHAPES = {"video": (1, 4, 3, 8), "audio": (1, 5, 2)}


def _grid(steps: int, shift: float) -> list[float]:
    sigmas = [1.0 - i / steps for i in range(steps + 1)]
    return [shift * s / (1 + (shift - 1) * s) for s in sigmas]


def _noise(seed: int) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return {m: torch.randn(shape, generator=gen).bfloat16() for m, shape in SHAPES.items()}


def _run(
    label: str = "strict",
    *,
    seed: int = 7,
    perturb: dict | None = None,
    keep_tensors: bool = False,
    operator_trace: str = "",
) -> tj.TrajectoryRecorder:
    """A deterministic fake trajectory; ``perturb`` = {(step, modality): delta} on velocity."""

    noise = _noise(seed)
    recorder = tj.TrajectoryRecorder(
        variant="FL2VA",
        label=label,
        seed=seed,
        sigma_grids={"video": _grid(STEPS, 12.0), "audio": _grid(STEPS, 3.0)},
        initial_noise=noise,
        shifts={"video": 12.0, "audio": 3.0},
        operator_trace_sha256=operator_trace,
        metadata={"note": "synthetic"},
        keep_tensors=keep_tensors,
    )
    latent = {m: t.clone() for m, t in noise.items()}
    gen = torch.Generator().manual_seed(1234)
    for step in range(STEPS):
        velocity, new_latent = {}, {}
        for m in recorder.modalities:
            v = torch.randn(SHAPES[m], generator=gen).bfloat16()
            delta = (perturb or {}).get((step, m))
            if delta is not None:
                v = v.clone()
                v.view(-1)[0] += delta
            velocity[m] = v
            grid = _grid(STEPS, 12.0 if m == "video" else 3.0)
            sigma, sigma_next = grid[step], grid[step + 1]
            r = sigma_next / sigma
            x0 = latent[m].float() + sigma * v.float()  # data-ward velocity
            new_latent[m] = (r * latent[m].float() + (1 - r) * x0).bfloat16()
        latent = new_latent
        recorder.record_step(step, velocity=velocity, latent=latent)
    return recorder


def test_artifact_is_deterministic_and_verifies() -> None:
    a, b = _run().seal(), _run().seal()
    assert a == b
    tj.verify(a)
    assert a["steps"] == STEPS and a["modalities"] == ["video", "audio"]
    assert a["final_latent_sha256"] == a["step_records"][-1]["latent_sha256"]


def test_label_or_trace_changes_the_seal_but_not_the_trajectory() -> None:
    a = _run("native").seal()
    b = _run("strict", operator_trace="ab" * 32).seal()
    assert a["artifact_sha256"] != b["artifact_sha256"]
    assert a["step_records"] == b["step_records"]


def test_json_roundtrip() -> None:
    artifact = _run().seal()
    assert tj.loads(tj.dumps(artifact)) == artifact


def test_sigma_grids_are_exact_fp32() -> None:
    artifact = _run().seal()
    for modality in tj.modalities_of(artifact):
        decoded = tj.sigmas_of(artifact, modality)
        assert decoded == [tj.f32_from_hex(tj.f32_hex(s)) for s in decoded]
        assert all(torch.tensor(s, dtype=torch.float32).item() == s for s in decoded)
    assert tj.f32_hex(0.1) != (0.1).hex()  # FP32 rounding is applied, not FP64 text


def test_tensor_digest_covers_bytes_dtype_and_shape() -> None:
    x = torch.arange(12, dtype=torch.float32)
    assert tj.tensor_digest(x) == tj.tensor_digest(x.clone())
    assert tj.tensor_digest(x) != tj.tensor_digest(x.reshape(3, 4))
    assert tj.tensor_digest(x) != tj.tensor_digest(x.to(torch.float64))
    y = x.clone()
    y[5] = torch.nextafter(y[5], torch.tensor(1e9))
    assert tj.tensor_digest(x) != tj.tensor_digest(y)
    assert tj.tensor_digest(x.reshape(3, 4).t()) == tj.tensor_digest(
        x.reshape(3, 4).t().contiguous()
    )
    assert tj.tensor_digest(torch.empty(0)) != tj.tensor_digest(
        torch.empty(0, dtype=torch.bfloat16)
    )


# --- recorder contract ------------------------------------------------------------------------


def test_recorder_enforces_step_order_and_completeness() -> None:
    recorder = _run()
    with pytest.raises(ValueError, match="only"):
        recorder.record_step(STEPS, velocity=_noise(0), latent=_noise(0))
    partial = tj.TrajectoryRecorder(
        variant="FL2VA",
        label="x",
        seed=0,
        sigma_grids={"video": _grid(2, 12.0)},
        initial_noise={"video": _noise(0)["video"]},
    )
    with pytest.raises(ValueError, match="expected step 0"):
        partial.record_step(
            1, velocity={"video": _noise(0)["video"]}, latent={"video": _noise(0)["video"]}
        )
    with pytest.raises(ValueError, match="recorded 0 of 2"):
        partial.seal()
    with pytest.raises(ValueError, match="exactly"):
        partial.record_step(0, velocity=_noise(0), latent={"video": _noise(0)["video"]})


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(variant="Nope"), "unknown variant"),
        (dict(sigma_grids={"video": [1.0]}), "length"),
        (
            dict(
                sigma_grids={"video": [1.0, 0.5], "audio": [1.0, 0.7, 0.0]},
                noise=("video", "audio"),
            ),
            "length",
        ),
        (dict(noise=("video", "audio")), "modalities"),
    ],
)
def test_recorder_rejects_bad_construction(kwargs, match) -> None:
    noise = _noise(0)
    args = dict(
        variant="FL2VA",
        label="x",
        seed=0,
        sigma_grids={"video": _grid(2, 12.0)},
        initial_noise={"video": noise["video"]},
    )
    names = kwargs.pop("noise", None)
    if names:
        args["initial_noise"] = {m: noise[m] for m in names}
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        tj.TrajectoryRecorder(**args)


# --- fail closed ---------------------------------------------------------------------------------


def _mutations():
    def edit_digest(a):
        a["step_records"][2]["velocity_sha256"]["video"] = "0" * 64

    def drop_step(a):
        a["step_records"].pop()

    def swap_steps(a):
        a["step_records"][1], a["step_records"][2] = a["step_records"][2], a["step_records"][1]

    def edit_grid(a):
        a["sigma_grids"]["video"][3] = tj.f32_hex(0.123)

    def edit_seed(a):
        a["seed"] = 8

    def edit_noise(a):
        a["initial_noise"]["audio"]["sha256"] = "1" * 64

    def edit_label(a):
        a["label"] = "native"

    return [edit_digest, drop_step, swap_steps, edit_grid, edit_seed, edit_noise, edit_label]


@pytest.mark.parametrize("mutate", _mutations(), ids=lambda f: f.__name__)
def test_any_edit_fails_verification(mutate) -> None:
    artifact = copy.deepcopy(_run().seal())
    mutate(artifact)
    with pytest.raises(ValueError):
        tj.verify(artifact)


def test_resealed_edit_is_still_caught_by_the_chain_or_grid() -> None:
    artifact = copy.deepcopy(_run().seal())
    artifact["step_records"][2]["velocity_sha256"]["video"] = "0" * 64
    artifact["artifact_sha256"] = tj._sha(
        {k: v for k, v in artifact.items() if k != "artifact_sha256"}
    )
    with pytest.raises(ValueError, match="hash chain"):
        tj.verify(artifact)


def test_non_monotone_grid_and_wrong_fingerprint_are_rejected() -> None:
    artifact = copy.deepcopy(_run().seal())
    artifact["architecture_fingerprint_sha256"] = "f" * 64
    artifact["artifact_sha256"] = tj._sha(
        {k: v for k, v in artifact.items() if k != "artifact_sha256"}
    )
    with pytest.raises(ValueError, match="fingerprint"):
        tj.verify(artifact)
    tj.verify(artifact, check_fingerprint=False)
    recorder = tj.TrajectoryRecorder(
        variant="FL2VA",
        label="x",
        seed=0,
        sigma_grids={"video": [1.0, 0.2, 0.5]},
        initial_noise={"video": _noise(0)["video"]},
    )
    for step in range(2):
        recorder.record_step(
            step, velocity={"video": _noise(0)["video"]}, latent={"video": _noise(0)["video"]}
        )
    with pytest.raises(ValueError, match="non-increasing"):
        tj.verify(recorder.seal())


def test_unsupported_schema() -> None:
    artifact = _run().seal()
    artifact["schema_version"] = "v0"
    with pytest.raises(ValueError, match="schema"):
        tj.verify(artifact)


# --- drift --------------------------------------------------------------------------------------


def test_identical_trajectories_have_no_drift() -> None:
    a, b = _run("strict").seal(), _run("strict").seal()
    assert tj.first_drift(a, b) is None
    report = tj.drift_report(a, b)
    assert report["bitwise_equal"] and report["differing"] == 0
    assert report["compared"] == STEPS * 2 * 2 and report["final_latent_equal"]


def test_first_drift_names_step_kind_and_modality() -> None:
    a = _run("strict").seal()
    b = _run("native", perturb={(2, "audio"): 0.25}).seal()
    drift = tj.first_drift(a, b)
    assert drift == {"step": 2, "kind": "velocity", "modality": "audio"}
    report = tj.drift_report(a, b)
    assert not report["bitwise_equal"] and not report["final_latent_equal"]
    # The perturbed velocity changes its own latent, which every later latent inherits;
    # the other modality and the other velocities are untouched.
    assert [(e["step"], e["kind"], e["modality"]) for e in report["entries"]] == [
        (2, "velocity", "audio"),
        (2, "latent", "audio"),
        (3, "latent", "audio"),
        (4, "latent", "audio"),
        (5, "latent", "audio"),
    ]
    assert report["first_drift"] == drift


def test_velocity_is_reported_before_the_latent_of_the_same_step() -> None:
    a = _run().seal()
    b = _run(perturb={(0, "video"): 1.0}).seal()
    assert tj.first_drift(a, b)["kind"] == "velocity"


def test_drift_with_tensors_reports_count_and_magnitude(tmp_path: Path) -> None:
    ra, rb = _run(keep_tensors=True), _run("native", perturb={(3, "video"): 0.5}, keep_tensors=True)
    ra.save_tensors(tmp_path / "a.safetensors")
    rb.save_tensors(tmp_path / "b.safetensors")
    drift = tj.first_drift(
        ra.seal(),
        rb.seal(),
        tensors_lhs=tmp_path / "a.safetensors",
        tensors_rhs=tmp_path / "b.safetensors",
    )
    assert (drift["step"], drift["kind"], drift["modality"]) == (3, "velocity", "video")
    assert drift["mismatch_count"] == 1
    assert drift["numel"] == 1 * 4 * 3 * 8
    assert drift["max_abs_diff"] > 0.0


def test_different_inputs_are_not_comparable() -> None:
    a, b = _run(seed=7).seal(), _run(seed=8).seal()
    with pytest.raises(ValueError, match="not comparable"):
        tj.first_drift(a, b)
    assert tj.first_drift(a, b, require_same_inputs=False) is not None


def test_save_tensors_needs_keep_tensors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="keep_tensors"):
        _run().save_tensors(tmp_path / "x.safetensors")


def test_artifact_json_is_plain_text() -> None:
    text = tj.dumps(_run().seal())
    parsed = json.loads(text)
    assert parsed["schema_version"] == tj.SCHEMA_VERSION
    assert all(isinstance(v, str) for v in parsed["sigma_grids"]["video"])
