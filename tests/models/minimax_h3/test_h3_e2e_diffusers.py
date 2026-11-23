# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CPU tests of ``tools/validation/models/h3_e2e_diffusers.py`` (capture, seal, compare)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from rl_engine.models.minimax_h3 import trajectory as tj
from tools.validation.models import h3_e2e_diffusers as e2e

scheduling = pytest.importorskip("diffusers.schedulers.scheduling_minimax_h3")


def _sample(capture, *, seed=0, drift_at=None, steps=10):
    video, audio = scheduling.MiniMaxH3Scheduler(12.0), scheduling.MiniMaxH3Scheduler(3.0)
    gen = torch.Generator().manual_seed(seed)
    xv, xa = torch.randn(6, 8, generator=gen), torch.randn(5, 4, generator=gen)
    with capture.scheduler(video, "video"), capture.scheduler(audio, "audio"):
        video.set_timesteps(steps, device="cpu")
        audio.set_timesteps(steps, device="cpu")
        for i, t in enumerate(video.timesteps):
            vv = torch.randn(6, 8, generator=gen)
            va = torch.randn(5, 4, generator=gen)
            if drift_at == i:
                vv = vv.clone()
                vv[0, 0] = vv[0, 0] + 1e-3
            xv = video.step(vv, t, xv, return_dict=False)[0]
            xa = audio.step(va, audio.timesteps[i], xa, return_dict=False)[0]
    assert "step" not in video.__dict__ and "step" not in audio.__dict__
    return video, audio


def _write_run(root, arm, case, repeat, *, drift_at=None, video_value=0.5):
    capture = e2e.StepCapture()
    video, audio = _sample(capture, seed=case, drift_at=drift_at)
    out = root / f"case{case}" / f"r{repeat}"
    out.mkdir(parents=True)
    artifact = capture.seal(
        label=f"{arm}-case{case}-r{repeat}",
        seed=42,
        grids={"video": video.sigmas.tolist(), "audio": audio.sigmas.tolist()},
        shifts={"video": 12.0, "audio": 3.0},
        trace_sha256="",
        metadata={"case": case},
        keep_tensors=True,
        out_dir=out,
    )
    v = np.full((2, 4, 4, 3), video_value, dtype=np.float32)
    a = np.zeros((2, 16), dtype=np.float32)
    np.save(out / "video.npy", v.astype(np.float16))
    np.save(out / "audio.npy", a)
    timer = e2e.PhaseTimer()
    timer.calls["transformer"] = [3.0, 1.0, 1.0]
    manifest = {
        "label": artifact["label"],
        "finite": capture.finite(),
        "seconds": 10.0,
        "timing": timer.summary(),
        "peak_allocated_bytes": 2**30,
        "scheduler_rows": {},
        "strict_boundaries": arm == "candidate",
        "provenance": {"checkpoint_hash_checked": True},
        "video_float32_sha256": e2e.media_digest(v),
        "audio_float32_sha256": e2e.media_digest(a),
    }
    (out / "manifest.json").write_text(json.dumps(manifest))
    return artifact


def test_the_capture_seals_a_verifiable_trajectory(tmp_path):
    artifact = _write_run(tmp_path, "candidate", 0, 0)
    tj.verify(artifact)
    assert artifact["steps"] == 9 and artifact["modalities"] == ["video", "audio"]
    loaded = tj.loads((tmp_path / "case0" / "r0" / "trajectory.json").read_text())
    assert loaded["artifact_sha256"] == artifact["artifact_sha256"]


def test_compare_passes_equal_repeats_and_reports_provider_drift(tmp_path):
    p, r = tmp_path / "p", tmp_path / "r"
    for repeat in (0, 1):
        _write_run(p, "provider", 0, repeat, drift_at=3)
        _write_run(r, "candidate", 0, repeat)
    out = e2e.compare_case(p, r, 0)
    assert out["candidate_repeats_bitwise"] and out["candidate_media_repeat_equal"]
    assert out["provider_repeats_bitwise"]
    pr = out["provider_vs_candidate"]
    assert not pr["bitwise_equal"]
    assert pr["first_drift"]["step"] == 3 and pr["first_drift"]["kind"] == "velocity"
    assert pr["first_drift"]["modality"] == "video" and "max_abs_diff" in pr["first_drift"]
    summary = e2e.summarize([out])
    assert summary["gate_candidate_repeats_bitwise"] and summary["gate_every_case_has_a_repeat"]
    assert summary["final_evidence"]
    assert summary["time_mean_first_run"]["candidate"]["transformer_steady_per_call"] == 1.0


def test_compare_fails_a_candidate_repeat_that_drifts(tmp_path):
    p, r = tmp_path / "p", tmp_path / "r"
    _write_run(p, "provider", 0, 0)
    _write_run(r, "candidate", 0, 0)
    _write_run(r, "candidate", 0, 1, drift_at=5)
    out = e2e.compare_case(p, r, 0)
    assert not out["candidate_repeats_bitwise"]
    assert out["candidate_repeats"][0]["first_drift"]["step"] == 5
    assert not e2e.summarize([out])["gate_candidate_repeats_bitwise"]


def test_a_case_without_a_repeat_fails_the_gate(tmp_path):
    p, r = tmp_path / "p", tmp_path / "r"
    _write_run(p, "provider", 0, 0)
    _write_run(r, "candidate", 0, 0)
    assert not e2e.summarize([e2e.compare_case(p, r, 0)])["gate_every_case_has_a_repeat"]


def test_failed_runs_are_not_compared(tmp_path):
    p, r = tmp_path / "p", tmp_path / "r"
    _write_run(p, "provider", 0, 0)
    _write_run(r, "candidate", 0, 0)
    (r / "case0" / "r1-failed-1").mkdir()
    assert [x.name for x in e2e._runs(r, 0)] == ["r0"]


def test_the_timer_restores_what_it_wrapped():
    class Box:
        def decode(self, x):
            return x + 1

    box, timer = Box(), e2e.PhaseTimer()
    with timer.method(box, "decode", "decode"):
        assert box.decode(1) == 2
    assert "decode" not in box.__dict__ and timer.calls["decode"] and box.decode(1) == 2


def test_the_contract_is_sealed_and_cases_select():
    contract = e2e.load_contract(e2e.source_root())
    assert len(contract["cases"]) == 8 and contract["num_inference_steps"] == 10
    assert [c["case"] for c in e2e.select_cases(contract, "3,1")] == [3, 1]
    with pytest.raises(ValueError, match="not in the contract"):
        e2e.select_cases(contract, "99")
