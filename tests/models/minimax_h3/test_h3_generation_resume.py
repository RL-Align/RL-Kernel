# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Resumable candidate generation: a preempted run continues to the same result.

A fake transformer and denoising loop stand in for the pipeline, so these run on CPU.
The loop feeds each call's output back into the next call's inputs (as the scheduler
does), so any wrong replay would change every later step.
"""

from __future__ import annotations

import types

import pytest
import torch

from rl_engine.validation.models.h3_generation_resume import (
    ResumableRun,
    ResumeError,
    StepStore,
    inputs_digest,
)

STEPS = 6
IDENTITY = {"commit": "abc", "seed": 42, "candidate": True, "preflight": False}


class FakeTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.computed = 0

    def forward(self, hidden_states, audio_hidden_states, *, return_dict=True):
        self.computed += 1
        video = torch.tanh(hidden_states) * 0.5
        audio = torch.sin(audio_hidden_states) * 0.25
        return _output((video, audio), {"return_dict": return_dict})


def _output(values, kwargs):
    if kwargs.get("return_dict", True):
        return types.SimpleNamespace(sample=values[0], audio_sample=values[1])
    return values


def _initial(seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(8, 4, generator=generator), torch.randn(3, 5, generator=generator)


def denoise(directory, *, resume, stop_after=None, initial=None, identity=IDENTITY):
    """Run the fake loop through a ResumableRun; return (latents, transformer, run)."""

    video, audio = initial if initial is not None else _initial()
    store = StepStore(directory)
    run = ResumableRun(store, identity, resume=resume)
    if not (directory / "initial.pt").exists():
        torch.save({"hidden_states": video, "audio_hidden_states": audio}, directory / "initial.pt")
    module = FakeTransformer()
    run.install(module, _output)

    def hook(mod, args, kwargs, output):
        values = (output.sample, output.audio_sample)
        run.record(args, kwargs, values)
        if stop_after is not None and run.recorded >= stop_after:
            raise KeyboardInterrupt("simulated preemption")

    module.register_forward_hook(hook, with_kwargs=True)
    for _ in range(STEPS):
        out = module(hidden_states=video, audio_hidden_states=audio)
        # The scheduler's update: the next inputs depend on this call's outputs.
        video = video + out.sample
        audio = audio - 0.5 * out.audio_sample
    return (video, audio), module, run


def _step_bytes(directory):
    return {
        p.name: p.read_bytes()
        for p in sorted(directory.iterdir())
        if p.name.startswith("step-") and p.suffix in (".pt", ".json")
    }


def test_uninterrupted_run_saves_every_step(tmp_path):
    (video, _), module, run = denoise(tmp_path / "a", resume=False)
    assert module.computed == STEPS
    assert len([p for p in (tmp_path / "a").glob("step-*.pt")]) == STEPS
    assert len([p for p in (tmp_path / "a").glob("step-*.in.json")]) == STEPS
    assert run.summary()["replayed_steps"] == 0


def test_resumed_run_matches_an_uninterrupted_one(tmp_path):
    whole, _, _ = denoise(tmp_path / "whole", resume=False)
    with pytest.raises(KeyboardInterrupt):
        denoise(tmp_path / "cut", resume=False, stop_after=4)
    resumed, module, run = denoise(tmp_path / "cut", resume=True)
    assert all(torch.equal(a, b) for a, b in zip(whole, resumed, strict=True))
    assert _step_bytes(tmp_path / "whole").keys() == _step_bytes(tmp_path / "cut").keys()
    for name, data in _step_bytes(tmp_path / "whole").items():
        if name.endswith(".pt"):
            assert data == _step_bytes(tmp_path / "cut")[name], name
    # Four steps were saved: three replayed, the fourth recomputed and verified, two new.
    summary = run.summary()
    assert summary["saved_steps_at_start"] == 4
    assert summary["replayed_steps"] == 3
    assert summary["recomputed_and_verified_step"] == 3
    assert module.computed == STEPS - 3
    assert [a["saved_steps"] for a in summary["attempts"]] == [0, 4]


def test_repeated_preemptions_still_converge(tmp_path):
    whole, _, _ = denoise(tmp_path / "whole", resume=False)
    for stop in (2, 3, 5):
        with pytest.raises(KeyboardInterrupt):
            denoise(tmp_path / "cut", resume=stop > 2, stop_after=stop)
    resumed, _, run = denoise(tmp_path / "cut", resume=True)
    assert all(torch.equal(a, b) for a, b in zip(whole, resumed, strict=True))
    assert len(run.summary()["attempts"]) == 4


def test_a_complete_run_resumes_by_recomputing_only_its_last_step(tmp_path):
    whole, _, _ = denoise(tmp_path / "a", resume=False)
    again, module, run = denoise(tmp_path / "a", resume=True)
    assert all(torch.equal(x, y) for x, y in zip(whole, again, strict=True))
    assert module.computed == 1
    assert run.summary()["recomputed_and_verified_step"] == STEPS - 1


def test_different_inputs_are_rejected_during_replay(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        denoise(tmp_path / "a", resume=False, stop_after=3)
    # Rejected at the first replayed call, not only at the recomputed last saved step.
    with pytest.raises(ResumeError, match="replayed step 0 has different inputs"):
        denoise(tmp_path / "a", resume=True, initial=_initial(seed=1))


def test_a_nondeterministic_recompute_is_rejected(tmp_path, monkeypatch):
    with pytest.raises(KeyboardInterrupt):
        denoise(tmp_path / "a", resume=False, stop_after=3)

    original = FakeTransformer.forward

    def noisy(self, hidden_states, audio_hidden_states, *, return_dict=True):
        out = original(self, hidden_states, audio_hidden_states, return_dict=return_dict)
        return _output((out.sample + 1e-6, out.audio_sample), {"return_dict": return_dict})

    monkeypatch.setattr(FakeTransformer, "forward", noisy)
    with pytest.raises(ResumeError, match="differs from the saved output"):
        denoise(tmp_path / "a", resume=True)


def test_a_different_identity_cannot_be_continued(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        denoise(tmp_path / "a", resume=False, stop_after=3)
    with pytest.raises(ResumeError, match="different identity"):
        denoise(tmp_path / "a", resume=True, identity={**IDENTITY, "commit": "other"})


def test_an_existing_directory_needs_resume(tmp_path):
    denoise(tmp_path / "a", resume=False, stop_after=None)
    with pytest.raises(ResumeError, match="resume"):
        denoise(tmp_path / "a", resume=False)


def test_a_corrupted_step_ends_the_valid_prefix(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        denoise(tmp_path / "a", resume=False, stop_after=4)
    (tmp_path / "a" / "step-002.pt").write_bytes(b"damaged")
    run = ResumableRun(StepStore(tmp_path / "a"), IDENTITY, resume=True)
    assert run.saved == 2


def test_leftover_temporary_files_are_ignored(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        denoise(tmp_path / "a", resume=False, stop_after=3)
    (tmp_path / "a" / "step-003.pt.tmp").write_bytes(b"half written")
    run = ResumableRun(StepStore(tmp_path / "a"), IDENTITY, resume=True)
    assert run.saved == 3


def test_a_run_without_initial_inputs_is_not_resumable(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        denoise(tmp_path / "a", resume=False, stop_after=3)
    (tmp_path / "a" / "initial.pt").unlink()
    assert ResumableRun(StepStore(tmp_path / "a"), IDENTITY, resume=True).saved == 0


def test_inputs_digest_tracks_value_dtype_and_shape_and_ignores_non_tensors():
    x = torch.arange(6.0).reshape(2, 3)
    base = inputs_digest((), {"a": x, "flag": True})
    assert base == inputs_digest((), {"a": x.clone(), "flag": False})
    assert base != inputs_digest((), {"a": x + 1})
    assert base != inputs_digest((), {"a": x.double()})
    assert base != inputs_digest((), {"a": x.reshape(3, 2)})
    assert base != inputs_digest((), {"b": x})
    assert inputs_digest((x,), {}) != inputs_digest((), {"a": x})
    assert inputs_digest((), {"s": torch.tensor(3.0)})  # zero-dim tensors hash
