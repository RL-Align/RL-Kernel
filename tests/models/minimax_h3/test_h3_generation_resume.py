# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Resumable candidate generation: a preempted run continues to the same result.

A fake transformer and denoising loop stand in for the pipeline, so these run on CPU.
The loop feeds each call's output back into the next call's inputs (as the scheduler
does), so any wrong replay would change every later step.
"""

from __future__ import annotations

import inspect
import subprocess
import types

import pytest
import torch

from rl_engine.validation.models import h3_generation_resume as resume_module
from rl_engine.validation.models.h3_generation_resume import (
    ResumableRun,
    ResumeError,
    StepStore,
    adopt_identity,
    inputs_digest,
    model_trees,
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


def denoise(
    directory, *, resume, stop_after=None, initial=None, identity=IDENTITY, verify_last_step=False
):
    """Run the fake loop through a ResumableRun; return (latents, transformer, run)."""

    video, audio = initial if initial is not None else _initial()
    store = StepStore(directory)
    run = ResumableRun(store, identity, resume=resume, verify_last_step=verify_last_step)
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


def test_resumed_run_with_verification_matches_an_uninterrupted_one(tmp_path):
    whole, _, _ = denoise(tmp_path / "whole", resume=False)
    with pytest.raises(KeyboardInterrupt):
        denoise(tmp_path / "cut", resume=False, stop_after=4)
    resumed, module, run = denoise(tmp_path / "cut", resume=True, verify_last_step=True)
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


def test_with_verification_a_complete_run_recomputes_only_its_last_step(tmp_path):
    whole, _, _ = denoise(tmp_path / "a", resume=False)
    again, module, run = denoise(tmp_path / "a", resume=True, verify_last_step=True)
    assert all(torch.equal(x, y) for x, y in zip(whole, again, strict=True))
    assert module.computed == 1
    assert run.summary()["recomputed_and_verified_step"] == STEPS - 1


def test_different_inputs_are_rejected_during_replay(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        denoise(tmp_path / "a", resume=False, stop_after=3)
    # Rejected at the first replayed call, not only at the recomputed last saved step.
    with pytest.raises(ResumeError, match="replayed step 0 has different inputs"):
        denoise(tmp_path / "a", resume=True, initial=_initial(seed=1))


def test_a_nondeterministic_recompute_is_rejected_when_verifying(tmp_path, monkeypatch):
    with pytest.raises(KeyboardInterrupt):
        denoise(tmp_path / "a", resume=False, stop_after=3)

    original = FakeTransformer.forward

    def noisy(self, hidden_states, audio_hidden_states, *, return_dict=True):
        out = original(self, hidden_states, audio_hidden_states, return_dict=return_dict)
        return _output((out.sample + 1e-6, out.audio_sample), {"return_dict": return_dict})

    monkeypatch.setattr(FakeTransformer, "forward", noisy)
    with pytest.raises(ResumeError, match="differs from the saved output"):
        denoise(tmp_path / "a", resume=True, verify_last_step=True)
    # Without verification nothing recomputes the saved steps, so the drift goes unnoticed.
    with pytest.raises(KeyboardInterrupt):
        denoise(tmp_path / "b", resume=False, stop_after=3)
    denoise(tmp_path / "b", resume=True)


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


def test_replay_wrapper_keeps_the_forward_signature(tmp_path):
    """The pipeline selects the transformer's layout inputs from inspect.signature(forward)."""

    module = FakeTransformer()
    before = list(inspect.signature(module.forward).parameters)
    (tmp_path / "a").mkdir()
    run = ResumableRun(StepStore(tmp_path / "a" / "x"), IDENTITY, resume=False)
    uninstall = run.install(module, _output)
    assert list(inspect.signature(module.forward).parameters) == before
    assert before == ["hidden_states", "audio_hidden_states", "return_dict"]
    uninstall()
    assert list(inspect.signature(module.forward).parameters) == before


def test_replay_wrapper_keeps_a_wrapped_forward_signature(tmp_path):
    """Same through an accelerate-style partial that itself carries __wrapped__."""

    import functools

    module = FakeTransformer()
    plain = module.forward

    def new_forward(m, *args, **kwargs):
        return plain(*args, **kwargs)

    hooked = functools.update_wrapper(functools.partial(new_forward, module), plain)
    module.forward = hooked
    before = list(inspect.signature(module.forward).parameters)
    assert before == ["hidden_states", "audio_hidden_states", "return_dict"]
    run = ResumableRun(StepStore(tmp_path / "b"), IDENTITY, resume=False)
    run.install(module, _output)
    assert list(inspect.signature(module.forward).parameters) == before


# --- default resume: replay everything saved, recompute nothing --------------------------


def test_default_resume_replays_every_saved_step_and_computes_only_the_rest(tmp_path):
    whole, _, _ = denoise(tmp_path / "whole", resume=False)
    with pytest.raises(KeyboardInterrupt):
        denoise(tmp_path / "cut", resume=False, stop_after=4)
    resumed, module, run = denoise(tmp_path / "cut", resume=True)
    assert all(torch.equal(a, b) for a, b in zip(whole, resumed, strict=True))
    summary = run.summary()
    assert (summary["saved_steps_at_start"], summary["replayed_steps"]) == (4, 4)
    assert summary["recomputed_and_verified_step"] is None
    assert summary["real_forward_calls"] == STEPS - 4 == module.computed
    for name, data in _step_bytes(tmp_path / "whole").items():
        if name.endswith(".pt"):
            assert data == _step_bytes(tmp_path / "cut")[name], name


def test_a_complete_run_resumes_with_no_forward_pass_at_all(tmp_path):
    whole, _, _ = denoise(tmp_path / "a", resume=False)
    again, module, run = denoise(tmp_path / "a", resume=True)
    assert all(torch.equal(x, y) for x, y in zip(whole, again, strict=True))
    assert module.computed == 0
    assert run.real_calls == 0


def test_each_attempt_only_needs_a_few_steps_to_make_progress(tmp_path):
    """Attempts that each finish one new step still add up to the whole run."""

    whole, _, _ = denoise(tmp_path / "whole", resume=False)
    saved = 0
    result = run = None
    for attempt in range(STEPS):
        stop = saved + 1 if saved + 1 < STEPS else None  # the last attempt is left to finish
        try:
            result, _, run = denoise(tmp_path / "cut", resume=attempt > 0, stop_after=stop)
        except KeyboardInterrupt:
            saved += 1
            continue
        break
    assert result is not None and saved == STEPS - 1
    assert all(torch.equal(a, b) for a, b in zip(whole, result, strict=True))
    assert run.real_calls == 1 and len(run.summary()["attempts"]) == STEPS


def test_the_validated_trace_survives_for_a_later_replay_only_attempt(tmp_path):
    store = StepStore(tmp_path / "a")
    assert store.load_trace() is None
    (tmp_path / "a").mkdir()
    trace = {"calls": {"block.0": 3}, "execution_order": ["block.0", "block.1"]}
    store.save_trace(trace)
    assert StepStore(tmp_path / "a").load_trace() == trace
    assert not list((tmp_path / "a").glob("*.tmp"))


# --- identity tied to the model paths, and adopting a commit-based run -------------------


def _git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "model").mkdir(parents=True)
    (root / "orchestration").mkdir()
    (root / "model" / "kernel.py").write_text("v1\n")
    (root / "orchestration" / "resume.py").write_text("v1\n")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "a")
    monkeypatch.setattr(resume_module, "MODEL_PATHS", ("model",))
    return root


def _commit(repo, path, text):
    (repo / path).write_text(text)
    _git(repo, "commit", "-q", "-am", f"change {path}")
    return _git(repo, "rev-parse", "HEAD")


def _old_run(directory, commit, trees):
    base = {k: v for k, v in IDENTITY.items() if k != "commit"}
    store = StepStore(directory)
    store.open({"commit": commit, **base}, resume=False)
    return {**base, "model_trees": trees}


def test_model_trees_ignore_orchestration_changes_but_not_model_changes(repo):
    first = model_trees("HEAD", cwd=repo)
    _commit(repo, "orchestration/resume.py", "v2\n")
    assert model_trees("HEAD", cwd=repo) == first
    _commit(repo, "model/kernel.py", "v2\n")
    assert model_trees("HEAD", cwd=repo) != first


def test_adopting_a_commit_based_run_lets_a_later_commit_continue_it(repo, tmp_path):
    old = _git(repo, "rev-parse", "HEAD")
    new_identity = _old_run(tmp_path / "run", old, model_trees("HEAD", cwd=repo))
    _commit(repo, "orchestration/resume.py", "v2\n")
    new_identity["model_trees"] = model_trees("HEAD", cwd=repo)
    adopt_identity(tmp_path / "run", new_identity, old[:8], cwd=repo)
    assert StepStore(tmp_path / "run").open(new_identity, resume=True) == 0
    attempts = StepStore(tmp_path / "run").attempts()
    assert attempts[-2]["adopted"]["from_commit"] == old
    assert attempts[-2]["adopted"]["model_paths_identical"] is True


def test_adoption_is_refused_when_a_model_path_changed(repo, tmp_path):
    old = _git(repo, "rev-parse", "HEAD")
    _old_run(tmp_path / "run", old, model_trees("HEAD", cwd=repo))
    _commit(repo, "model/kernel.py", "v2\n")
    new_identity = {
        **{k: v for k, v in IDENTITY.items() if k != "commit"},
        "model_trees": model_trees("HEAD", cwd=repo),
    }
    before = (tmp_path / "run" / "run.json").read_text()
    with pytest.raises(ResumeError, match="model paths changed"):
        adopt_identity(tmp_path / "run", new_identity, old[:8], cwd=repo)
    assert (tmp_path / "run" / "run.json").read_text() == before


def test_adoption_refuses_other_differences_and_dry_run_writes_nothing(repo, tmp_path):
    old = _git(repo, "rev-parse", "HEAD")
    identity = _old_run(tmp_path / "run", old, model_trees("HEAD", cwd=repo))
    before = (tmp_path / "run" / "run.json").read_text()
    with pytest.raises(ResumeError, match="more than the commit"):
        adopt_identity(tmp_path / "run", {**identity, "seed": 7}, old[:8], cwd=repo)
    with pytest.raises(ResumeError, match="not abc"):
        adopt_identity(tmp_path / "run", identity, "abc", cwd=repo)
    adopt_identity(tmp_path / "run", identity, old[:8], cwd=repo, dry_run=True)
    assert (tmp_path / "run" / "run.json").read_text() == before
    adopt_identity(tmp_path / "run", identity, old[:8], cwd=repo)
    with pytest.raises(ResumeError, match="already"):
        adopt_identity(tmp_path / "run", identity, old[:8], cwd=repo)
