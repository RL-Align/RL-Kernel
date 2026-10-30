# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Resumable denoising for the full-H3 candidate generation (RFC #420).

A candidate generation takes about 20 hours per seed and its cluster partition
preempts jobs every few hours, so a restart must not redo finished steps. The
generation already keeps every step's predicted velocities (``step-NNN.pt``);
``StepStore`` adds, next to each, a sidecar with the digest of the transformer's
inputs for that step and the digest of the saved file, both written atomically
(the sidecar last, so a step counts only when both exist).

On resume every saved transformer call is *replayed*: the saved velocities are
returned instead of recomputed, and the pipeline's own scheduler update turns them
into the next latents exactly as before. The input digest of every replayed call is
checked against the one saved, so the replay is proven to have rebuilt the trajectory
the earlier attempt had. Recomputing the last saved step (``verify_last_step``)
additionally checks that the forward pass reproduces its stored output bitwise; it
costs one full step per restart, which on a heavily preempted partition is most of an
attempt, so it is opt-in. The execution trace is saved after the first real step of
an attempt, so a run whose remaining work is only decoding keeps a complete trace.

A run is tied to the numerics it started with (``model_trees``: the git tree of every
path in ``MODEL_PATHS``), not to a commit, so a commit that changes only orchestration
code can continue it. ``adopt_identity`` moves a run started under a commit-based
identity onto this one and refuses unless those paths are byte-identical.
"""

from __future__ import annotations

import datetime
import functools
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable

import torch

RUN_FILE = "run.json"
TRACE_FILE = "trace.json"

# Everything that determines the generated numbers. The resume and orchestration code is
# deliberately absent: changing it must not invalidate saved steps.
MODEL_PATHS = (
    "rl_engine/backends",
    "rl_engine/contracts",
    "rl_engine/ops",
    "rl_engine/reference",
    "rl_engine/runtime",
    "rl_engine/validation/models/h3_block.py",
    "rl_engine/validation/models/h3_cases.py",
    "rl_engine/validation/models/h3_full_manifest.json",
    "rl_engine/validation/models/h3_full_model.py",
    "rl_engine/validation/models/h3_provider.py",
    "rl_engine/validation/models/h3_weights.py",
    "csrc",
    "build_tools",
    "examples/models/minimax_h3/generation-contract.json",
    "examples/models/minimax_h3/generation-inputs.json",
    "examples/models/minimax_h3/generation-prompt.txt",
)


class ResumeError(ValueError):
    """A saved run cannot be continued, or a replay disagrees with what was saved."""


def tensor_digest(named: dict[str, torch.Tensor]) -> str:
    """sha256 over tensor names, dtypes, shapes and raw bytes, in name order."""

    digest = hashlib.sha256()
    for name in sorted(named):
        tensor = named[name].detach().cpu().contiguous()
        digest.update(f"{name}|{tensor.dtype}|{tuple(tensor.shape)}|".encode())
        if tensor.numel():
            digest.update(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()))
    return digest.hexdigest()


def inputs_digest(args: tuple, kwargs: dict[str, Any]) -> str:
    """Digest of every tensor passed to one transformer call (non-tensors are ignored)."""

    named = {f"arg{i}": v for i, v in enumerate(args) if isinstance(v, torch.Tensor)}
    named.update({k: v for k, v in kwargs.items() if isinstance(v, torch.Tensor)})
    return tensor_digest(named)


def model_trees(revision: str = "HEAD", cwd: Path | None = None) -> dict[str, str]:
    """The git tree (or blob) id of every model path at ``revision``."""

    import subprocess

    trees = {}
    for path in MODEL_PATHS:
        trees[path] = subprocess.check_output(
            ["git", "rev-parse", f"{revision}:{path}"], text=True, cwd=cwd
        ).strip()
    return trees


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 24):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, write: Callable[[Path], object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    write(temporary)
    os.replace(temporary, path)


class StepStore:
    """The per-step velocities and input digests of one seed, in one directory."""

    def __init__(self, directory: Path):
        self.directory = directory

    def _step(self, step: int) -> Path:
        return self.directory / f"step-{step:03}.pt"

    def _sidecar(self, step: int) -> Path:
        return self.directory / f"step-{step:03}.in.json"

    def open(self, identity: dict[str, Any], *, resume: bool) -> int:
        """Create the directory, or on ``resume`` validate it; return the saved step count.

        ``identity`` (commit, contract, seed, flags, ...) must equal the one stored,
        so steps from a different source or setting can never be mixed in.
        """

        run_file = self.directory / RUN_FILE
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if not self.directory.exists():
            self.directory.mkdir(parents=True)
        elif not resume:
            raise ResumeError(f"{self.directory} exists; pass resume to continue it")
        if run_file.exists():
            stored = json.loads(run_file.read_text())
            if stored["identity"] != identity:
                raise ResumeError("saved run has a different identity; refusing to continue it")
        else:
            # ``run.json`` is written before any step, so saved steps without it mean it was
            # removed: their identity is unknown and they must not be adopted by this one.
            orphans = sorted(path.name for path in self.directory.glob("step-*"))
            if orphans:
                raise ResumeError(
                    f"{self.directory} holds saved steps ({orphans[0]}, ...) but no {RUN_FILE}; "
                    "their identity is unknown, refusing to continue them"
                )
            stored = {"identity": identity, "attempts": []}
        saved = self._complete_steps()
        stored["attempts"].append({"started": now, "saved_steps": saved})
        _atomic_write(run_file, lambda p: p.write_text(json.dumps(stored, indent=2) + "\n"))
        return saved

    def attempts(self) -> list[dict[str, Any]]:
        return json.loads((self.directory / RUN_FILE).read_text())["attempts"]

    def _complete_steps(self) -> int:
        count = 0
        while self._step(count).exists() and self._sidecar(count).exists():
            sidecar = json.loads(self._sidecar(count).read_text())
            if _file_sha256(self._step(count)) != sidecar["output_sha256"]:
                break
            count += 1
        if count and not (self.directory / "initial.pt").exists():
            return 0
        return count

    def save(self, step: int, values: tuple[torch.Tensor, ...], digest: str) -> None:
        """Persist one step's outputs, then its sidecar; both atomically."""

        _atomic_write(self._step(step), lambda p: torch.save(tuple(values), p))
        sidecar = {"inputs_sha256": digest, "output_sha256": _file_sha256(self._step(step))}
        _atomic_write(
            self._sidecar(step), lambda p: p.write_text(json.dumps(sidecar, indent=2) + "\n")
        )

    def save_trace(self, trace: dict[str, Any]) -> None:
        """Keep the validated execution trace of this seed's real forward passes."""

        _atomic_write(
            self.directory / TRACE_FILE, lambda p: p.write_text(json.dumps(trace, indent=2) + "\n")
        )

    def load_trace(self) -> dict[str, Any] | None:
        path = self.directory / TRACE_FILE
        return json.loads(path.read_text()) if path.exists() else None

    def saved_digest(self, step: int) -> str:
        return json.loads(self._sidecar(step).read_text())["inputs_sha256"]

    def load(self, step: int) -> tuple[torch.Tensor, ...]:
        sidecar = json.loads(self._sidecar(step).read_text())
        if _file_sha256(self._step(step)) != sidecar["output_sha256"]:
            raise ResumeError(f"saved step {step} changed on disk")
        return tuple(torch.load(self._step(step), map_location="cpu", weights_only=True))


class ResumableRun:
    """Replays saved steps and records new ones for one seed's transformer calls."""

    def __init__(
        self,
        store: StepStore,
        identity: dict[str, Any],
        *,
        resume: bool,
        verify_last_step: bool = False,
    ):
        self.store = store
        self.saved = store.open(identity, resume=resume)
        self.verify_last_step = verify_last_step
        self.replay = self.saved - 1 if verify_last_step and self.saved else self.saved
        self.verified_step: int | None = None
        self._forward_calls = 0
        self.recorded = 0
        self.real_calls = 0

    def install(self, module: torch.nn.Module, make_output: Callable[[tuple, dict], Any]):
        """Serve the first ``replay`` calls of ``module`` from disk; return an uninstaller."""

        original = module.forward

        # The pipeline picks the transformer's layout inputs from inspect.signature(forward),
        # so the wrapper must keep the original signature (functools.wraps sets __wrapped__).
        @functools.wraps(original)
        def forward(*args, **kwargs):
            index = self._forward_calls
            self._forward_calls += 1
            if index >= self.replay:
                return original(*args, **kwargs)
            if inputs_digest(args, kwargs) != self.store.saved_digest(index):
                raise ResumeError(f"replayed step {index} has different inputs than when saved")
            device = next(
                v.device for v in (*args, *kwargs.values()) if isinstance(v, torch.Tensor)
            )
            values = tuple(v.to(device) for v in self.store.load(index))
            return make_output(values, kwargs)

        module.forward = forward

        def uninstall():
            module.forward = original

        return uninstall

    def record(self, args: tuple, kwargs: dict[str, Any], values: tuple[torch.Tensor, ...]):
        """Handle one finished call (forward hook): save it, or verify a recomputed step."""

        index = self.recorded
        self.recorded += 1
        if index < self.replay:
            return
        self.real_calls += 1
        digest = inputs_digest(args, kwargs)
        values = tuple(v.detach().cpu() for v in values)
        if index < self.saved:
            if digest != self.store.saved_digest(index):
                raise ResumeError(f"recomputed step {index} has different inputs than saved")
            stored = self.store.load(index)
            if len(stored) != len(values) or not all(
                torch.equal(a, b) for a, b in zip(stored, values, strict=True)
            ):
                raise ResumeError(f"recomputed step {index} differs from the saved output")
            self.verified_step = index
            return
        self.store.save(index, values, digest)

    def summary(self) -> dict[str, Any]:
        return dict(
            saved_steps_at_start=self.saved,
            replayed_steps=self.replay,
            verify_last_step=self.verify_last_step,
            recomputed_and_verified_step=self.verified_step,
            real_forward_calls=self.real_calls,
            attempts=self.store.attempts(),
        )


def adopt_identity(
    directory: Path,
    new_identity: dict[str, Any],
    from_commit: str,
    cwd: Path | None = None,
    *,
    dry_run: bool = False,
) -> None:
    """Move a run started under a commit-based identity onto a ``model_trees`` identity.

    Refuses unless everything but the commit matches and the model paths at
    ``from_commit`` are byte-identical to the ones in ``new_identity``; records the
    adoption in the run's attempt list. ``dry_run`` performs every check and writes nothing.
    """

    run_file = directory / RUN_FILE
    stored = json.loads(run_file.read_text())
    old = stored["identity"]
    if "model_trees" in old:
        raise ResumeError("run already has a model-tree identity")
    if old.get("commit", "").startswith(from_commit) is False:
        raise ResumeError(f"run was started at {old.get('commit')}, not {from_commit}")
    rest_old = {k: v for k, v in old.items() if k != "commit"}
    rest_new = {k: v for k, v in new_identity.items() if k != "model_trees"}
    if rest_old != rest_new:
        raise ResumeError("run differs from the new identity in more than the commit")
    then = model_trees(old["commit"], cwd=cwd)
    if then != new_identity["model_trees"]:
        changed = sorted(k for k in then if then[k] != new_identity["model_trees"][k])
        raise ResumeError(f"model paths changed since {from_commit}: {changed}")
    if dry_run:
        return
    stored["identity"] = new_identity
    stored["attempts"].append(
        {
            "adopted": {
                "from_commit": old["commit"],
                "model_paths_identical": True,
                "when": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
        }
    )
    _atomic_write(run_file, lambda p: p.write_text(json.dumps(stored, indent=2) + "\n"))
