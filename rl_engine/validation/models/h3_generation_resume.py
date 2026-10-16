# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Resumable denoising for the full-H3 candidate generation (RFC #420).

A candidate generation takes about 20 hours per seed and its cluster partition
preempts jobs every few hours, so a restart must not redo finished steps. The
generation already keeps every step's predicted velocities (``step-NNN.pt``);
``StepStore`` adds, next to each, a sidecar with the digest of the transformer's
inputs for that step and the digest of the saved file, both written atomically
(the sidecar last, so a step counts only when both exist).

On resume the first ``saved - 1`` transformer calls are *replayed*: the saved
velocities are returned instead of recomputed, and the pipeline's own scheduler
update turns them into the next latents exactly as before. The input digest of
every replayed call is checked against the one saved, so the replay is proven to
have rebuilt the trajectory the earlier attempt had. The last saved step is
always recomputed, and must reproduce its stored output bitwise: that checks the
resumed run is equivalent to an uninterrupted one, and guarantees a real forward
pass, so the execution trace stays complete.
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

    def saved_digest(self, step: int) -> str:
        return json.loads(self._sidecar(step).read_text())["inputs_sha256"]

    def load(self, step: int) -> tuple[torch.Tensor, ...]:
        sidecar = json.loads(self._sidecar(step).read_text())
        if _file_sha256(self._step(step)) != sidecar["output_sha256"]:
            raise ResumeError(f"saved step {step} changed on disk")
        return tuple(torch.load(self._step(step), map_location="cpu", weights_only=True))


class ResumableRun:
    """Replays saved steps and records new ones for one seed's transformer calls."""

    def __init__(self, store: StepStore, identity: dict[str, Any], *, resume: bool):
        self.store = store
        self.saved = store.open(identity, resume=resume)
        self.replay = max(self.saved - 1, 0)
        self.verified_step: int | None = None
        self._forward_calls = 0
        self.recorded = 0

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
            recomputed_and_verified_step=self.verified_step,
            attempts=self.store.attempts(),
        )
