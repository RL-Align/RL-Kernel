# SPDX-License-Identifier: Apache-2.0
"""Immutable inputs and tensor evidence for Qwen3 PPO qualification."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch

from rl_engine.integrations.common.ppo import PPOConfig

CONFIG = PPOConfig(epochs=2, minibatch_size=2, microbatch_size=1)
OPTIMIZER = dict(lr=1e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def sha(path):
    """Return the hex SHA-256 digest of a file, read in 8 MiB chunks."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(data)
    return h.hexdigest()


def read_tensor(root, row):
    """Load one evidence tensor from ``root`` and check it against its manifest ``row``.

    Raises ``ValueError`` if the path escapes ``root``, the SHA-256 or shape differs from
    ``row``, or the tensor has nonfinite values.
    """
    path = root / row["path"]
    if not path.resolve().is_relative_to(root.resolve()) or sha(path) != row["sha256"]:
        raise ValueError(f"tensor identity changed: {path}")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if list(value.shape) != row["shape"] or not torch.isfinite(value).all():
        raise ValueError(f"invalid tensor evidence: {path}")
    return value


def write_tensor(root, name, tensor):
    """Save a detached CPU copy of ``tensor`` under ``root`` and return its manifest row.

    The file name is the SHA-256 of ``name``; the row holds ``path``, ``sha256``, ``shape``
    and ``dtype``. Raises ``ValueError`` on nonfinite values or an already written ``name``.
    """
    value = tensor.detach().cpu().contiguous()
    if not torch.isfinite(value).all():
        raise ValueError(f"nonfinite evidence: {name}")
    path = root / (hashlib.sha256(name.encode()).hexdigest() + ".pt")
    if path.exists():
        raise ValueError(f"duplicate evidence: {name}")
    torch.save(value, path)
    return dict(path=path.name, sha256=sha(path), shape=list(value.shape), dtype=str(value.dtype))


def identity():
    """Return run provenance: commit, Slurm job id, model revision, torch, PPO and optimizer config.

    Raises ``ValueError`` when ``git status --porcelain`` is not empty (uncommitted changes or
    untracked files).
    """
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("qualification requires clean source")
    return dict(
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        job_id=os.environ.get("SLURM_JOB_ID"),
        model_revision=REVISION,
        torch=torch.__version__,
        config=asdict(CONFIG),
        optimizer=OPTIMIZER,
    )


def verify_checkpoint(root):
    """Check checkpoint files against ``qwen3-8b-manifest.json`` and return the manifest SHA-256.

    Every listed file under ``root`` must match its recorded size and SHA-256, and the
    manifest must name ``REVISION``; otherwise ``ValueError`` is raised.
    """
    manifest = Path(__file__).with_name("qwen3-8b-manifest.json")
    data = json.loads(manifest.read_text())
    if data["revision"] != REVISION:
        raise ValueError("unexpected checkpoint revision")
    for row in data["files"]:
        p = root / row["path"]
        if p.stat().st_size != row["size"] or sha(p) != row["sha256"]:
            raise ValueError(f"checkpoint changed: {p}")
    return sha(manifest)


def load_packet(root):
    """Return the frozen ``rollout.pt`` packet as a CPU dict after verifying its manifest.

    Raises ``ValueError`` if ``manifest.json`` names another model revision or a different
    rollout SHA-256; the packet must also validate as a version-0 ``PPORollout``.
    """
    index = json.loads((root / "manifest.json").read_text())
    if index["model_revision"] != REVISION:
        raise ValueError("wrong rollout model")
    if sha(root / "rollout.pt") != index["rollout_sha256"]:
        raise ValueError("fixed rollout changed")
    packet = torch.load(root / "rollout.pt", map_location="cpu", weights_only=True)
    from rl_engine.integrations.common.ppo import PPORollout

    PPORollout(**packet).validate(actor_version=0, critic_version=0)
    return packet


def load_targets(root):
    """Return the frozen ``targets.pt`` dict on CPU after checking it against ``targets.json``."""
    manifest = json.loads((root / "targets.json").read_text())
    if sha(root / "targets.pt") != manifest["sha256"]:
        raise ValueError("frozen PPO targets changed")
    return torch.load(root / "targets.pt", map_location="cpu", weights_only=True)


def save_report(root, report):
    """Write ``report`` as indented JSON to ``root/report.json``."""
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")


def verify_reference(root, manifest, *, metadata_only=False):
    """Return a frozen acceptance manifest after checking the hashes of its files under ``root``.

    ``metadata_only`` checks only the ``.json`` entries. Raises ``ValueError`` if the manifest
    is not frozen for ``REVISION`` or a file escapes ``root`` or has a different SHA-256.
    """
    data = json.loads(manifest.read_text())
    if data["status"] != "frozen" or data["model_revision"] != REVISION:
        raise ValueError("acceptance manifest is not frozen for Qwen3")
    for name, expected in data["files"].items():
        if metadata_only and not name.endswith(".json"):
            continue
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or sha(path) != expected:
            raise ValueError(f"reference artifact changed: {name}")
    return data
