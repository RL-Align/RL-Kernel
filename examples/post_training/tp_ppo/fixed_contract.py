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
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(data)
    return h.hexdigest()


def read_tensor(root, row):
    path = root / row["path"]
    if not path.resolve().is_relative_to(root.resolve()) or sha(path) != row["sha256"]:
        raise ValueError(f"tensor identity changed: {path}")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if list(value.shape) != row["shape"] or not torch.isfinite(value).all():
        raise ValueError(f"invalid tensor evidence: {path}")
    return value


def write_tensor(root, name, tensor):
    value = tensor.detach().cpu().contiguous()
    if not torch.isfinite(value).all():
        raise ValueError(f"nonfinite evidence: {name}")
    path = root / (hashlib.sha256(name.encode()).hexdigest() + ".pt")
    if path.exists():
        raise ValueError(f"duplicate evidence: {name}")
    torch.save(value, path)
    return dict(path=path.name, sha256=sha(path), shape=list(value.shape), dtype=str(value.dtype))


def identity():
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
    manifest = json.loads((root / "targets.json").read_text())
    if sha(root / "targets.pt") != manifest["sha256"]:
        raise ValueError("frozen PPO targets changed")
    return torch.load(root / "targets.pt", map_location="cpu", weights_only=True)


def save_report(root, report):
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")


def verify_reference(root, manifest, *, metadata_only=False):
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
