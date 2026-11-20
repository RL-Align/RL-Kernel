# SPDX-License-Identifier: Apache-2.0
"""Fail-closed binding for the additive AITER diagnostic replay API."""

import functools
import hashlib
import importlib
import json
from pathlib import Path

from rl_engine.contracts.operators.learned_route import AITER_REVISION
from rl_engine.contracts.operators.router_training import REPLAY_CONTRACT_ID


@functools.lru_cache(maxsize=1)
def implementation():
    folder = Path(__file__).with_name("patches")
    manifest = json.loads((folder / "router-training.json").read_text())
    if manifest["aiter_revision"] != AITER_REVISION:
        raise RuntimeError("router training AITER revision mismatch")
    for name, digest in manifest["patch_sha256"].items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("router training patch fingerprint mismatch")
    try:
        api = importlib.import_module("aiter.ops.triton.moe.moe_routing.router_replay")
    except ImportError as exc:
        raise RuntimeError("task 33 requires the pinned AITER router-training companion") from exc
    if getattr(api, "ROUTER_REPLAY_CONTRACT_ID", None) != REPLAY_CONTRACT_ID:
        raise RuntimeError("router replay API contract mismatch")
    for name, digest in manifest["source_sha256"].items():
        module = importlib.import_module(name)
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != digest:
            raise RuntimeError("router training source fingerprint mismatch")
    return api, manifest
