# SPDX-License-Identifier: Apache-2.0
"""Pinned AITER auxiliary loss; all arithmetic and launch identities fail closed."""

import functools
import hashlib
import importlib
import json
from pathlib import Path

from rl_engine.contracts.operators.learned_route import AITER_REVISION
from rl_engine.contracts.operators.router_aux_loss import AITER_CONTRACT_ID, LAUNCH


@functools.lru_cache(maxsize=1)
def implementation():
    folder = Path(__file__).with_name("patches")
    manifest = json.loads((folder / "router-aux-loss.json").read_text())
    if manifest["aiter_revision"] != AITER_REVISION:
        raise RuntimeError("auxiliary loss AITER revision mismatch")
    for name, digest in manifest["patch_sha256"].items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("auxiliary loss patch fingerprint mismatch")
    try:
        api = importlib.import_module("aiter.ops.triton.moe.moe_routing.router_aux_loss")
    except ImportError as exc:
        raise RuntimeError("task 34 requires the pinned AITER auxiliary loss companion") from exc
    if getattr(api, "AUX_LOSS_CONTRACT_ID", None) != AITER_CONTRACT_ID or api.LAUNCH != LAUNCH:
        raise RuntimeError("auxiliary loss arithmetic/launch contract mismatch")
    for name, digest in manifest["source_sha256"].items():
        module = importlib.import_module(name)
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != digest:
            raise RuntimeError("auxiliary loss source fingerprint mismatch")
    return api, manifest
