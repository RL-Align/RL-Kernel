# SPDX-License-Identifier: Apache-2.0
"""Companion-patched AITER learned router; validate every dependency identity."""

import functools
import hashlib
import importlib
import json
from pathlib import Path

import torch

from rl_engine.contracts.operators.learned_route import (
    AITER_CONTRACT_ID,
    AITER_REVISION,
    CONTRACT_ID,
    LAUNCH,
)

API_MODULE = "aiter.ops.triton.moe.moe_routing.learned_route"


@functools.lru_cache(maxsize=1)
def implementation():
    folder = Path(__file__).with_name("patches")
    manifest = json.loads((folder / "learned-route.json").read_text())
    if manifest["aiter_revision"] != AITER_REVISION:
        raise RuntimeError("learned router AITER base revision mismatch")
    for name, digest in manifest["patch_sha256"].items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("learned router companion patch identity mismatch")
    try:
        api = importlib.import_module(API_MODULE)
    except ImportError as exc:
        raise RuntimeError(
            "task 32 requires pinned learned-router AITER companion patches"
        ) from exc
    if getattr(api, "LEARNED_ROUTE_CONTRACT_ID", None) != AITER_CONTRACT_ID or api.LAUNCH != LAUNCH:
        raise RuntimeError("learned router arithmetic/launch contract mismatch")
    for name, digest in manifest["source_sha256"].items():
        module = importlib.import_module(name)
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != digest:
            raise RuntimeError("learned router AITER source fingerprint mismatch")
    return api, manifest


def execution_record(logits):
    import triton

    _, manifest = implementation()
    props = torch.cuda.get_device_properties(logits.device)
    return {
        "task": 32,
        "contract_id": CONTRACT_ID,
        "actual_backend": "aiter.triton.learned_route",
        "aiter_base_revision": AITER_REVISION,
        "source_sha256": dict(manifest["source_sha256"]),
        "patch_sha256": dict(manifest["patch_sha256"]),
        "select_entry": API_MODULE + ".learned_route_select",
        "weight_kernel": API_MODULE + "._weights",
        "backward_kernel": API_MODULE + "._weight_vjp",
        "gpu": props.name,
        "arch": props.gcnArchName,
        "device": str(logits.device),
        "torch": torch.__version__,
        "rocm": torch.version.hip,
        "triton": triton.__version__,
        "config": dict(LAUNCH),
        "input_shape": list(logits.shape),
        "input_stride": list(logits.stride()),
        "input_dtype": "float32",
        "output_dtype": "float32",
        "index_dtype": "int64",
        "bias_role": "selection_only.no_gradient",
        "weight_source": "pre_bias_sqrt_softplus_scores.direct_gather",
        "normalization": "mask_then_ascending_slot_sum.clamp_min(1e-20).scale_once",
        "normal_selection_recomputed": True,
        "graph": False,
        "kernel_launched": logits.shape[0] != 0,
    }
