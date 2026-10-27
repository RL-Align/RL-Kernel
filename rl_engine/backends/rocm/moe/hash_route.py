# SPDX-License-Identifier: Apache-2.0
"""Exact companion-patched AITER task-30 backend; never a fallback."""

import functools
import hashlib
import importlib
import json
from pathlib import Path

import torch

from rl_engine.contracts.operators.hash_route import AITER_CONTRACT_ID, AITER_REVISION, CONTRACT_ID

API_MODULE = "aiter.ops.triton.moe.moe_routing.hash_route"


@functools.lru_cache(maxsize=1)
def implementation():
    folder = Path(__file__).with_name("patches")
    manifest = json.loads((folder / "hash-route.json").read_text())
    if manifest["aiter_revision"] != AITER_REVISION:
        raise RuntimeError("Hash route AITER base revision mismatch")
    for patch, digest in manifest["patch_sha256"].items():
        if hashlib.sha256((folder / patch).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Hash route companion patch fingerprint mismatch")
    try:
        api = importlib.import_module(API_MODULE)
    except ImportError as exc:
        raise RuntimeError(
            "task 30 requires the pinned AITER Hash route companion patches"
        ) from exc
    if getattr(api, "HASH_ROUTE_CONTRACT_ID", None) != AITER_CONTRACT_ID:
        raise RuntimeError("Hash route AITER arithmetic contract mismatch")
    for module_name, digest in manifest["source_sha256"].items():
        module = importlib.import_module(module_name)
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Hash route AITER source fingerprint mismatch")
    return api, manifest


def execution_record(logits):
    import triton

    _, manifest = implementation()
    properties = torch.cuda.get_device_properties(logits.device)
    return {
        "task": 30,
        "contract_id": CONTRACT_ID,
        "actual_backend": "aiter.triton.hash_route.fixed_slot",
        "forward_kernel": (
            "aiter.ops.triton._triton_kernels.moe.moe_routing.hash_route._hash_routing"
        ),
        "backward_kernel": API_MODULE + "._hash_route_vjp",
        "aiter_base_revision": AITER_REVISION,
        "source_sha256": dict(manifest["source_sha256"]),
        "patch_sha256": dict(manifest["patch_sha256"]),
        "gpu": properties.name,
        "arch": properties.gcnArchName,
        "device": str(logits.device),
        "torch": torch.__version__,
        "rocm": torch.version.hip,
        "triton": triton.__version__,
        "forward_config": {
            "block_m": 32,
            "block_n": 128,
            "num_warps": 8,
            "enable_fp_fusion": False,
        },
        "backward_config": {
            "slots_padded": 8,
            "num_warps": 1,
            "num_stages": 1,
            "enable_fp_fusion": False,
        },
        "denominator": "ascending_slot_sum.clamp_min(1e-20)",
        "slot_order": "tid2eid_column_order",
        "graph": False,
        "input_shape": list(logits.shape),
        "input_stride": list(logits.stride()),
        "input_dtype": "float32",
        "weight_dtype": "float32",
        "index_dtype": "int64",
        "kernel_launched": logits.shape[0] != 0,
    }
