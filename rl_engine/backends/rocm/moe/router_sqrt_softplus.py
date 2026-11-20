# SPDX-License-Identifier: Apache-2.0
"""AITER task-29 binding; all numerical kernels live in the companion patch."""

import functools
import hashlib
import importlib
import json
from pathlib import Path

import torch

from rl_engine.contracts.operators.router_sqrt_softplus import (
    AITER_CONTRACT_ID,
    AITER_REVISION,
    BLOCK_SIZE,
    CONTRACT_ID,
    LAUNCH,
)

API_MODULE = "aiter.ops.triton.moe.moe_routing.router_sqrt_softplus"
SCORE_MODULE = "aiter.ops.triton._triton_kernels.moe.moe_routing.topk"
WRAPPER_MODULE = "aiter.ops.triton.moe.moe_routing.topk"


@functools.lru_cache(maxsize=1)
def patch_manifest():
    directory = Path(__file__).with_name("patches")
    manifest = json.loads((directory / "router-sqrt-softplus.json").read_text())
    digest = hashlib.sha256((directory / "router-sqrt-softplus.patch").read_bytes()).hexdigest()
    if manifest["aiter_revision"] != AITER_REVISION or digest != manifest["patch_sha256"]:
        raise RuntimeError("task-29 AITER companion patch fingerprint changed")
    return manifest


@functools.lru_cache(maxsize=1)
def implementation():
    try:
        api = importlib.import_module(API_MODULE)
    except ImportError as exc:
        raise RuntimeError(
            "task 29 requires the pinned AITER companion patch; see "
            "docs/operators/router-sqrt-softplus.md"
        ) from exc
    if getattr(api, "ROUTER_SQRT_SOFTPLUS_CONTRACT_ID", None) != AITER_CONTRACT_ID:
        raise RuntimeError("AITER sqrt-softplus arithmetic contract does not match task 29")
    if api.BLOCK != BLOCK_SIZE or api.LAUNCH != LAUNCH:
        raise RuntimeError("AITER sqrt-softplus launch configuration changed")
    scores = importlib.import_module(SCORE_MODULE)
    wrapper = importlib.import_module(WRAPPER_MODULE)
    hashes = {
        name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for name, module in ((API_MODULE, api), (SCORE_MODULE, scores), (WRAPPER_MODULE, wrapper))
    }
    if hashes != patch_manifest()["source_sha256"]:
        raise RuntimeError("AITER task-29 sources do not match the pinned companion patch")
    return api, hashes


def execution_record(x):
    import triton

    _, hashes = implementation()
    props = torch.cuda.get_device_properties(x.device)
    return {
        "task": 29,
        "contract_id": CONTRACT_ID,
        "actual_backend": "aiter.triton.router_sqrt_softplus",
        "aiter_base_revision": AITER_REVISION,
        "aiter_companion_patch": "router-sqrt-softplus.patch",
        "aiter_patch_sha256": patch_manifest()["patch_sha256"],
        "source_sha256": dict(hashes),
        "forward_kernel": API_MODULE + "._router_sqrt_softplus_fwd",
        "backward_kernel": API_MODULE + "._router_sqrt_softplus_bwd",
        "gpu": props.name,
        "arch": props.gcnArchName,
        "device": str(x.device),
        "torch": torch.__version__,
        "rocm": torch.version.hip,
        "triton": triton.__version__,
        "config": {"block_size": BLOCK_SIZE, **LAUNCH},
        "graph": False,
        "input_dtype": "float32",
        "output_dtype": "float32",
        "input_shape": list(x.shape),
        "input_stride": list(x.stride()),
        "input_domain": "finite FP32",
        "kernel_launched": x.numel() != 0,
    }
