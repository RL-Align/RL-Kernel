# SPDX-License-Identifier: Apache-2.0
"""Exact companion AITER binding for task 31; no fallback or post-selection sort."""

import functools
import hashlib
import importlib
import json
from pathlib import Path

import torch

from rl_engine.contracts.operators.stable_topk6 import (
    AITER_CONTRACT_ID,
    AITER_REVISION,
    CONTRACT_ID,
    LAUNCH,
)

API_MODULE = "aiter.ops.triton.moe.moe_routing.stable_topk6"
SHARED_MODULE = "aiter.ops.triton._triton_kernels.moe.moe_routing.topk"
KERNEL_MODULE = "aiter.ops.triton._triton_kernels.moe.moe_routing.stable_topk6"


@functools.lru_cache(maxsize=1)
def implementation():
    directory = Path(__file__).with_name("patches")
    manifest = json.loads((directory / "stable-topk6.json").read_text())
    if manifest["aiter_revision"] != AITER_REVISION or (
        hashlib.sha256((directory / "stable-topk6.patch").read_bytes()).hexdigest()
        != manifest["patch_sha256"]
    ):
        raise RuntimeError("stable Top-6 companion patch identity mismatch")
    try:
        api = importlib.import_module(API_MODULE)
    except ImportError as exc:
        raise RuntimeError(
            "task 31 requires the pinned AITER stable Top-6 companion patch"
        ) from exc
    if getattr(api, "STABLE_TOPK6_CONTRACT_ID", None) != AITER_CONTRACT_ID or api.LAUNCH != LAUNCH:
        raise RuntimeError("stable Top-6 AITER arithmetic/launch contract mismatch")
    for name, expected in manifest["source_sha256"].items():
        module = importlib.import_module(name)
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != expected:
            raise RuntimeError("stable Top-6 AITER source fingerprint mismatch")
    shared = importlib.import_module(SHARED_MODULE)
    digest = hashlib.sha256(Path(shared.__file__).read_bytes()).hexdigest()
    profiles = manifest["shared_source_profiles"]
    matching = [name for name, expected in profiles.items() if digest == expected]
    if len(matching) != 1:
        raise RuntimeError("stable Top-6 shared AITER source has an unvalidated fingerprint")
    return api, manifest, matching[0], digest


def execution_record(scores):
    import triton

    _, manifest, profile, digest = implementation()
    properties = torch.cuda.get_device_properties(scores.device)
    return {
        "task": 31,
        "contract_id": CONTRACT_ID,
        "actual_backend": "aiter.triton.stable_topk6",
        "forward_kernel": KERNEL_MODULE + "._stable_topk6_fwd",
        "backward_kernel": KERNEL_MODULE + "._stable_topk6_vjp",
        "aiter_base_revision": AITER_REVISION,
        "shared_source_profile": profile,
        "source_sha256": {**manifest["source_sha256"], SHARED_MODULE: digest},
        "patch_sha256": manifest["patch_sha256"],
        "gpu": properties.name,
        "arch": properties.gcnArchName,
        "device": str(scores.device),
        "torch": torch.__version__,
        "rocm": torch.version.hip,
        "triton": triton.__version__,
        "config": {"block_m": 1, "block_n": 128, "padded_k": 8, **LAUNCH},
        "ordering": "score_descending.global_expert_id_ascending",
        "signed_zero_ranking": "equal",
        "payload": "original_fp32_bits",
        "input_shape": list(scores.shape),
        "input_stride": list(scores.stride()),
        "input_dtype": "float32",
        "output_dtype": "float32",
        "index_dtype": "int64",
        "graph": False,
        "kernel_launched": scores.shape[0] != 0,
        "forward_temporary_global_workspace_bytes": 0,
    }
