# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""The task-9 AITER GEMM interface used by the task-28 BF16 gate.

No tuning database, persistent scheduling, split-K, bias, or activation. This
is a separate FP32-output contract from RLK's BF16-node deterministic GEMM.
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import inspect
from pathlib import Path

import torch

CONTRACT_ID = "rlkernel.aiter_fixed_bf16_fp32.v1"
KERNEL_MODULE = "aiter.ops.triton._triton_kernels.gemm.basic.gemm_a16w16"
WRAPPER_MODULE = "aiter.ops.triton.gemm.basic.gemm_a16w16"


def fixed_gemm_config(k: int) -> dict:
    """Only K controls the K span; M never controls arithmetic or launch options."""
    return {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 1,
        "NUM_KSPLIT": 1,
        "SPLITK_BLOCK_SIZE": ((k + 31) // 32) * 32,
        "cache_modifier": ".cg",
        "num_warps": 4,
        "num_stages": 2,
        "matrix_instr_nonkdim": 16,
        "kpack": 1,
    }


@functools.lru_cache(maxsize=1)
def _implementation():
    # Import failure is intentional: never fall back to torch.mm or RLK BF16 GEMM.
    wrapper = importlib.import_module(WRAPPER_MODULE)
    kernel = importlib.import_module(KERNEL_MODULE)
    gemm = wrapper.gemm_a16w16
    parameters = inspect.signature(gemm).parameters
    if not {"dtype", "config", "skip_reduce"} <= parameters.keys():
        raise RuntimeError("AITER gemm_a16w16 lacks the fixed FP32-output interface")
    # The older Triton-only API and the RFC-pinned multi-backend API are both
    # explicit profiles. In the latter, prohibit automatic backend selection.
    options = {"backend": "triton", "persistent": False} if "backend" in parameters else {}
    sources = {
        name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for name, module in ((WRAPPER_MODULE, wrapper), (KERNEL_MODULE, kernel))
    }
    return gemm, options, sources


def fixed_gemm_execution(x: torch.Tensor) -> dict:
    _, options, sources = _implementation()
    from aiter._version import __version__ as aiter_version
    import triton

    props = torch.cuda.get_device_properties(x.device)
    return {
        "contract_id": CONTRACT_ID,
        "actual_backend": "aiter.triton.gemm_a16w16",
        "kernel": KERNEL_MODULE + "._gemm_a16_w16_kernel",
        "api_profile": "explicit_triton" if options else "legacy_triton_only",
        "source_sha256": dict(sources),
        "gpu": props.name,
        "arch": props.gcnArchName,
        "device": str(x.device),
        "torch": torch.__version__,
        "rocm": torch.version.hip,
        "aiter": aiter_version,
        "triton": triton.__version__,
        "config": fixed_gemm_config(x.shape[1]),
        "output_dtype": "float32",
        "bias": False,
        "activation": None,
        "graph": False,
        "workspace_bytes": 0,
    }


def aiter_fixed_linear_fp32(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Raw forward: BF16 [M,K] @ BF16 [N,K].T -> FP32 [M,N]."""
    gemm, options, _ = _implementation()
    with torch.cuda.device(x.device):
        result = gemm(
            x,
            weight,
            dtype=torch.float32,
            config=fixed_gemm_config(x.shape[1]),
            bias=None,
            activation=None,
            skip_reduce=False,
            **options,
        )
    if result.dtype != torch.float32 or result.shape != (x.shape[0], weight.shape[0]):
        raise RuntimeError("AITER violated the gate FP32-output contract")
    return result
