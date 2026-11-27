# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Backward of the DSv4 sparse attention core.
Note: Backward determinism is not required for now.
"""

from __future__ import annotations

import importlib.metadata
import os

import torch
from torch import Tensor

CUDNN_FRONTEND_VERSION = "1.31.0"
# Validated too: 1.32.3, which vLLM main's FlashInfer 0.7.4 requires (env rlk41, torch 2.14;
# the DSv4 sparse / attention backward and the DSv4.1 training tests pass, 2026-11-26).
CUDNN_FRONTEND_VALIDATED = (CUDNN_FRONTEND_VERSION, "1.32.3")
ALLOW_UNPINNED_ENV = "RL_KERNEL_CUDNN_DSA_ALLOW_UNPINNED"
DETERMINISTIC = False  # dkv uses FP32 atomics on SM90


def cudnn_dsa_available() -> bool:
    try:
        require_cudnn_dsa()
    except RuntimeError:
        return False
    return True


def require_cudnn_dsa():
    """The cuDNN-frontend SM90 DSA backward entry point, or ``RuntimeError``."""
    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise RuntimeError("the DSA backward requires NVIDIA CUDA")
    if torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("the DSv4 sparse attention backward is wired for SM90 only")
    try:
        version = importlib.metadata.version("nvidia-cudnn-frontend")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"nvidia-cudnn-frontend=={CUDNN_FRONTEND_VERSION} is required for the DSv4 "
            "attention backward"
        ) from exc
    if version not in CUDNN_FRONTEND_VALIDATED and os.environ.get(ALLOW_UNPINNED_ENV) != "1":
        raise RuntimeError(
            f"nvidia-cudnn-frontend {version} is not one of the validated "
            f"{CUDNN_FRONTEND_VALIDATED} (set {ALLOW_UNPINNED_ENV}=1 to accept it for development)"
        )
    from cudnn.deepseek_sparse_attention.sparse_attention_backward import (
        sparse_attention_backward_wrapper,
    )

    return sparse_attention_backward_wrapper


def sparse_attention_backward(
    q: Tensor,
    workspace: Tensor,
    indices: Tensor,
    lens: Tensor,
    sink: Tensor,
    scale: float,
    out: Tensor,
    lse: Tensor,
    dout: Tensor,
) -> tuple[Tensor, Tensor]:
    wrapper = require_cudnn_dsa()
    t, h, d = q.shape
    res = wrapper(
        q.contiguous(),
        workspace.reshape(-1, d).contiguous(),
        out.contiguous(),
        dout.contiguous(),
        lse.float().contiguous(),
        sink.float().contiguous(),
        indices.reshape(t, -1).to(torch.int32).contiguous(),
        softmax_scale=float(scale),
        topk_length=lens.to(torch.int32).contiguous(),
    )
    return res["dq"], res["dkv"]
