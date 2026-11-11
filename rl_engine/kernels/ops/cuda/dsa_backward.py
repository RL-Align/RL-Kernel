# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Backward of the DSv4 sparse attention core (cuDNN-frontend ``dsa_bwd_sm90``).

The forward is FlashMLA's ``flash_mla_sparse_fwd`` (``dsv4_attention.sparse_attention``).
The pinned FlashMLA has no sparse backward on any arch. This wraps the SM90 CuTe-DSL
kernel from ``nvidia-cudnn-frontend``, which Megatron-Core and verl run for DSv4 on Hopper.
Its contract matches ours exactly:
- head dim 512 with K = V and one KV head (MQA);
- the sink folded into FlashMLA's KV-only LSE;
- per-query global index lists with ``-1`` padding and ``topk_length``.

**Not deterministic (temporary, user decision 2026-11-11).**
- ``dq`` is: each work tile owns one query and writes it once.
- ``dkv`` is accumulated with FP32 ``atomicAdd`` over every (query, slot) that selected a
  row, so its bytes vary run to run. ``deterministic=True`` exists only on SM100.
- The planned fix: fork ``dsa_bwd_sm90`` with int64 fixed-point dKV accumulation
  (``docs/architecture/dsv4-attention-backward-proposal.md``).

Fails closed unless the pinned ``nvidia-cudnn-frontend`` is installed;
``RL_KERNEL_CUDNN_DSA_ALLOW_UNPINNED=1`` accepts another version for development.
"""

from __future__ import annotations

import importlib.metadata
import os

import torch
from torch import Tensor

CUDNN_FRONTEND_VERSION = "1.31.0"
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
    if version != CUDNN_FRONTEND_VERSION and os.environ.get(ALLOW_UNPINNED_ENV) != "1":
        raise RuntimeError(
            f"nvidia-cudnn-frontend {version} is not the pinned {CUDNN_FRONTEND_VERSION} "
            f"(set {ALLOW_UNPINNED_ENV}=1 to accept it for development)"
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
    """``(dq BF16 [T, H, 512], dworkspace BF16 [S, 512])`` for ``sparse_attention``.

    Arguments as in the forward: ``q`` BF16 ``[T, H, 512]``, ``workspace`` BF16
    ``[S, 1, 512]`` or ``[S, 512]``, ``indices`` int32 ``[T, 1, W]`` or ``[T, W]`` (``-1``
    padding), ``lens`` int32 ``[T]``, ``sink`` FP32 ``[H]``; ``out`` / ``lse`` from the
    forward (``lse`` FP32 ``[T, H]``, KV-only, natural log); ``dout`` BF16 ``[T, H, 512]``.
    ``dworkspace`` is not deterministic (see the module docstring).
    """
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
