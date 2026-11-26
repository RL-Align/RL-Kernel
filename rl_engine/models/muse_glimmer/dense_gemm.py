# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Strict BF16 linear projection shared by Muse-Glimmer training and rollout."""

import functools
from typing import Literal

import torch

from rl_engine.models.muse_glimmer.operator_trace import MuseGlimmerSpec
from rl_engine.ops.gemm import fixed_leaf_linear


@functools.lru_cache(maxsize=1)
def _weight_shapes() -> frozenset[tuple[int, int]]:
    return frozenset(MuseGlimmerSpec.from_manifest().gemm_shapes().values())


def dense_gemm(
    x: torch.Tensor, weight: torch.Tensor, *, backend: Literal["cuda", "triton"]
) -> torch.Tensor:
    """Compute one independent projection from X[M,K] and native weight W[N,K]."""
    if backend not in ("cuda", "triton"):
        raise ValueError(f"unknown Muse-Glimmer dense GEMM backend {backend!r}")
    if x.ndim != 2 or weight.ndim != 2 or x.size(1) != weight.size(1):
        raise ValueError("Muse-Glimmer dense GEMM requires X[M,K] and W[N,K] with the same K")
    if x.size(0) == 0:
        raise ValueError("Muse-Glimmer dense GEMM requires positive M")
    if tuple(weight.shape) not in _weight_shapes():
        raise ValueError("Muse-Glimmer dense GEMM requires a single projection weight shape")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("Muse-Glimmer dense GEMM requires BF16 inputs and weights")
    if not x.is_cuda or x.device != weight.device:
        raise RuntimeError("Muse-Glimmer dense GEMM requires inputs on the same GPU")
    return fixed_leaf_linear(x, weight, backend=backend)
