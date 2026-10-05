# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Finite-value checks with recoverable CUDA-graph replay errors."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

import torch
from torch import Tensor

from rl_engine.kernels.p2.errors import P2FailClosedError, P2Status

_CAPTURE_CHECKS: ContextVar[list[tuple[Tensor, str]] | None] = ContextVar(
    "p2_capture_finite_checks", default=None
)


def capture_finite_checks() -> list[tuple[Tensor, str]] | None:
    return _CAPTURE_CHECKS.get()


def require_finite(
    tensors: tuple[Tensor, ...],
    message: str,
    *,
    capture_checks: list[tuple[Tensor, str]] | None = None,
) -> None:
    capturing = False
    if tensors[0].is_cuda:
        with torch.cuda.device(tensors[0].device):
            capturing = torch.cuda.is_current_stream_capturing()
    checks = None
    if capturing:
        checks = capture_checks if capture_checks is not None else _CAPTURE_CHECKS.get()
    if capturing and checks is None:
        raise P2FailClosedError(
            P2Status.UNSUPPORTED_CAPABILITY,
            "T06 capture requires CheckedCUDAGraph.capture() for finite-value checks",
        )
    finite = torch.stack([torch.isfinite(t).all() for t in tensors]).all()
    if checks is not None:
        checks.append((finite, message))
    elif not finite.item():
        raise P2FailClosedError(P2Status.NON_FINITE, message)


class CheckedCUDAGraph(torch.cuda.CUDAGraph):
    """Check captured finite flags after replay, without device assertions."""

    def __init__(self) -> None:
        super().__init__()
        self._finite_checks: list[tuple[Tensor, str]] = []

    @contextmanager
    def capture(self, **kwargs):
        self._finite_checks = []
        token = _CAPTURE_CHECKS.set(self._finite_checks)
        try:
            with torch.cuda.graph(self, **kwargs):
                yield
        finally:
            _CAPTURE_CHECKS.reset(token)

    def replay(self) -> None:
        super().replay()
        for finite, message in self._finite_checks:
            if not finite.item():
                raise P2FailClosedError(P2Status.NON_FINITE, message)

    def reset(self) -> None:
        super().reset()
        self._finite_checks.clear()
