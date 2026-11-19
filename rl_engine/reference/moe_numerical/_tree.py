# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Arithmetic helper shared only by the fixed-order CPU references."""

import threading
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from rl_engine.contracts.operators.moe_numerical import MoEContractError

_SUBNORMAL_LOCK = threading.RLock()


@contextmanager
def preserve_subnormals() -> Iterator[None]:
    """Preserve FP32 subnormals without changing the caller's mode on exit."""
    with _SUBNORMAL_LOCK:
        # The setter reports success, not the previous mode. Probe input and
        # output flushing separately; the setter can only restore them together.
        tiny = torch.tensor(2.0**-65, dtype=torch.float32, device="cpu")
        flush_outputs = (tiny * tiny).view(torch.int32).item() == 0
        raw = torch.tensor(1, dtype=torch.int32, device="cpu")
        flush_inputs = raw.view(torch.float32).double().item() == 0
        if flush_inputs != flush_outputs:
            raise MoEContractError("cannot restore mixed CPU subnormal modes")
        if not flush_outputs:
            yield
            return
        if not torch.set_flush_denormal(False):
            raise MoEContractError("cannot disable CPU subnormal flushing")
        try:
            yield
        finally:
            if not torch.set_flush_denormal(True):
                raise MoEContractError("cannot restore CPU subnormal flushing")


def require_cpu(value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
        raise MoEContractError("MoE numerical references require CPU tensors")


def adjacent_tree(values: torch.Tensor) -> torch.Tensor:
    """Reduce dimension 1 using explicit FP32 pair adds, carrying odd nodes.

    No zero padding: adding +0 can change the sign of a -0 leaf. No in-place
    accumulation or implicit reduction primitive may change this grouping.
    """
    nodes = list(values.unbind(dim=1))
    while len(nodes) > 1:
        merged = [nodes[i] + nodes[i + 1] for i in range(0, len(nodes) - 1, 2)]
        if len(nodes) % 2:
            merged.append(nodes[-1])
        nodes = merged
    return nodes[0]
