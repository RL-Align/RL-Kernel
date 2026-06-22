# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.distributed as dist

DETERMINISTIC_NCCL_ENV = {
    "NCCL_ALGO": "Ring",
    "NCCL_PROTO": "Simple",
    "NCCL_MIN_NCHANNELS": "1",
    "NCCL_MAX_NCHANNELS": "1",
}


@dataclass(frozen=True)
class DeterministicAllReduceConfig:
    """Configuration for :func:`deterministic_all_reduce`."""

    mode: Literal["nccl_ring", "ordered_rank_fallback"] = "nccl_ring"
    op: Literal["sum", "mean"] = "sum"
    force_fp32_accumulation: bool = True
    async_op: bool = False
    group: Optional[dist.ProcessGroup] = None


def configure_deterministic_nccl_env(*, overwrite: bool = False) -> dict[str, Optional[str]]:
    """Configure the opt-in NCCL single-ring fast path environment.

    NCCL reads these variables during process-group initialization, so callers
    should run this helper before ``torch.distributed.init_process_group``. The
    helper returns the previous values so callers can log or restore them.

    Existing environment values are preserved by default. Pass
    ``overwrite=True`` to force the RL-Kernel deterministic NCCL settings.
    """

    if dist.is_available() and dist.is_initialized():
        warnings.warn(
            "configure_deterministic_nccl_env() was called after "
            "torch.distributed was initialized; NCCL may have already read its "
            "collective configuration.",
            RuntimeWarning,
            stacklevel=2,
        )

    previous: dict[str, Optional[str]] = {}
    for key, value in DETERMINISTIC_NCCL_ENV.items():
        previous[key] = os.environ.get(key)
        if overwrite or key not in os.environ:
            os.environ[key] = value
        elif os.environ[key] != value:
            warnings.warn(
                f"{key} is already set to {os.environ[key]!r}; leaving it unchanged. "
                f"Pass overwrite=True to set {value!r}.",
                RuntimeWarning,
                stacklevel=2,
            )
    return previous


def deterministic_all_reduce(
    tensor: torch.Tensor,
    config: Optional[DeterministicAllReduceConfig] = None,
) -> torch.Tensor:
    """Reduce ``tensor`` in place and return it.

    ``mode="nccl_ring"`` uses ``torch.distributed.all_reduce``. Call
    :func:`configure_deterministic_nccl_env` before process-group initialization
    when using NCCL and single-ring/single-channel behavior is desired.

    ``mode="ordered_rank_fallback"`` gathers rank inputs, accumulates them on
    rank 0 in ascending global rank order, and broadcasts the result. This path
    is slow and memory-heavy, but it gives a concrete reference order for smoke
    tests and unsupported hardware fallbacks.
    """

    cfg = config or DeterministicAllReduceConfig()
    _validate_config(tensor, cfg)

    if cfg.async_op:
        raise NotImplementedError("deterministic_all_reduce currently requires async_op=False")

    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable in this PyTorch build")

    if not dist.is_initialized():
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise RuntimeError(
                "torch.distributed is not initialized, but WORLD_SIZE indicates a "
                "multi-rank launch"
            )
        return tensor

    world_size = dist.get_world_size(group=cfg.group)
    if world_size == 1:
        return tensor

    if cfg.mode == "nccl_ring":
        return _all_reduce_fast_path(tensor, cfg, world_size)
    if cfg.mode == "ordered_rank_fallback":
        return _ordered_rank_fallback(tensor, cfg, world_size)
    raise ValueError(f"unsupported deterministic all-reduce mode: {cfg.mode!r}")


def _validate_config(tensor: torch.Tensor, cfg: DeterministicAllReduceConfig) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"tensor must be a torch.Tensor, got {type(tensor)!r}")
    if cfg.op not in {"sum", "mean"}:
        raise ValueError(f"unsupported reduction op: {cfg.op!r}")
    if cfg.mode not in {"nccl_ring", "ordered_rank_fallback"}:
        raise ValueError(f"unsupported deterministic all-reduce mode: {cfg.mode!r}")
    if cfg.op == "mean" and not (tensor.is_floating_point() or tensor.is_complex()):
        raise TypeError("op='mean' requires a floating-point or complex tensor")


def _all_reduce_fast_path(
    tensor: torch.Tensor,
    cfg: DeterministicAllReduceConfig,
    world_size: int,
) -> torch.Tensor:
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=cfg.group, async_op=False)
    if cfg.op == "mean":
        tensor.div_(world_size)
    return tensor


def _ordered_rank_fallback(
    tensor: torch.Tensor,
    cfg: DeterministicAllReduceConfig,
    world_size: int,
) -> torch.Tensor:
    send = tensor.detach().contiguous()
    gathered = [torch.empty_like(send) for _ in range(world_size)]
    dist.all_gather(gathered, send, group=cfg.group)

    rank = dist.get_rank(group=cfg.group)
    result = torch.empty_like(send)
    if rank == 0:
        accumulation_dtype = _accumulation_dtype(send, cfg.force_fp32_accumulation)
        reduced = gathered[0].to(dtype=accumulation_dtype)
        for rank_tensor in gathered[1:]:
            reduced.add_(rank_tensor.to(dtype=accumulation_dtype))
        if cfg.op == "mean":
            reduced.div_(world_size)
        result.copy_(reduced.to(dtype=send.dtype))

    dist.broadcast(result, src=0, group=cfg.group)
    tensor.copy_(result.view_as(tensor))
    return tensor


def _accumulation_dtype(tensor: torch.Tensor, force_fp32_accumulation: bool) -> torch.dtype:
    if not force_fp32_accumulation or not tensor.is_floating_point():
        return tensor.dtype
    if tensor.dtype == torch.float64:
        return torch.float64
    return torch.float32
