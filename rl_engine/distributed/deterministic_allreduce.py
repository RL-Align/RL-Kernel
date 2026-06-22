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
    """Options for :func:`deterministic_all_reduce`."""

    mode: Literal["torch_all_reduce", "ordered_rank_reference"] = "torch_all_reduce"
    op: Literal["sum", "mean"] = "sum"
    force_fp32_accumulation: bool = True
    async_op: bool = False
    group: Optional[dist.ProcessGroup] = None


def configure_deterministic_nccl_env(*, overwrite: bool = False) -> dict[str, Optional[str]]:
    """Set best-effort NCCL ring settings before process-group init."""

    if dist.is_available() and dist.is_initialized():
        warnings.warn(
            "NCCL environment was configured after torch.distributed initialization",
            RuntimeWarning,
            stacklevel=2,
        )

    previous: dict[str, Optional[str]] = {}
    for key, value in DETERMINISTIC_NCCL_ENV.items():
        previous[key] = os.environ.get(key)
        if overwrite or key not in os.environ:
            os.environ[key] = value
            continue
        if os.environ[key] != value:
            warnings.warn(
                f"{key} is {os.environ[key]!r}; expected {value!r}",
                RuntimeWarning,
                stacklevel=2,
            )
    return previous


def deterministic_all_reduce(
    tensor: torch.Tensor,
    config: Optional[DeterministicAllReduceConfig] = None,
) -> torch.Tensor:
    """Reduce ``tensor`` in place and return it."""

    cfg = config or DeterministicAllReduceConfig()
    _validate(tensor, cfg)

    if cfg.async_op:
        raise NotImplementedError("async deterministic all-reduce is not implemented")
    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable")
    if not dist.is_initialized():
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise RuntimeError("torch.distributed is not initialized")
        return tensor

    world_size = dist.get_world_size(group=cfg.group)
    if world_size == 1:
        return tensor

    if cfg.mode == "torch_all_reduce":
        return _torch_all_reduce(tensor, cfg, world_size)
    return _ordered_rank_reference(tensor, cfg, world_size)


def _validate(tensor: torch.Tensor, cfg: DeterministicAllReduceConfig) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"tensor must be a torch.Tensor, got {type(tensor)!r}")
    if cfg.mode not in {"torch_all_reduce", "ordered_rank_reference"}:
        raise ValueError(f"unsupported all-reduce mode: {cfg.mode!r}")
    if cfg.op not in {"sum", "mean"}:
        raise ValueError(f"unsupported reduction op: {cfg.op!r}")
    if cfg.op == "mean" and not (tensor.is_floating_point() or tensor.is_complex()):
        raise TypeError("op='mean' requires a floating-point or complex tensor")


def _torch_all_reduce(
    tensor: torch.Tensor,
    cfg: DeterministicAllReduceConfig,
    world_size: int,
) -> torch.Tensor:
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=cfg.group, async_op=False)
    if cfg.op == "mean":
        tensor.div_(world_size)
    return tensor


def _ordered_rank_reference(
    tensor: torch.Tensor,
    cfg: DeterministicAllReduceConfig,
    world_size: int,
) -> torch.Tensor:
    send = tensor.detach().contiguous()
    gathered = [torch.empty_like(send) for _ in range(world_size)]
    dist.all_gather(gathered, send, group=cfg.group)

    result = torch.empty_like(send)
    if dist.get_rank(group=cfg.group) == 0:
        dtype = _accumulation_dtype(send, cfg.force_fp32_accumulation)
        reduced = gathered[0].to(dtype=dtype)
        for item in gathered[1:]:
            reduced.add_(item.to(dtype=dtype))
        if cfg.op == "mean":
            reduced.div_(world_size)
        result.copy_(reduced.to(dtype=send.dtype))

    dist.broadcast(result, src=_group_root_global_rank(cfg.group), group=cfg.group)
    tensor.copy_(result.view_as(tensor))
    return tensor


def _group_root_global_rank(group: Optional[dist.ProcessGroup]) -> int:
    if group is None:
        return 0
    try:
        return int(dist.get_global_rank(group, 0))
    except AttributeError as exc:
        raise RuntimeError(
            "custom process groups require torch.distributed.get_global_rank"
        ) from exc


def _accumulation_dtype(tensor: torch.Tensor, force_fp32: bool) -> torch.dtype:
    if not force_fp32 or not tensor.is_floating_point():
        return tensor.dtype
    if tensor.dtype == torch.float64:
        return torch.float64
    return torch.float32
