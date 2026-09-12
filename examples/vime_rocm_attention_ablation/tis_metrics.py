# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Mismatch metrics hook that deliberately leaves Vime's policy loss unchanged."""

from __future__ import annotations

import itertools
import os
import threading
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

SIDECAR_SCHEMA_VERSION = "rlkernel.vime_rocm_attention_mismatch_sidecar.v1"
SIDECAR_DIRECTORY_ENV = "RL_KERNEL_MISMATCH_SIDECAR_DIR"

_CALL_COUNTER = itertools.count()
_CALL_COUNTER_LOCK = threading.Lock()


def _cpu_vector(value: Any, *, label: str) -> torch.Tensor:
    try:
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    except Exception as exc:
        raise ValueError(f"{label} is not tensor-like") from exc
    return tensor.detach().to(device="cpu").reshape(-1).contiguous()


def _global_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    try:
        return int(os.environ.get("RANK", "0"))
    except ValueError as exc:
        raise ValueError("RANK must be an integer when torch.distributed is unavailable") from exc


def _write_sidecar(
    args: Any,
    *,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
) -> None:
    directory_value = os.environ.get(SIDECAR_DIRECTORY_ENV)
    if not directory_value:
        raise RuntimeError(
            f"{SIDECAR_DIRECTORY_ENV} must name an arm-local directory for mismatch evidence"
        )
    count = len(train_log_probs)
    fields = {
        "rollout_log_probs": rollout_log_probs,
        "loss_masks": loss_masks,
        "total_lengths": total_lengths,
        "response_lengths": response_lengths,
    }
    if any(len(value) != count for value in fields.values()):
        lengths = {"train_log_probs": count, **{key: len(value) for key, value in fields.items()}}
        raise ValueError(f"mismatch sidecar fields have different sample counts: {lengths}")

    rank = _global_rank()
    with _CALL_COUNTER_LOCK:
        call_index = next(_CALL_COUNTER)
    payload = {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "rank": rank,
        "call_index": call_index,
        "tensor_parallel_size": int(args.tensor_model_parallel_size),
        "context_parallel_size": int(args.context_parallel_size),
        "train_log_probs": [
            _cpu_vector(value, label="train_log_probs") for value in train_log_probs
        ],
        "rollout_log_probs": [
            _cpu_vector(value, label="rollout_log_probs") for value in rollout_log_probs
        ],
        "loss_masks": [_cpu_vector(value, label="loss_masks") for value in loss_masks],
        "total_lengths": [int(value) for value in total_lengths],
        "response_lengths": [int(value) for value in response_lengths],
    }

    directory = Path(directory_value)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"rank{rank:05d}.call{call_index:08d}.pt"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def metrics_only_tis(
    args: Any,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    **_: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    """Return TIS diagnostics without applying TIS weights or rejection masks.

    Current Vime requires ``--custom-tis-function-path`` whenever
    ``--get-mismatch-metrics`` is enabled.  Its built-in fallback multiplies
    ``pg_loss`` by clipped importance weights, which would introduce a second
    experimental variable into this Attention-only matrix.  This hook reports
    the same ratio diagnostics while returning both loss inputs verbatim.
    """

    training = torch.cat(train_log_probs, dim=0).detach()
    rollout = torch.cat(rollout_log_probs, dim=0).detach()
    if training.shape != rollout.shape:
        raise ValueError(
            "training and rollout log probabilities must have identical shapes: "
            f"{tuple(training.shape)} != {tuple(rollout.shape)}"
        )

    ratio = torch.exp(training - rollout)
    clipped = torch.clamp(ratio, min=args.tis_clip_low, max=args.tis_clip)
    metrics = {
        "tis": ratio,
        "tis_clipfrac": (clipped != ratio).to(dtype=ratio.dtype),
        "tis_abs": (ratio - 1).abs(),
    }
    _write_sidecar(
        args,
        train_log_probs=train_log_probs,
        rollout_log_probs=rollout_log_probs,
        loss_masks=loss_masks,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
    )
    return pg_loss, loss_masks, metrics
