# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Mapping

import torch


def enabled() -> bool:
    return os.getenv("RL_KERNEL_ALIGNMENT_DIAGNOSTICS", "").lower() in {"1", "true", "yes", "on"}


def environment(directory: Path) -> dict[str, str]:
    """Explicitly pass diagnostics through Ray's environment boundary."""
    if not enabled():
        return {}
    values = {
        key: value for key, value in os.environ.items() if key.startswith("RL_KERNEL_ALIGNMENT_")
    }
    values["RL_KERNEL_ALIGNMENT_DIAGNOSTICS_DIR"] = str(directory)
    return values


def snapshot(value: Any) -> Any:
    """Copy before a later fused/in-place operation can mutate the evidence."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True).contiguous()
    if isinstance(value, Mapping):
        return {str(k): snapshot(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [snapshot(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported diagnostic value: {type(value).__name__}")


def fingerprint(value: torch.Tensor) -> str:
    tensor = snapshot(value)
    digest = hashlib.sha256(str((tuple(tensor.shape), tensor.dtype)).encode())
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def bitdiff(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    """Compare storage bits (including signed zero), rejecting nonfinite data."""
    left, right = snapshot(left), snapshot(right)
    if left.shape != right.shape or left.dtype != right.dtype:
        return {
            "comparable": False,
            "reason": "shape_or_dtype",
            "left": [list(left.shape), str(left.dtype)],
            "right": [list(right.shape), str(right.dtype)],
        }
    if not left.numel():
        return {"comparable": False, "reason": "empty_tensor"}
    lb = left.reshape(-1).view(torch.uint8).reshape(-1, left.element_size())
    rb = right.reshape(-1).view(torch.uint8).reshape_as(lb)
    mismatch = (lb != rb).any(dim=1)
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    indices = mismatch.nonzero().flatten()
    first = int(indices[0]) if indices.numel() else None
    coordinates = None
    if first is not None:
        coordinates = []
        remaining = first
        for size in reversed(left.shape):
            coordinates.insert(0, remaining % size)
            remaining //= size
    return {
        "comparable": True,
        "finite": finite,
        "equal": finite and not bool(mismatch.any()),
        "elements": left.numel(),
        "bitwise_mismatches": int(mismatch.sum()),
        "first_index": coordinates,
        "left_bits": lb[first].tolist() if first is not None else None,
        "right_bits": rb[first].tolist() if first is not None else None,
        "max_abs_diff": float((left.double() - right.double()).abs().max()) if finite else None,
    }


def atomic_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(snapshot(payload), temporary)
    os.replace(temporary, path)


def _stitch(records: list[Mapping[str, Any]], stage: str, selected: set[int]):
    """Gather logical token/features, verify overlapping TP/CP replicas bitwise."""
    rows: dict[int, dict[int, torch.Tensor]] = {}
    errors = []
    widths = set()
    dtypes = set()
    for record in records:
        value = record["tensors"].get(stage)
        if value is None:
            continue
        dtypes.add(value.dtype)
        if value.ndim != 2:
            errors.append(f"{stage}: expected normalized [rows, features] tensor")
            continue
        metadata = record["metadata"]
        widths.add(metadata.get("feature_sizes", {}).get(stage))
        positions = metadata["stage_positions"][stage].tolist()
        offset = int(metadata.get("feature_offsets", {}).get(stage, 0))
        if len(positions) != value.shape[0]:
            errors.append(f"{stage}: token map length differs from tensor")
            continue
        for index, position in enumerate(positions):
            if position not in selected:
                continue
            pieces = rows.setdefault(position, {})
            row = value[index].reshape(-1)
            if offset in pieces:
                if not bitdiff(pieces[offset], row).get("equal"):
                    errors.append(
                        f"{stage}: replica disagreement at token {position}, feature {offset}"
                    )
            else:
                pieces[offset] = row
    result = {}
    if len(dtypes) > 1:
        errors.append(f"{stage}: shard dtypes differ; implicit promotion is not bitwise evidence")
    for position, pieces in rows.items():
        cursor = 0
        joined = []
        for offset, value in sorted(pieces.items()):
            if offset != cursor:
                errors.append(f"{stage}: gap/overlap at token {position}, feature {cursor}")
                break
            cursor += value.numel()
            joined.append(value)
        else:
            if len(widths) != 1 or None in widths or cursor != next(iter(widths)):
                errors.append(
                    f"{stage}: incomplete or unknown feature coverage at token {position}"
                )
            result[position] = torch.cat(joined)
    return result, errors
