# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Observe native paged decode, comparing logical KV rather than allocator IDs."""

from __future__ import annotations

from typing import Callable, Sequence

import torch

from .events import EventCapture, LogicalTensor


def paged_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    block_table: torch.Tensor,
    lengths: torch.Tensor,
    *,
    sample_ids: Sequence[int],
    head_offset: int = 0,
    total_heads: int | None = None,
) -> dict:
    """Normalize actual [page, slot, head, dim] storage without recomputation.

    Supply explicit lossless views for other native layouts. Only active slots
    are read: allocator padding and uninitialized future tokens are not evidence.
    Head ownership is explicit, so different TP shardings can be compared.
    Packed/quantized caches require a separate adapter, not an implicit cast.
    """
    if key.ndim != 4 or key.shape != value.shape or key.dtype != value.dtype:
        raise ValueError("KV storage must share [page, slot, head, dim] shape/dtype")
    if key.device != value.device or key.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        raise ValueError("KV storage must be floating tensors on the same device")
    pages, page_size, heads, dim = key.shape
    if min(key.shape) < 1:
        raise ValueError("empty physical KV storage")
    if (
        block_table.ndim != 2
        or lengths.ndim != 1
        or lengths.numel() != block_table.shape[0]
        or len(sample_ids) != lengths.numel()
        or not sample_ids
        or len(set(sample_ids)) != len(sample_ids)
        or any(type(s) is not int or s < 0 for s in sample_ids)
        or block_table.dtype not in (torch.int32, torch.int64)
        or lengths.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("invalid sample/block/length metadata")
    total_heads = heads if total_heads is None else total_heads
    if (
        type(total_heads) is not int
        or type(head_offset) is not int
        or head_offset < 0
        or head_offset + heads > total_heads
    ):
        raise ValueError("invalid KV head ownership")
    table, sizes = block_table.detach().cpu().tolist(), lengths.detach().cpu().tolist()
    rows, keys, values = [], [], []
    for sample, blocks, size in zip(sample_ids, table, sizes, strict=True):
        if not 0 < size <= len(blocks) * page_size:
            raise ValueError("active KV length is empty or exceeds its block table")
        active = blocks[: (size + page_size - 1) // page_size]
        if any(not 0 <= block < pages for block in active) or len(set(active)) != len(active):
            raise ValueError("invalid/aliased active pages within a sequence")
        page_ids = torch.tensor(
            [blocks[position // page_size] for position in range(size)],
            device=key.device,
        )
        slots = torch.arange(size, device=key.device) % page_size
        keys.append(key[page_ids, slots].reshape(size, heads * dim))
        values.append(value[page_ids, slots].reshape(size, heads * dim))
        rows.extend((sample, position) for position in range(size))
    return {
        "key": LogicalTensor(torch.cat(keys), rows, head_offset * dim, total_heads * dim),
        "value": LogicalTensor(torch.cat(values), rows, head_offset * dim, total_heads * dim),
        "lengths": {str(s): n for s, n in zip(sample_ids, sizes, strict=True)},
        "head_dim": dim,
        "total_heads": total_heads,
    }


class DecodeCapture:
    """Call an original native decode once, observing cache before AND after it."""

    @staticmethod
    def boundaries(prefix: str = "decode") -> list[dict]:
        return [
            {"id": f"{prefix}.{stage}", "module": prefix, "stage": stage}
            for stage in ("input", "output")
        ]

    def __init__(self, capture: EventCapture, cache: Callable[[], dict], *, prefix="decode"):
        self.capture, self.cache, self.prefix = capture, cache, prefix
        self.calls = 0

    def run(
        self,
        forward: Callable,
        *args,
        query_rows: Sequence[tuple[int, int]],
        frame: str | None = None,
        inputs: Callable | None = None,
        outputs: Callable | None = None,
        **kwargs,
    ):
        frame = f"decode:{self.calls}" if frame is None else frame
        self.calls += 1
        try:
            if (
                not query_rows
                or any(
                    len(row) != 2 or any(type(i) is not int or i < 0 for i in row)
                    for row in query_rows
                )
                or len(set(query_rows)) != len(query_rows)
            ):
                raise ValueError("declare unique actual decode sample/position rows")
            parent = self.capture.record(
                f"{self.prefix}.input",
                {
                    "query_rows": list(query_rows),
                    "arguments": (
                        inputs(args, kwargs) if inputs else {"args": args, "kwargs": kwargs}
                    ),
                    "cache": self.cache(),
                },
                frame=frame,
            )
            result = forward(*args, **kwargs)
            self.capture.record(
                f"{self.prefix}.output",
                {"result": outputs(result) if outputs else result, "cache": self.cache()},
                frame=frame,
                inputs=[parent],
            )
            return result
        except Exception as exc:
            self.capture.errors.append(f"decode execution failed: {exc}")
            raise
