# SPDX-License-Identifier: Apache-2.0
"""INK-40: use the shared global-tile vocab logprob for Inkling Mini."""

import torch
import torch.distributed as dist

PADDED = 201024
VALID = 200058
TILES = 64


def vocab_bounds(tp: int) -> tuple[tuple[int, int], ...]:
    if tp not in (1, 2, 4, 8):
        raise ValueError("expected TP1/2/4/8")
    width = PADDED // tp
    return tuple((rank * width, (rank + 1) * width) for rank in range(tp))


def vocab_logprob(
    raw_logits: torch.Tensor, targets: torch.Tensor, *, platform: str, tp_group=None,
    role: str = "infer",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply cap exactly once to raw FP32 shard, then shared WS2 logprob.

    TP is vocabulary-only. Group members must supply identical target rows.
    TP1 uses the same 64 global tiles as TP2/4/8, not the INK-32 local path.
    """
    if platform not in ("cuda", "rocm") or not raw_logits.is_cuda:
        raise RuntimeError("requires CUDA or ROCm GPU logits")
    if (torch.version.hip is not None) != (platform == "rocm"):
        raise RuntimeError("platform mismatch")
    if tp_group is not None:
        if not dist.is_initialized() or dist.get_backend(tp_group) != "nccl":
            raise RuntimeError("requires NCCL/RCCL TP group")
        tp, rank = dist.get_world_size(tp_group), dist.get_rank(tp_group)
    else:
        tp, rank = 1, 0
    bounds = vocab_bounds(tp)
    if not 0 <= rank < tp:
        raise ValueError("not a member of the TP group")
    if raw_logits.ndim != 2 or raw_logits.dtype != torch.float32 or not raw_logits.shape[0]:
        raise ValueError("expected nonempty FP32 logits")
    if raw_logits.shape[1] != bounds[rank][1] - bounds[rank][0]:
        raise ValueError("unexpected vocabulary shard width")
    if targets.shape != raw_logits.shape[:1] or targets.device != raw_logits.device:
        raise ValueError("targets must match local token rows and device")
    if targets.dtype not in (torch.int32, torch.int64):
        raise TypeError("targets must be integer IDs")
    from rl_engine.backends.shared.triton.logprob.vocab_parallel_logp import (
        TritonVocabParallelLogprobOp,
    )
    from rl_engine.contracts.operators.logprob import (
        LogprobContract, MaskSpec, ReductionSpec, ShardingSpec,
    )

    contract = LogprobContract(
        role=role, dtype="fp32", mask=MaskSpec(targets.numel(), tuple((targets != -100).tolist())),
        sharding=ShardingSpec(rank, tp, bounds, VALID, PADDED), reduction=ReductionSpec(),
    )
    # Mask padded inputs before tanh so their backward cannot propagate NaNs.
    valid = torch.arange(bounds[rank][0], bounds[rank][1], device=raw_logits.device) < VALID
    safe = torch.where(valid, raw_logits, 0.0)
    capped = 30.0 * torch.tanh(safe / 30.0)
    return TritonVocabParallelLogprobOp().apply(
        capped, targets, contract=contract, tp_group=tp_group,
        num_vocab_tiles=TILES, validate=True, deterministic=True,
    )
