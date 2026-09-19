# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import os
from typing import Any, Optional

import torch

from rl_engine.distributed.collectives import collective_for_group

# Backward token-chunk target: process at most this many ``[chunk, V]`` logit
# elements per cuBLAS step so peak backward memory stays ~``chunk*V`` instead of
# ``N*V``.
BWD_CHUNK_ELEMS = 1 << 24
_LOW_PRECISION_DTYPES = (torch.float16, torch.bfloat16)
_TP_VOCAB_PARTITION_CACHE: dict[tuple[int, str, int, int, Optional[int], int], int] = {}


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _validate_tp_targets_enabled() -> bool:
    return _env_flag("RL_KERNEL_LINEAR_LOGP_VALIDATE_TP_TARGETS") or _env_flag(
        "VIME_RL_KERNEL_VALIDATE_TP_TARGETS"
    )


def _use_fp32_matmul(*tensors: torch.Tensor) -> bool:
    return any(tensor.dtype in _LOW_PRECISION_DTYPES for tensor in tensors)


def _matmul_operand(tensor: torch.Tensor, use_fp32: bool) -> torch.Tensor:
    return tensor.float() if use_fp32 else tensor


def _linear_logits(
    hidden_2d: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    *,
    use_fp32: bool,
) -> torch.Tensor:
    logits = torch.matmul(
        _matmul_operand(hidden_2d, use_fp32),
        _matmul_operand(weight, use_fp32).t(),
    )
    if bias is not None:
        logits = logits + _matmul_operand(bias, use_fp32)
    return logits


def _require_distributed_initialized():
    import torch.distributed as dist

    if not dist.is_available():
        raise RuntimeError("tensor-parallel linear_logp requires torch.distributed.")
    if not dist.is_initialized():
        raise RuntimeError("tensor-parallel linear_logp requires an initialized process group.")
    return dist


def _tensor_parallel_world_size(tp_group: Any) -> int:
    if tp_group is None:
        return 1
    dist = _require_distributed_initialized()
    return dist.get_world_size(group=tp_group)


def should_use_tensor_parallel_linear_logp(
    tp_group: Any,
    vocab_start_index: int,
    global_vocab_size: Optional[int],
    local_vocab_size: int,
) -> bool:
    """Whether a linear_logp call describes a vocab-parallel weight shard."""
    explicit_tp = tp_group is not None or vocab_start_index != 0 or global_vocab_size is not None
    if local_vocab_size <= 0 and not explicit_tp:
        raise ValueError("lm_head_weight must contain at least one vocab row.")
    if not explicit_tp:
        return False

    world_size = _tensor_parallel_world_size(tp_group)
    if local_vocab_size <= 0 and world_size <= 1:
        raise ValueError("lm_head_weight must contain at least one vocab row.")
    if world_size <= 1:
        if vocab_start_index != 0:
            raise ValueError("vocab_start_index requires a tensor-parallel group.")
        if global_vocab_size is not None and int(global_vocab_size) != local_vocab_size:
            raise ValueError(
                "global_vocab_size differs from the local vocab size, but no "
                "multi-rank tensor-parallel group was provided."
            )
        return False
    return True


def _validate_tp_vocab_partition(
    *,
    tp_group: Any,
    device: torch.device,
    vocab_start_index: int,
    local_vocab_size: int,
    global_vocab_size: Optional[int],
) -> int:
    dist = _require_distributed_initialized()
    local_end = vocab_start_index + local_vocab_size
    local_range = torch.tensor([vocab_start_index, local_end], device=device, dtype=torch.long)
    ranges_t = [torch.empty_like(local_range) for _ in range(dist.get_world_size(tp_group))]
    dist.all_gather(ranges_t, local_range, group=tp_group)

    ranges = sorted((int(r[0].item()), int(r[1].item())) for r in ranges_t)
    expected_start = 0
    for start, end in ranges:
        if end <= start:
            raise ValueError(f"invalid TP vocab shard range [{start}, {end}).")
        if start != expected_start:
            raise ValueError(
                "TP vocab shards must form a contiguous [0, V) partition; " f"got ranges={ranges}."
            )
        expected_start = end

    covered_vocab_size = expected_start
    global_size = torch.tensor(
        [
            1 if global_vocab_size is not None else 0,
            0 if global_vocab_size is None else int(global_vocab_size),
        ],
        device=device,
        dtype=torch.long,
    )
    global_sizes_t = [torch.empty_like(global_size) for _ in range(dist.get_world_size(tp_group))]
    dist.all_gather(global_sizes_t, global_size, group=tp_group)
    invalid_sizes = [
        int(value[1].item())
        for value in global_sizes_t
        if int(value[0].item()) and int(value[1].item()) != covered_vocab_size
    ]
    if invalid_sizes:
        raise ValueError(
            "global_vocab_size must match the TP vocab partition size: "
            f"got {invalid_sizes[0]}, covered {covered_vocab_size}."
        )
    return covered_vocab_size if global_vocab_size is None else int(global_vocab_size)


def _validate_tp_vocab_partition_cached(
    *,
    tp_group: Any,
    device: torch.device,
    vocab_start_index: int,
    local_vocab_size: int,
    global_vocab_size: Optional[int],
) -> int:
    dist = _require_distributed_initialized()
    world_size = dist.get_world_size(group=tp_group)
    key = (
        id(tp_group),
        str(device),
        int(vocab_start_index),
        int(local_vocab_size),
        None if global_vocab_size is None else int(global_vocab_size),
        int(world_size),
    )
    cached = _TP_VOCAB_PARTITION_CACHE.get(key)
    if cached is not None:
        return cached
    covered = _validate_tp_vocab_partition(
        tp_group=tp_group,
        device=device,
        vocab_start_index=vocab_start_index,
        local_vocab_size=local_vocab_size,
        global_vocab_size=global_vocab_size,
    )
    _TP_VOCAB_PARTITION_CACHE[key] = int(covered)
    return int(covered)


def _validate_even_tp_vocab_partition_local(
    *,
    tp_group: Any,
    vocab_start_index: int,
    local_vocab_size: int,
    global_vocab_size: Optional[int],
) -> int:
    """Validate the strict equal-shard contract without a data-plane collective."""

    dist = _require_distributed_initialized()
    rank = dist.get_rank(group=tp_group)
    world_size = dist.get_world_size(group=tp_group)
    expected_global_size = local_vocab_size * world_size
    expected_start = rank * local_vocab_size
    if local_vocab_size <= 0:
        raise ValueError("strict TP linear_logp requires a non-empty vocab shard.")
    if vocab_start_index != expected_start:
        raise ValueError(
            "strict TP linear_logp requires equal rank-ordered vocab shards: "
            f"rank={rank} expected vocab_start_index={expected_start}, "
            f"got {vocab_start_index}."
        )
    if global_vocab_size is not None and int(global_vocab_size) != expected_global_size:
        raise ValueError(
            "strict TP linear_logp global_vocab_size must equal "
            f"local_vocab_size * world_size={expected_global_size}, "
            f"got {global_vocab_size}."
        )
    return expected_global_size


def _validate_global_targets(
    target_1d: torch.Tensor,
    global_vocab_size: int,
    tp_group: Any = None,
) -> None:
    invalid = (target_1d < 0) | (target_1d >= global_vocab_size)
    local_invalid = bool(invalid.any().item())
    if tp_group is not None:
        dist = _require_distributed_initialized()
        invalid_flag = torch.tensor(int(local_invalid), device=target_1d.device, dtype=torch.int32)
        dist.all_reduce(invalid_flag, op=dist.ReduceOp.MAX, group=tp_group)
        if target_1d.numel():
            min_target = torch.tensor(
                int(target_1d.min().item()), device=target_1d.device, dtype=torch.long
            )
            max_target = torch.tensor(
                int(target_1d.max().item()), device=target_1d.device, dtype=torch.long
            )
        else:
            min_target = torch.tensor(global_vocab_size, device=target_1d.device, dtype=torch.long)
            max_target = torch.tensor(-1, device=target_1d.device, dtype=torch.long)
        dist.all_reduce(min_target, op=dist.ReduceOp.MIN, group=tp_group)
        dist.all_reduce(max_target, op=dist.ReduceOp.MAX, group=tp_group)
        local_invalid = bool(invalid_flag.item())
        t_min, t_max = int(min_target.item()), int(max_target.item())
    elif local_invalid:
        t_min, t_max = int(target_1d.min().item()), int(target_1d.max().item())
    if local_invalid:
        raise ValueError(
            f"target_ids out of range: expected [0, {global_vocab_size - 1}], "
            f"got [{t_min}, {t_max}]. Mask or filter padding / ignore-index values "
            "(e.g. -100) before this op."
        )


def _assert_global_targets_async(
    target_1d: torch.Tensor,
    global_vocab_size: int,
) -> None:
    """Range-check targets without synchronizing the CUDA hot path."""
    torch._assert_async(
        ((target_1d >= 0) & (target_1d < global_vocab_size)).all(),
        f"target_ids out of range: expected [0, {global_vocab_size - 1}]. "
        "Mask or filter padding / ignore-index values (e.g. -100) before this op.",
    )


def _chunked_local_linear_logp_stats(
    hidden_2d: torch.Tensor,
    weight: torch.Tensor,
    target_1d: torch.Tensor,
    bias_t: torch.Tensor,
    *,
    has_bias: bool,
    vocab_start_index: int,
    validate_owner_count: bool = False,
    chunk_elems: int = BWD_CHUNK_ELEMS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    n = hidden_2d.size(0)
    local_vocab = weight.size(0)
    device = hidden_2d.device

    local_max = torch.full((n,), -torch.inf, device=device, dtype=torch.float32)
    local_sum = torch.zeros(n, device=device, dtype=torch.float32)
    local_target_logit = torch.zeros(n, device=device, dtype=torch.float32)
    owner_count = torch.zeros(n, device=device, dtype=torch.int32) if validate_owner_count else None
    rows = torch.arange(n, device=device)
    use_fp32 = _use_fp32_matmul(hidden_2d, weight)

    vocab_chunk = max(1, min(local_vocab, chunk_elems // max(n, 1)))
    for v0 in range(0, local_vocab, vocab_chunk):
        v1 = min(v0 + vocab_chunk, local_vocab)
        logits = _linear_logits(
            hidden_2d,
            weight[v0:v1],
            bias_t[v0:v1] if has_bias else None,
            use_fp32=use_fp32,
        )
        logits_f = logits.float()

        tile_max = logits_f.max(dim=-1).values
        new_max = torch.maximum(local_max, tile_max)
        local_sum = local_sum * torch.exp(local_max - new_max) + torch.exp(
            logits_f - new_max.unsqueeze(1)
        ).sum(dim=-1)
        local_max = new_max

        global_v0 = vocab_start_index + v0
        global_v1 = vocab_start_index + v1
        owns_target = (target_1d >= global_v0) & (target_1d < global_v1)
        safe_local_idx = (target_1d - global_v0).clamp(0, max(v1 - v0 - 1, 0)).long()
        tile_target_logit = logits_f[rows, safe_local_idx]
        local_target_logit = torch.where(owns_target, tile_target_logit, local_target_logit)
        if owner_count is not None:
            owner_count += owns_target.to(torch.int32)

    return local_max, local_sum, local_target_logit, owner_count


def _merge_tp_local_logp(
    local_lse: torch.Tensor,
    local_target_logit: torch.Tensor,
    *,
    tp_group: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge one or more contiguous vocab summaries in global vocab order.

    The leading optional summary dimension lets a coarser physical TP rank
    expose the same virtual vocab shards as a finer TP layout.  Flattening
    physical rank then local summary preserves ascending global-vocab order.
    The historical one-summary-per-rank path keeps its original tensor layout.
    """
    dist = _require_distributed_initialized()
    world_size = dist.get_world_size(group=tp_group)
    if local_lse.shape != local_target_logit.shape:
        raise ValueError("local LSE and selected-logit summaries must have matching shapes")
    if local_lse.ndim == 1:
        summaries_per_rank = 1
        local_lse_summaries = local_lse.unsqueeze(0)
        local_target_summaries = local_target_logit.unsqueeze(0)
    elif local_lse.ndim == 2:
        summaries_per_rank = local_lse.size(0)
        if summaries_per_rank <= 0:
            raise ValueError("linear_logp requires at least one local vocab summary")
        local_lse_summaries = local_lse
        local_target_summaries = local_target_logit
    else:
        raise ValueError("linear_logp summaries must be [tokens] or [summaries, tokens]")

    if world_size <= 1 and summaries_per_rank == 1:
        global_lse = local_lse
        return local_target_logit - global_lse, global_lse

    local_stats = torch.stack(
        (local_lse_summaries, local_target_summaries), dim=0
    ).contiguous()
    if world_size <= 1:
        gathered = local_stats.unsqueeze(0)
    elif local_stats.is_cuda:
        collective = collective_for_group(
            tp_group,
            min_size_bytes=local_stats.numel() * local_stats.element_size(),
            device=local_stats.device,
        )
        if collective is None:
            raise RuntimeError("tensor-parallel linear_logp requires an RL-Kernel collective")
        gathered = collective.all_gather(
            local_stats,
            validate_signature=False,
        ).reshape(world_size, *local_stats.shape)
    else:
        gathered = torch.empty(
            (world_size, *local_stats.shape),
            device=local_stats.device,
            dtype=local_stats.dtype,
        )
        try:
            dist.all_gather_into_tensor(gathered, local_stats, group=tp_group)
        except (AttributeError, RuntimeError):
            chunks = [torch.empty_like(local_stats) for _ in range(world_size)]
            dist.all_gather(chunks, local_stats, group=tp_group)
            gathered = torch.stack(chunks, dim=0)

    # [contract] Explicit rank-ordered reduction trees. ``torch.logsumexp`` and
    # ``Tensor.sum`` own their internal reduction order, which is not part of
    # any contract; the merge order is pinned here instead:
    #   * global max: successive maxima in ascending rank order (order-free),
    #   * rescaled sumexp: ascending-rank sequential chain,
    #   * target logit: ascending-rank sequential chain (exactly one rank
    #     owns a nonzero owner value, so the chain is exact).
    local_lse = gathered[:, 0, :, :].reshape(
        world_size * summaries_per_rank, local_lse_summaries.size(-1)
    )
    local_zt = gathered[:, 1, :, :].reshape(
        world_size * summaries_per_rank, local_target_summaries.size(-1)
    )
    summary_count = local_lse.size(0)
    global_max = local_lse[0].clone()
    for summary in range(1, summary_count):
        global_max = torch.maximum(global_max, local_lse[summary])
    sumexp = torch.exp(local_lse[0] - global_max)
    for summary in range(1, summary_count):
        sumexp = sumexp + torch.exp(local_lse[summary] - global_max)
    global_lse = global_max + torch.log(sumexp)
    target_logit = local_zt[0].clone()
    for summary in range(1, summary_count):
        target_logit = target_logit + local_zt[summary]
    return target_logit - global_lse, global_lse


def _deterministic_tp_all_reduce_(tensor: torch.Tensor, tp_group: Any) -> torch.Tensor:
    if not tensor.is_cuda:
        dist = _require_distributed_initialized()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=tp_group)
        return tensor
    collective = collective_for_group(
        tp_group,
        min_size_bytes=tensor.numel() * tensor.element_size(),
        device=tensor.device,
    )
    if collective is None:
        raise RuntimeError("tensor-parallel linear_logp requires an RL-Kernel collective")
    return collective.all_reduce(
        tensor.contiguous(),
        out=tensor,
        validate_signature=False,
    )


def tensor_parallel_linear_logp_backward(
    grad_logp: torch.Tensor,
    hidden_2d: torch.Tensor,
    weight: torch.Tensor,
    bias_t: torch.Tensor,
    target_1d: torch.Tensor,
    global_lse: torch.Tensor,
    *,
    has_bias: bool,
    lead_shape,
    hidden_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    bias_dtype,
    vocab_start_index: int,
    tp_group: Any,
    compute_grad_hidden: bool = True,
    compute_grad_weight: bool = True,
    compute_grad_bias: bool = True,
    chunk_elems: int = BWD_CHUNK_ELEMS,
):
    n, d = hidden_2d.shape
    local_vocab = weight.shape[0]
    dt = weight.dtype
    g = grad_logp.reshape(-1).to(torch.float32)

    grad_h = torch.empty_like(hidden_2d, dtype=torch.float32) if compute_grad_hidden else None
    grad_w = (
        torch.zeros(local_vocab, d, device=weight.device, dtype=torch.float32)
        if compute_grad_weight
        else None
    )
    grad_b = (
        torch.zeros(local_vocab, device=weight.device, dtype=torch.float32)
        if has_bias and compute_grad_bias
        else None
    )
    use_fp32 = _use_fp32_matmul(hidden_2d, weight)

    chunk = max(1, min(n, chunk_elems // local_vocab))
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        x = hidden_2d[i0:i1]
        logits = _linear_logits(
            x,
            weight,
            bias_t if has_bias else None,
            use_fp32=use_fp32,
        )

        dz = -torch.exp(logits.float() - global_lse[i0:i1].unsqueeze(1))
        local_idx = target_1d[i0:i1] - int(vocab_start_index)
        owns_target = (local_idx >= 0) & (local_idx < local_vocab)
        rows = torch.arange(i1 - i0, device=dz.device)
        safe_local_idx = local_idx.clamp(0, max(local_vocab - 1, 0)).long()
        dz[rows, safe_local_idx] += owns_target.to(dz.dtype)
        dz *= g[i0:i1].unsqueeze(1)

        if use_fp32:
            if grad_h is not None:
                grad_h[i0:i1] = torch.matmul(dz, weight.float()).float()
            if grad_w is not None:
                grad_w += torch.matmul(dz.t(), x.float()).float()
        else:
            dz_dt = dz.to(dt)
            if grad_h is not None:
                grad_h[i0:i1] = torch.matmul(dz_dt, weight).float()
            if grad_w is not None:
                grad_w += torch.matmul(dz_dt.t(), x).float()
        if grad_b is not None:
            grad_b += dz.sum(0)

    grad_hidden = None
    if grad_h is not None:
        _deterministic_tp_all_reduce_(grad_h, tp_group)
        grad_hidden = grad_h.to(hidden_dtype).reshape((*tuple(lead_shape), d))
    grad_weight = grad_w.to(weight_dtype) if grad_w is not None else None
    grad_bias = grad_b.to(bias_dtype) if grad_b is not None else None
    return grad_hidden, grad_weight, grad_bias


class _TensorParallelLinearLogpFunction(torch.autograd.Function):
    """Autograd path for vocab-sharded LM-head tensor parallelism."""

    @staticmethod
    def forward(
        ctx,
        hidden,
        lm_head_weight,
        bias,
        target_ids,
        vocab_start_index,
        global_vocab_size,
        tp_group,
    ):
        hidden_2d = hidden.reshape(-1, hidden.size(-1)).contiguous()
        weight = lm_head_weight.contiguous()
        target_1d = (
            target_ids.reshape(-1).to(device=hidden_2d.device, dtype=torch.long).contiguous()
        )
        bias_t = bias.contiguous() if bias is not None else hidden_2d
        vocab_start_index = int(vocab_start_index)
        global_vocab_size = _validate_tp_vocab_partition_cached(
            tp_group=tp_group,
            device=hidden_2d.device,
            vocab_start_index=vocab_start_index,
            local_vocab_size=weight.size(0),
            global_vocab_size=global_vocab_size,
        )
        validate_tp_targets = _validate_tp_targets_enabled()
        if validate_tp_targets:
            _validate_global_targets(target_1d, global_vocab_size, tp_group)

        local_max, local_sum, local_target_logit, owner_count = _chunked_local_linear_logp_stats(
            hidden_2d,
            weight,
            target_1d,
            bias_t,
            has_bias=bias is not None,
            vocab_start_index=vocab_start_index,
            validate_owner_count=validate_tp_targets,
        )

        if owner_count is not None:
            dist = _require_distributed_initialized()
            global_owner_count = owner_count.clone()
            dist.all_reduce(global_owner_count, op=dist.ReduceOp.SUM, group=tp_group)
            if bool((global_owner_count != 1).any().item()):
                raise ValueError(
                    "target_ids must be covered by exactly one TP vocab shard; check "
                    "vocab_start_index and global_vocab_size."
                )

        local_lse = local_max + torch.log(local_sum)
        log_prob, lse = _merge_tp_local_logp(local_lse, local_target_logit, tp_group=tp_group)
        ctx.save_for_backward(hidden_2d, weight, bias_t, target_1d, lse)
        ctx.has_bias = bias is not None
        ctx.lead_shape = hidden.shape[:-1]
        ctx.hidden_dtype = hidden.dtype
        ctx.weight_dtype = lm_head_weight.dtype
        ctx.bias_dtype = bias.dtype if bias is not None else None
        ctx.vocab_start_index = vocab_start_index
        ctx.tp_group = tp_group
        return log_prob.reshape(hidden.shape[:-1])

    @staticmethod
    def backward(ctx, grad_logp):
        hidden_2d, weight, bias_t, target_1d, lse = ctx.saved_tensors
        grad_hidden, grad_weight, grad_bias = tensor_parallel_linear_logp_backward(
            grad_logp,
            hidden_2d,
            weight,
            bias_t,
            target_1d,
            lse,
            has_bias=ctx.has_bias,
            lead_shape=ctx.lead_shape,
            hidden_dtype=ctx.hidden_dtype,
            weight_dtype=ctx.weight_dtype,
            bias_dtype=ctx.bias_dtype,
            vocab_start_index=ctx.vocab_start_index,
            tp_group=ctx.tp_group,
            compute_grad_hidden=ctx.needs_input_grad[0],
            compute_grad_weight=ctx.needs_input_grad[1],
            compute_grad_bias=ctx.needs_input_grad[2],
        )
        return grad_hidden, grad_weight, grad_bias, None, None, None, None


def tensor_parallel_linear_logp(
    hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    target_ids: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    tp_group: Any,
    vocab_start_index: int = 0,
    global_vocab_size: Optional[int] = None,
) -> torch.Tensor:
    if hidden.shape[:-1] != target_ids.shape:
        raise ValueError(
            f"hidden leading shape {tuple(hidden.shape[:-1])} must match "
            f"target_ids shape {tuple(target_ids.shape)}"
        )
    if lm_head_weight.size(-1) != hidden.size(-1):
        raise ValueError(
            f"hidden dim {hidden.size(-1)} must match lm_head_weight dim "
            f"{lm_head_weight.size(-1)}"
        )
    if lm_head_weight.device != hidden.device:
        raise ValueError(
            f"lm_head_weight device {lm_head_weight.device} must match hidden "
            f"device {hidden.device}"
        )
    if bias is not None:
        if bias.ndim != 1 or bias.numel() != lm_head_weight.size(0):
            raise ValueError(
                f"bias must be 1-D with local V={lm_head_weight.size(0)} elements, "
                f"got shape {tuple(bias.shape)}"
            )
        if bias.device != hidden.device:
            raise ValueError(f"bias device {bias.device} must match hidden device {hidden.device}")

    return _TensorParallelLinearLogpFunction.apply(
        hidden,
        lm_head_weight,
        bias,
        target_ids,
        int(vocab_start_index),
        None if global_vocab_size is None else int(global_vocab_size),
        tp_group,
    )


def chunked_linear_logp_backward(
    grad_logp: torch.Tensor,
    hidden_2d: torch.Tensor,
    weight: torch.Tensor,
    target_1d: torch.Tensor,
    bias_t: torch.Tensor,
    *,
    has_bias: bool,
    lead_shape,
    hidden_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    bias_dtype,
    chunk_elems: int = BWD_CHUNK_ELEMS,
    compute_grad_hidden: bool = True,
    compute_grad_weight: bool = True,
    compute_grad_bias: bool = True,
):
    # Liger-style chunked backward shared by the Triton and CUDA SM90 fused ops.
    n, d = hidden_2d.shape
    v = weight.shape[0]
    dt = weight.dtype
    g = grad_logp.reshape(-1).to(torch.float32)

    grad_h = torch.empty_like(hidden_2d, dtype=torch.float32) if compute_grad_hidden else None
    grad_w = (
        torch.zeros(v, d, device=weight.device, dtype=torch.float32)
        if compute_grad_weight
        else None
    )
    grad_b = (
        torch.zeros(v, device=weight.device, dtype=torch.float32)
        if has_bias and compute_grad_bias
        else None
    )
    use_fp32 = _use_fp32_matmul(hidden_2d, weight)

    chunk = max(1, min(n, chunk_elems // v))
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        x = hidden_2d[i0:i1]  # [C, D]
        logits = _linear_logits(
            x,
            weight,
            bias_t if has_bias else None,
            use_fp32=use_fp32,
        )

        # dz = g * (onehot - softmax(logits)), recomputed from scratch so it is
        # self-normalizing and independent of the forward's saved lse.
        dz = torch.softmax(logits.float(), dim=-1).neg_()  # [C, V] fp32
        rows = torch.arange(i1 - i0, device=dz.device)
        dz[rows, target_1d[i0:i1].long()] += 1.0
        dz *= g[i0:i1].unsqueeze(1)

        if use_fp32:
            if grad_h is not None:
                grad_h[i0:i1] = torch.matmul(dz, weight.float()).float()  # [C, D]
            if grad_w is not None:
                grad_w += torch.matmul(dz.t(), x.float()).float()  # [V, D]
        else:
            dz_dt = dz.to(dt)
            if grad_h is not None:
                grad_h[i0:i1] = torch.matmul(dz_dt, weight).float()  # [C, D]
            if grad_w is not None:
                grad_w += torch.matmul(dz_dt.t(), x).float()  # [V, D]
        if grad_b is not None:
            grad_b += dz.sum(0)

    grad_hidden = (
        grad_h.to(hidden_dtype).reshape(tuple(lead_shape) + (d,)) if grad_h is not None else None
    )
    grad_weight = grad_w.to(weight_dtype) if grad_w is not None else None
    grad_bias = grad_b.to(bias_dtype) if grad_b is not None else None
    return grad_hidden, grad_weight, grad_bias


class NativeLinearLogpOp:
    """Naive PyTorch reference for fused linear log-prob.

    Materializes the full ``[N, V]`` logits with a single ``F.linear`` and runs
    ``log_softmax`` + ``gather``. This is the obviously-correct oracle the fused
    kernels are validated against (and the baseline the benchmark measures the
    VRAM win against); it is also the CPU / Triton-less fallback. Differentiable
    w.r.t. ``hidden``, ``lm_head_weight`` and ``bias`` through autograd.
    """

    def __init__(self) -> None:
        pass

    def __call__(
        self,
        hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        target_ids: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        *,
        tp_group: Any = None,
        vocab_start_index: int = 0,
        global_vocab_size: Optional[int] = None,
    ) -> torch.Tensor:
        return self.apply(
            hidden,
            lm_head_weight,
            target_ids,
            bias,
            tp_group=tp_group,
            vocab_start_index=vocab_start_index,
            global_vocab_size=global_vocab_size,
        )

    def apply(
        self,
        hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        target_ids: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        *,
        tp_group: Any = None,
        vocab_start_index: int = 0,
        global_vocab_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Selected-token log-prob ``z[t] - logsumexp(z)``, returned in float32."""
        if hidden.shape[:-1] != target_ids.shape:
            raise ValueError(
                f"hidden leading shape {tuple(hidden.shape[:-1])} must match "
                f"target_ids shape {tuple(target_ids.shape)}"
            )
        if lm_head_weight.size(-1) != hidden.size(-1):
            raise ValueError(
                f"hidden dim {hidden.size(-1)} must match lm_head_weight dim "
                f"{lm_head_weight.size(-1)}"
            )
        if should_use_tensor_parallel_linear_logp(
            tp_group,
            int(vocab_start_index),
            global_vocab_size,
            lm_head_weight.size(0),
        ):
            return tensor_parallel_linear_logp(
                hidden,
                lm_head_weight,
                target_ids,
                bias,
                tp_group=tp_group,
                vocab_start_index=vocab_start_index,
                global_vocab_size=global_vocab_size,
            )

        lead_shape = hidden.shape[:-1]
        hidden_2d = hidden.reshape(-1, hidden.size(-1))
        logits = torch.nn.functional.linear(hidden_2d, lm_head_weight, bias)
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        target_1d = target_ids.reshape(-1).to(device=logits.device, dtype=torch.long)
        selected = torch.gather(log_probs, dim=-1, index=target_1d.unsqueeze(1)).squeeze(-1)
        return selected.reshape(lead_shape)
