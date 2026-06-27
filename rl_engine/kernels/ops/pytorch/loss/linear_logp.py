# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

from typing import Any, Optional

import torch

# Backward token-chunk target: process at most this many ``[chunk, V]`` logit
# elements per cuBLAS step so peak backward memory stays ~``chunk*V`` instead of
# ``N*V``.
BWD_CHUNK_ELEMS = 1 << 24
_LOW_PRECISION_DTYPES = (torch.float16, torch.bfloat16)


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
        [0, 0 if global_vocab_size is None else int(global_vocab_size)],
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


def _chunked_local_linear_logp_stats(
    hidden_2d: torch.Tensor,
    weight: torch.Tensor,
    target_1d: torch.Tensor,
    bias_t: torch.Tensor,
    *,
    has_bias: bool,
    vocab_start_index: int,
    chunk_elems: int = BWD_CHUNK_ELEMS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = hidden_2d.size(0)
    local_vocab = weight.size(0)
    device = hidden_2d.device

    local_max = torch.full((n,), -torch.inf, device=device, dtype=torch.float32)
    local_sum = torch.zeros(n, device=device, dtype=torch.float32)
    local_target_logit = torch.zeros(n, device=device, dtype=torch.float32)
    owner_count = torch.zeros(n, device=device, dtype=torch.int32)
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
        if bool(owns_target.any().item()):
            local_idx = (target_1d[owns_target] - global_v0).long()
            local_target_logit[owns_target] = logits_f[rows[owns_target], local_idx]
            owner_count[owns_target] += 1

    return local_max, local_sum, local_target_logit, owner_count


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
        dist = _require_distributed_initialized()
        hidden_2d = hidden.reshape(-1, hidden.size(-1)).contiguous()
        weight = lm_head_weight.contiguous()
        target_1d = (
            target_ids.reshape(-1).to(device=hidden_2d.device, dtype=torch.long).contiguous()
        )
        bias_t = bias.contiguous() if bias is not None else hidden_2d
        vocab_start_index = int(vocab_start_index)
        global_vocab_size = _validate_tp_vocab_partition(
            tp_group=tp_group,
            device=hidden_2d.device,
            vocab_start_index=vocab_start_index,
            local_vocab_size=weight.size(0),
            global_vocab_size=global_vocab_size,
        )
        _validate_global_targets(target_1d, global_vocab_size, tp_group)

        local_max, local_sum, local_target_logit, owner_count = _chunked_local_linear_logp_stats(
            hidden_2d,
            weight,
            target_1d,
            bias_t,
            has_bias=bias is not None,
            vocab_start_index=vocab_start_index,
        )

        global_max = local_max.clone()
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=tp_group)
        global_sum = local_sum * torch.exp(local_max - global_max)
        dist.all_reduce(global_sum, op=dist.ReduceOp.SUM, group=tp_group)

        target_logit = local_target_logit.clone()
        dist.all_reduce(target_logit, op=dist.ReduceOp.SUM, group=tp_group)

        global_owner_count = owner_count.clone()
        dist.all_reduce(global_owner_count, op=dist.ReduceOp.SUM, group=tp_group)
        if bool((global_owner_count != 1).any().item()):
            raise ValueError(
                "target_ids must be covered by exactly one TP vocab shard; check "
                "vocab_start_index and global_vocab_size."
            )

        lse = global_max + torch.log(global_sum)
        ctx.save_for_backward(hidden_2d, weight, bias_t, target_1d, lse)
        ctx.has_bias = bias is not None
        ctx.lead_shape = hidden.shape[:-1]
        ctx.hidden_dtype = hidden.dtype
        ctx.weight_dtype = lm_head_weight.dtype
        ctx.bias_dtype = bias.dtype if bias is not None else None
        ctx.vocab_start_index = vocab_start_index
        ctx.tp_group = tp_group
        return (target_logit - lse).reshape(hidden.shape[:-1])

    @staticmethod
    def backward(ctx, grad_logp):
        dist = _require_distributed_initialized()
        hidden_2d, weight, bias_t, target_1d, lse = ctx.saved_tensors
        n, d = hidden_2d.shape
        local_vocab = weight.shape[0]
        dt = weight.dtype
        g = grad_logp.reshape(-1).to(torch.float32)

        grad_h = torch.empty_like(hidden_2d, dtype=torch.float32)
        grad_w = torch.zeros(local_vocab, d, device=weight.device, dtype=torch.float32)
        grad_b = (
            torch.zeros(local_vocab, device=weight.device, dtype=torch.float32)
            if ctx.has_bias
            else None
        )
        use_fp32 = _use_fp32_matmul(hidden_2d, weight)

        chunk = max(1, min(n, BWD_CHUNK_ELEMS // local_vocab))
        for i0 in range(0, n, chunk):
            i1 = min(i0 + chunk, n)
            x = hidden_2d[i0:i1]
            logits = _linear_logits(
                x,
                weight,
                bias_t if ctx.has_bias else None,
                use_fp32=use_fp32,
            )

            dz = -torch.exp(logits.float() - lse[i0:i1].unsqueeze(1))
            local_idx = target_1d[i0:i1] - ctx.vocab_start_index
            owns_target = (local_idx >= 0) & (local_idx < local_vocab)
            if bool(owns_target.any().item()):
                rows = torch.arange(i1 - i0, device=dz.device)[owns_target]
                dz[rows, local_idx[owns_target].long()] += 1.0
            dz *= g[i0:i1].unsqueeze(1)

            if use_fp32:
                grad_h[i0:i1] = torch.matmul(dz, weight.float()).float()
                grad_w += torch.matmul(dz.t(), x.float()).float()
            else:
                dz_dt = dz.to(dt)
                grad_h[i0:i1] = torch.matmul(dz_dt, weight).float()
                grad_w += torch.matmul(dz_dt.t(), x).float()
            if grad_b is not None:
                grad_b += dz.sum(0)

        dist.all_reduce(grad_h, op=dist.ReduceOp.SUM, group=ctx.tp_group)
        grad_hidden = grad_h.to(ctx.hidden_dtype).reshape((*ctx.lead_shape, d))
        grad_weight = grad_w.to(ctx.weight_dtype)
        grad_bias = grad_b.to(ctx.bias_dtype) if grad_b is not None else None
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
):
    # Liger-style chunked backward shared by the Triton and CUDA SM90 fused ops.
    n, d = hidden_2d.shape
    v = weight.shape[0]
    dt = weight.dtype
    g = grad_logp.reshape(-1).to(torch.float32)

    grad_h = torch.empty_like(hidden_2d, dtype=torch.float32)
    grad_w = torch.zeros(v, d, device=weight.device, dtype=torch.float32)
    grad_b = torch.zeros(v, device=weight.device, dtype=torch.float32) if has_bias else None
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
            grad_h[i0:i1] = torch.matmul(dz, weight.float()).float()  # [C, D]
            grad_w += torch.matmul(dz.t(), x.float()).float()  # [V, D]
        else:
            dz_dt = dz.to(dt)
            grad_h[i0:i1] = torch.matmul(dz_dt, weight).float()  # [C, D]
            grad_w += torch.matmul(dz_dt.t(), x).float()  # [V, D]
        if grad_b is not None:
            grad_b += dz.sum(0)

    grad_hidden = grad_h.to(hidden_dtype).reshape(tuple(lead_shape) + (d,))
    grad_weight = grad_w.to(weight_dtype)
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
