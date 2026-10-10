# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""ROCm-only training optimizations for the deterministic Qwen3 FFN."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from rl_engine.backends.rocm.ffn import ffn as common


def _cp_sharded_weight_gradient(
    a: Tensor,
    grad: Tensor,
    *,
    collective: Any,
    tp_world: int,
    column: bool,
    disable_split_k: bool,
) -> Tensor:
    """Compute independent output rows once across context-parallel ranks."""

    world_size = int(collective.world_size)
    output_rows = int(grad.size(1))
    if output_rows % world_size:
        raise ValueError(
            "CP-sharded FFN weight-gradient rows must divide evenly across ranks, "
            f"got {output_rows} rows and world_size={world_size}."
        )
    rows_per_rank = output_rows // world_size
    row_start = int(collective.rank) * rows_per_rank
    local = common._canonical_tp_weight_gradient(
        a,
        grad.narrow(1, row_start, rows_per_rank).contiguous(),
        tp_world=tp_world,
        column=column,
        disable_split_k=disable_split_k,
    )
    return collective.all_gather(local.contiguous())


def _packed_gate_up_weight_gradients(
    a: Tensor,
    grad_gate: Tensor,
    grad_up: Tensor,
    *,
    tp_world: int,
    disable_split_k: bool,
    cp_collective: Any = None,
) -> tuple[Tensor, Tensor]:
    """Share one deterministic GEMM across packed gate/up rows."""

    grad_gate_up = torch.cat((grad_gate, grad_up), dim=1).contiguous()
    if cp_collective is None:
        grad_gate_up_weight = common._canonical_tp_weight_gradient(
            a,
            grad_gate_up,
            tp_world=tp_world,
            column=True,
            disable_split_k=disable_split_k,
        )
    else:
        grad_gate_up_weight = _cp_sharded_weight_gradient(
            a,
            grad_gate_up,
            collective=cp_collective,
            tp_world=tp_world,
            column=True,
            disable_split_k=disable_split_k,
        )
    return grad_gate_up_weight.chunk(2, dim=0)


class _RocmTrainingFFNFunction(common._DeterministicFFNFunction):
    """Keep the rollout implementation immutable while specializing training."""

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor):
        if ctx.packed_gate_up:
            (
                rmsnorm_output,
                gate_up,
                activated,
                gate_weight,
                up_weight,
                down_weight,
            ) = ctx.saved_tensors
        else:
            (
                rmsnorm_output,
                gate,
                up,
                activated,
                gate_weight,
                up_weight,
                down_weight,
            ) = ctx.saved_tensors
        tp_collective = ctx.tp_collective
        cp_collective = ctx.cp_collective
        disable_split_k = ctx.disable_split_k
        grad_output = grad_output.reshape(-1, grad_output.size(-1)).contiguous()
        if ctx.sequence_parallel:
            grad_output = common._all_gather_tokens(grad_output, tp_collective)

        grad_activated = common._canonical_tp_input_gradient(
            grad_output,
            down_weight,
            tp_world=ctx.tp_world,
            column=False,
            disable_split_k=disable_split_k,
        )
        if ctx.packed_gate_up:
            grad_gate, grad_up = common._C.swiglu_packed_backward(grad_activated, gate_up)
        else:
            grad_gate, grad_up = common._C.swiglu_backward(grad_activated, gate, up)

        if cp_collective is not None or ctx.cp_layout is not None:
            gather = (
                (lambda *values, **kw: values)
                if cp_collective is None
                else common._all_gather_packed_tokens
            )
            (
                activated_full,
                grad_output_full,
                rmsnorm_full,
                grad_gate_full,
                grad_up_full,
            ) = gather(
                activated,
                grad_output,
                rmsnorm_output,
                grad_gate,
                grad_up,
                collective=cp_collective,
            )
            if ctx.cp_layout is not None:
                activated_full, grad_output_full, rmsnorm_full, grad_gate_full, grad_up_full = (
                    ctx.cp_layout.ordered(value)
                    for value in (
                        activated_full,
                        grad_output_full,
                        rmsnorm_full,
                        grad_gate_full,
                        grad_up_full,
                    )
                )
            if cp_collective is None:
                grad_down_weight = common._canonical_tp_weight_gradient(
                    activated_full,
                    grad_output_full,
                    tp_world=ctx.tp_world,
                    column=False,
                    disable_split_k=disable_split_k,
                )
            else:
                grad_down_weight = _cp_sharded_weight_gradient(
                    activated_full,
                    grad_output_full,
                    collective=cp_collective,
                    tp_world=ctx.tp_world,
                    column=False,
                    disable_split_k=disable_split_k,
                )
            if ctx.packed_gate_up:
                grad_gate_weight, grad_up_weight = _packed_gate_up_weight_gradients(
                    rmsnorm_full,
                    grad_gate_full,
                    grad_up_full,
                    tp_world=ctx.tp_world,
                    disable_split_k=disable_split_k,
                    cp_collective=cp_collective,
                )
            elif cp_collective is None:
                grad_gate_weight = common._canonical_tp_weight_gradient(
                    rmsnorm_full,
                    grad_gate_full,
                    tp_world=ctx.tp_world,
                    column=True,
                    disable_split_k=disable_split_k,
                )
                grad_up_weight = common._canonical_tp_weight_gradient(
                    rmsnorm_full,
                    grad_up_full,
                    tp_world=ctx.tp_world,
                    column=True,
                    disable_split_k=disable_split_k,
                )
            else:
                grad_gate_weight = _cp_sharded_weight_gradient(
                    rmsnorm_full,
                    grad_gate_full,
                    collective=cp_collective,
                    tp_world=ctx.tp_world,
                    column=True,
                    disable_split_k=disable_split_k,
                )
                grad_up_weight = _cp_sharded_weight_gradient(
                    rmsnorm_full,
                    grad_up_full,
                    collective=cp_collective,
                    tp_world=ctx.tp_world,
                    column=True,
                    disable_split_k=disable_split_k,
                )
        else:
            grad_down_weight = common._canonical_tp_weight_gradient(
                activated,
                grad_output,
                tp_world=ctx.tp_world,
                column=False,
                disable_split_k=disable_split_k,
            )
            if ctx.packed_gate_up:
                grad_gate_weight, grad_up_weight = _packed_gate_up_weight_gradients(
                    rmsnorm_output,
                    grad_gate,
                    grad_up,
                    tp_world=ctx.tp_world,
                    disable_split_k=disable_split_k,
                )
            else:
                grad_gate_weight = common._canonical_tp_weight_gradient(
                    rmsnorm_output,
                    grad_gate,
                    tp_world=ctx.tp_world,
                    column=True,
                    disable_split_k=disable_split_k,
                )
                grad_up_weight = common._canonical_tp_weight_gradient(
                    rmsnorm_output,
                    grad_up,
                    tp_world=ctx.tp_world,
                    column=True,
                    disable_split_k=disable_split_k,
                )

        grad_rmsnorm_from_gate = common._canonical_tp_input_gradient(
            grad_gate,
            gate_weight,
            tp_world=ctx.tp_world,
            column=True,
            disable_split_k=disable_split_k,
        )
        grad_rmsnorm_from_up = common._canonical_tp_input_gradient(
            grad_up,
            up_weight,
            tp_world=ctx.tp_world,
            column=True,
            disable_split_k=disable_split_k,
        )
        if ctx.sequence_parallel:
            grad_rmsnorm_from_gate, grad_rmsnorm_from_up = tp_collective.reduce_scatter_many(
                (grad_rmsnorm_from_gate, grad_rmsnorm_from_up)
            )
        elif tp_collective is not None:
            grad_rmsnorm_from_gate = common._all_reduce_inplace(
                grad_rmsnorm_from_gate, tp_collective
            )
            grad_rmsnorm_from_up = common._all_reduce_inplace(grad_rmsnorm_from_up, tp_collective)

        grad_rmsnorm_output = grad_rmsnorm_from_gate.add_(grad_rmsnorm_from_up)
        return (
            grad_rmsnorm_output.reshape(ctx.input_shape),
            grad_gate_weight,
            grad_up_weight,
            grad_down_weight,
            None,
            None,
            None,
            None,
            None,
        )


def qwen3_ffn_training(
    rmsnorm_output: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
    *,
    fused_gate_up_weight: Tensor | None = None,
    tp_group: Any = None,
    cp_group: Any = None,
    sequence_parallel: bool = False,
    deterministic: bool | None = None,
    disable_split_k: bool | None = None,
) -> Tensor:
    """Run strict ROCm training with canonical CP order and packed gradients."""

    if not isinstance(sequence_parallel, bool):
        raise TypeError("sequence_parallel must be a bool.")
    deterministic = common._resolve_deterministic_mode(deterministic, disable_split_k)
    common._validate_ffn_inputs(
        rmsnorm_output,
        gate_weight,
        up_weight,
        down_weight,
        fused_gate_up_weight,
    )
    common._require_ffn_kernels(
        disable_split_k=deterministic,
        packed_gate_up=fused_gate_up_weight is not None and deterministic,
    )
    return _RocmTrainingFFNFunction.apply(
        rmsnorm_output,
        gate_weight,
        up_weight,
        down_weight,
        fused_gate_up_weight,
        tp_group,
        cp_group,
        sequence_parallel,
        deterministic,
    )


__all__ = ["qwen3_ffn_training"]
