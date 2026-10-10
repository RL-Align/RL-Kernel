# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Tensor-parallel H3 AdaLN projection (RFC #420 ``tp_adaln_3mod``).

Ownership: rank ``r`` of ``tp`` owns the contiguous output columns
``[r * N / tp, (r + 1) * N / tp)`` of the ``N = 3 * 6 * H`` projection, i.e. the
same rows of ``adaln_proj.linear.weight``/``bias``. ``N / tp`` must be a
multiple of the 64-row ``d_input`` chunk of contract ``h3-det-linear-v1``.

Every floating-point operation is the WS1 kernel itself, so every rank ends
with the WS1 bytes:

* forward: each column depends only on ``x`` and its own weight row, so the
  local GEMV produces the WS1 columns; a rank-ordered all-gather (a copy)
  rebuilds the ``(T, N)`` table, and the six tensors and ``3T`` modality rows
  are views of it exactly as in WS1;
* ``dW``/``db``: rows of the shard, computed locally (no cross-rank sum);
* ``d_temb``: each rank computes the WS1 chunk partials of its columns, the
  all-gather puts them in global chunk order, and every rank runs the WS1
  ascending fold. No collective reduction arithmetic is used.

The backward expects the table gradient to be identical on every TP rank
(the modulation consumes the full replicated table); it reads its own columns.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from rl_engine.ops.autograd.backward_runtime import record_backward
from rl_engine.backends.cuda.model_specific.minimax_h3.det_linear import (
    CONTRACT,
    DINPUT_CHUNK,
    det_linear_available,
    det_linear_backward_input_partials,
    det_linear_backward_weight,
    det_linear_fold_chunks,
    det_linear_forward,
    silu_backward_fp32,
)
from rl_engine.backends.cuda.model_specific.minimax_h3.ws2_comm import gather_rows
from rl_engine.reference.minimax_h3 import H3_ADALN_CHUNKS, H3_MODALITY_NUM
from rl_engine.reference.minimax_h3.adaln_projection import split_adaln_table

KERNEL_ID = "rl_engine._C.h3_det_linear_forward[column_shard]+all_gather[rank_order]"
BACKWARD_IMPL = "local_dweight+all_gather_chunk_partials+ascending_fold"
BACKWARD_KERNEL_ID = (
    "rl_engine._C.h3_det_linear_backward_weight"
    "+rl_engine._C.h3_det_linear_backward_input_partials"
    "+all_gather+rl_engine._C.h3_det_linear_fold_chunks"
)


@dataclass(frozen=True)
class AdaLNColumnShard:
    """The output columns a TP rank owns, and the WS1 objects they map to."""

    tp: int
    rank: int
    n_total: int
    begin: int
    end: int

    @property
    def hidden(self) -> int:
        return self.n_total // (H3_MODALITY_NUM * H3_ADALN_CHUNKS)

    def slots(self) -> list[tuple[int, int, int, int]]:
        """``(modality, chunk, h_begin, h_end)`` pieces covered by this shard."""

        out, col = [], self.begin
        while col < self.end:
            slot, h0 = divmod(col, self.hidden)
            h1 = min(self.hidden, h0 + self.end - col)
            out.append((*divmod(slot, H3_ADALN_CHUNKS), h0, h1))
            col += h1 - h0
        return out


def adaln_column_shard(n_total: int, tp: int, rank: int) -> AdaLNColumnShard:
    if tp < 1 or not 0 <= rank < tp:
        raise ValueError(f"need 0 <= rank < tp, got rank={rank}, tp={tp}")
    if n_total % (H3_MODALITY_NUM * H3_ADALN_CHUNKS):
        raise ValueError(f"N={n_total} is not 3 modalities x 6 chunks x H")
    if n_total % (tp * DINPUT_CHUNK):
        # A shard boundary inside a d_input chunk would split one WS1 partial.
        raise ValueError(
            f"tp={tp} does not split N={n_total} on {DINPUT_CHUNK}-column chunk boundaries"
        )
    width = n_total // tp
    return AdaLNColumnShard(tp, rank, n_total, rank * width, (rank + 1) * width)


def shard_adaln_projection(weight: torch.Tensor, bias: torch.Tensor, tp: int, rank: int):
    """This rank's rows of the full projection weight and bias (views)."""

    shard = adaln_column_shard(weight.shape[0], tp, rank)
    return weight[shard.begin : shard.end], bias[shard.begin : shard.end]


def _gather_columns(collective, local: torch.Tensor) -> torch.Tensor:
    """``(T, N / tp)`` column shards -> the ``(T, N)`` table, in rank order (a copy)."""

    num_t, width = local.shape
    rows = gather_rows(collective, local)  # (tp * T, N / tp), rank-major
    return rows.view(-1, num_t, width).transpose(0, 1).reshape(num_t, -1)


def _validate_shard_inputs(temb, weight, bias, shard: AdaLNColumnShard) -> None:
    for name, tensor in (("temb", temb), ("weight", weight), ("bias", bias)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if not tensor.is_cuda or tensor.device != temb.device:
            raise ValueError(f"{name} must be on this rank's CUDA device, got {tensor.device}")
    if temb.dtype != torch.float32:
        raise TypeError(f"temb must be float32 (the SiLU runs before the cast), got {temb.dtype}")
    if weight.dtype not in (torch.bfloat16, torch.float32) or bias.dtype != weight.dtype:
        raise TypeError(
            f"weight/bias must share bfloat16 or float32, got {weight.dtype}/{bias.dtype}"
        )
    width = shard.end - shard.begin
    if weight.dim() != 2 or weight.shape[0] != width or bias.shape != (width,):
        raise ValueError(
            f"rank {shard.rank} of {shard.tp} owns {width} rows, got weight "
            f"{tuple(weight.shape)} and bias {tuple(bias.shape)}"
        )
    if temb.dim() != 2 or temb.shape[0] == 0 or temb.shape[1] != weight.shape[1]:
        raise ValueError(
            f"temb must be a non-empty (T, {weight.shape[1]}) matrix, got {tuple(temb.shape)}"
        )


class _TPAdaLNProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, temb, weight_shard, bias_shard, collective, shard):
        temb = temb.contiguous()
        act = F.silu(temb).to(weight_shard.dtype)
        (local,) = det_linear_forward(act, weight_shard, bias_shard)
        ctx.save_for_backward(temb, act, weight_shard)
        ctx.collective, ctx.shard = collective, shard
        return _gather_columns(collective, local)

    @staticmethod
    def backward(ctx, grad_table):
        temb, act, weight = ctx.saved_tensors
        shard = ctx.shard
        grad_local = grad_table[:, shard.begin : shard.end].float().contiguous()
        dw, db = det_linear_backward_weight(grad_local, act, weight.dtype)
        partial = det_linear_backward_input_partials(grad_local, weight)
        d_act = det_linear_fold_chunks(gather_rows(ctx.collective, partial), torch.float32)
        record_backward(
            "tp_adaln_3mod", kernel_id=BACKWARD_KERNEL_ID, impl=BACKWARD_IMPL, family="cuda"
        )
        return silu_backward_fp32(d_act, temb), dw, db, None, None


class H3TPAdaLNProjectionCudaOp:
    """Column-parallel ``adaln_projection_3mod``, byte-equal to WS1 on every rank.

    ``collective`` is a rank-ordered all-gather with ``rank``, ``world_size``
    and ``backend_id`` (e.g. ``DeterministicCollective``). Inputs are the replicated
    FP32 ``temb`` and this rank's weight/bias rows.
    """

    op_class = "reduction"
    kernel_id = KERNEL_ID
    backward_impl = BACKWARD_IMPL
    contract = CONTRACT

    def __init__(self, collective, n_total: int) -> None:
        if not det_linear_available():
            raise RuntimeError(
                "rl_engine._C lacks h3_det_linear_*; rebuild with csrc/cuda/h3/det_linear.cu"
            )
        self.collective = collective
        self.shard = adaln_column_shard(n_total, collective.world_size, collective.rank)

    def __call__(self, temb, weight_shard, bias_shard):
        return self.forward(temb, weight_shard, bias_shard)

    def forward(self, temb, weight_shard, bias_shard) -> tuple[torch.Tensor, ...]:
        table = self.forward_table(temb, weight_shard, bias_shard)
        return split_adaln_table(table, self.shard.hidden)

    def forward_table(self, temb, weight_shard, bias_shard) -> torch.Tensor:
        _validate_shard_inputs(temb, weight_shard, bias_shard, self.shard)
        return _TPAdaLNProjection.apply(temb, weight_shard, bias_shard, self.collective, self.shard)

    def readback(self) -> dict:
        """Runtime identity of this rank's projection, for evidence and strict traces."""

        return {
            "op": "tp_adaln_3mod",
            "kernel_id": KERNEL_ID,
            "backward_kernel_id": BACKWARD_KERNEL_ID,
            "contract": CONTRACT,
            "collective_backend": getattr(
                self.collective, "backend_id", type(self.collective).__name__
            ),
            "collective_ops": ["all_gather"],
            "reduction_order": "ws1_ascending_chunk_fold_after_rank_ordered_gather",
            "tp": self.shard.tp,
            "rank": self.shard.rank,
            "columns": [self.shard.begin, self.shard.end],
            "slots": [list(s) for s in self.shard.slots()],
            "fallback": None,
        }
