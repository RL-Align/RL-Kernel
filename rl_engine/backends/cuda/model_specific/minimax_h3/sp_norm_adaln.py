# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Sequence-parallel H3 RMSNorm / AdaLN modulation / gated residual (RFC #420 ``sp_norm_adaln``).

Ownership: rank ``r`` of ``sp`` holds packed positions ``[r * S // sp, (r + 1) * S // sp)``
of every batch item. The full ``(S,)`` row index is replicated; it is small, and every
rank derives the same reduction plan from it.

Forward and the row-local gradients (``dx``, ``d_sublayer``, ``d_residual``) are the WS1
kernels on the local rows. Rows are independent, so they are byte-equal by construction.

The cross-row reductions (``d_norm_weight``, ``d_shift``/``d_scale``, ``d_gate``) are WS1
two-level folds: FP32 sums over fixed 256-element tiles, then an ascending fold of the
tiles. Every rank builds the global WS1 tile list. A tile is computed by the rank that
holds its first row; the tile's rows from other ranks are all-gathered beforehand (only
rows of tiles that straddle a shard boundary move). The tile partials are all-gathered,
put back in WS1 tile order, and every rank runs the WS1 fold. The partial and fold kernels
are the WS1 kernels, so every gradient equals WS1 on every rank. The only collectives are
rank-ordered all-gathers (copies).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rl_engine.ops.autograd.backward_runtime import record_backward
from rl_engine.backends.extension import _C, _EXT_AVAILABLE
from rl_engine.backends.cuda.model_specific.minimax_h3.adaln_row_gather import (
    BACKWARD_TILE,
    _segment_tiles,
)
from rl_engine.backends.cuda.model_specific.minimax_h3.ws2_comm import gather_rows
from rl_engine.reference.minimax_h3.gate_residual import validate_h3_gate_residual
from rl_engine.reference.minimax_h3.rmsnorm import (
    H3_NORM_EPS,
    validate_h3_modulation,
    validate_h3_rmsnorm,
)

_SYMBOLS = (
    "h3_rmsnorm_forward",
    "h3_rmsnorm_backward_dx",
    "h3_rmsnorm_backward_partials",
    "h3_rmsnorm_fold_partials",
    "h3_gate_residual_forward",
    "h3_gate_residual_backward_dy",
    "h3_gate_grad_partials",
    "h3_gate_grad_fold",
)
BACKWARD_IMPL = "row_local_ws1+gather_straddling_rows+ws1_tile_partials+gather+ws1_fold"


def sp_norm_adaln_available() -> bool:
    return bool(_EXT_AVAILABLE and all(hasattr(_C, name) for name in _SYMBOLS))


@dataclass(frozen=True)
class SPRowLayout:
    """Which packed positions each sequence-parallel rank holds."""

    seq_len: int
    batch: int
    sp: int
    rank: int

    def bounds(self, rank: int) -> tuple[int, int]:
        return rank * self.seq_len // self.sp, (rank + 1) * self.seq_len // self.sp

    @property
    def lo(self) -> int:
        return self.bounds(self.rank)[0]

    @property
    def hi(self) -> int:
        return self.bounds(self.rank)[1]

    @property
    def local_len(self) -> int:
        return self.hi - self.lo


def sp_row_layout(seq_len: int, batch: int, sp: int, rank: int) -> SPRowLayout:
    if sp < 1 or not 0 <= rank < sp:
        raise ValueError(f"need 0 <= rank < sp, got rank={rank}, sp={sp}")
    if batch < 1 or seq_len < sp:
        raise ValueError(f"every rank needs a row: seq_len={seq_len}, sp={sp}, batch={batch}")
    return SPRowLayout(seq_len, batch, sp, rank)


def _ordinal_by_group(group: torch.Tensor, groups: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-group counts, and each element's position within its group (in input order)."""

    counts = torch.bincount(group, minlength=groups)
    order = torch.argsort(group, stable=True)
    ordinal = torch.empty_like(group)
    first = torch.cumsum(counts, 0) - counts
    ordinal[order] = torch.arange(group.numel(), device=group.device) - first[group[order]]
    return counts, ordinal


class _Plan:
    """Global WS1 tiles of one backward, and this rank's part in computing them.

    ``families`` are ``(rows, tile_begin, tile_end)`` over global flattened rows
    ``b * S + s``; consecutive tiles cover ``rows`` in order. Every rank builds the same
    plan from replicated metadata, so collective calls and shapes agree.
    """

    def __init__(self, layout: SPRowLayout, families, device) -> None:
        self.layout, sp = layout, layout.sp
        self._his = torch.tensor([layout.bounds(r)[1] for r in range(sp)], device=device)

        # Rows of a tile computed by another rank: the only activation rows that move.
        tile_owner, shared = [], []
        for rows, begin, end in families:
            owner = self.owner(rows[begin])
            tile_owner.append(owner)
            shared.append(rows[torch.repeat_interleave(owner, end - begin) != self.owner(rows)])
        shared = torch.unique(torch.cat(shared))
        shared_owner = self.owner(shared)
        counts, slot = _ordinal_by_group(shared_owner, sp)
        self.max_send = int(counts.max()) if shared.numel() else 0

        # Buffer row of every global row this rank reads: its own rows first, then
        # every rank's send block (padded to max_send) in rank order.
        own = self.global_rows(layout.rank)
        buffer_index = torch.full((layout.batch * layout.seq_len,), -1, device=device)
        buffer_index[own] = torch.arange(own.numel(), device=device)
        foreign = shared_owner != layout.rank
        buffer_index[shared[foreign]] = (
            own.numel() + shared_owner[foreign] * self.max_send + slot[foreign]
        )
        mine = ~foreign
        self.send_local = self.local_index(shared[mine][torch.argsort(slot[mine])])
        self.recv_global = torch.zeros(sp * self.max_send, dtype=torch.long, device=device)
        self.recv_global[shared_owner * self.max_send + slot] = shared  # padding: never read

        self.families = [
            self._my_tiles(rows, begin, end, owner, buffer_index)
            for (rows, begin, end), owner in zip(families, tile_owner)
        ]

    def owner(self, rows: torch.Tensor) -> torch.Tensor:
        return torch.searchsorted(self._his, rows % self.layout.seq_len, right=True)

    def global_rows(self, rank: int) -> torch.Tensor:
        lo, hi = self.layout.bounds(rank)
        s = torch.arange(lo, hi, device=self._his.device)
        b = torch.arange(self.layout.batch, device=self._his.device)
        return (b[:, None] * self.layout.seq_len + s[None, :]).reshape(-1)

    def local_index(self, rows: torch.Tensor) -> torch.Tensor:
        b, s = rows // self.layout.seq_len, rows % self.layout.seq_len
        return b * self.layout.local_len + s - self.layout.lo

    def _my_tiles(self, rows, begin, end, owner, buffer_index):
        mine = torch.nonzero(owner == self.layout.rank).flatten()
        lengths = (end - begin)[mine]
        my_end = torch.cumsum(lengths, 0)
        my_begin = my_end - lengths
        elems = torch.repeat_interleave(begin[mine] - my_begin, lengths)
        elems = elems + torch.arange(elems.numel(), device=rows.device)
        counts, ordinal = _ordinal_by_group(owner, self.layout.sp)
        width = int(counts.max()) if owner.numel() else 0
        return {
            "tiles": (buffer_index[rows[elems]].contiguous(), my_begin, my_end),
            "width": width,  # tiles per rank in the padded partial gather
            "gather_index": owner * width + ordinal,  # WS1 tile order within that gather
        }

    def exchange(self, collective, *tensors: torch.Tensor) -> list[torch.Tensor]:
        """For each ``(M_local, ...)`` tensor: local rows, then the gathered send blocks."""

        out = []
        for t in tensors:
            if self.max_send == 0:
                out.append(t)
                continue
            send = t.new_zeros((self.max_send, *t.shape[1:]))
            send[: self.send_local.numel()] = t[self.send_local]
            out.append(torch.cat([t, gather_rows(collective, send)]))
        return out

    def gather_partials(self, collective, family: int, partial: torch.Tensor) -> torch.Tensor:
        """This rank's tile partials -> every tile's partial, in WS1 tile order."""

        fam = self.families[family]
        padded = partial.new_zeros((fam["width"], *partial.shape[1:]))
        padded[: partial.shape[0]] = partial
        return gather_rows(collective, padded).index_select(0, fam["gather_index"]).contiguous()


def _dweight_tiles(total: int, device):
    begin = torch.arange(0, total, BACKWARD_TILE, device=device)
    return torch.arange(total, device=device), begin, torch.clamp(begin + BACKWARD_TILE, max=total)


@dataclass(frozen=True)
class SPPlan:
    """Everything an SP backward needs that depends only on the layout and the row index.

    Built once per ``(row index, table rows)`` and shared by every norm and gated
    residual that uses that index (every block of a forward pass).
    """

    layout: SPRowLayout
    plan: _Plan
    local_index: torch.Tensor | None  # (S_local,) table row of each local position
    index_buf: torch.Tensor | None  # table row of each exchange-buffer row
    seg_first_tile: torch.Tensor | None
    num_rows: int | None


def sp_plan(
    layout: SPRowLayout, index_full=None, num_rows: int | None = None, *, device=None
) -> SPPlan:
    """The reduction plan for one row index (``None``: the plain norm's ``dweight`` only)."""

    device = index_full.device if index_full is not None else torch.device(device or "cuda")
    families = [_dweight_tiles(layout.batch * layout.seq_len, device)]
    if index_full is None:
        return SPPlan(layout, _Plan(layout, families, device), None, None, None, None)
    *seg_tiles, seg_first_tile = _segment_tiles(index_full.repeat(layout.batch), num_rows)
    plan = _Plan(layout, [*families, tuple(seg_tiles)], device)
    local_index = index_full[layout.lo : layout.hi].clone(memory_format=torch.contiguous_format)
    received = index_full[plan.recv_global % layout.seq_len]
    index_buf = torch.cat([local_index.repeat(layout.batch), received]).contiguous()
    return SPPlan(layout, plan, local_index, index_buf, seg_first_tile, num_rows)


def sp_rmsnorm_backward(grad, x, weight, rstd, shift, scale, sp: SPPlan, collective):
    """One rank's backward on its ``(B * S_local, H)`` rows: ``(dx, dweight, d_shift, d_scale)``.

    ``dx`` is row-local; the other gradients are the full WS1 reductions, identical on
    every rank. ``shift``/``scale`` are ``None`` for the plain norm.
    """

    weight, plan, modulated = weight.contiguous(), sp.plan, sp.local_index is not None
    dx = _C.h3_rmsnorm_backward_dx(grad, x, weight, rstd, shift, scale, sp.local_index)
    g_buf, x_buf, rstd_buf = plan.exchange(collective, grad, x, rstd)
    seg = plan.families[1]["tiles"] if modulated else (None, None, None)
    partials = _C.h3_rmsnorm_backward_partials(
        g_buf, x_buf, weight, rstd_buf, shift, scale, sp.index_buf,
        *plan.families[0]["tiles"], *seg,
    )  # fmt: skip
    gathered = [plan.gather_partials(collective, i, p) for i, p in enumerate(partials)]
    record_backward(
        "sp_norm_adaln",
        kernel_id="rl_engine._C.h3_rmsnorm_backward_partials",
        impl=BACKWARD_IMPL,
        family="cuda",
    )
    if not modulated:
        (dweight,) = _C.h3_rmsnorm_fold_partials(gathered[0], weight)
        return dx, dweight, None, None
    dweight, d_shift, d_scale = _C.h3_rmsnorm_fold_partials(
        gathered[0], weight, gathered[1], sp.seg_first_tile
    )
    return dx, dweight, d_shift.to(shift.dtype), d_scale.to(scale.dtype)


def sp_gate_residual_backward(grad, y, gate, sp: SPPlan, collective):
    """One rank's backward on its rows: ``(d_sublayer, d_gate)``; ``d_residual`` is ``grad``."""

    plan = sp.plan
    dy = _C.h3_gate_residual_backward_dy(grad, gate, sp.local_index)
    g_buf, y_buf = plan.exchange(collective, grad, y)
    partial = _C.h3_gate_grad_partials(g_buf, y_buf, *plan.families[1]["tiles"])
    dgate = _C.h3_gate_grad_fold(
        plan.gather_partials(collective, 1, partial), sp.seg_first_tile, gate.dtype
    )
    record_backward(
        "sp_norm_adaln",
        kernel_id="rl_engine._C.h3_gate_grad_partials",
        impl=BACKWARD_IMPL,
        family="cuda",
    )
    return dy, dgate


class _SPRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, shift, scale, sp, collective, eps):
        x2 = x.contiguous().view(-1, x.shape[-1])
        y, rstd = _C.h3_rmsnorm_forward(
            x2, weight.contiguous(), float(eps), shift, scale, sp.local_index
        )
        ctx.save_for_backward(x2, weight, rstd, shift, scale)
        ctx.sp, ctx.collective, ctx.x_shape = sp, collective, x.shape
        return y.view(x.shape)

    @staticmethod
    def backward(ctx, grad):
        x2, weight, rstd, shift, scale = ctx.saved_tensors
        g2 = grad.contiguous().view_as(x2)
        dx, dweight, d_shift, d_scale = sp_rmsnorm_backward(
            g2, x2, weight, rstd, shift, scale, ctx.sp, ctx.collective
        )
        return dx.view(ctx.x_shape), dweight, d_shift, d_scale, None, None, None


class _SPGateResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, residual, y, gate, sp, collective):
        hidden = residual.shape[-1]
        res2 = residual.contiguous().view(-1, hidden)
        y2 = y.contiguous().view(-1, hidden)
        out = _C.h3_gate_residual_forward(res2, y2, gate, sp.local_index)
        ctx.save_for_backward(y2, gate)
        ctx.sp, ctx.collective, ctx.shape = sp, collective, residual.shape
        return out.view(residual.shape)

    @staticmethod
    def backward(ctx, grad):
        y2, gate = ctx.saved_tensors
        g2 = grad.contiguous().view_as(y2)
        dy, dgate = sp_gate_residual_backward(g2, y2, gate, ctx.sp, ctx.collective)
        return grad, dy.view(ctx.shape), dgate, None, None


class H3SPNormAdaLNCudaOp:
    """Sequence-parallel ``h3_rmsnorm`` (+ modulation) and ``adaln_gate_residual``.

    ``collective`` is a rank-ordered all-gather with ``rank``, ``world_size`` and
    ``backend_id`` (e.g. ``DeterministicCollective``). Activations are this rank's
    ``(B, S_local, H)`` rows; the norm weight, the table views and the ``(S,)`` row
    index are the full, replicated tensors. Gradients of the replicated tensors come
    back identical on every rank, equal to WS1.
    """

    op_class = "reduction"
    backward_impl = BACKWARD_IMPL

    def __init__(self, collective, seq_len: int, batch: int) -> None:
        if not sp_norm_adaln_available():
            raise RuntimeError("rl_engine._C lacks the H3 SP symbols; rebuild the CUDA extension")
        self.collective = collective
        self.layout = sp_row_layout(seq_len, batch, collective.world_size, collective.rank)
        self._plans: list[tuple[object, int, int | None, torch.device, SPPlan]] = []

    def plan(self, index: torch.Tensor | None, num_rows: int | None, *, device=None) -> SPPlan:
        """The cached plan for this index object (rebuilt if it was modified in place)."""

        device = index.device if index is not None else torch.device(device or "cuda")
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        version = -1 if index is None else index._version
        for obj, ver, rows, cached_device, plan in self._plans:
            if obj is index and ver == version and rows == num_rows and cached_device == device:
                return plan
        plan = sp_plan(self.layout, index, num_rows, device=device)
        # Holding the index keeps its storage alive, so a new tensor can never alias it.
        self._plans = [(index, version, num_rows, device, plan), *self._plans[:3]]
        return plan

    def _check(self, x: torch.Tensor, index: torch.Tensor | None, num_rows: int | None) -> None:
        expected = (self.layout.batch, self.layout.local_len)
        if not isinstance(x, torch.Tensor) or x.dim() != 3 or tuple(x.shape[:2]) != expected:
            shape = tuple(x.shape) if isinstance(x, torch.Tensor) else type(x).__name__
            raise ValueError(f"activations must be this rank's {expected} + (H,) rows, got {shape}")
        if not x.is_cuda:
            raise ValueError("H3SPNormAdaLNCudaOp needs CUDA tensors")
        if index is None:
            return
        if index.dim() != 1 or index.shape[0] != self.layout.seq_len:
            raise ValueError(
                f"index must be the full ({self.layout.seq_len},) row index, "
                f"got {tuple(index.shape)}"
            )
        if index.numel() and (int(index.min()) < 0 or int(index.max()) >= num_rows):
            raise IndexError(f"index must be in [0, {num_rows})")

    def norm(self, x, weight, eps: float = H3_NORM_EPS) -> torch.Tensor:
        self._check(x, None, None)
        validate_h3_rmsnorm(x, weight, eps)
        return _SPRMSNorm.apply(
            x, weight, None, None, self.plan(None, None, device=x.device), self.collective, eps
        )

    def norm_modulated(self, x, weight, shift, scale, index, eps: float = H3_NORM_EPS):
        self._check(x, index, shift.shape[0])
        validate_h3_rmsnorm(x, weight, eps)
        validate_h3_modulation(x, shift, scale, index[self.layout.lo : self.layout.hi])
        if shift.stride(1) != 1 or scale.stride(1) != 1 or shift.stride(0) != scale.stride(0):
            shift, scale = shift.contiguous(), scale.contiguous()
        sp = self.plan(index, shift.shape[0])
        return _SPRMSNorm.apply(x, weight, shift, scale, sp, self.collective, eps)

    def gate_residual(self, residual, y, gate, index) -> torch.Tensor:
        self._check(residual, index, gate.shape[0])
        validate_h3_gate_residual(residual, y, gate, index[self.layout.lo : self.layout.hi])
        if gate.stride(1) != 1:
            gate = gate.contiguous()
        sp = self.plan(index, gate.shape[0])
        return _SPGateResidual.apply(residual, y, gate, sp, self.collective)

    def readback(self) -> dict:
        return {
            "op": "sp_norm_adaln",
            "backward_impl": BACKWARD_IMPL,
            "collective_backend": getattr(
                self.collective, "backend_id", type(self.collective).__name__
            ),
            "collective_ops": ["all_gather"],
            "reduction_order": "ws1_256_element_tiles_then_ascending_fold",
            "sp": self.layout.sp,
            "rank": self.layout.rank,
            "positions": [self.layout.lo, self.layout.hi],
            "batch": self.layout.batch,
            "fallback": None,
        }
