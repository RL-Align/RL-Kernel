# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Batch-invariant DSv4 MoE block (forward and backward), single GPU or expert parallel.

    router (P3)  ->  route plan  ->  dispatch [-> all-to-all]  ->  routed experts (P5, SM90)
                 [-> all-to-all back]  ->  combine (slot order) + shared expert + residual

* Router: :class:`~rl_engine.models.deepseek_v4.router.Dsv4Router`.
* Plan / combine: ``rl_engine.kernels.ops.pytorch.moe.route_plan`` (pairs sorted by
  (expert, token, slot); combine in slot order, P6 policy), built by the CUDA counting
  sort ``build_plan_cuda`` and summed by the CUDA slot sum.
* Routed experts: P5 ``Sm90FusedMoeMlp`` (MXFP8 activations x MXFP4 frozen weights on
  FP8 WGMMA, clamp-SwiGLU, ``* p_s`` once); needs the ``KERNEL_ALIGN_MOE_SM90=1``
  build. Each row's bytes depend only on that row and its expert.
* Shared expert: ``DeepGemmSharedExpertProvider`` (FP8 DeepGEMM, clamped SwiGLU).

Activations are MXFP8-quantized once per token; the codes / scales are gathered per
row (the quantization is row-local, so this equals quantizing the rows) and are what
travels between ranks.

Expert parallelism (``ep_group`` of size R): rank r owns experts
``[r * E / R, (r + 1) * E / R)`` and its own tokens; the global token id is
rank-major. A rank's rows leave already sorted by (expert, token, slot); the receiver
stable-sorts what arrives (source-rank major) by local expert, which gives
(expert, global token, slot), the EP = 1 order. Results go back along the reverse
splits, so the sender gets them in its own plan order and combines exactly as with
EP = 1. Every rank joins every collective, with or without tokens. Hence the block
output does not depend on R (``scripts/check_dsv4_moe_ep.py``). The collectives are
NCCL ``all_to_all_single`` (pure byte movement); the split sizes are read on the host
(one sync per forward).

Backward (``_MoEBlockFn``; expert, shared-expert and gate weights frozen): the output
gradient goes to every row of its token, the routed experts run the P5 SM90 backward
(``z`` recomputed) on the ranks that own them, along the forward's splits; ``dx`` is
the slot-ordered fan-in plus the shared expert's ``dx``, plus the router's ``dx`` from
``dL/dweights``, summed in FP32 and cast once.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from torch import Tensor

from rl_engine.kernels.ops.cuda.moe import build_plan_cuda, combine_cuda, mxfp8_act_quant_fwd_cuda
from rl_engine.kernels.ops.pytorch.moe.route_plan import RoutePlan
from rl_engine.models.deepseek_v4.router import Dsv4Router
from rl_engine.moe.contract import ExpertBatch
from rl_engine.moe.mx_format import MXTensor


def _routed_backend():
    from rl_engine.moe.backends.sm90_fused_mlp import Sm90FusedMoeMlp

    return Sm90FusedMoeMlp()


def _shared_backend():
    from rl_engine.moe.backends.deepgemm_shared_expert import DeepGemmSharedExpertProvider

    return DeepGemmSharedExpertProvider()


@dataclass
class _Rows:
    """Dispatched rows in expert order: MXFP8 codes / scales, route weights, offsets."""

    codes: Tensor  # uint8 [Q, H]
    scales: Tensor  # uint8 [Q, H / 32]
    p_s: Tensor  # FP32 [Q]
    offsets: Tensor  # int32 [E_local + 1]


@dataclass
class _Exchange:
    """What the return all-to-all needs from the dispatch one."""

    local_perm: Tensor  # int64 [Q]: row q of the expert-ordered batch is arrival row local_perm[q]
    send_splits: list[int]
    recv_splits: list[int]


def _all_to_all(src: Tensor, out_rows: int, out_splits, in_splits, group) -> Tensor:
    out = torch.empty((out_rows, *src.shape[1:]), dtype=src.dtype, device=src.device)
    dist.all_to_all_single(
        out,
        src.contiguous(),
        output_split_sizes=out_splits,
        input_split_sizes=in_splits,
        group=group,
    )
    return out


def ep_dispatch(rows: _Rows, plan: RoutePlan, group) -> tuple[_Rows, _Exchange]:
    """Send each rank the rows of its experts; return them in (expert, global token, slot)."""
    world = dist.get_world_size(group)
    experts = plan.num_experts
    if experts % world:
        raise ValueError(f"{experts} experts do not split over {world} ranks")
    local = experts // world
    send_counts = (rows.offsets[1:] - rows.offsets[:-1]).contiguous()  # int32 [E]
    recv_counts = torch.empty_like(send_counts)  # [R * E_local]: source-major
    dist.all_to_all_single(recv_counts, send_counts, group=group)
    host = torch.stack([send_counts, recv_counts]).cpu()  # the one host sync
    send_splits = host[0].view(world, local).sum(1).tolist()
    recv_splits = host[1].view(world, local).sum(1).tolist()
    total = sum(recv_splits)

    codes = _all_to_all(rows.codes, total, recv_splits, send_splits, group)
    scales = _all_to_all(rows.scales, total, recv_splits, send_splits, group)
    p_s = _all_to_all(rows.p_s, total, recv_splits, send_splits, group)

    expert_of_arrival = torch.repeat_interleave(
        torch.arange(local, device=codes.device).repeat(world),
        recv_counts.to(torch.int64),
        output_size=total,
    )
    sorted_e, local_perm = torch.sort(expert_of_arrival, stable=True)
    bounds = torch.arange(local + 1, device=codes.device)
    offsets = torch.searchsorted(sorted_e, bounds).to(torch.int32)
    out = _Rows(codes[local_perm], scales[local_perm], p_s[local_perm], offsets)
    return out, _Exchange(local_perm, send_splits, recv_splits)


def ep_send(t: Tensor, ex: _Exchange, group) -> Tensor:
    """Send an extra per-row tensor (plan order) along the dispatch splits; the receiver
    gets it in its expert order, like the dispatched rows."""
    arrival = _all_to_all(t, sum(ex.recv_splits), ex.recv_splits, ex.send_splits, group)
    return arrival[ex.local_perm]


def ep_return(y_local: Tensor, ex: _Exchange, group) -> Tensor:
    """Send expert outputs back; the sender receives them in its own plan order."""
    arrival = torch.empty_like(y_local)
    arrival[ex.local_perm] = y_local
    return _all_to_all(arrival, sum(ex.send_splits), ex.send_splits, ex.recv_splits, group)


@dataclass
class Dsv4MoE:
    router: Dsv4Router  # full gate: [E, H]
    w1: MXTensor  # e2m1 [E_local, 2F, H] (gate rows then up rows), frozen
    w2: MXTensor  # e2m1 [E_local, H, F], frozen
    shared_fc1: Tensor | None = None  # BF16 [2F_s, H]
    shared_fc2: Tensor | None = None  # BF16 [H, F_s]
    # Checkpoint FP8 shared expert (used instead of shared_fc1 / shared_fc2):
    # ``((q [2F_s, H], scale), (q [H, F_s], scale))``, 128x128 FP32 block scales.
    shared_fp8: tuple | None = None
    ep_group: object = None  # torch.distributed process group; None = EP 1
    path: str = "two_launch"  # P5 path: "two_launch" (faster) | "fused"
    check_finite: bool = True  # MXFP8 quant fail-closed check (one host sync; off for graphs)
    lora: object = None  # MoeLora (lora.py): routed fc1 / fc2 and shared-expert adapters
    _routed: object = field(default=None, init=False, repr=False)
    _shared: object = field(default=None, init=False, repr=False)
    _shared_w: object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._routed = _routed_backend()
        if self.shared_fp8 is not None:
            if self.shared_fc1 is not None:
                raise ValueError("give shared_fc1 / shared_fc2 or shared_fp8, not both")
            self._shared = _shared_backend()
            self._shared_w = self._shared.prepare_fp8(*self.shared_fp8)
        elif self.shared_fc1 is not None:
            self._shared = _shared_backend()
            self._shared_w = self._shared.prepare(self.shared_fc1, self.shared_fc2)
        world = 1 if self.ep_group is None else dist.get_world_size(self.ep_group)
        if int(self.w1.shape[0]) * world != self.num_experts:
            raise ValueError(
                f"w1 holds {self.w1.shape[0]} experts; expected {self.num_experts} / {world}"
            )

    @property
    def num_experts(self) -> int:
        return int(self.router.gate_weight.shape[0])

    def _expert_batch(self, rows: _Rows, hidden: int) -> tuple[ExpertBatch, MXTensor]:
        from rl_engine.moe.backends.sm90_fused_mlp import PROFILE

        q = rows.codes.shape[0]
        x_q = MXTensor(rows.codes, rows.scales, "e4m3", (q, hidden))
        # P5 reads only x_q; ``x`` carries the shape for its checks (no storage).
        shape_only = torch.zeros(1, 1, dtype=torch.bfloat16, device=rows.codes.device)
        batch = ExpertBatch(
            x=shape_only.expand(q, hidden),
            expert_offsets=rows.offsets,
            p_s=rows.p_s,
            w1=self.w1,
            w2=self.w2,
            lora=None,
            output_slot=torch.zeros(1, dtype=torch.int32, device=rows.codes.device).expand(q),
            numeric_profile=PROFILE,
        )
        return batch, x_q

    def _expert_forward(self, rows: _Rows, hidden: int) -> Tensor:
        if rows.codes.shape[0] == 0:
            return torch.empty(0, hidden, dtype=torch.bfloat16, device=rows.codes.device)
        batch, x_q = self._expert_batch(rows, hidden)
        return self._routed.forward(batch, x_q, path=self.path)

    def _expert_backward(self, rows: _Rows, dy: Tensor, hidden: int) -> tuple[Tensor, Tensor]:
        q = rows.codes.shape[0]
        if q == 0:
            dev = rows.codes.device
            return torch.empty(0, hidden, device=dev), torch.empty(0, device=dev)
        batch, x_q = self._expert_batch(rows, hidden)
        return self._routed.backward(batch, x_q, dy)

    def routed_rows(
        self, x: Tensor, weights: Tensor, plan: RoutePlan
    ) -> tuple[Tensor, tuple[_Rows, _Exchange | None]]:
        """Weighted expert outputs BF16 ``[T * 6, H]`` in this rank's plan order, plus the
        state :meth:`routed_backward` needs."""
        t, hidden = x.shape
        if t:
            x_q = mxfp8_act_quant_fwd_cuda(x, check_finite=self.check_finite)
            codes, scales = x_q.codes, x_q.scales
        else:
            codes = torch.empty(0, hidden, dtype=torch.uint8, device=x.device)
            scales = torch.empty(0, hidden // 32, dtype=torch.uint8, device=x.device)
        token = plan.token
        rows = _Rows(
            codes.index_select(0, token),
            scales.index_select(0, token),
            weights.reshape(-1).to(torch.float32).index_select(0, plan.perm),
            plan.expert_offsets,
        )
        if self._routed_lora():
            return self._routed_lora_forward(x, rows, plan, hidden)
        if self.ep_group is None:
            return self._expert_forward(rows, hidden), (rows, None)
        local, ex = ep_dispatch(rows, plan, self.ep_group)
        return ep_return(self._expert_forward(local, hidden), ex, self.ep_group), (local, ex)

    def routed_backward(
        self, dy_rows: Tensor, state: tuple[_Rows, _Exchange | None]
    ) -> tuple[Tensor, Tensor]:
        """``(dx_rows FP32 [T * 6, H], dp_s FP32 [T * 6])`` in plan order from ``dy_rows``
        (BF16, plan order); with EP the rows travel along the forward's splits."""
        rows, ex = state[0], state[1]
        hidden = dy_rows.shape[1]
        if len(state) == 3:
            raise RuntimeError("routed-expert LoRA state: use routed_lora_backward")
        if ex is None:
            return self._expert_backward(rows, dy_rows, hidden)
        arrival = _all_to_all(
            dy_rows, sum(ex.recv_splits), ex.recv_splits, ex.send_splits, self.ep_group
        )
        dx_l, dps_l = self._expert_backward(rows, arrival[ex.local_perm], hidden)
        return ep_return(dx_l, ex, self.ep_group), ep_return(dps_l, ex, self.ep_group)

    # LoRA (lora.py, profile dsv4-lora-v1) --------------------------------------
    def _routed_lora(self) -> bool:
        return self.lora is not None and (self.lora.fc1 is not None or self.lora.fc2 is not None)

    def _shared_lora(self) -> bool:
        return self.lora is not None and (
            self.lora.shared_fc1 is not None or self.lora.shared_fc2 is not None
        )

    def _routed_lora_rows(self, rows: _Rows, hidden: int, z_add, z_token):
        """Expert compute on this rank's rows with the routed adapters:
        fc1_z (+ ``z_add[z_token]``) -> clamp-SwiGLU * p_s -> MXFP8 -> fc3 (+ fc2 delta).
        Returns ``(y BF16 [Q, H], h BF16 [Q, F])``."""
        from rl_engine.models.deepseek_v4.lora import lora_delta

        lo = self.lora
        q = rows.codes.shape[0]
        dev = rows.codes.device
        if q == 0:
            f = int(self.w1.shape[1]) // 2
            return (
                torch.empty(0, hidden, dtype=torch.bfloat16, device=dev),
                torch.empty(0, f, dtype=torch.bfloat16, device=dev),
            )
        batch, x_q = self._expert_batch(rows, hidden)
        z = self._routed.fc1_z(batch, x_q)
        h_q, h = self._routed.swiglu_quant(z, rows.p_s, z_add, z_token)
        del z
        y = self._routed.fc3(batch, h_q)
        if lo.fc2 is not None:
            y = lora_delta(h, lo.fc2).add_(y).to(torch.bfloat16)
        return y, h

    def _routed_lora_forward(self, x: Tensor, rows: _Rows, plan: RoutePlan, hidden: int):
        """Routed experts with the shared routed adapters, EP = 1 or EP > 1."""
        from rl_engine.models.deepseek_v4.lora import lora_delta, lora_hidden, lora_up

        lo = self.lora
        state = {"x": x, "token": plan.token, "inv": plan.inv}
        if self.ep_group is None:
            # The fc1 delta depends only on the token: once per token, gathered by row.
            z_tok = lora_delta(x, lo.fc1) if lo.fc1 is not None else None
            y, h = self._routed_lora_rows(rows, hidden, z_tok, plan.token)
            state["h"] = h
            return y, (rows, None, state)
        # EP: u1 = BF16(x A1^T) travels with the row; the expert's rank applies B1.
        u1 = None
        if lo.fc1 is not None:
            u1 = lora_hidden(x, lo.fc1).index_select(0, plan.token)
        local, ex = ep_dispatch(rows, plan, self.ep_group)
        z_add = token = None
        if u1 is not None:
            state["u1_local"] = ep_send(u1, ex, self.ep_group)
            z_add = lora_up(state["u1_local"], lo.fc1)
            token = torch.arange(z_add.shape[0], device=z_add.device)
        y_local, h = self._routed_lora_rows(local, hidden, z_add, token)
        state["h"] = h
        return ep_return(y_local, ex, self.ep_group), (local, ex, state)

    def routed_lora_backward(self, dy_rows: Tensor, state, grads: dict):
        """``(dx_rows FP32, dp_s FP32, dx_tokens FP32 | None)`` in plan order; adds the
        routed adapters' ``(dA, dB)`` to ``grads``, summed over the EP group.

        The fc1 adapter's dX and dA are formed per token (``du`` summed over the token's
        rows in slot order); with EP, ``du`` returns along the forward's splits.
        """
        from rl_engine.models.deepseek_v4.lora import (
            lora_backward,
            lora_delta,
            lora_hidden,
            lora_up,
        )

        rows, ex, ls = state
        lo = self.lora
        hidden = dy_rows.shape[1]
        group = self.ep_group
        dev = dy_rows.device
        if ex is not None:
            dy_local = ep_send(dy_rows, ex, group)
            u1_local = ls.get("u1_local")
        else:
            dy_local = dy_rows
            u1_local = None
            if lo.fc1 is not None:
                u1_local = lora_hidden(ls["x"], lo.fc1).index_select(0, ls["token"])
        q = rows.codes.shape[0]
        if q:
            batch, x_q = self._expert_batch(rows, hidden)
            z_add = token = None
            if lo.fc1 is not None:
                if ex is None:
                    z_add, token = lora_delta(ls["x"], lo.fc1), ls["token"]
                else:
                    z_add = lora_up(u1_local, lo.fc1)
            dh_add = None
            if lo.fc2 is not None:
                dh_add, da, db = lora_backward(dy_local, ls["h"], lo.fc2)
                grads["fc2"] = (da, db)
            if z_add is not None and token is not None:
                z_add = z_add.index_select(0, token)
            dx_local, dps_local, dz = self._routed.backward(
                batch, x_q, dy_local, z_add=z_add, dh_add=dh_add, return_dz=True
            )
        else:
            dx_local = torch.empty(0, hidden, device=dev)
            dps_local = torch.empty(0, device=dev)
            dz = None
            if lo.fc2 is not None:
                grads["fc2"] = (
                    torch.zeros(lo.fc2.a.shape, device=dev),
                    torch.zeros(lo.fc2.b.shape, device=dev),
                )
        du_local = None
        if lo.fc1 is not None:
            ad = lo.fc1
            if q:
                dd = dz.float() * ad.scale  # [Q, 2F]
                db1 = dd.t() @ u1_local.float()
                du_local = dd @ ad.b.detach().float()  # [Q, r]
            else:
                db1 = torch.zeros(ad.b.shape, device=dev)
                du_local = torch.empty(0, ad.rank, device=dev)
        if ex is not None:
            dx_rows = ep_return(dx_local, ex, group)
            dp_s = ep_return(dps_local, ex, group)
            du_rows = ep_return(du_local, ex, group) if du_local is not None else None
        else:
            dx_rows, dp_s, du_rows = dx_local, dps_local, du_local
        dx_tokens = None
        if lo.fc1 is not None:
            ad, x = lo.fc1, ls["x"]
            if du_rows.shape[0]:
                du_tok = du_rows[ls["inv"]].sum(dim=1)  # [T, r], slot order
            else:
                du_tok = torch.zeros(x.shape[0], ad.rank, device=dev)
            grads["fc1"] = (du_tok.t() @ x.float(), db1)
            dx_tokens = du_tok @ ad.a.detach().float()
        if ex is not None:
            for name in ("fc1", "fc2"):
                if name in grads:
                    for t in grads[name]:
                        dist.all_reduce(t, group=group)
        return dx_rows, dp_s, dx_tokens

    def shared_forward(self, x: Tensor):
        """Shared-expert output BF16 ``[T, H]`` and its saved state (LoRA-aware)."""
        if not self._shared_lora():
            return self._shared.forward(x, self._shared_w)
        from rl_engine.kernels.ops.cuda.activation import silu_and_mul_clamp as swiglu
        from rl_engine.models.deepseek_v4.lora import lora_apply

        lo, w = self.lora, self._shared_w
        z = w.fc1(x.contiguous())
        if lo.shared_fc1 is not None:
            z = lora_apply(z, x, lo.shared_fc1)
        if not z.shape[0]:
            empty = torch.empty(0, w.fc2.out_features, device=x.device, dtype=torch.bfloat16)
            return empty, {"z": z, "x": x, "h": None}
        q, scale, h = swiglu.silu_and_mul_clamp_quant_cuda(
            z, self._shared.swiglu_limit, return_h=True
        )
        y = w.fc2.forward_quantized((q, scale))
        if lo.shared_fc2 is not None:
            y = lora_apply(y, h, lo.shared_fc2)
        return y, {"z": z, "x": x, "h": h}

    def shared_backward(self, dout: Tensor, saved, grads: dict) -> Tensor:
        """``dL/dx`` FP32 through the shared expert; adds its adapters' ``(dA, dB)``."""
        if not self._shared_lora():
            return self._shared.backward_dx(dout, saved)
        from rl_engine.kernels.ops.cuda.activation import silu_and_mul_clamp as swiglu
        from rl_engine.models.deepseek_v4.lora import lora_backward

        if self._shared.backward != "fp8":
            raise NotImplementedError("shared-expert LoRA needs the FP8 backward")
        lo, w = self.lora, self._shared_w
        dy = dout.to(torch.bfloat16).contiguous()
        if not dy.shape[0]:
            return torch.empty(0, w.fc1.in_features, device=dy.device, dtype=torch.float32)
        dh = w.fc2.backward_dx(dy)
        if lo.shared_fc2 is not None:
            dx_l, da, db = lora_backward(dy, saved["h"], lo.shared_fc2)
            grads["shared_fc2"] = (da, db)
            dh = (dh.float() + dx_l).to(torch.bfloat16)
        dz = swiglu.silu_and_mul_clamp_backward_cuda(dh, saved["z"], self._shared.swiglu_limit)
        dx = w.fc1.backward_dx(dz).float()
        if lo.shared_fc1 is not None:
            dx_l, da, db = lora_backward(dz, saved["x"], lo.shared_fc1)
            grads["shared_fc1"] = (da, db)
            dx = dx + dx_l
        return dx

    def lora_params(self) -> list[Tensor]:
        """The adapters' ``A`` / ``B`` tensors, in the order the autograd returns grads."""
        if self.lora is None:
            return []
        return [t for _, ad in self.lora.adapters() for t in (ad.a, ad.b)]

    def forward(
        self, x: Tensor, residual: Tensor | None = None, input_ids: Tensor | None = None
    ) -> Tensor:
        """MoE block output BF16 ``[T, H]`` (``+ residual`` once when given).

        Differentiable in ``x`` (and ``residual``) when they require grad; the expert,
        shared-expert and gate weights are frozen.
        """
        return _MoEBlockFn.apply(x.contiguous(), residual, input_ids, self, *self.lora_params())


class _MoEBlockFn(torch.autograd.Function):
    """Whole-block autograd so ``dx`` is summed once in FP32 (P6 ``fused_dx_fanin``).

    Backward, all FP32 until the final BF16 cast:
        dy_rows   = dout[token]                         (exact gather, BF16)
        dx_rows, dp_s = routed experts backward           (P5 SM90, EP along the same splits)
        dweights  = dp_s[inv];  dx_router = router backward (frozen gate)
        dx        = (slot-ordered sum of dx_rows + dx_shared) + dx_router  -> BF16
    With LoRA (``moe.lora``), the routed fc1 adapter's per-token dX joins before the cast,
    and the adapters' ``A`` / ``B`` tensors (inputs after ``moe``) get their gradients.
    """

    @staticmethod
    def forward(ctx, x, residual, input_ids, moe, *lora_params):
        from rl_engine.kernels.ops.cuda.router import route_topk_cuda
        from rl_engine.models.deepseek_v4.router import gate_logits

        r = moe.router
        logits = gate_logits(x, r.gate_weight)
        hash_ids = input_ids if r.tid2eid is not None else None
        if r.tid2eid is not None and input_ids is None:
            raise ValueError("DSv4 hash routing requires input_ids")
        weights, ids = route_topk_cuda(
            logits, r.bias, r.routed_scaling_factor, input_ids=hash_ids, tid2eid=r.tid2eid
        )
        plan = build_plan_cuda(ids, moe.num_experts)
        y_rows, state = moe.routed_rows(x, weights, plan)
        shared, shared_saved = None, None
        if moe._shared is not None:
            shared, shared_saved = moe.shared_forward(x)
        out = combine_cuda(y_rows, plan, shared, residual)
        need = ctx.needs_input_grad[0] or any(ctx.needs_input_grad[4:])
        if need:
            ctx.moe, ctx.plan, ctx.state, ctx.shared_saved = moe, plan, state, shared_saved
            ctx.save_for_backward(logits, ids)
            ctx.x_dtype = x.dtype
        return out

    @staticmethod
    def backward(ctx, dout):
        from rl_engine.kernels.ops.cuda.moe import dispatch_backward_cuda
        from rl_engine.kernels.ops.cuda.router import route_backward_dx_cuda
        from rl_engine.kernels.ops.pytorch.moe.route_plan import weights_backward

        n_params = len(ctx.needs_input_grad) - 4
        d_res = dout if ctx.needs_input_grad[1] else None
        if not ctx.needs_input_grad[0] and not any(ctx.needs_input_grad[4:]):
            return (None, d_res, None, None) + (None,) * n_params
        moe, plan = ctx.moe, ctx.plan
        logits, ids = ctx.saved_tensors
        dout = dout.to(torch.bfloat16).contiguous()
        dy_rows = dout.index_select(0, plan.token)
        grads: dict = {}
        dx_tokens = None
        if len(ctx.state) == 3:
            dx_rows, dp_s, dx_tokens = moe.routed_lora_backward(dy_rows, ctx.state, grads)
        else:
            dx_rows, dp_s = moe.routed_backward(dy_rows, ctx.state)
        dx_shared = None
        if moe._shared is not None:
            dx_shared = moe.shared_backward(dout, ctx.shared_saved, grads)
        dx = dispatch_backward_cuda(dx_rows, plan, dx_shared)
        r = moe.router
        dx_router = route_backward_dx_cuda(
            logits,
            ids,
            weights_backward(dp_s, plan),
            r.gate_weight,
            r.routed_scaling_factor,
            torch.float32,
        )
        dx = dx + dx_router
        if dx_tokens is not None:
            dx = dx + dx_tokens
        dx = dx.to(ctx.x_dtype) if ctx.needs_input_grad[0] else None
        lora_grads = []
        if moe.lora is not None:
            for name, ad in moe.lora.adapters():
                if name not in grads:  # no rows reached this adapter
                    grads[name] = (torch.zeros_like(ad.a), torch.zeros_like(ad.b))
                da, db = grads[name]
                lora_grads += [da.to(ad.a.dtype), db.to(ad.b.dtype)]
        return (dx, d_res, None, None, *lora_grads)
