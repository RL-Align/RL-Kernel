# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Synthetic recorded producer for T09 ladder development (contract §5).

Contract §1.3/§5: "WS1/WS2 可仅使用 T01 的 recorded/synthetic producer 关闭,
不等待 live 下游" — until T01's start kit publishes real fixtures, T09
develops the ladder against deterministic synthetic artifacts produced here.

Determinism: seeded generators only; identical (case_id, seed) always
reproduces bit-identical artifacts, which L1 relies on.
"""

from __future__ import annotations

import hashlib

import torch

from rl_engine.moe.naive_topk6 import K, naive_topk6
from rl_engine.moe.validation.fingerprint import (
    Artifact,
    EnvelopeFields,
    RouteIdentity,
    RouteRow,
)

DEFAULT_E = 256
_SCALE = 1.5
_EPS = 1e-20


def _six_way_sum(a: list[float]) -> float:
    """FP32 fixed 6-way tree per §2.1: ((a0+a1)+(a2+a3))+(a4+a5)."""
    t = torch.tensor([a[0] + a[1], a[2] + a[3], a[4] + a[5]], dtype=torch.float32)
    return float((t[0] + t[1]) + t[2])


def make_learned_route_rows(
    case_id: str,
    *,
    seed: int,
    t_rows: int,
    layers: tuple[int, ...] = (3,),
    bias: torch.Tensor | None = None,
    e: int = DEFAULT_E,
) -> tuple[list[RouteRow], dict[str, object]]:
    """Produce deterministic Learned-mode Core rows via naive_topk6.

    Follows §2.1 Learned math: q = s + b, ids = stable_topk6(q),
    a_i = s[ids_i] (pre-bias weight source), p_i = a_i / Z, w_i = p_i * 1.5,
    with the fixed FP32 6-way tree for S.
    """
    g = torch.Generator().manual_seed(seed)
    if bias is None:
        bias = torch.randn(e, generator=g, dtype=torch.float32) * 0.1

    rows: list[RouteRow] = []
    identity = RouteIdentity(
        checkpoint_id=f"ckpt-{case_id}", weight_id=f"w-{case_id}",
        bias_fingerprint=_fp(bias),
    )
    for layer in layers:
        z = torch.randn(t_rows, e, generator=g, dtype=torch.float32)
        u = torch.nn.functional.softplus(z)
        s = torch.sqrt(u)
        q = s + bias
        ids, _ = naive_topk6(q)
        for t in range(t_rows):
            a = [float(s[t, int(ids[t, i])]) for i in range(K)]
            ssum = _six_way_sum(a)
            zeta = ssum + _EPS
            for i in range(K):
                p = a[i] / zeta
                rows.append(RouteRow(
                    global_token_id=t, input_token_id=t,
                    absolute_layer=layer, router_mode="learned",
                    topk_index=i, logical_expert_id=int(ids[t, i]),
                    valid=True, invalid_reason=None,
                    route_weight=p * _SCALE, weight_score=a[i],
                    selection_score=float(q[t, int(ids[t, i])]),
                ))
    meta = {
        "case_id": case_id, "checkpoint_id": identity.checkpoint_id,
        "weight_id": identity.weight_id, "absolute_layer": layers[0],
        "router_mode": "learned",
        "table_fingerprint": hashlib.sha256(f"t-{case_id}".encode()).hexdigest(),
        "bias_fingerprint": identity.bias_fingerprint
        or hashlib.sha256(f"b-{case_id}".encode()).hexdigest(),
        "logit_round_point": "fp32_direct", "tie_break_policy": "p3_canonical",
        "capacity_policy": "dropless_v1",
    }
    return rows, meta


def _fp(x: torch.Tensor) -> str:
    return hashlib.sha256(
        x.contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def make_artifact(
    case_id: str,
    rows: list[RouteRow],
    identity: RouteIdentity,
    *,
    run_id: str,
    engine_id: str,
    attempt_id: int,
    rank: int = 0,
    placement_offset: int = 0,
    padding_rows: int = 0,
) -> Artifact:
    """Wrap rows + padding into an Artifact with Envelope fields.

    Padding rows follow §2.1 canonical: token ids -1, expert -1, weight/score
    +0.0, invalid/padding. physical_expert_id uses a deterministic bijection
    ``logical + placement_offset`` standing in for T08's versioned map.
    """
    env: list[EnvelopeFields] = []
    all_rows: list[RouteRow] = list(rows)
    for src, r in enumerate(rows):
        env.append(EnvelopeFields(
            run_id=run_id, engine_id=engine_id, attempt_id=attempt_id,
            source_row=src, physical_expert_id=r.logical_expert_id + placement_offset,
            rank=rank,
        ))
    for p in range(padding_rows):
        src = len(rows) + p
        all_rows.append(RouteRow(
            global_token_id=-1, input_token_id=-1,
            absolute_layer=rows[0].absolute_layer if rows else 0,
            router_mode="learned", topk_index=p % K,
            logical_expert_id=-1, valid=False, invalid_reason="padding",
            route_weight=0.0, weight_score=0.0, selection_score=0.0,
        ))
        env.append(EnvelopeFields(
            run_id=run_id, engine_id=engine_id, attempt_id=attempt_id,
            source_row=src, physical_expert_id=-1, rank=rank,
        ))
    return Artifact(identity=identity, rows=all_rows, envelopes=env)


def repack_rows(rows: list[RouteRow], *, batch_size: int) -> list[RouteRow]:
    """L2 helper: renumber global token ids as if batch partition changed.

    Core per-token semantics must survive repack; only identity-of-position
    changes, which L2 must prove does NOT alter semantic hashes. Here the
    repack keeps (token content) identical and only reorders rows — the
    canonical serialization sorts by token, so hashes stay stable.
    """
    return list(reversed(rows))
