# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""T01 synthetic recorded producer for router validation development.

WS1/WS2 may be closed using only recorded/synthetic
producers, without waiting for live downstream — until the start kit
publishes real fixtures, validation is
developed against deterministic synthetic artifacts produced here.

Determinism: seeded generators only; identical (case_id, seed) always
reproduces bit-identical artifacts, which L1 relies on.
"""

from __future__ import annotations

import hashlib

import torch

from rl_engine.moe.naive_topk import DEFAULT_E, K, naive_topk
from rl_engine.moe.validation.t01_fingerprint import (
    Artifact,
    EnvelopeFields,
    RouteIdentity,
    RouteRow,
)

_SCALE = 1.5
_EPS = 1e-20


def _six_way_sum(a: list[float]) -> float:
    """FP32 fixed 6-way tree: ((a0+a1)+(a2+a3))+(a4+a5).

    The reduction order is contract-defined; replacing it with ``torch.sum``
    or regrouping the additions can change the result bits.
    """
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
    """Produce deterministic Learned-mode Core rows via naive_topk.

    Follows Learned math: q = s + b, ids = stable_topk(q),
    a_i = s[ids_i] (pre-bias weight source), p_i = a_i / Z, w_i = p_i * 1.5,
    with the fixed FP32 6-way tree for S.
    """
    g = torch.Generator().manual_seed(seed)
    if bias is None:
        bias = torch.randn(e, generator=g, dtype=torch.float32) * 0.1

    rows: list[RouteRow] = []
    identity = RouteIdentity(
        checkpoint_id=f"ckpt-{case_id}",
        weight_id=f"w-{case_id}",
        bias_fingerprint=_fp(bias),
    )
    for layer in layers:
        z = torch.randn(t_rows, e, generator=g, dtype=torch.float32)
        u = torch.nn.functional.softplus(z)
        s = torch.sqrt(u)
        q = s + bias
        ids, _ = naive_topk(q)
        for t in range(t_rows):
            a = [float(s[t, int(ids[t, i])]) for i in range(K)]
            ssum = _six_way_sum(a)
            zeta = torch.tensor(ssum, dtype=torch.float32) + torch.tensor(_EPS, dtype=torch.float32)
            for i in range(K):
                p = torch.tensor(a[i], dtype=torch.float32) / zeta
                rows.append(
                    RouteRow(
                        global_token_id=t,
                        input_token_id=t,
                        absolute_layer=layer,
                        router_mode="learned",
                        topk_index=i,
                        logical_expert_id=int(ids[t, i]),
                        valid=True,
                        invalid_reason=None,
                        route_weight=float(p * _SCALE),
                        weight_score=a[i],
                        selection_score=float(q[t, int(ids[t, i])]),
                    )
                )
    meta = {
        "case_id": case_id,
        "checkpoint_id": identity.checkpoint_id,
        "weight_id": identity.weight_id,
        "absolute_layer": layers[0],
        "router_mode": "learned",
        # learned layers have no tid2eid table: the irrelevant fingerprint
        # stays absent (encoded as present=false), never a fabricated value
        "table_fingerprint": None,
        "bias_fingerprint": identity.bias_fingerprint
        or hashlib.sha256(f"b-{case_id}".encode()).hexdigest(),
        "logit_round_point": "fp32_direct",
        "tie_break_policy": "q_desc_id_asc",
        "capacity_policy": "dropless_v1",
    }
    return rows, meta


def _fp(x: torch.Tensor) -> str:
    return hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


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

    Padding rows are canonical: token ids -1, expert -1, weight/score
    +0.0, invalid/padding. physical_expert_id uses a deterministic bijection
    ``logical + placement_offset`` standing in for the versioned placement map.
    """
    env: list[EnvelopeFields] = []
    all_rows: list[RouteRow] = list(rows)
    for src, r in enumerate(rows):
        env.append(
            EnvelopeFields(
                run_id=run_id,
                engine_id=engine_id,
                attempt_id=attempt_id,
                source_row=src,
                physical_expert_id=r.logical_expert_id + placement_offset,
                rank=rank,
            )
        )
    for p in range(padding_rows):
        src = len(rows) + p
        all_rows.append(
            RouteRow(
                global_token_id=-1,
                input_token_id=-1,
                absolute_layer=rows[0].absolute_layer if rows else 0,
                router_mode=rows[0].router_mode if rows else "learned",
                topk_index=p % K,
                logical_expert_id=-1,
                valid=False,
                invalid_reason="padding",
                route_weight=0.0,
                weight_score=0.0,
                selection_score=0.0,
            )
        )
        env.append(
            EnvelopeFields(
                run_id=run_id,
                engine_id=engine_id,
                attempt_id=attempt_id,
                source_row=src,
                physical_expert_id=-1,
                rank=rank,
            )
        )
    return Artifact(identity=identity, rows=all_rows, envelopes=env)


def repack_rows(rows: list[RouteRow], *, batch_size: int) -> list[RouteRow]:
    """L2 helper: physically reorder rows as if batch packing changed.

    Rows are reversed within each ``batch_size`` chunk; per-token content
    is untouched. L2 must prove this physical-layout change does NOT alter
    semantic hashes — canonical serialization keys routing-decision units
    by ``(absolute_layer, global_token_id)`` and orders rows within a unit
    by ``topk_index``, so hashes stay stable under any row permutation.
    """
    out: list[RouteRow] = []
    for start in range(0, len(rows), batch_size):
        out.extend(reversed(rows[start : start + batch_size]))
    return out


def shard_rows(rows: list[RouteRow], parts: int) -> list[list[RouteRow]]:
    """WS2 helper: distribute whole routing units into ``parts`` shards.

    Stands in for sequence partition (CP/DP): every token lands on exactly
    one shard **with all of its slot rows** — cutting mid-token would split
    one routing decision across ranks and falsely trip the duplicate-
    ownership check. Rows are grouped by ``global_token_id`` (first-seen
    order) and dealt round-robin to the shards; per-token content is
    untouched. The WS2 cross-config check must prove the union of shards
    carries the same per-unit semantic hashes as the unsharded base.
    """
    if parts < 1:
        raise ValueError(f"parts must be >= 1, got {parts}")
    units: list[list[RouteRow]] = []
    index_of: dict[int, int] = {}
    for r in rows:
        if r.global_token_id not in index_of:
            index_of[r.global_token_id] = len(units)
            units.append([])
        units[index_of[r.global_token_id]].append(r)
    shards: list[list[RouteRow]] = [[] for _ in range(parts)]
    for i, unit in enumerate(units):
        shards[i % parts].extend(unit)
    return shards
