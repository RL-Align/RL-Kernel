# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Synthetic P2-F-ATTN / P2-F-OPROJ recorded cases. T01 may replace blobs later."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from rl_engine.kernels.p2.candidate_plan import CandidatePlan
from rl_engine.kernels.p2.contract import (
    CONCAT_Z_DIM,
    GROUP_FLAT_DIM,
    HEAD_DIM,
    HIDDEN_SIZE,
    N_O_PROJ_GROUPS,
    N_Q_HEADS,
    O_LORA_RANK,
    LayerType,
)
from rl_engine.kernels.p2.fixtures.recorded import tensor_checksum
from rl_engine.kernels.p2.o_proj.rope_consumer import fixture_cos_sin
from rl_engine.kernels.p2.state_gate import StateGateVerdict, synthetic_pass_verdict


@dataclass
class AttnCase:
    name: str
    q: Tensor
    k: Tensor
    v: Tensor
    sink: Tensor
    plan: CandidatePlan
    state_gate: StateGateVerdict


@dataclass
class OProjCase:
    name: str
    o: Tensor
    w_a: Tensor
    w_b: Tensor
    cos: Tensor
    sin: Tensor
    positions: Tensor


def make_attn_case(
    name: str,
    *,
    layer_type: LayerType | str,
    tokens: int,
    n_compressed: int,
    n_recent: int,
    seed: int = 0,
    sink_mode: str = "normal",
    invalid_prefix: int = 0,
    invalid_recent: int = 0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    state_status: str = "PASS",
) -> AttnCase:
    torch.manual_seed(seed)
    n = n_compressed + n_recent
    q = torch.randn(tokens, N_Q_HEADS, HEAD_DIM, dtype=dtype, device=device) * 0.02
    k = torch.randn(n, HEAD_DIM, dtype=dtype, device=device) * 0.02
    v = torch.randn(n, HEAD_DIM, dtype=dtype, device=device) * 0.02
    valid = torch.ones(n, dtype=torch.bool, device=device)
    if n and invalid_prefix:
        valid[: min(invalid_prefix, n_compressed)] = False
    if n and invalid_recent and n_recent:
        valid[n_compressed : n_compressed + min(invalid_recent, n_recent)] = False
    kind = None
    if n:
        kind = torch.zeros(n, dtype=torch.int64, device=device)
        if n_recent:
            kind[n_compressed:] = 1
    if sink_mode == "dominates":
        sink = torch.full((tokens, N_Q_HEADS), 80.0, dtype=torch.float32, device=device)
    elif sink_mode == "shared":
        sink = torch.randn(N_Q_HEADS, dtype=torch.float32, device=device) * 0.1
    else:
        sink = torch.randn(tokens, N_Q_HEADS, dtype=torch.float32, device=device) * 0.1
    plan = CandidatePlan(
        layer_type=layer_type,
        n_compressed=n_compressed,
        n_recent=n_recent,
        valid=valid,
        kind=kind,
    )
    if state_status == "PASS":
        gate = synthetic_pass_verdict(tag=name)
    else:
        gate = StateGateVerdict(status=state_status, source="synthetic_recorded", details=name)
    return AttnCase(name=name, q=q, k=k, v=v, sink=sink, plan=plan, state_gate=gate)


def named_attn_catalog(device: str = "cpu") -> dict[str, AttnCase]:
    specs = (
        ("c0_recent_only", "C0", 2, 0, 8, 1, "normal", 0, 0),
        ("c0_recent_1", "C0", 1, 0, 1, 2, "normal", 0, 0),
        ("csa_selected_c4", "C4", 2, 8, 8, 3, "normal", 0, 0),
        ("hca_all_c128", "C128", 2, 3, 8, 4, "normal", 0, 0),
        ("sink_dominates", "C4", 1, 4, 4, 5, "dominates", 0, 0),
        ("candidate_empty", "C0", 2, 0, 0, 6, "normal", 0, 0),
        ("candidate_partial", "C4", 2, 6, 6, 7, "normal", 2, 1),
        ("shared_sink", "C0", 2, 0, 4, 8, "shared", 0, 0),
        ("c0_recent_127", "C0", 1, 0, 127, 9, "normal", 0, 0),
        ("c0_recent_128", "C0", 1, 0, 128, 10, "normal", 0, 0),
        ("csa_prefix_512", "C4", 1, 512, 128, 11, "normal", 0, 0),
        ("hca_zero_completed", "C128", 1, 0, 8, 12, "normal", 0, 0),
        ("hca_one_completed", "C128", 1, 1, 8, 13, "normal", 0, 0),
    )
    catalog = {}
    for name, layer, tokens, n_c, n_r, seed, sink_mode, inv_p, inv_r in specs:
        catalog[name] = make_attn_case(
            name,
            layer_type=layer,
            tokens=tokens,
            n_compressed=n_c,
            n_recent=n_r,
            seed=seed,
            sink_mode=sink_mode,
            invalid_prefix=inv_p,
            invalid_recent=inv_r,
            device=device,
        )
    return catalog


def catalog_checksums(device: str = "cpu") -> dict[str, str]:
    """Stable CPU seed checksums for T06 recorded fixtures (T01 may replace)."""

    out = {}
    for name, case in named_attn_catalog(device=device).items():
        out[name] = "|".join(
            [
                tensor_checksum(case.q),
                tensor_checksum(case.k),
                tensor_checksum(case.v),
                tensor_checksum(case.sink),
                tensor_checksum(case.plan.valid),
            ]
        )
    return out


def make_oproj_case(
    name: str,
    *,
    tokens: int = 1,
    seed: int = 0,
    identifiable_groups: bool = False,
    device: str = "cpu",
) -> OProjCase:
    torch.manual_seed(seed)
    o = torch.randn(tokens, N_Q_HEADS, HEAD_DIM, device=device) * 0.02
    w_a = torch.randn(N_O_PROJ_GROUPS, O_LORA_RANK, GROUP_FLAT_DIM, device=device) * 0.02
    w_b = torch.randn(HIDDEN_SIZE, CONCAT_Z_DIM, device=device) * 0.02
    if identifiable_groups:
        o.zero_()
        w_a.zero_()
        w_b.zero_()
        for group in range(N_O_PROJ_GROUPS):
            o[:, group * 8 : (group + 1) * 8, :] = float(group + 1)
            # W_a[g] maps flattened group to a one-hot-ish rank-0 channel
            w_a[group, 0, :] = 1.0 / GROUP_FLAT_DIM
        # W_b maps concat Z group slots back: Y[g] = Z[g*1024]
        for group in range(N_O_PROJ_GROUPS):
            w_b[group, group * O_LORA_RANK] = 1.0
    positions = torch.arange(tokens, device=device)
    if identifiable_groups:
        cos = torch.ones(tokens, 32, device=device)
        sin = torch.zeros(tokens, 32, device=device)
    else:
        cos, sin = fixture_cos_sin(positions)
    return OProjCase(name=name, o=o, w_a=w_a, w_b=w_b, cos=cos, sin=sin, positions=positions)
