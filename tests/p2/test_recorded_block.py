# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Recorded full-chain T06 verification: attention -> o-proj and long seq."""

from __future__ import annotations

import torch

from rl_engine.kernels.p2.attention.mqa_joint_attention_sink import MqaJointAttentionSinkOp
from rl_engine.kernels.p2.attention.oracle import mqa_joint_attention_sink_fwd
from rl_engine.kernels.p2.block import recorded_attention_block
from rl_engine.kernels.p2.contract import HIDDEN_SIZE
from rl_engine.kernels.p2.fixtures.catalog import (
    make_attn_case,
    make_oproj_case,
    named_attn_catalog,
)


def test_recorded_block_c0_csa_hca_output_shape():
    oproj = make_oproj_case("blk", tokens=1, seed=21)
    layers = (("C0", 0, 8), ("C4", 8, 8), ("C128", 3, 8))
    for layer, n_c, n_r in layers:
        case = make_attn_case(
            f"block-{layer}",
            layer_type=layer,
            tokens=1,
            n_compressed=n_c,
            n_recent=n_r,
            seed=21,
        )
        result = recorded_attention_block(
            case.q,
            case.k,
            case.v,
            case.sink,
            case.plan,
            oproj.w_a,
            oproj.w_b,
            oproj.cos,
            oproj.sin,
            case.state_gate,
        )
        assert result.y.shape == (1, HIDDEN_SIZE)
        assert torch.isfinite(result.y).all()
        mass = result.attention.debug["p"].sum(-1) + result.attention.debug["p_sink"]
        assert torch.allclose(mass, torch.ones_like(mass), atol=1e-6)


def test_repeated_oracle_same_state_and_output_bytes():
    case = named_attn_catalog()["csa_selected_c4"]
    op = MqaJointAttentionSinkOp(backend="oracle")
    rows = []
    for repeat in range(4):
        out = op.forward_fp32(
            case.q, case.k, case.v, case.sink, case.plan, compare=True, state_gate=case.state_gate
        )
        rows.append((repeat, out.o.clone()))
    ref = rows[0][1]
    for repeat, o in rows[1:]:
        assert torch.equal(o, ref), f"repeat {repeat} drifted"


def test_long_sequence_attention_finite():
    case = make_attn_case(
        "long",
        layer_type="C4",
        tokens=32,
        n_compressed=32,
        n_recent=128,
        seed=42,
    )
    saved = mqa_joint_attention_sink_fwd(case.q, case.k, case.v, case.sink, case.plan)
    assert saved.o.shape == (32, 64, 512)
    assert torch.isfinite(saved.o).all()
    assert saved.p.shape[-1] == 160
    mass = saved.p.sum(-1) + saved.p_sink
    assert torch.allclose(mass, torch.ones_like(mass), atol=1e-5)


def test_layer_table_c0_c4_c128_independent_recorded_rows():
    """Same recorded Q family, three layer types, no compressor — T06 branch only."""
    op = MqaJointAttentionSinkOp(backend="oracle")
    outs = []
    for layer, n_c, n_r in (("C0", 0, 8), ("C4", 8, 8), ("C128", 2, 8)):
        case = make_attn_case(
            f"table-{layer}",
            layer_type=layer,
            tokens=1,
            n_compressed=n_c,
            n_recent=n_r,
            seed=3,
        )
        out = op.forward_fp32(
            case.q, case.k, case.v, case.sink, case.plan, compare=True, state_gate=case.state_gate
        )
        outs.append(out.o)
        assert torch.isfinite(out.o).all()
    # Different candidate sets must not collapse to the same row by accident.
    assert not torch.equal(outs[0], outs[1])
    assert not torch.equal(outs[1], outs[2])
