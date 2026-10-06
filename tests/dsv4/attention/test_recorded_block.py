# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Recorded full-chain T06 verification: attention -> o-proj and long seq."""

from __future__ import annotations

import pytest
import torch

from rl_engine.kernels.dsv4.attention.mqa_joint_attention_sink import MqaJointAttentionSinkOp
from rl_engine.kernels.dsv4.attention.oracle import mqa_joint_attention_sink_fwd
from rl_engine.kernels.dsv4.attention.block import recorded_attention_block
from rl_engine.kernels.dsv4.attention.contract import HIDDEN_SIZE
from rl_engine.kernels.dsv4.attention.cuda_runtime import ensure_t06_cuda_kernel
from rl_engine.kernels.dsv4.attention.fixtures.catalog import (
    make_attn_case,
    make_oproj_case,
    named_attn_catalog,
)
from rl_engine.kernels.dsv4.attention.o_proj.o_proj_grouped import OProjGroupedOp


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
    """Different candidate sets must produce different attention rows."""
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
    assert not torch.equal(outs[0], outs[1])
    assert not torch.equal(outs[1], outs[2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
@pytest.mark.parametrize("o_proj_backend", ["det_gemm", "auto"])
@pytest.mark.usefixtures("strict_det_gemm")
def test_recorded_block_cuda_rounds_attention_at_det_gemm_boundary(o_proj_backend):
    ensure_t06_cuda_kernel()
    case = make_attn_case(
        "block-cuda",
        layer_type="C4",
        tokens=2,
        n_compressed=2,
        n_recent=2,
        seed=31,
        device="cuda",
        dtype=torch.bfloat16,
    )
    projection = make_oproj_case("block-cuda-projection", tokens=2, seed=32, device="cuda")
    w_a, w_b = projection.w_a.bfloat16(), projection.w_b.bfloat16()
    result = recorded_attention_block(
        case.q,
        case.k,
        case.v,
        case.sink,
        case.plan,
        w_a,
        w_b,
        projection.cos,
        projection.sin,
        case.state_gate,
        attn_backend="cuda",
        o_proj_backend=o_proj_backend,
    )
    assert result.attention.provenance.backend == "cuda"
    assert result.attention.o.dtype == torch.float32
    assert result.o_proj.provenance.backend == "det_gemm"
    assert result.o_proj.saved.o.dtype == torch.bfloat16
    assert torch.equal(result.o_proj.saved.o, result.attention.o.bfloat16())
    assert result.y.shape == (2, HIDDEN_SIZE)
    assert result.y.dtype == torch.float32
    assert torch.isfinite(result.y).all()
    expected = OProjGroupedOp(backend="det_gemm").forward_fp32(
        result.attention.o.bfloat16(), w_a, w_b, projection.cos, projection.sin
    )
    assert torch.equal(result.y, expected.y)
