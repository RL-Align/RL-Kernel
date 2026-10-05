# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import torch

from rl_engine.kernels.p2.contract import (
    HEAD_DIM,
    MAIN_ROTARY_END,
    MAIN_ROTARY_START,
    N_O_PROJ_GROUPS,
)
from rl_engine.kernels.p2.fixtures.catalog import make_oproj_case
from rl_engine.kernels.p2.o_proj.o_proj_grouped import OProjGroupedOp
from rl_engine.kernels.p2.o_proj.oracle import o_proj_grouped_fwd, sequential_linear, split_groups
from rl_engine.kernels.p2.o_proj.rope_consumer import (
    apply_gptj_interleaved_partial,
    fixture_cos_sin,
)


def test_inverse_rope_is_out_of_place():
    case = make_oproj_case("rope", tokens=1, seed=0)
    original = case.o.clone()
    out = apply_gptj_interleaved_partial(case.o, case.cos, case.sin, inverse=True)
    assert torch.equal(case.o, original)
    assert out.data_ptr() != case.o.data_ptr()
    nope = slice(0, MAIN_ROTARY_START)
    assert torch.equal(out[..., nope], original[..., nope])
    assert torch.equal(out[..., MAIN_ROTARY_END:], original[..., MAIN_ROTARY_END:])


def test_inverse_then_forward_roundtrip():
    case = make_oproj_case("rt", tokens=2, seed=1)
    inv = apply_gptj_interleaved_partial(case.o, case.cos, case.sin, inverse=True)
    back = apply_gptj_interleaved_partial(inv, case.cos, case.sin, inverse=False)
    assert torch.allclose(back, case.o, atol=1e-5, rtol=1e-5)


def test_group_order_identifiable():
    case = make_oproj_case("id", tokens=1, identifiable_groups=True)
    saved = o_proj_grouped_fwd(case.o, case.w_a, case.w_b, case.cos, case.sin)
    for group in range(N_O_PROJ_GROUPS):
        assert torch.allclose(saved.y[0, group], torch.tensor(float(group + 1)), atol=1e-5)


def test_wrong_concat_changes_output():
    case = make_oproj_case("concat", tokens=1, identifiable_groups=True)
    saved = o_proj_grouped_fwd(case.o, case.w_a, case.w_b, case.cos, case.sin)
    swapped = torch.cat([saved.z_groups[1], saved.z_groups[0], *saved.z_groups[2:]], dim=-1)
    y_wrong = sequential_linear(swapped, case.w_b)
    assert not torch.allclose(saved.y, y_wrong, atol=1e-6)


def test_o_proj_forward_and_backward():
    case = make_oproj_case("bwd", tokens=1, seed=4)
    op = OProjGroupedOp(backend="oracle")
    result = op.forward_fp32(case.o, case.w_a, case.w_b, case.cos, case.sin)
    assert result.y.shape == (1, 4096)
    assert torch.isfinite(result.y).all()
    d_o, d_w_a, d_w_b = op.backward(torch.ones_like(result.y), result.saved, case.cos, case.sin)
    assert d_o.shape == case.o.shape
    assert d_w_a.shape == case.w_a.shape
    assert d_w_b.shape == case.w_b.shape
    assert torch.isfinite(d_o).all()


def test_split_eight_groups():
    o = torch.arange(64 * HEAD_DIM, dtype=torch.float32).reshape(1, 64, HEAD_DIM)
    groups = split_groups(o)
    assert len(groups) == 8
    assert groups[0].shape == (1, 4096)
    assert torch.equal(groups[0][0, :HEAD_DIM], o[0, 0])
    assert torch.equal(groups[7][0, :HEAD_DIM], o[0, 56])


def test_rope_autograd_through_inverse():
    case = make_oproj_case("ag", tokens=1, seed=2)
    o = case.o.clone().requires_grad_(True)
    y = apply_gptj_interleaved_partial(o, case.cos, case.sin, inverse=True)
    assert y.data_ptr() != o.data_ptr()
    y.sum().backward()
    assert o.grad is not None and torch.isfinite(o.grad).all()


def test_catalog_checksums_are_seed_stable():
    from rl_engine.kernels.p2.fixtures.catalog import catalog_checksums

    a = catalog_checksums()
    b = catalog_checksums()
    assert a == b
    assert "c0_recent_127" in a and "csa_prefix_512" in a


def test_fixture_tables_are_fp32():
    cos, sin = fixture_cos_sin(torch.arange(3))
    assert cos.dtype == torch.float32 and sin.dtype == torch.float32
    assert cos.shape == (3, 32)
