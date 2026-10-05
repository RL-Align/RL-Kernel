# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Exercise OProjGroupedOp with the available native or JIT DetGemm runtime."""

import pytest
import torch

from rl_engine.kernels.ops import base
from rl_engine.kernels.p2.attention import mqa_joint_attention_sink as mqa
from rl_engine.kernels.p2.contract import (
    CONCAT_Z_DIM,
    GROUP_FLAT_DIM,
    HEAD_DIM,
    HEADS_PER_GROUP,
    HIDDEN_SIZE,
    N_O_PROJ_GROUPS,
    N_Q_HEADS,
    O_LORA_RANK,
)
from rl_engine.kernels.p2.errors import P2FailClosedError, P2Status
from rl_engine.kernels.p2.fixtures.catalog import make_oproj_case
from rl_engine.kernels.p2.o_proj.o_proj_grouped import OProjGroupedOp
from rl_engine.kernels.p2.o_proj.rope_consumer import fixture_cos_sin

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")


@pytest.fixture(scope="module", autouse=True)
def det_gemm_runtime():
    required = (
        "det_gemm_fwd_rhs_transposed", "det_gemm_fwd", "det_gemm_db_transposed",
        "det_gemm_sm90_compiled",
    )
    for native in (base._C, mqa._C):
        if native is not None and all(hasattr(native, name) for name in required):
            return native
    pytest.skip("DetGemm native or JIT runtime is unavailable")


def test_runtime_extension_exposes_det_gemm_symbols(det_gemm_runtime):
    assert isinstance(det_gemm_runtime.det_gemm_sm90_compiled(), bool)


def test_o_proj_det_gemm_runs_on_available_runtime():
    case = make_oproj_case("det", tokens=1, seed=0)
    o = case.o.cuda().to(torch.bfloat16)
    w_a = case.w_a.cuda().to(torch.bfloat16)
    w_b = case.w_b.cuda().to(torch.bfloat16)
    cos = case.cos.cuda()
    sin = case.sin.cuda()
    result = OProjGroupedOp(backend="det_gemm").forward(o, w_a, w_b, cos, sin)
    assert result.y.shape == (1, HIDDEN_SIZE)
    assert result.y.dtype == torch.bfloat16
    assert torch.isfinite(result.y).all()
    assert result.provenance.backend == "det_gemm"
    ref = OProjGroupedOp(backend="torch_fp32").forward_fp32(
        o.float(), w_a.float(), w_b.float(), cos, sin
    )
    delta = (result.y.float() - ref.y.float()).abs()
    assert float(delta.max()) < 5e-3
    assert o.data_ptr() == result.saved.o.data_ptr()


def test_o_proj_det_gemm_refuses_silent_fp32_downcast():
    case = make_oproj_case("fp32", tokens=1, seed=1)
    with pytest.raises(P2FailClosedError) as exc:
        OProjGroupedOp(backend="det_gemm").forward(
            case.o.cuda(),
            case.w_a.cuda(),
            case.w_b.cuda(),
            case.cos.cuda(),
            case.sin.cuda(),
        )
    assert exc.value.status is P2Status.ROUND_POINT_MISMATCH


def test_o_proj_det_gemm_forward_fp32_returns_fp32():
    case = make_oproj_case("fp32out", tokens=1, seed=3)
    o = case.o.cuda().to(torch.bfloat16)
    w_a = case.w_a.cuda().to(torch.bfloat16)
    w_b = case.w_b.cuda().to(torch.bfloat16)
    result = OProjGroupedOp(backend="det_gemm").forward_fp32(
        o, w_a, w_b, case.cos.cuda(), case.sin.cuda()
    )
    assert result.y.dtype == torch.float32
    assert result.y.shape == (1, HIDDEN_SIZE)
    assert torch.isfinite(result.y).all()


def _reference_rotation(x, cos, sin, *, inverse):
    """Independent pair formula; never call the production RoPE consumer."""
    result = x.float().clone()
    even, odd = x[..., 448:512:2].float(), x[..., 449:512:2].float()
    c, s = cos[:, None], sin[:, None]
    if inverse:
        result[..., 448:512:2] = even * c + odd * s
        result[..., 449:512:2] = odd * c - even * s
    else:
        result[..., 448:512:2] = even * c - odd * s
        result[..., 449:512:2] = odd * c + even * s
    return result


def test_o_proj_det_gemm_op_backward():
    # Full contract shapes, three tokens, and nontrivial RoPE. Sparse weights
    # give independently calculable activation VJPs without importing the
    # DetGemm reduction tree into this test's reference implementation.
    tokens = 3
    o = torch.arange(tokens * N_Q_HEADS * HEAD_DIM, device="cuda", dtype=torch.float32)
    o = ((o.remainder(251) - 125) / 512).reshape(tokens, N_Q_HEADS, HEAD_DIM)
    o = o.bfloat16().requires_grad_()
    w_a = torch.zeros(
        N_O_PROJ_GROUPS, O_LORA_RANK, GROUP_FLAT_DIM, device="cuda", dtype=torch.bfloat16
    )
    w_b = torch.zeros(HIDDEN_SIZE, CONCAT_Z_DIM, device="cuda", dtype=torch.bfloat16)
    for group in range(N_O_PROJ_GROUPS):
        w_a[group, 0, 448] = 0.5 if group % 2 else 1
        w_a[group, 1, 449] = 0.25
        w_b[2 * group, group * O_LORA_RANK] = 1
        w_b[2 * group + 1, group * O_LORA_RANK + 1] = 0.5
    w_a.requires_grad_()
    w_b.requires_grad_()
    cos, sin = fixture_cos_sin(torch.tensor([1, 7, 19], device="cuda"))
    # These FP32 gradients deliberately differ from their BF16 conversion.
    d_y = torch.arange(tokens * HIDDEN_SIZE, device="cuda", dtype=torch.float32)
    d_y = ((d_y.remainder(47) + 1025) / 1024).reshape(tokens, HIDDEN_SIZE)
    op = OProjGroupedOp(backend="det_gemm")
    result = op.forward(o, w_a, w_b, cos, sin)

    with torch.no_grad():
        d_o, d_w_a, d_w_b = op.backward(d_y, result.saved, cos, sin)
        rotated = _reference_rotation(o, cos, sin, inverse=True).bfloat16()
        z = torch.zeros(tokens, CONCAT_Z_DIM, device="cuda", dtype=torch.bfloat16)
        d_z = torch.zeros_like(z)
        d_rotated = torch.zeros_like(o, dtype=torch.float32)
        expected_w_a = torch.zeros_like(w_a, dtype=torch.float32)
        rounded_d_y = d_y.bfloat16().float()
        for group in range(N_O_PROJ_GROUPS):
            head = group * HEADS_PER_GROUP
            rank = group * O_LORA_RANK
            scale = 0.5 if group % 2 else 1
            z[:, rank] = rotated[:, head, 448] * scale
            z[:, rank + 1] = rotated[:, head, 449] * 0.25
            d_z[:, rank] = rounded_d_y[:, 2 * group]
            d_z[:, rank + 1] = rounded_d_y[:, 2 * group + 1] * 0.5
            d_rotated[:, head, 448] = d_z[:, rank].float() * scale
            d_rotated[:, head, 449] = d_z[:, rank + 1].float() * 0.25
            group_input = rotated[:, head : head + HEADS_PER_GROUP].reshape(tokens, -1)
            # A double matrix product is independent of the sequential
            # production VJP; allow only FP32 accumulation roundoff below.
            expected_w_a[group] = (
                d_z[:, rank : rank + O_LORA_RANK].double().T @ group_input.double()
            ).float()
        expected_o = _reference_rotation(d_rotated, cos, sin, inverse=False)
        expected_w_b = (d_y.double().T @ z.double()).float()
        torch.testing.assert_close(result.saved.o_tilde, rotated, rtol=0, atol=0)
        torch.testing.assert_close(result.saved.z, z, rtol=0, atol=0)
        assert d_o.dtype == d_w_a.dtype == d_w_b.dtype == torch.float32
        torch.testing.assert_close(d_o, expected_o, rtol=0, atol=0)
        torch.testing.assert_close(d_w_a, expected_w_a, rtol=0, atol=1e-7)
        torch.testing.assert_close(d_w_b, expected_w_b, rtol=0, atol=1e-7)

    result.y.backward(d_y.bfloat16())
    assert o.grad.dtype == w_a.grad.dtype == w_b.grad.dtype == torch.bfloat16
    torch.testing.assert_close(o.grad, expected_o.bfloat16(), rtol=0, atol=0)
    torch.testing.assert_close(w_a.grad, expected_w_a.bfloat16(), rtol=0, atol=0)
    expected_autograd_w_b = (rounded_d_y.double().T @ z.double()).bfloat16()
    torch.testing.assert_close(w_b.grad, expected_autograd_w_b, rtol=0, atol=0)
    # Explicit dW_a retains FP32 output; explicit dW_b also retains the original
    # FP32 dY. Neither is the BF16 weight VJP returned by DetGemm autograd.
    assert torch.any(d_w_a != w_a.grad.float())
    assert torch.any(d_w_b != w_b.grad.float())
    assert torch.any(d_w_b.bfloat16() != w_b.grad)


def test_o_proj_det_gemm_backward_finite():
    case = make_oproj_case("bwd", tokens=1, seed=2)
    o = case.o.cuda().to(torch.bfloat16).clone().requires_grad_(True)
    w_a = case.w_a.cuda().to(torch.bfloat16).clone().requires_grad_(True)
    w_b = case.w_b.cuda().to(torch.bfloat16)
    y = OProjGroupedOp(backend="det_gemm").forward(
        o, w_a, w_b, case.cos.cuda(), case.sin.cuda()
    ).y
    y.float().sum().backward()
    assert o.grad is not None and torch.isfinite(o.grad).all()
    assert w_a.grad is not None and torch.isfinite(w_a.grad).all()
