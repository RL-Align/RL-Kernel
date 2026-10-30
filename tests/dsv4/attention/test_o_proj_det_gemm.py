# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Exercise OProjGroupedOp with the available native or JIT DetGemm runtime."""

from types import SimpleNamespace

import pytest
import torch

from rl_engine.kernels.ops import base
from rl_engine.kernels.ops.cuda.matmul import det_gemm
from rl_engine.kernels.dsv4.attention import cuda_runtime
from rl_engine.kernels.dsv4.attention import mqa_joint_attention_sink as mqa
from rl_engine.kernels.dsv4.attention.contract import (
    CONCAT_Z_DIM,
    GROUP_FLAT_DIM,
    HEAD_DIM,
    HEADS_PER_GROUP,
    HIDDEN_SIZE,
    N_O_PROJ_GROUPS,
    N_Q_HEADS,
    O_LORA_RANK,
)
from rl_engine.kernels.dsv4.attention.errors import DSv4FailClosedError, DSv4Status
from rl_engine.kernels.dsv4.attention.fixtures.catalog import make_oproj_case
from rl_engine.kernels.dsv4.attention.o_proj.o_proj_grouped import OProjGroupedOp, _det_gemm_linear
from rl_engine.kernels.dsv4.attention.o_proj.rope_consumer import fixture_cos_sin

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")


@pytest.mark.parametrize("backend", ["det_gemm", "auto"])
@pytest.mark.parametrize("field", ["w_a", "w_b", "cos", "sin"])
def test_cpu_operand_rejected_before_backend_load(monkeypatch, backend, field):
    inputs = {name: torch.empty(0, device="cuda") for name in ("o", "w_a", "w_b", "cos", "sin")}
    inputs[field] = torch.empty(0)
    op = OProjGroupedOp(backend=backend)

    def unexpected_backend(*args):
        pytest.fail("backend loaded before device validation")

    monkeypatch.setattr(op, "_linear", unexpected_backend)
    with pytest.raises(DSv4FailClosedError, match="same device") as exc:
        op.forward(**inputs)
    assert exc.value.status is DSv4Status.SCHEMA_MISMATCH


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
def test_mixed_cuda_devices_rejected_before_gemm(monkeypatch):
    op = OProjGroupedOp(backend="det_gemm")

    def unexpected_backend(*args):
        pytest.fail("backend loaded before device validation")

    monkeypatch.setattr(op, "_linear", unexpected_backend)
    with pytest.raises(DSv4FailClosedError, match="same device"):
        op.forward(
            torch.empty(0, device="cuda:1"), torch.empty(0, device="cuda:0"),
            *(torch.empty(0, device="cuda:1") for _ in range(3)),
        )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
def test_linear_uses_noncurrent_cuda_device(det_gemm_runtime):
    with torch.cuda.device(1):
        try:
            _det_gemm_linear()
        except DSv4FailClosedError as exc:
            if "refusing naive fallback" in str(exc) or torch.cuda.get_device_capability(1)[0] != 9:
                pytest.skip("strict DetGemm unavailable on GPU 1")
            raise
    x = torch.ones(32, 64, device="cuda:1", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.ones(64, 64, device="cuda:1", dtype=torch.bfloat16, requires_grad=True)
    with torch.cuda.device(0):
        linear, _, _ = OProjGroupedOp(backend="det_gemm")._linear(x)
        y = linear(x, weight)
        y.sum().backward()
        assert torch.cuda.current_device() == 0
    assert y.device == x.grad.device == weight.grad.device == x.device
    torch.testing.assert_close(y, torch.full_like(y, 64), rtol=0, atol=0)
    torch.testing.assert_close(x.grad, torch.full_like(x, 64), rtol=0, atol=0)
    torch.testing.assert_close(weight.grad, torch.full_like(weight, 32), rtol=0, atol=0)


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


@pytest.mark.parametrize("backend", ["det_gemm", "auto"])
@pytest.mark.parametrize("missing", ["det_gemm_fwd", "det_gemm_db_transposed"])
def test_partial_det_gemm_is_rejected_before_cuda_forward(
    det_gemm_runtime, monkeypatch, backend, missing
):
    native = SimpleNamespace(
        **{
            name: getattr(det_gemm_runtime, name)
            for name in (
                "det_gemm_fwd_rhs_transposed",
                "det_gemm_fwd",
                "det_gemm_db_transposed",
            )
            if name != missing
        }
    )

    def failed_jit():
        raise RuntimeError("JIT unavailable")

    monkeypatch.setattr(cuda_runtime, "ensure_native_kernels", failed_jit)
    monkeypatch.setattr(base, "_C", native)
    monkeypatch.setattr(base, "_EXT_AVAILABLE", True)
    monkeypatch.setattr(det_gemm, "_C", native)
    monkeypatch.setattr(det_gemm, "_EXT_AVAILABLE", True)
    case = make_oproj_case("partial-native", tokens=1, device="cuda")
    with pytest.raises(DSv4FailClosedError) as exc:
        OProjGroupedOp(backend=backend).forward(
            case.o.bfloat16(), case.w_a.bfloat16(), case.w_b.bfloat16(), case.cos, case.sin
        )
    assert exc.value.status is DSv4Status.UNSUPPORTED_CAPABILITY


@pytest.mark.usefixtures("strict_det_gemm")
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
    with pytest.raises(DSv4FailClosedError) as exc:
        OProjGroupedOp(backend="det_gemm").forward(
            case.o.cuda(),
            case.w_a.cuda(),
            case.w_b.cuda(),
            case.cos.cuda(),
            case.sin.cuda(),
        )
    assert exc.value.status is DSv4Status.ROUND_POINT_MISMATCH


@pytest.mark.usefixtures("strict_det_gemm")
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
    """Independent pairwise RoPE reference."""
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


@pytest.mark.usefixtures("strict_det_gemm")
def test_o_proj_det_gemm_op_backward():
    # Sparse weights make activation VJPs independently calculable.
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
            # Compare to a double-precision reference within FP32 roundoff.
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
    # Explicit weight VJPs retain FP32 dY and output.
    assert torch.any(d_w_a != w_a.grad.float())
    assert torch.any(d_w_b != w_b.grad.float())
    assert torch.any(d_w_b.bfloat16() != w_b.grad)


@pytest.mark.usefixtures("strict_det_gemm")
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


@pytest.mark.parametrize("backend", ["det_gemm", "auto"])
def test_uncompiled_sm90_rejected_without_cuda_gemm(det_gemm_runtime, monkeypatch, backend):
    def unexpected_gemm(*args):
        pytest.fail("native GEMM launched without compiled SM90 support")

    monkeypatch.setenv("RL_KERNEL_DET_GEMM_BACKEND", "sm90")
    monkeypatch.setattr(det_gemm_runtime, "det_gemm_sm90_compiled", lambda: False)
    for name in ("det_gemm_fwd_rhs_transposed", "det_gemm_fwd", "det_gemm_db_transposed"):
        monkeypatch.setattr(det_gemm_runtime, name, unexpected_gemm)
    with pytest.raises(DSv4FailClosedError, match="refusing naive fallback") as exc:
        OProjGroupedOp(backend=backend)._linear(torch.ones(1, device="cuda", dtype=torch.bfloat16))
    assert exc.value.status is DSv4Status.UNSUPPORTED_CAPABILITY
    torch.cuda.synchronize()
