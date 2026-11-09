import pytest
import torch

from rl_engine.backends.musa.gemm.det_gemm import MusaDetGemmOp
from rl_engine.kernels.registry import KernelRegistry


def _musa_available() -> bool:
    return hasattr(torch, "musa") and torch.musa.is_available()


def _two_musa_devices_available() -> bool:
    return _musa_available() and torch.musa.device_count() >= 2


def _reference(a, b):
    return (a.float() @ b.float()).to(a.dtype)


def _cpu_fp32_reference(a, b):
    result = a.detach().cpu().float() @ b.detach().cpu().float()
    return result.to(a.device)


@pytest.mark.skipif(not _musa_available(), reason="requires a MUSA device")
@pytest.mark.parametrize("shape", [(1, 7, 5), (31, 70, 65), (128, 128, 128)])
def test_musa_det_gemm_forward_matches_reference(shape):
    m, k, n = shape
    torch.manual_seed(2026)
    a = torch.randn(m, k, device="musa", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="musa", dtype=torch.bfloat16)
    op = MusaDetGemmOp()
    actual = op(a, b)
    expected = _reference(a, b)
    assert torch.allclose(actual.float(), expected.float(), atol=0.25, rtol=0.02)


@pytest.mark.skipif(not _musa_available(), reason="requires a MUSA device")
def test_musa_det_gemm_is_batch_invariant():
    torch.manual_seed(2027)
    a = torch.randn(128, 64, device="musa", dtype=torch.bfloat16)
    b = torch.randn(64, 96, device="musa", dtype=torch.bfloat16)
    op = MusaDetGemmOp()
    full = op(a, b)
    chunks = torch.cat((op(a[:31], b), op(a[31:], b)), dim=0)
    assert torch.equal(full, chunks)


@pytest.mark.skipif(not _musa_available(), reason="requires a MUSA device")
def test_musa_det_gemm_backward_matches_reference():
    torch.manual_seed(2028)
    a = torch.randn(8, 32, device="musa", dtype=torch.bfloat16, requires_grad=True)
    b = torch.randn(32, 24, device="musa", dtype=torch.bfloat16, requires_grad=True)
    grad = torch.randn(8, 24, device="musa", dtype=torch.bfloat16)
    MusaDetGemmOp()(a, b).backward(grad)
    grad_a, grad_b = a.grad.float(), b.grad.float()

    ar = a.detach().float().requires_grad_(True)
    br = b.detach().float().requires_grad_(True)
    (_reference(ar, br).float()).backward(grad.float())
    assert torch.allclose(grad_a, ar.grad, atol=2e-2, rtol=2e-2)
    assert torch.allclose(grad_b, br.grad, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not _musa_available(), reason="requires a MUSA device")
def test_musa_det_gemm_forward_fp32_backward_casts_both_gradients():
    torch.manual_seed(2030)
    a = torch.randn(7, 13, device="musa", dtype=torch.bfloat16, requires_grad=True)
    b = torch.randn(13, 11, device="musa", dtype=torch.bfloat16, requires_grad=True)
    upstream = torch.randn(7, 11, device="musa", dtype=torch.float32)

    MusaDetGemmOp().forward_fp32(a, b).backward(upstream)

    upstream_bf16 = upstream.to(torch.bfloat16).float()
    expected_a = (upstream_bf16 @ b.detach().float().t()).to(torch.bfloat16)
    expected_b = (a.detach().float().t() @ upstream_bf16).to(torch.bfloat16)
    assert torch.allclose(a.grad.float(), expected_a.float(), atol=0.25, rtol=0.02)
    assert torch.allclose(b.grad.float(), expected_b.float(), atol=0.25, rtol=0.02)


@pytest.mark.skipif(not _musa_available(), reason="requires a MUSA device")
@pytest.mark.parametrize(
    ("a_dtype", "b_dtype"),
    [
        (torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.bfloat16),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.float32),
    ],
)
def test_musa_det_gemm_forward_accum_fp32_accepts_fp32_intermediates(a_dtype, b_dtype):
    torch.manual_seed(2031)
    a = torch.randn(5, 9, device="musa", dtype=a_dtype, requires_grad=True)
    b = torch.randn(9, 6, device="musa", dtype=b_dtype, requires_grad=True)
    upstream = torch.randn(5, 6, device="musa", dtype=torch.float32)

    output = MusaDetGemmOp().forward_accum_fp32(a, b)
    assert output.dtype == torch.float32
    torch.testing.assert_close(output, _cpu_fp32_reference(a, b), atol=1e-6, rtol=1e-6)

    output.backward(upstream)
    expected_a = _cpu_fp32_reference(upstream, b.detach().t()).to(a_dtype)
    expected_b = _cpu_fp32_reference(a.detach().t(), upstream).to(b_dtype)
    torch.testing.assert_close(a.grad, expected_a, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(b.grad, expected_b, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not _musa_available(), reason="requires a MUSA device")
@pytest.mark.parametrize(
    ("m", "n", "k", "max_grid_x", "message"),
    [
        (2**62, 4, 1, 2**31 - 1, "output element count"),
        (2**62, 1, 4, 2**31 - 1, "A input element count"),
        (1, 2**62, 4, 2**31 - 1, "B input element count"),
        (1, 513, 0, 2, "required block count"),
    ],
)
def test_musa_det_gemm_launch_validation_rejects_unsafe_shapes(m, n, k, max_grid_x, message):
    from rl_engine import _C

    with pytest.raises(RuntimeError, match=message):
        _C.det_gemm_validate_launch(m, n, k, max_grid_x)


@pytest.mark.skipif(not _musa_available(), reason="requires a MUSA device")
def test_musa_det_gemm_launch_validation_accepts_grid_boundary():
    from rl_engine import _C

    assert _C.det_gemm_validate_launch(1, 512, 0, 2) == 2


@pytest.mark.skipif(not _musa_available(), reason="requires a MUSA device")
def test_musa_det_gemm_linear_weight_layout_and_registry():
    torch.manual_seed(2029)
    a = torch.randn(4, 16, device="musa", dtype=torch.bfloat16)
    weight = torch.randn(9, 16, device="musa", dtype=torch.bfloat16)
    actual = MusaDetGemmOp().linear(a, weight)
    expected = (a.float() @ weight.float().t()).to(a.dtype)
    assert torch.equal(actual, expected)
    assert KernelRegistry().get_op("det_gemm", device="musa").__class__.__name__ == "MusaDetGemmOp"


@pytest.mark.skipif(
    not _two_musa_devices_available(),
    reason="requires two MUSA devices",
)
def test_musa_det_gemm_uses_input_device_when_current_device_differs():
    previous_device = torch.musa.current_device()
    try:
        torch.musa.set_device(0)
        input_device = torch.device("musa:1")
        a = torch.randn(8, 16, device=input_device, dtype=torch.bfloat16)
        b = torch.randn(16, 12, device=input_device, dtype=torch.bfloat16)

        actual = MusaDetGemmOp()(a, b)
        expected = _reference(a, b)

        assert actual.device == input_device
        assert torch.allclose(actual.float(), expected.float(), atol=0.25, rtol=0.02)
        assert torch.musa.current_device() == 0
    finally:
        torch.musa.set_device(previous_device)
