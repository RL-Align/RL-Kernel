import pytest
import torch

from rl_engine.kernels.ops.musa.matmul.det_gemm import MusaDetGemmOp
from rl_engine.kernels.registry import KernelRegistry


def _musa_available() -> bool:
    return hasattr(torch, "musa") and torch.musa.is_available()


def _reference(a, b):
    return (a.float() @ b.float()).to(a.dtype)


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
def test_musa_det_gemm_linear_weight_layout_and_registry():
    torch.manual_seed(2029)
    a = torch.randn(4, 16, device="musa", dtype=torch.bfloat16)
    weight = torch.randn(9, 16, device="musa", dtype=torch.bfloat16)
    actual = MusaDetGemmOp().linear(a, weight)
    expected = (a.float() @ weight.float().t()).to(a.dtype)
    assert torch.equal(actual, expected)
    assert KernelRegistry().get_op("det_gemm", device="musa").__class__.__name__ == "MusaDetGemmOp"
