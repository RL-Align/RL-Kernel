# SPDX-License-Identifier: Apache-2.0
"""Contract acceptance for task 28, distinct from routing tasks 29–34."""

import pytest
import torch

from rl_engine.models.deepseek_v4.gate import DeepSeekV4Gate
from rl_engine.ops.moe.router_gate import router_gate_projection

ROCM = torch.version.hip is not None and torch.cuda.is_available()
gpu = pytest.mark.skipif(not ROCM, reason="requires a ROCm GPU")


def project(x, w):
    return router_gate_projection(x, w, model_version="synthetic-task28-v1")


def bits(x):
    return x.contiguous().view(torch.int32)


def layout(x, kind):
    if kind == "transpose":
        return x.T.contiguous().T
    if kind == "padded":
        out = torch.empty((x.shape[0], 2 * x.shape[1] + 3), device=x.device, dtype=x.dtype)
        out = out[:, 1 : 2 * x.shape[1] + 1 : 2]
        out.copy_(x)
        return out
    return x


@gpu
@pytest.mark.parametrize("k,n", [(1, 1), (33, 7), (65, 67), (4096, 256)])
@pytest.mark.parametrize("kind", ["contiguous", "padded", "transpose"])
def test_gate_batch_stride_padding_and_grad_mode_bytes(k, n, kind):
    torch.manual_seed(28)
    row = torch.randn(1, k, dtype=torch.bfloat16).cuda()
    w = layout(torch.randn(n, k, dtype=torch.bfloat16).cuda(), kind)
    expected = project(row, w).logits
    for m in (1, 16, 17, 32, 65):
        x = torch.randn(m, k, dtype=torch.bfloat16).cuda()
        indices = sorted({0, m // 2, m - 1})
        x[indices] = row
        x = layout(x, kind).requires_grad_()
        train = project(x, w)
        with torch.inference_mode():
            infer = project(x, w)
        assert train.logits.dtype == torch.float32
        assert torch.equal(bits(train.logits), bits(infer.logits))
        for idx in indices:
            assert torch.equal(bits(train.logits[idx]), bits(expected[0]))
        assert train.execution["config"] == project(row, w).execution["config"]
        ref = x.detach().double() @ w.double().T
        torch.testing.assert_close(train.logits.double(), ref, atol=2e-4, rtol=3e-5)


@gpu
def test_close_scores_are_not_bf16_then_cast_to_fp32():
    # 1 + 2^-17 survives FP32 but not BF16. This detects premature rounding.
    row = torch.zeros((1, 4096), dtype=torch.bfloat16, device="cuda")
    row[0, :2] = torch.tensor([1.0, 2.0**-10], device="cuda")
    w = torch.zeros((256, 4096), dtype=torch.bfloat16, device="cuda")
    w[:2, 0] = 1
    w[1, 1] = 2.0**-7
    for m in (1, 16, 17):
        x = row.repeat(m, 1)
        scores = project(x, w).logits
        assert scores[0, 0].item() == 1.0
        assert scores[0, 1].item() == 1.0 + 2.0**-17
        assert scores[0, 0].bfloat16() == scores[0, 1].bfloat16()
        assert torch.equal(bits(scores), bits(scores[0:1].expand_as(scores)))


@gpu
@pytest.mark.parametrize("m", [0, 1, 16, 17])
def test_gate_gradients_keep_fp32_upstream_and_repeat_deterministically(m):
    torch.manual_seed(281)
    x = torch.randn(m, 33, device="cuda", dtype=torch.bfloat16).requires_grad_()
    w = torch.randn(7, 33, device="cuda", dtype=torch.bfloat16).requires_grad_()
    grad = torch.randn(m, 7, device="cuda", dtype=torch.float32)
    expected_dx = (grad.double() @ w.detach().double()).to(torch.bfloat16)
    expected_dw = (grad.double().T @ x.detach().double()).to(torch.bfloat16)
    first = torch.autograd.grad(project(x, w).logits, (x, w), grad)
    second = torch.autograd.grad(project(x, w).logits, (x, w), grad)
    for a, b, ref in zip(first, second, (expected_dx, expected_dw), strict=True):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16))
        torch.testing.assert_close(a, ref, atol=2e-3, rtol=8e-3)


@gpu
def test_adapter_uses_live_parameter_after_in_place_version_update():
    w = torch.nn.Parameter(torch.ones(7, 33, dtype=torch.bfloat16, device="cuda"))
    x = torch.ones(17, 33, dtype=torch.bfloat16, device="cuda")
    gate = DeepSeekV4Gate(w, model_version="v1")
    assert gate.weight is w
    before = gate(x)
    with torch.no_grad():
        w.data.mul_(2)  # Deliberately does not update the Parameter version counter.
    gate.model_version = "v2"
    after = gate(x)
    assert torch.equal(after, before * 2)
    assert gate.last_execution["model_version"] == "v2"


@gpu
def test_graph_rejected_explicitly():
    x = torch.ones(1, 33, device="cuda", dtype=torch.bfloat16)
    w = torch.ones(7, 33, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="Graph capture"):
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            project(x, w)


@pytest.mark.parametrize(
    "x,w,error",
    [
        (torch.ones(2, 3, 4), torch.ones(7, 4), ValueError),
        (torch.ones(2, 3), torch.ones(7, 4), ValueError),
        (torch.ones(2, 3), torch.ones(7, 3), TypeError),
        (
            torch.ones(2, 3, dtype=torch.bfloat16),
            torch.ones(7, 3, dtype=torch.bfloat16),
            RuntimeError,
        ),
    ],
)
def test_unsupported_inputs_fail_before_backend_import(x, w, error):
    with pytest.raises(error):
        project(x, w)


def test_model_version_required():
    with pytest.raises(ValueError, match="model_version"):
        router_gate_projection(torch.empty(0), torch.empty(0), model_version="")


@gpu
def test_unsupported_dtype_and_expanded_stride():
    x = torch.ones(1, 33, dtype=torch.bfloat16, device="cuda")
    w = torch.ones(7, 33, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(TypeError, match="BF16"):
        project(x, w.float())
    with pytest.raises(ValueError, match="positive strides"):
        project(x.expand(17, -1), w)


@gpu
def test_aiter_failure_propagates_without_a_fallback(monkeypatch):
    from rl_engine.backends.rocm.gemm import aiter_fixed_gemm

    def unavailable():
        raise ImportError("test missing AITER")

    monkeypatch.setattr(aiter_fixed_gemm, "_implementation", unavailable)
    x = torch.ones(1, 33, dtype=torch.bfloat16, device="cuda")
    w = torch.ones(7, 33, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ImportError, match="missing AITER"):
        project(x, w)


@pytest.mark.cuda_only
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires NVIDIA CUDA for the two upstream reference entries",
)
def test_two_cuda_reference_entries(record_property):
    from rl_engine.reference.moe.router_gate import cuda_gate_reference

    torch.manual_seed(28)
    w = torch.randn(256, 4096, device="cuda", dtype=torch.bfloat16)
    row = torch.randn(1, 4096, device="cuda", dtype=torch.bfloat16)
    baseline = cuda_gate_reference(row, w, entry="mm")
    for m in (1, 16, 17):
        x = row.repeat(m, 1)
        mm = cuda_gate_reference(x, w, entry="mm")
        assert mm.dtype == torch.float32
        record_property(
            f"mm_m{m}_row_bit_mismatches", int((bits(mm[0]) != bits(baseline[0])).sum().item())
        )
        torch.testing.assert_close(mm, x.float() @ w.float().T, atol=2e-3, rtol=3e-5)
        if m <= 16:
            cute = cuda_gate_reference(x, w, entry="cute")
            assert cute.dtype == torch.float32
            record_property(
                f"cute_vs_mm_m{m}_bit_mismatches", int((bits(cute) != bits(mm)).sum().item())
            )
            torch.testing.assert_close(cute, mm, atol=2e-3, rtol=3e-5)
